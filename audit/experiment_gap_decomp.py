"""Measure the variational gap of the joint GTM instead of inferring it from bias.

The residual joint-coefficient bias (0.024 at bias-min, single100k) must be
KL(q || true posterior) > 0 (Hansen: at KL=0 the ELBO maximizer IS the marginal MLE,
so bias -> 0 mod O(1/sqrt N)). This script MEASURES that gap and decomposes it
(Cremer et al. 2018, "Inference Suboptimality in VAEs"):

    total gap        = log p(x) - ELBO_amortized
    amortization gap = ELBO_refined - ELBO_amortized   (q optimized per-doc, same family)
    approximation gap= log p(x)    - ELBO_refined       (best q in family still misses)
    total = amortization + approximation

log p(x) is estimated by IWAE with large K using the REFINED q (tightest available).
All ELBO/IWAE terms reuse the model's own _recon_loglik / _posterior_loglik / prior, and
the encoder's _mog_unpack / _mog_sample, so the numbers are the model's, not a re-derivation.

Decision rule:
  total gap ~ 0           -> bound already tight, KL~0: the bias is NOT the variational gap
                            (look to finite-N, identification/centering, or lambda-optim).
  gap large, AMORT-heavy  -> semi-amortized VI / better encoder optimization is the lever.
  gap large, APPROX-heavy -> the family can't reach the posterior; DReG-IWAE territory.

Set DECOMP_SMOKE=1 for a ~2-min end-to-end validation on tiny config.
Units: nats/doc (and nats/word = /L). Refit [256,256] MoG-20 joint to the bias-min."""
import os, json, time, numpy as np, scipy.sparse, tempfile, torch
import torch.nn.functional as F
from torch.distributions import Normal, MultivariateNormal
from scipy.optimize import linear_sum_assignment
from sklearn.feature_extraction.text import CountVectorizer
from deeplatent import Corpus, GTM, generate_documents

SMOKE = os.environ.get("DECOMP_SMOKE") == "1"
K, C, VOCAB, L = 5, 3, 200, 10
HIDDEN, COMP = [256, 256], 20
SEED = 1000
OPTIM = {"main": {"lr": 1e-3, "weight_decay": 0.0},
         "prior": {"lr": 1e-4, "weight_decay": 0.0}}
if SMOKE:
    N, STEPS, M_SAMPLE, K_LIST, REFINE_STEPS, REFINE_MC, EVAL_K, REFINE_LR = 3000, 6000, 200, [1, 5, 25], 300, 16, 50, 0.01
else:
    N, STEPS, M_SAMPLE, K_LIST, REFINE_STEPS, REFINE_MC, EVAL_K, REFINE_LR = 100000, 60000, 3000, [1, 5, 25, 100], 400, 24, 200, 0.01
OUT = "audit/results_gap_decomp.json"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

lambda_fixed = (np.random.default_rng(7).standard_normal((C + 1, K)) * 0.5)
Lc = (lambda_fixed.T - lambda_fixed.T.mean(0, keepdims=True))

def align(tt, et):
    r, c = linear_sum_assignment(-(tt.T @ et)); return {int(i): int(j) for i, j in zip(r, c)}
def center_rows(m): return m - m.mean(0, keepdims=True)
def densify(corpus):
    mm = corpus.processed_modalities["text"]["bow"]["matrix"]
    if scipy.sparse.issparse(mm):
        corpus.processed_modalities["text"]["bow"]["matrix"] = np.asarray(mm.todense(), np.float32)

# ---------------- data + refit ----------------
print(f"{'[SMOKE] ' if SMOKE else ''}gap decomposition | N={N}, {L} w/doc, {HIDDEN} MoG-{COMP}, refit {STEPS} steps", flush=True)
dft, df, tw, lam, _ = generate_documents(num_docs=N, num_topics=K, vocab_size=VOCAB, num_covs=C,
    doc_topic_prior="logistic_normal", min_words=L, max_words=L, lambda_=lambda_fixed, random_seed=SEED)
true_theta = dft[[f"Topic{i}" for i in range(K)]].values.astype(np.float64)
vec = CountVectorizer(); vec.fit(df["doc_clean_0"])
mods = {"text": {"column": "doc_clean_0", "views": {"bow": {"type": "bow", "vectorizer": vec}}}}
corpus = Corpus(df, modalities=mods, prevalence="~ cov_1 + cov_2 + cov_3"); densify(corpus)

t0 = time.time()
model = GTM(train_data=corpus, n_topics=K, ae_type="vae", vi_type="mixture_of_gaussians",
            mixture_components=COMP, doc_topic_prior="logistic_normal", update_prior=True, w_prior=1.0,
            encoder_args={"text_bow": {"hidden_dims": HIDDEN}}, decoder_args={"text_bow": {"hidden_dims": []}},
            batch_size=256, num_steps=STEPS, num_workers=0 if SMOKE else 4, print_every_n_steps=10**9,
            optim_args=OPTIM, return_best_model=False, ckpt_folder=tempfile.mkdtemp(), seed=SEED, device=device)
print(f"refit done ({time.time()-t0:.0f}s)", flush=True)

# sanity: reproduce the baseline bias
th = model.get_doc_topic_distribution(corpus, num_samples=10).astype(np.float64)
mp = align(true_theta, th)
W = model.prior.mean_net.weight.detach().cpu().numpy().astype(np.float64).copy()
W[:, 0] += model.prior.mean_net.bias.detach().cpu().numpy().astype(np.float64)
Wc = center_rows(np.stack([W[mp[t]] for t in range(K)]))
bias = float(np.abs((Wc - Lc)[:, 1:]).mean())
print(f"sanity: mean|bias|={bias:.3f} (baseline bias-min ~0.024)", flush=True)

model.encoder.eval(); model.decoders.eval(); model.prior.eval()
for module in (model.encoder, model.decoders, model.prior):   # refinement optimizes ONLY the per-doc
    if module is not None:                                     # variational params; freeze the model so
        for p in module.parameters():                         # repeated backward doesn't reuse its graph
            p.requires_grad_(False)
enc = model.encoder
has_full_cov = hasattr(model.prior, "sigma")

# ---------------- shared ELBO/IWAE machinery (the model's own pieces) ----------------
def batch_context(data):
    for k, v in data.items():
        if isinstance(v, torch.Tensor): data[k] = v.to(device)
    prevalence = data.get("M_prevalence_covariates", None)
    content = data.get("M_content_covariates", None)
    modality_inputs = {}
    for key in enc.encoders.keys():
        mod, view = key.split("_", 1) if "_" in key else (key, key)
        vd = data["modalities"]["text"]["bow"]            # single bow modality in this experiment
        x = vd.to(device)
        if prevalence is not None: x = torch.cat([x, prevalence], dim=1)
        modality_inputs[key] = x
    B = prevalence.shape[0]
    if has_full_cov:
        mu_p, Sigma_p = model.prior.get_prior_params(prevalence, return_full_cov=True)
        prior_dist = MultivariateNormal(mu_p.detach(),
                                        covariance_matrix=Sigma_p.detach().unsqueeze(0).expand(B, -1, -1))
    else:
        mu_p, logvar_p = model.prior.get_prior_params(prevalence)
        prior_dist = Normal(mu_p.detach(), torch.exp(0.5 * logvar_p.detach()))
    return modality_inputs, prevalence, content, prior_dist, data

def logpz(prior_dist, z):
    lp = prior_dist.log_prob(z)
    return lp if has_full_cov else lp.sum(1)

def amortized_params(modality_inputs, prevalence):
    _, _, info = enc(modality_inputs, prevalence_covariates=prevalence)
    _, means, logvars, pi = info[-1]
    return means.detach(), logvars.detach(), pi.detach()

def log_weights(means, logvars, pi, prior_dist, content, data, S):
    """Return [B,S] importance log-weights log p(x|z)+log p(z)-log q(z) for S draws."""
    lw = []
    pi_t = ("mog", means, torch.clamp(logvars, -8.0, 8.0), pi)
    for _ in range(S):
        z = enc._mog_sample(means, torch.clamp(logvars, -8.0, 8.0), pi)
        theta = F.softmax(z, dim=1)
        logpx = model._recon_loglik(theta, data, corpus, content)
        lw.append(logpx + logpz(prior_dist, z) - model._posterior_loglik(pi_t, z))
    return torch.stack(lw, dim=1)

def elbo_iwae(means, logvars, pi, ctx, Keval):
    _, _, content, prior_dist, data = ctx
    with torch.no_grad():
        lw = log_weights(means, logvars, pi, prior_dist, content, data, Keval)
    return lw.mean(1), torch.logsumexp(lw, dim=1) - np.log(Keval)   # [B],[B]

def refine(means0, logvars0, pi0, ctx):
    """Per-doc SVI: optimize the variational params (init = encoder output) to max ELBO."""
    _, _, content, prior_dist, data = ctx
    means = means0.clone().requires_grad_(True)
    logvars = logvars0.clone().requires_grad_(True)
    pi_logits = torch.log(pi0 + 1e-12).clone().requires_grad_(True)
    opt = torch.optim.Adam([means, logvars, pi_logits], lr=REFINE_LR)
    for _ in range(REFINE_STEPS):
        opt.zero_grad()
        pi = F.softmax(pi_logits, dim=1)
        lw = log_weights(means, logvars, pi, prior_dist, content, data, REFINE_MC)
        (-lw.mean()).backward()
        opt.step()
    return means.detach(), logvars.detach(), F.softmax(pi_logits, dim=1).detach()

# ---------------- run decomposition on a doc sample ----------------
rng = np.random.default_rng(0)
idx = np.sort(rng.choice(N, size=min(M_SAMPLE, N), replace=False))
sdf = df.iloc[idx].reset_index(drop=True)
scorp = Corpus(sdf, modalities=mods, prevalence="~ cov_1 + cov_2 + cov_3"); densify(scorp)
loader = torch.utils.data.DataLoader(scorp, batch_size=256, shuffle=False, num_workers=0)

acc = {k: [] for k in ["elbo_a", "iwae_a", "elbo_r", "iwae_r"]}
t1 = time.time()
for bi, data in enumerate(loader):
    ctx = batch_context(data)
    modality_inputs, prevalence = ctx[0], ctx[1]
    mA, lA, pA = amortized_params(modality_inputs, prevalence)
    e_a, w_a = elbo_iwae(mA, lA, pA, ctx, EVAL_K)
    mR, lR, pR = refine(mA, lA, pA, ctx)
    e_r, w_r = elbo_iwae(mR, lR, pR, ctx, EVAL_K)
    for k, t in zip(acc, (e_a, w_a, e_r, w_r)): acc[k].append(t.cpu().numpy())
    if bi % 5 == 0:
        print(f"  batch {bi:>3} | ELBO_a={e_a.mean():.3f} IWAE_a={w_a.mean():.3f} "
              f"ELBO_r={e_r.mean():.3f} IWAE_r={w_r.mean():.3f}  ({time.time()-t1:.0f}s)", flush=True)
agg = {k: float(np.concatenate(v).mean()) for k, v in acc.items()}

# total-gap K-sweep (amortized q, whole sample, model's own estimator)
ksweep = {}
for kk in K_LIST:
    iw, el = model.estimate_marginal_log_likelihood(scorp, n_samples=kk, reduce="mean")
    ksweep[kk] = {"iwae": float(iw), "elbo": float(el)}
    print(f"  IWAE-sweep K={kk:>3}: IWAE={float(iw):.3f} ELBO={float(el):.3f} gap={float(iw-el):.3f}", flush=True)

elbo_a, iwae_a, elbo_r, iwae_r = agg["elbo_a"], agg["iwae_a"], agg["elbo_r"], agg["iwae_r"]
logpx = iwae_r                               # tightest available estimate of log p(x)
total = logpx - elbo_a
amort = elbo_r - elbo_a
approx = logpx - elbo_r
results = {"smoke": SMOKE, "N": N, "steps": STEPS, "M_sample": int(len(idx)), "mean_abs_bias": bias,
           "agg": agg, "ksweep": {str(k): v for k, v in ksweep.items()},
           "gaps_nats_per_doc": {"total": total, "amortization": amort, "approximation": approx},
           "gaps_nats_per_word": {"total": total / L, "amortization": amort / L, "approximation": approx / L}}
json.dump(results, open(OUT, "w"), indent=2)

print(f"\n=== variational gap decomposition (nats/doc; /{L} words in [], mean over {len(idx)} docs) ===")
print(f"{'ELBO_amortized':>20} {elbo_a:>9.3f}")
print(f"{'ELBO_refined':>20} {elbo_r:>9.3f}")
print(f"{'log p(x) ~ IWAE_r':>20} {logpx:>9.3f}")
print(f"{'-'*32}")
print(f"{'TOTAL gap':>20} {total:>9.3f}   [{total/L:.3f}/word]")
print(f"{'  amortization':>20} {amort:>9.3f}   [{amort/L:.3f}/word]  ({100*amort/total:.0f}% of total)" if total>1e-6 else f"{'  amortization':>20} {amort:>9.3f}")
print(f"{'  approximation':>20} {approx:>9.3f}   [{approx/L:.3f}/word]  ({100*approx/total:.0f}% of total)" if total>1e-6 else f"{'  approximation':>20} {approx:>9.3f}")
print(f"\nbias(mean|.|)={bias:.3f} | saved -> {OUT}", flush=True)
