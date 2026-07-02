"""MEASURE the variational gap of the SUPERVISED joint model (y in encoder), not infer it.

Claim under test (mine): the topic->y coefficient attenuation (c~0.65 at sigma=0.05) is because the
amortized q(theta|x,y) is WIDER than the true posterior p(theta|x,y) -> a LOOSE bound. If that is right:
  (1) the measured joint gap log p(x,y) - ELBO_amortized is large, and
  (2) recomputing the coefficient from the REFINED (per-doc SVI) posterior moves c toward 1.
If instead the bound is TIGHT (small gap) yet c stays ~0.65, my story is WRONG and the attenuation is
NOT the variational gap.

Joint ELBO/IWAE log-weights:  log p(x|theta) + log p(y|theta) + log p(z) - log q(z|x,y),  z ~ q, theta=softmax(z).
  log p(y|theta) = -0.5[(y - yhat)^2/sigma^2 + log(2 pi sigma^2)],  yhat = head(theta),  sigma^2 = exp(noise_log_var).
The encoder is fed the TRUE y (as in training) so q = q(theta|x,y). Sigma pinned to I; learned sigma^2.

Coefficient from q-moments (the head's own optimum given q):
  b = [sum_i E_q(theta theta')]^{-1} sum_i E_q(theta) y_i,   E_q estimated by MC over q.
Compare c_amortized (should reproduce ~0.65) vs c_refined. Aligned to true topics + centered.

Reuses the model's _recon_loglik / _posterior_loglik / _mog_sample. Set SUPGAP_SMOKE=1 for a quick check.
"""
import os, json, time, numpy as np, scipy.sparse, tempfile, torch
import torch.nn.functional as F
from torch.distributions import MultivariateNormal
from scipy.optimize import linear_sum_assignment
from sklearn.feature_extraction.text import CountVectorizer
from deeplatent import Corpus, GTM, generate_documents

SMOKE = os.environ.get("SUPGAP_SMOKE") == "1"
K, C, VOCAB, L = 5, 3, 200, 10
HIDDEN, COMP = [256, 256], 20
SEED = 1000
SIGMA = float(os.environ.get("SUPGAP_SIGMA", "1.0"))
PIN_COV = os.environ.get("SUPGAP_PIN_COV", "0") == "1"   # default: LEARN Sigma (no oracle). prior cov.
OPTIM = {"main": {"lr": 1e-3, "weight_decay": 0.0}, "prior": {"lr": 1e-4, "weight_decay": 0.0}}
PRED_ARGS = {"y": {"hidden_dims": [], "loss_weight": 1.0}}
if SMOKE:
    N, STEPS, M_SAMPLE, MOM_S, REFINE_STEPS, REFINE_MC, EVAL_K, REFINE_LR = 3000, 4000, 600, 64, 200, 16, 64, 0.01
else:
    N, STEPS, M_SAMPLE, MOM_S, REFINE_STEPS, REFINE_MC, EVAL_K, REFINE_LR = 100000, 60000, 3000, 128, 400, 24, 200, 0.01
OUT = f"audit/results_supervised_gap_s{SIGMA}.json"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

lambda_fixed = np.random.default_rng(7).standard_normal((C + 1, K)) * 0.5
label_coeffs = np.array([2.0, 1.0, 0.0, -1.0, -2.0])
true_b = label_coeffs - label_coeffs.mean()

def align(tt, et):
    r, c = linear_sum_assignment(-(tt.T @ et)); return {int(i): int(j) for i, j in zip(r, c)}
def center(v): return v - v.mean()
def densify(c):
    mm = c.processed_modalities["text"]["bow"]["matrix"]
    if scipy.sparse.issparse(mm):
        c.processed_modalities["text"]["bow"]["matrix"] = np.asarray(mm.todense(), np.float32)
def fit_c(b):
    b = center(b)
    return float((b @ true_b) / (true_b @ true_b)), float(np.corrcoef(b, true_b)[0, 1])

print(f"{'[SMOKE] ' if SMOKE else ''}supervised gap | N={N}, {L} w/doc, {HIDDEN} MoG-{COMP}, sigma={SIGMA}, "
      f"prior_cov={'PINNED I' if PIN_COV else 'LEARNED'}, refit {STEPS}", flush=True)
dft, df, tw, lam, lc = generate_documents(num_docs=N, num_topics=K, vocab_size=VOCAB, num_covs=C,
    doc_topic_prior="logistic_normal", min_words=L, max_words=L, lambda_=lambda_fixed,
    label_type="regression", label_coeffs=label_coeffs, random_seed=SEED)
true_theta = dft[[f"Topic{i}" for i in range(K)]].values.astype(np.float64)
# regenerate y at the requested sigma (generate_documents hardcodes 0.05)
rng = np.random.default_rng(2024 + int(SIGMA * 1000))
df["label"] = (true_theta @ label_coeffs) + rng.normal(0, SIGMA, N)
vec = CountVectorizer(); vec.fit(df["doc_clean_0"])
mods = {"text": {"column": "doc_clean_0", "views": {"bow": {"type": "bow", "vectorizer": vec}}}}
labels_cfg = {"y": {"column": "label", "type": "regression"}}
corpus = Corpus(df, modalities=mods, prevalence="~ cov_1 + cov_2 + cov_3", labels=labels_cfg); densify(corpus)

t0 = time.time()
model = GTM(train_data=corpus, n_topics=K, ae_type="vae", vi_type="mixture_of_gaussians",
            mixture_components=COMP, doc_topic_prior="logistic_normal", update_prior=True, w_prior=1.0,
            learn_prior_cov=not PIN_COV, labels_in_encoder=True, predictor_args=PRED_ARGS,
            encoder_args={"text_bow": {"hidden_dims": HIDDEN}}, decoder_args={"text_bow": {"hidden_dims": []}},
            batch_size=256, num_steps=STEPS, num_workers=0 if SMOKE else 4, print_every_n_steps=10**9,
            optim_args=OPTIM, return_best_model=False, ckpt_folder=tempfile.mkdtemp(), seed=SEED, device=device)
print(f"refit done ({time.time()-t0:.0f}s)", flush=True)

# sanity: the head's own coefficient (the c~0.65 we are explaining)
th_full = model.get_doc_topic_distribution(corpus, num_samples=10).astype(np.float64)
perm_full = [align(true_theta, th_full)[t] for t in range(K)]
Wh = model.predictor.predictors["y"].neural_net["pred_0"].weight.detach().cpu().numpy().astype(np.float64)
c_head, corr_head = fit_c(np.array([Wh[0, perm_full[t]] for t in range(K)]))
sig2_y = float(torch.exp(model.predictor.noise_log_var["y"]).detach().cpu())
print(f"sanity: head c={c_head:.3f} corr={corr_head:.3f} | sig2_hat={sig2_y:.4f} (true {SIGMA**2:.4f})", flush=True)

for module in (model.encoder, model.decoders, model.prior, model.predictor):
    for p in module.parameters():
        p.requires_grad_(False)
enc = model.encoder
W_head = model.predictor.predictors["y"].neural_net["pred_0"].weight.detach()  # [1,K]
b_head = model.predictor.predictors["y"].neural_net["pred_0"].bias.detach()    # [1]
inv2s2 = 1.0 / (2.0 * sig2_y); logZ_y = 0.5 * np.log(2 * np.pi * sig2_y)

def batch_context(data):
    for k, v in data.items():
        if isinstance(v, torch.Tensor): data[k] = v.to(device)
    prevalence = data["M_prevalence_covariates"]
    content = data.get("M_content_covariates", None)
    y = data["M_labels"][:, 0]                                  # [B]
    x = data["modalities"]["text"]["bow"].to(device)
    x = torch.cat([x, prevalence], dim=1)
    x = torch.cat([x, data["M_labels"].float()], dim=1)        # labels_in_encoder: q sees true y
    modality_inputs = {list(enc.encoders.keys())[0]: x}
    B = prevalence.shape[0]
    mu_p, Sig = model.prior.get_prior_params(prevalence, return_full_cov=True)
    prior_dist = MultivariateNormal(mu_p.detach(), covariance_matrix=Sig.detach().unsqueeze(0).expand(B, -1, -1))
    return modality_inputs, prevalence, content, prior_dist, data, y

def logpy(theta, y):
    yhat = (theta @ W_head.t()).squeeze(1) + b_head            # [B]
    return -inv2s2 * (y - yhat) ** 2 - logZ_y                  # [B]

def amortized_params(modality_inputs, prevalence):
    _, _, info = enc(modality_inputs, prevalence_covariates=prevalence)
    _, means, logvars, pi = info[-1]
    return means.detach(), logvars.detach(), pi.detach()

def log_weights(means, logvars, pi, ctx, S):
    _, _, content, prior_dist, data, y = ctx
    lv = torch.clamp(logvars, -8.0, 8.0)
    pi_t = ("mog", means, lv, pi)
    lw = []
    for _ in range(S):
        z = enc._mog_sample(means, lv, pi)
        theta = model.latent_to_theta(z)
        lp = (model._recon_loglik(theta, data, corpus, content) + logpy(theta, y)
              + prior_dist.log_prob(z) - model._posterior_loglik(pi_t, z))
        lw.append(lp)
    return torch.stack(lw, dim=1)                              # [B,S]

def elbo_iwae(means, logvars, pi, ctx, Keval):
    with torch.no_grad():
        lw = log_weights(means, logvars, pi, ctx, Keval)
    return lw.mean(1), torch.logsumexp(lw, dim=1) - np.log(Keval)

def refine(m0, l0, p0, ctx):
    means = m0.clone().requires_grad_(True)
    logvars = l0.clone().requires_grad_(True)
    pi_logits = torch.log(p0 + 1e-12).clone().requires_grad_(True)
    opt = torch.optim.Adam([means, logvars, pi_logits], lr=REFINE_LR)
    for _ in range(REFINE_STEPS):
        opt.zero_grad()
        lw = log_weights(means, logvars, F.softmax(pi_logits, dim=1), ctx, REFINE_MC)
        (-lw.mean()).backward()
        opt.step()
    return means.detach(), logvars.detach(), F.softmax(pi_logits, dim=1).detach()

def q_moments(means, logvars, pi, y, S):
    """Accumulate sum_i E_q[theta theta'] [K,K], sum_i E_q[theta]*y_i [K] over the batch."""
    lv = torch.clamp(logvars, -8.0, 8.0)
    Eth = torch.zeros(means.shape[0], K, device=device)
    Ethth = torch.zeros(means.shape[0], K, K, device=device)
    for _ in range(S):
        th = enc.latent_to_theta(enc._mog_sample(means, lv, pi))   # [B,K]
        Eth += th
        Ethth += th.unsqueeze(2) * th.unsqueeze(1)
    Eth /= S; Ethth /= S
    return Ethth.sum(0), (Eth * y.unsqueeze(1)).sum(0)

# ---- run on a doc sample ----
rng2 = np.random.default_rng(0)
idx = np.sort(rng2.choice(N, size=min(M_SAMPLE, N), replace=False))
sdf = df.iloc[idx].reset_index(drop=True)
scorp = Corpus(sdf, modalities=mods, prevalence="~ cov_1 + cov_2 + cov_3", labels=labels_cfg); densify(scorp)
true_theta_s = true_theta[idx]
th_s = model.get_doc_topic_distribution(scorp, num_samples=10).astype(np.float64)
perm = [align(true_theta_s, th_s)[t] for t in range(K)]
loader = torch.utils.data.DataLoader(scorp, batch_size=256, shuffle=False, num_workers=0)

acc = {k: [] for k in ["elbo_a", "iwae_a", "elbo_r", "iwae_r"]}
MA = torch.zeros(K, K, device=device); rA = torch.zeros(K, device=device)
MR = torch.zeros(K, K, device=device); rR = torch.zeros(K, device=device)
t1 = time.time()
for bi, data in enumerate(loader):
    ctx = batch_context(data)
    mi, prev, y = ctx[0], ctx[1], ctx[5]
    mA, lA, pA = amortized_params(mi, prev)
    e_a, w_a = elbo_iwae(mA, lA, pA, ctx, EVAL_K)
    mR, lR, pR = refine(mA, lA, pA, ctx)
    e_r, w_r = elbo_iwae(mR, lR, pR, ctx, EVAL_K)
    with torch.no_grad():
        dM, dr = q_moments(mA, lA, pA, y, MOM_S); MA += dM; rA += dr
        dM, dr = q_moments(mR, lR, pR, y, MOM_S); MR += dM; rR += dr
    for k, t in zip(acc, (e_a, w_a, e_r, w_r)): acc[k].append(t.cpu().numpy())
    if bi % 4 == 0:
        print(f"  batch {bi:>3} | ELBO_a={e_a.mean():.3f} IWAE_a={w_a.mean():.3f} "
              f"ELBO_r={e_r.mean():.3f} IWAE_r={w_r.mean():.3f}  ({time.time()-t1:.0f}s)", flush=True)
agg = {k: float(np.concatenate(v).mean()) for k, v in acc.items()}

def solve_c(M, r):
    b = torch.linalg.solve(M, r).cpu().numpy().astype(np.float64)  # model-topic order
    cc, corr = fit_c(np.array([b[perm[t]] for t in range(K)]))
    return cc, corr
c_amort, corr_amort = solve_c(MA, rA)
c_refined, corr_refined = solve_c(MR, rR)

elbo_a, iwae_a, elbo_r, iwae_r = agg["elbo_a"], agg["iwae_a"], agg["elbo_r"], agg["iwae_r"]
logpxy = iwae_r
total, amort, approx = logpxy - elbo_a, elbo_r - elbo_a, logpxy - elbo_r
results = {"smoke": SMOKE, "sigma": SIGMA, "sig2_hat": sig2_y, "c_head": c_head,
           "c_amort": c_amort, "c_refined": c_refined, "agg": agg,
           "gaps_nats_per_doc": {"total": total, "amortization": amort, "approximation": approx}}
json.dump(results, open(OUT, "w"), indent=2)

print(f"\n=== SUPERVISED joint gap (sigma={SIGMA}, nats/doc, {len(idx)} docs) ===")
print(f"{'ELBO_amortized':>20} {elbo_a:>9.3f}")
print(f"{'ELBO_refined':>20} {elbo_r:>9.3f}")
print(f"{'log p(x,y) ~ IWAE_r':>20} {logpxy:>9.3f}")
print(f"{'-'*32}")
print(f"{'TOTAL gap':>20} {total:>9.3f}")
print(f"{'  amortization':>20} {amort:>9.3f}")
print(f"{'  approximation':>20} {approx:>9.3f}")
print(f"\ncoefficient slope c (1 = unbiased):")
print(f"   head (trained)   c={c_head:.3f} corr={corr_head:.3f}")
print(f"   amortized q      c={c_amort:.3f} corr={corr_amort:.3f}")
print(f"   refined  q       c={c_refined:.3f} corr={corr_refined:.3f}")
print("INTERP: if total gap small AND c_refined ~ c_amort ~ 0.65 -> attenuation is NOT the variational gap.")
print(f"        if c_refined -> 1 -> the amortization gap WAS the cause. saved -> {OUT}", flush=True)
