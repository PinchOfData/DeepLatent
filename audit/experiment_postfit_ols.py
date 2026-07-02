"""POST-FIT OLS at large N: is "fit the joint model, then run the regression as usual" valid,
and does the regressor's conditioning set decide it?

Theory chain under test (single N=100k run so sampling noise cannot fool us; robust SEs give the
yardstick — a real bias shows up as |t| >> 2, a calibrated readout as |t| = O(1)):

  regressor theta_hat                      predicted OLS of y on theta_hat
  --------------------------------------   -------------------------------------------------
  true theta (oracle)                      c = 1 (finite-N floor)
  two-step fit, encoder readout            ATTENUATED (classical EIV; established c~0.86 at N=10k)
  joint fit, encoder @ y=0                 INFLATED (encoder weight on x-evidence assumes y observed;
                                             feeding y=0 = conditioning on y=0, not marginalizing ->
                                             theta_hat ~ c_shrink*E[theta|x], OLS rescales by 1/c_shrink)
  joint fit, encoder @ y=true              INFLATED (endogeneity: regressor absorbs eps)
  joint fit, SNIS E[theta|x]   (no y term) c = 1  <- THE MONEY ROW (calibrated posterior mean,
                                             Berkson-type error -> Omega=0 in reg_unstruct's Thm 1)
  joint fit, SNIS E[theta|x,y] (y in w)    INFLATED (calibrated but endogenous)

SNIS: samples from the encoder proposal, self-normalized weights
  x-only:     log w = log p(x|z) + log p(z|x^p) - log q(z | x, y=0)   [proposal std inflated]
  supervised: log w = log p(x|z) + log p(y|theta) + log p(z|x^p) - log q(z | x, y_true)
Validation: per-doc SVI refinement of the x-only ELBO (frozen generative model) on a subsample must
agree with the SNIS E[theta|x] (mean abs diff ~ MC noise). ESS reported for both SNIS arms.

Reuses model internals (_recon_loglik/_posterior_loglik/_mog_sample) exactly as
audit/experiment_supervised_gap.py. DGP + joint config mirror that script (established: c_head~0.985
at 60k steps). Two-step arm mirrors experiment_mc_calibrate.py (fixed standard logistic-normal).

NOTE (v0.2.0): logistic-normal latents are now (K-1)-dim contrast coordinates. Checkpoints
saved by pre-0.2.0 runs of this script CANNOT be loaded by >=0.2.0 (load_model raises; check
out the pre-0.2.0 commit to read them). Fresh runs retrain and save 0.2.0-format ckpts.
POSTFIT_SMOKE=1 for a quick end-to-end check.
"""
import os, json, time, gc, numpy as np, scipy.sparse, tempfile, torch
import torch.nn.functional as F
from torch.distributions import MultivariateNormal
from scipy.optimize import linear_sum_assignment
from sklearn.feature_extraction.text import CountVectorizer
from deeplatent import Corpus, GTM, generate_documents

SMOKE = os.environ.get("POSTFIT_SMOKE") == "1"
K, C, VOCAB, L = 5, 3, 200, 10
HIDDEN, COMP = [256, 256], 20
SEED = 1000
SIGMA = 1.0
INFLATE = float(os.environ.get("POSTFIT_INFLATE", "1.5"))   # proposal std inflation, x-only SNIS arm
OPTIM = {"main": {"lr": 1e-3, "weight_decay": 0.0}, "prior": {"lr": 1e-4, "weight_decay": 0.0}}
PRED_ARGS = {"y": {"hidden_dims": [], "loss_weight": 1.0}}
if SMOKE:
    N, STEPS, S_X, S_XY, ENC_S, REF_N, REF_STEPS, REF_MC = 3000, 3000, 64, 32, 20, 512, 150, 16
else:
    N, STEPS, S_X, S_XY, ENC_S, REF_N, REF_STEPS, REF_MC = 100000, 60000, 128, 64, 50, 3000, 400, 24
OUT = "audit/results_postfit_ols.json"
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

def ols_report(theta_hat, y, perm, tag):
    """No-intercept OLS of y on theta_hat (rows sum to 1 -> constant absorbed), HC1 robust SEs,
    coefficients + vcov mapped through topic alignment then centered (simplex identifiability)."""
    X = np.column_stack([theta_hat[:, perm[t]] for t in range(K)]).astype(np.float64)
    n = X.shape[0]
    XtX = X.T @ X
    b = np.linalg.solve(XtX, X.T @ y)
    e = y - X @ b
    meat = X.T @ (X * (e ** 2)[:, None])
    V = np.linalg.solve(XtX, np.linalg.solve(XtX, meat).T).T * (n / (n - K))
    A = np.eye(K) - np.ones((K, K)) / K
    bc, Vc = A @ b, A @ V @ A.T
    se = np.sqrt(np.maximum(np.diag(Vc), 1e-300))
    bias = bc - true_b
    tstat = bias / se
    cover = (np.abs(bias) <= 1.96 * se)
    c = float((bc @ true_b) / (true_b @ true_b))
    corr = float(np.corrcoef(bc, true_b)[0, 1])
    rec = {"tag": tag, "b_hat": bc.tolist(), "bias": bias.tolist(), "se": se.tolist(),
           "t": tstat.tolist(), "cover_95": int(cover.sum()), "slope": c, "corr": corr,
           "mab": float(np.abs(bias).mean())}
    print(f"  {tag:<28} c={c:6.3f} corr={corr:.3f} mab={np.abs(bias).mean():.3f} "
          f"cover={int(cover.sum())}/5 max|t|={np.abs(tstat).max():7.1f}", flush=True)
    return rec

# ---------------- DGP (identical to experiment_supervised_gap.py) ----------------
print(f"{'[SMOKE] ' if SMOKE else ''}postfit OLS | N={N}, {L} w/doc, {HIDDEN} MoG-{COMP}, "
      f"sigma={SIGMA}, steps={STEPS}, S_x={S_X} (inflate {INFLATE}), S_xy={S_XY}", flush=True)
dft, df, tw, lam, lc = generate_documents(num_docs=N, num_topics=K, vocab_size=VOCAB, num_covs=C,
    doc_topic_prior="logistic_normal", min_words=L, max_words=L, lambda_=lambda_fixed,
    label_type="regression", label_coeffs=label_coeffs, random_seed=SEED)
true_theta = dft[[f"Topic{i}" for i in range(K)]].values.astype(np.float64)
rng = np.random.default_rng(2024 + int(SIGMA * 1000))
y = (true_theta @ label_coeffs) + rng.normal(0, SIGMA, N)
df["label"] = y
vec = CountVectorizer(); vec.fit(df["doc_clean_0"])
mods = {"text": {"column": "doc_clean_0", "views": {"bow": {"type": "bow", "vectorizer": vec}}}}
labels_cfg = {"y": {"column": "label", "type": "regression"}}
corpus = Corpus(df, modalities=mods, prevalence="~ cov_1 + cov_2 + cov_3", labels=labels_cfg); densify(corpus)
corpus_u = Corpus(df, modalities=mods, prevalence="~ cov_1 + cov_2 + cov_3"); densify(corpus_u)

results = {"smoke": SMOKE, "config": {"N": N, "steps": STEPS, "sigma": SIGMA, "hidden": HIDDEN,
           "comp": COMP, "S_x": S_X, "S_xy": S_XY, "inflate": INFLATE}, "true_b": true_b.tolist(),
           "arms": []}
identity_perm = list(range(K))

print("\n--- arm: oracle OLS on true theta (finite-N floor) ---", flush=True)
results["arms"].append(ols_report(true_theta, y, identity_perm, "oracle_true_theta"))

# ---------------- JOINT model ----------------
t0 = time.time()
model = GTM(train_data=corpus, n_topics=K, ae_type="vae", vi_type="mixture_of_gaussians",
            mixture_components=COMP, doc_topic_prior="logistic_normal", update_prior=True, w_prior=1.0,
            learn_prior_cov=True, labels_in_encoder=True, predictor_args=PRED_ARGS,
            encoder_args={"text_bow": {"hidden_dims": HIDDEN}}, decoder_args={"text_bow": {"hidden_dims": []}},
            batch_size=256, num_steps=STEPS, num_workers=0, print_every_n_steps=10**9,
            optim_args=OPTIM, return_best_model=False, ckpt_folder=tempfile.mkdtemp(), seed=SEED, device=device)
print(f"joint fit done ({time.time()-t0:.0f}s)", flush=True)

th_y0 = model.get_doc_topic_distribution(corpus, num_samples=ENC_S).astype(np.float64)
perm_j = [align(true_theta, th_y0)[t] for t in range(K)]

Wh = model.predictor.predictors["y"].neural_net["pred_0"].weight.detach().cpu().numpy().astype(np.float64)
head_b = center(np.array([Wh[0, perm_j[t]] for t in range(K)]))
sig2_y = float(torch.exp(model.predictor.noise_log_var["y"]).detach().cpu())
c_head = float((head_b @ true_b) / (true_b @ true_b))
results["head"] = {"b_hat": head_b.tolist(), "slope": c_head, "sigma2_hat": sig2_y,
                   "mab": float(np.abs(head_b - true_b).mean())}
print(f"\njoint head reference: c={c_head:.3f} mab={np.abs(head_b-true_b).mean():.3f} "
      f"sig2_hat={sig2_y:.3f} (established ~0.985 at this config)", flush=True)

print("\n--- arm: joint encoder readout @ y=0 (standard get_doc_topic_distribution) ---", flush=True)
results["arms"].append(ols_report(th_y0, y, perm_j, "joint_encoder_y0"))

# ---------------- manual encoder plumbing (pattern from experiment_supervised_gap.py) ----------------
for module in (model.encoder, model.decoders, model.prior, model.predictor):
    for p in module.parameters():
        p.requires_grad_(False)
enc = model.encoder
W_head = model.predictor.predictors["y"].neural_net["pred_0"].weight.detach()
b_head = model.predictor.predictors["y"].neural_net["pred_0"].bias.detach()
inv2s2 = 1.0 / (2.0 * sig2_y); logZ_y = 0.5 * np.log(2 * np.pi * sig2_y)
enc_key = list(enc.encoders.keys())[0]

def batch_context(data, label_mode):
    """label_mode: 'true' | 'zero'. Returns encoder inputs + prior + tensors on device."""
    for k, v in data.items():
        if isinstance(v, torch.Tensor): data[k] = v.to(device)
    prevalence = data["M_prevalence_covariates"]
    yb = data["M_labels"][:, 0].float()
    lab = data["M_labels"].float() if label_mode == "true" else torch.zeros_like(data["M_labels"].float())
    x = torch.cat([data["modalities"]["text"]["bow"].to(device), prevalence, lab], dim=1)
    B = prevalence.shape[0]
    mu_p, Sig = model.prior.get_prior_params(prevalence, return_full_cov=True)
    prior_dist = MultivariateNormal(mu_p.detach(), covariance_matrix=Sig.detach().unsqueeze(0).expand(B, -1, -1))
    return {enc_key: x}, prevalence, prior_dist, data, yb

def mog_params(modality_inputs, prevalence):
    _, _, info = enc(modality_inputs, prevalence_covariates=prevalence)
    _, means, logvars, pi = info[-1]
    return means.detach(), torch.clamp(logvars.detach(), -8.0, 8.0), pi.detach()

def logpy(theta, yb):
    yhat = (theta @ W_head.t()).squeeze(1) + b_head
    return -inv2s2 * (yb - yhat) ** 2 - logZ_y

def snis_batch(means, logvars, pi, prior_dist, data, content, yb, S, with_y, inflate):
    """Self-normalized IS posterior mean of theta. Proposal = MoG(means, logvars*, pi) with
    std inflated by `inflate` (density evaluated under the SAME inflated params)."""
    lv_q = logvars + 2.0 * np.log(inflate)
    pi_t = ("mog", means, lv_q, pi)
    lws, ths = [], []
    for _ in range(S):
        z = enc._mog_sample(means, lv_q, pi)
        theta = model.latent_to_theta(z)
        lp = (model._recon_loglik(theta, data, corpus, content)
              + prior_dist.log_prob(z) - model._posterior_loglik(pi_t, z))
        if with_y:
            lp = lp + logpy(theta, yb)
        lws.append(lp); ths.append(theta)
    lw = torch.stack(lws, dim=1)                       # [B,S]
    th = torch.stack(ths, dim=1)                       # [B,S,K]
    w = torch.softmax(lw, dim=1)
    ess = 1.0 / (w ** 2).sum(dim=1)
    return (w.unsqueeze(2) * th).sum(dim=1), ess       # [B,K], [B]

def snis_readout(S, with_y, label_mode, inflate, tag):
    loader = torch.utils.data.DataLoader(corpus, batch_size=256, shuffle=False, num_workers=0)
    out, esss = [], []
    t1 = time.time()
    with torch.no_grad():
        for bi, data in enumerate(loader):
            mi, prev, prior_dist, data, yb = batch_context(data, label_mode)
            content = data.get("M_content_covariates", None)
            m, l, p = mog_params(mi, prev)
            tm, ess = snis_batch(m, l, p, prior_dist, data, content, yb, S, with_y, inflate)
            out.append(tm.cpu().numpy()); esss.append(ess.cpu().numpy())
    th = np.concatenate(out).astype(np.float64); ess = np.concatenate(esss)
    print(f"  [{tag}] ESS/S: mean={ess.mean()/S:.2f} median={np.median(ess)/S:.2f} "
          f"min={ess.min():.1f}/{S}  ({time.time()-t1:.0f}s)", flush=True)
    return th, {"mean": float(ess.mean()), "median": float(np.median(ess)), "min": float(ess.min()), "S": S}

print("\n--- arm: joint encoder readout @ y=TRUE (endogenous by construction) ---", flush=True)
loader = torch.utils.data.DataLoader(corpus, batch_size=256, shuffle=False, num_workers=0)
th_list = []
with torch.no_grad():
    for data in loader:
        mi, prev, _, data, _ = batch_context(data, "true")
        m, l, p = mog_params(mi, prev)
        acc = torch.zeros(m.shape[0], K, device=device)
        for _ in range(ENC_S):
            acc += enc.latent_to_theta(enc._mog_sample(m, l, p))
        th_list.append((acc / ENC_S).cpu().numpy())
th_ytrue = np.concatenate(th_list).astype(np.float64)
results["arms"].append(ols_report(th_ytrue, y, perm_j, "joint_encoder_ytrue"))

print("\n--- arm: joint SNIS E[theta|x]  (x-only weights; THE MONEY ROW) ---", flush=True)
th_snis_x, ess_x = snis_readout(S_X, with_y=False, label_mode="zero", inflate=INFLATE, tag="snis_x")
rec = ols_report(th_snis_x, y, perm_j, "joint_snis_x_only"); rec["ess"] = ess_x
results["arms"].append(rec)

print("\n--- arm: joint SNIS E[theta|x,y]  (y in weights; endogenous, calibrated) ---", flush=True)
th_snis_xy, ess_xy = snis_readout(S_XY, with_y=True, label_mode="true", inflate=1.0, tag="snis_xy")
rec = ols_report(th_snis_xy, y, perm_j, "joint_snis_xy"); rec["ess"] = ess_xy
results["arms"].append(rec)

# ---------------- SVI validation of the SNIS x-only readout (subsample) ----------------
print(f"\n--- validation: per-doc SVI refinement of x-only ELBO on {REF_N} docs ---", flush=True)
rng2 = np.random.default_rng(0)
ridx = np.sort(rng2.choice(N, size=min(REF_N, N), replace=False))
sub = torch.utils.data.Subset(corpus, ridx.tolist())
rloader = torch.utils.data.DataLoader(sub, batch_size=256, shuffle=False, num_workers=0)
ref_means = []
t1 = time.time()
for data in rloader:
    mi, prev, prior_dist, data, yb = batch_context(data, "zero")
    content = data.get("M_content_covariates", None)
    m0, l0, p0 = mog_params(mi, prev)
    means = m0.clone().requires_grad_(True)
    logvars = l0.clone().requires_grad_(True)
    pi_logits = torch.log(p0 + 1e-12).clone().requires_grad_(True)
    opt = torch.optim.Adam([means, logvars, pi_logits], lr=0.01)
    for _ in range(REF_STEPS):
        opt.zero_grad()
        pi_c = F.softmax(pi_logits, dim=1)
        lv_c = torch.clamp(logvars, -8.0, 8.0)
        pi_t = ("mog", means, lv_c, pi_c)
        lw = []
        for _ in range(REF_MC):
            z = enc._mog_sample(means, lv_c, pi_c)
            theta = model.latent_to_theta(z)
            lw.append(model._recon_loglik(theta, data, corpus, content)
                      + prior_dist.log_prob(z) - model._posterior_loglik(pi_t, z))
        (-torch.stack(lw, dim=1).mean()).backward()
        opt.step()
    with torch.no_grad():
        pi_c = F.softmax(pi_logits, dim=1); lv_c = torch.clamp(logvars, -8.0, 8.0)
        acc = torch.zeros(means.shape[0], K, device=device)
        for _ in range(64):
            acc += enc.latent_to_theta(enc._mog_sample(means, lv_c, pi_c))
        ref_means.append((acc / 64).cpu().numpy())
th_ref = np.concatenate(ref_means).astype(np.float64)
diff = np.abs(th_ref - th_snis_x[ridx])
agree_corr = [float(np.corrcoef(th_ref[:, k], th_snis_x[ridx][:, k])[0, 1]) for k in range(K)]
results["svi_validation"] = {"n": int(len(ridx)), "mean_abs_diff": float(diff.mean()),
                             "p95_abs_diff": float(np.percentile(diff, 95)), "per_coord_corr": agree_corr}
print(f"  SVI-refined vs SNIS E[theta|x]: mean|diff|={diff.mean():.4f} p95={np.percentile(diff,95):.4f} "
      f"corr={np.round(agree_corr,3).tolist()}  ({time.time()-t1:.0f}s)", flush=True)

del loader, rloader
gc.collect(); torch.cuda.empty_cache()

# ---------------- TWO-STEP baseline ----------------
print("\n--- arm: two-step (unsupervised fit, fixed standard logistic-normal prior) ---", flush=True)
t0 = time.time()
model2 = GTM(train_data=corpus_u, n_topics=K, ae_type="vae", vi_type="mixture_of_gaussians",
             mixture_components=COMP, doc_topic_prior="logistic_normal", update_prior=False, w_prior=1.0,
             learn_prior_cov=False,
             encoder_args={"text_bow": {"hidden_dims": HIDDEN}}, decoder_args={"text_bow": {"hidden_dims": []}},
             batch_size=256, num_steps=STEPS, num_workers=0, print_every_n_steps=10**9,
             optim_args=OPTIM, return_best_model=False, ckpt_folder=tempfile.mkdtemp(), seed=SEED, device=device)
print(f"two-step fit done ({time.time()-t0:.0f}s)", flush=True)
th_2s = model2.get_doc_topic_distribution(corpus_u, num_samples=ENC_S).astype(np.float64)
perm_2s = [align(true_theta, th_2s)[t] for t in range(K)]
results["arms"].append(ols_report(th_2s, y, perm_2s, "two_step_encoder"))

json.dump(results, open(OUT, "w"), indent=2)
print(f"\n=== SUMMARY (true_b = {true_b.tolist()}) ===")
print(f"{'arm':<28} {'c':>7} {'corr':>6} {'mab':>6} {'cover':>6} {'max|t|':>8}")
for a in results["arms"]:
    print(f"{a['tag']:<28} {a['slope']:>7.3f} {a['corr']:>6.3f} {a['mab']:>6.3f} "
          f"{a['cover_95']:>4}/5 {max(abs(t) for t in a['t']):>8.1f}")
print(f"joint head reference c={c_head:.3f}")
print(f"saved -> {OUT}", flush=True)
