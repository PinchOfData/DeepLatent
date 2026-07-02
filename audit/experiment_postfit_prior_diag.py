"""Diagnose the residual +7% inflation of the post-fit OLS on SNIS E[theta|x]
(audit/experiment_postfit_ols.py money row: c=1.068, max|t|=7.8 at N=100k).

Hypothesis: the readout inherits the FITTED PRIOR's miscalibration (learned Sigma temperature
deflation -> prior too tight -> posterior means over-shrunk -> OLS inflated), not a decoder or
Monte Carlo problem. Test by swapping the prior used in the SNIS weights while keeping the fitted
decoder and encoder proposal:

  prior in weights                         predicted OLS slope c
  A. fitted prior (mean_net + learned Sig) ~1.07  (replicates the money row)
  B. TRUE DGP prior  N(M_prev @ lambda_true, I)   -> 1.00 if the prior is the culprit
  C. fitted mean_net, Sigma = I            between A and B isolates cov-temperature vs mean error
  D. arm A with S=512 on a 25k subsample   ~ A on same subsample -> SNIS MC bias is NOT the story

Also retrains the same joint model (seed 1000, identical config) and SAVES it to
audit/postfit_joint_model.ckpt so future readout experiments need no retraining.

NOTE (v0.2.0): logistic-normal latents are now (K-1)-dim contrast coordinates. Checkpoints
saved by pre-0.2.0 runs of this script CANNOT be loaded by >=0.2.0 (load_model raises; check
out the pre-0.2.0 commit to read them). Fresh runs retrain and save 0.2.0-format ckpts.
"""
import os, json, time, numpy as np, scipy.sparse, tempfile, torch
import torch.nn.functional as F
from torch.distributions import MultivariateNormal
from scipy.optimize import linear_sum_assignment
from sklearn.feature_extraction.text import CountVectorizer
from deeplatent import Corpus, GTM, generate_documents

SMOKE = os.environ.get("PRIORDIAG_SMOKE") == "1"
K, C, VOCAB, L = 5, 3, 200, 10
HIDDEN, COMP = [256, 256], 20
SEED, SIGMA, INFLATE = 1000, 1.0, 1.5
OPTIM = {"main": {"lr": 1e-3, "weight_decay": 0.0}, "prior": {"lr": 1e-4, "weight_decay": 0.0}}
PRED_ARGS = {"y": {"hidden_dims": [], "loss_weight": 1.0}}
N, STEPS, S, S_BIG, SUB_N = (3000, 3000, 32, 128, 1000) if SMOKE else (100000, 60000, 128, 512, 25000)
OUT = "audit/results_postfit_prior_diag.json"
CKPT = "audit/postfit_joint_model.ckpt"
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

def ols_report(theta_hat, yv, perm, tag):
    X = np.column_stack([theta_hat[:, perm[t]] for t in range(K)]).astype(np.float64)
    n = X.shape[0]
    XtX = X.T @ X
    b = np.linalg.solve(XtX, X.T @ yv)
    e = yv - X @ b
    meat = X.T @ (X * (e ** 2)[:, None])
    V = np.linalg.solve(XtX, np.linalg.solve(XtX, meat).T).T * (n / (n - K))
    A = np.eye(K) - np.ones((K, K)) / K
    bc, Vc = A @ b, A @ V @ A.T
    se = np.sqrt(np.maximum(np.diag(Vc), 1e-300))
    bias = bc - true_b
    c = float((bc @ true_b) / (true_b @ true_b))
    rec = {"tag": tag, "n": int(n), "b_hat": bc.tolist(), "bias": bias.tolist(), "se": se.tolist(),
           "slope": c, "corr": float(np.corrcoef(bc, true_b)[0, 1]),
           "mab": float(np.abs(bias).mean()), "max_t": float(np.abs(bias / se).max())}
    print(f"  {tag:<26} c={c:6.3f} mab={rec['mab']:.3f} max|t|={rec['max_t']:7.1f} "
          f"b={np.round(bc,3).tolist()}", flush=True)
    return rec

print(f"{'[SMOKE] ' if SMOKE else ''}prior diag | N={N}, steps={STEPS}, S={S}/{S_BIG}", flush=True)
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

t0 = time.time()
model = GTM(train_data=corpus, n_topics=K, ae_type="vae", vi_type="mixture_of_gaussians",
            mixture_components=COMP, doc_topic_prior="logistic_normal", update_prior=True, w_prior=1.0,
            learn_prior_cov=True, labels_in_encoder=True, predictor_args=PRED_ARGS,
            encoder_args={"text_bow": {"hidden_dims": HIDDEN}}, decoder_args={"text_bow": {"hidden_dims": []}},
            batch_size=256, num_steps=STEPS, num_workers=0, print_every_n_steps=10**9,
            optim_args=OPTIM, return_best_model=False, ckpt_folder=tempfile.mkdtemp(), seed=SEED, device=device)
print(f"joint fit done ({time.time()-t0:.0f}s)", flush=True)
try:
    model.save_model(CKPT)
    print(f"model saved -> {CKPT}", flush=True)
except Exception as exc:                                   # save failure must not kill the diagnostic
    print(f"WARNING: save_model failed ({exc}); continuing", flush=True)

th_y0 = model.get_doc_topic_distribution(corpus, num_samples=20).astype(np.float64)
perm_j = [align(true_theta, th_y0)[t] for t in range(K)]
Wh = model.predictor.predictors["y"].neural_net["pred_0"].weight.detach().cpu().numpy().astype(np.float64)
head_b = center(np.array([Wh[0, perm_j[t]] for t in range(K)]))
c_head = float((head_b @ true_b) / (true_b @ true_b))
print(f"head sanity: c={c_head:.3f} (expect ~0.985)", flush=True)

for module in (model.encoder, model.decoders, model.prior, model.predictor):
    for p in module.parameters():
        p.requires_grad_(False)
enc = model.encoder
enc_key = list(enc.encoders.keys())[0]
# Since v0.2.0 the latent is (K-1)-dim contrast coords eta = V^T z. The TRUE prior
# z ~ N(M lambda, I_K) projects EXACTLY to eta ~ N(M (lambda V), I_{K-1}).
V_basis = model.prior.V  # [K, K-1]
lam_t = torch.tensor(lambda_fixed, dtype=torch.float32, device=device) @ V_basis  # [(C+1), K-1]
eyeL = torch.eye(model.n_latent, device=device)
# fitted prior covariance drift diagnostic (temperature story: learned Sigma deflates below I)
with torch.no_grad():
    probe = torch.zeros(1, C + 1, device=device); probe[0, 0] = 1.0
    _, Sig_f = model.prior.get_prior_params(probe, return_full_cov=True)
print(f"fitted prior Sigma diag (contrast space): "
      f"{np.round(torch.diagonal(Sig_f).cpu().numpy(), 3).tolist()} (true: 1.0)", flush=True)

def snis_theta(prior_mode, S_use, doc_limit=None, tag=""):
    """SNIS E[theta|x] with the y-free weights; prior_mode selects the prior in the weights."""
    loader = torch.utils.data.DataLoader(corpus, batch_size=256, shuffle=False, num_workers=0)
    out, esss, seen = [], [], 0
    t1 = time.time()
    with torch.no_grad():
        for data in loader:
            for k, v in data.items():
                if isinstance(v, torch.Tensor): data[k] = v.to(device)
            prevalence = data["M_prevalence_covariates"]
            lab0 = torch.zeros_like(data["M_labels"].float())
            x = torch.cat([data["modalities"]["text"]["bow"].to(device), prevalence, lab0], dim=1)
            content = data.get("M_content_covariates", None)
            B = prevalence.shape[0]
            if prior_mode == "fitted":
                mu_p, Sig = model.prior.get_prior_params(prevalence, return_full_cov=True)
                prior_dist = MultivariateNormal(mu_p.detach(), covariance_matrix=Sig.detach().unsqueeze(0).expand(B, -1, -1))
            elif prior_mode == "true":
                mu_p = prevalence.float() @ lam_t
                prior_dist = MultivariateNormal(mu_p, covariance_matrix=eyeL.unsqueeze(0).expand(B, -1, -1))
            elif prior_mode == "fitted_mean_unit_cov":
                mu_p, _ = model.prior.get_prior_params(prevalence, return_full_cov=True)
                prior_dist = MultivariateNormal(mu_p.detach(), covariance_matrix=eyeL.unsqueeze(0).expand(B, -1, -1))
            _, _, info = enc({enc_key: x}, prevalence_covariates=prevalence)
            _, means, logvars, pi = info[-1]
            means, pi = means.detach(), pi.detach()
            lv_q = torch.clamp(logvars.detach(), -8.0, 8.0) + 2.0 * np.log(INFLATE)
            pi_t = ("mog", means, lv_q, pi)
            lws, ths = [], []
            for _ in range(S_use):
                z = enc._mog_sample(means, lv_q, pi)
                theta = model.latent_to_theta(z)
                lp = (model._recon_loglik(theta, data, corpus, content)
                      + prior_dist.log_prob(z) - model._posterior_loglik(pi_t, z))
                lws.append(lp); ths.append(theta)
            lw = torch.stack(lws, dim=1)
            th = torch.stack(ths, dim=1)
            w = torch.softmax(lw, dim=1)
            esss.append((1.0 / (w ** 2).sum(dim=1)).cpu().numpy())
            out.append((w.unsqueeze(2) * th).sum(dim=1).cpu().numpy())
            seen += B
            if doc_limit is not None and seen >= doc_limit:
                break
    th = np.concatenate(out).astype(np.float64)
    ess = np.concatenate(esss)
    if doc_limit is not None:
        th = th[:doc_limit]; ess = ess[:doc_limit]
    print(f"  [{tag}] ESS/S mean={ess.mean()/S_use:.2f} min={ess.min():.1f}/{S_use} ({time.time()-t1:.0f}s)", flush=True)
    return th

results = {"smoke": SMOKE, "c_head": c_head,
           "fitted_prior_sigma_diag": torch.diagonal(Sig_f).cpu().numpy().tolist(), "arms": []}

print("\nA. fitted prior (money-row replication)", flush=True)
thA = snis_theta("fitted", S, tag="A_fitted")
results["arms"].append(ols_report(thA, y, perm_j, "A_fitted_prior"))

print("\nB. TRUE prior N(M lambda_true, I)", flush=True)
thB = snis_theta("true", S, tag="B_true")
results["arms"].append(ols_report(thB, y, perm_j, "B_true_prior"))

print("\nC. fitted mean, Sigma=I", flush=True)
thC = snis_theta("fitted_mean_unit_cov", S, tag="C_mean_unitcov")
results["arms"].append(ols_report(thC, y, perm_j, "C_fitted_mean_unit_cov"))

print(f"\nD. MC-bias check: fitted prior, S={S} vs S={S_BIG}, first {SUB_N} docs", flush=True)
thD1 = thA[:SUB_N]
thD2 = snis_theta("fitted", S_BIG, doc_limit=SUB_N, tag=f"D_S{S_BIG}")
results["arms"].append(ols_report(thD1, y[:SUB_N], perm_j, f"D_fitted_S{S}_sub"))
results["arms"].append(ols_report(thD2, y[:SUB_N], perm_j, f"D_fitted_S{S_BIG}_sub"))
results["theta_diff_S"] = float(np.abs(thD1 - thD2).mean())
print(f"  mean|theta(S={S}) - theta(S={S_BIG})| = {results['theta_diff_S']:.4f}", flush=True)

json.dump(results, open(OUT, "w"), indent=2)
print(f"\nINTERP: B~1.00 -> prior is the culprit; C tells cov-temperature vs mean share; "
      f"D1~D2 -> not MC bias.\nsaved -> {OUT}", flush=True)
