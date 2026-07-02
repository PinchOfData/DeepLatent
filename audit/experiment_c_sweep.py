"""C-SWEEP: words-per-doc C in {25, 50, 100} at N=100k (C=10 cell = audit/results_postfit_ols.json +
results_postfit_prior_diag.json — identical config/seed, merge when reporting).

Bridges to Battaglia-Christensen-Hansen-Sacher's kappa = sqrt(n)*E[1/C]: C=10 -> kappa~32 (extreme),
C=25 -> 12.6, C=50 -> 6.3, C=100 -> 3.2. Predictions as C grows:
  (1) fitted prior Sigma-hat diag -> 1.0 (logit temperature gets pinned by the multinomial likelihood;
      at C=10 it deflated to ~0.5-0.7 and inflated the post-fit OLS readout by +6.8%)
  (2) post-fit OLS on SNIS E[theta|x] -> c=1 with max|t| = O(1)  (the "fit once, regress as usual"
      route becomes coverage-valid)
  (3) two-step attenuation shrinks with kappa (their Theorem 1: bias proportional to kappa)
  (4) the y=0 encoder readout inflation also shrinks (y's share of the posterior -> 0)

Fixed protocol per cell (identical to the C=10 run): joint fit 60k steps -> head, Sigma probe,
y=0 readout OLS, SNIS-x readout OLS (S=128, proposal std x1.5); separate two-step fit 60k steps ->
readout OLS; oracle OLS. HC1 robust SEs, Hungarian-aligned + centered coefficients. Joint models saved
as audit/postfit_model_C{c}.ckpt. Partial results dumped after every cell (crash-safe).

NOTE (v0.2.0): logistic-normal latents are now (K-1)-dim contrast coordinates. Checkpoints
saved by pre-0.2.0 runs of this script CANNOT be loaded by >=0.2.0 (load_model raises; check
out the pre-0.2.0 commit to read them). Fresh runs retrain and save 0.2.0-format ckpts.
CSWEEP_SMOKE=1 for plumbing check; CSWEEP_C to override the cell list.
"""
import os, json, time, gc, numpy as np, scipy.sparse, tempfile, torch
import torch.nn.functional as F
from torch.distributions import MultivariateNormal
from scipy.optimize import linear_sum_assignment
from sklearn.feature_extraction.text import CountVectorizer
from deeplatent import Corpus, GTM, generate_documents

SMOKE = os.environ.get("CSWEEP_SMOKE") == "1"
CELLS = json.loads(os.environ.get("CSWEEP_C", "[25, 50, 100]"))
K, C_COV, VOCAB = 5, 3, 200
HIDDEN, COMP = [256, 256], 20
SEED, SIGMA, INFLATE, S = 1000, 1.0, 1.5, 128
OPTIM = {"main": {"lr": 1e-3, "weight_decay": 0.0}, "prior": {"lr": 1e-4, "weight_decay": 0.0}}
PRED_ARGS = {"y": {"hidden_dims": [], "loss_weight": 1.0}}
N, STEPS, ENC_S = (3000, 2000, 10) if SMOKE else (100000, 60000, 50)
if SMOKE: S = 32
OUT = "audit/results_c_sweep.json"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

lambda_fixed = np.random.default_rng(7).standard_normal((C_COV + 1, K)) * 0.5
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
    rec = {"tag": tag, "b_hat": bc.tolist(), "bias": bias.tolist(), "se": se.tolist(),
           "slope": float((bc @ true_b) / (true_b @ true_b)),
           "corr": float(np.corrcoef(bc, true_b)[0, 1]), "mab": float(np.abs(bias).mean()),
           "cover_95": int((np.abs(bias) <= 1.96 * se).sum()), "max_t": float(np.abs(bias / se).max())}
    print(f"    {tag:<22} c={rec['slope']:6.3f} mab={rec['mab']:.3f} cover={rec['cover_95']}/5 "
          f"max|t|={rec['max_t']:7.1f}", flush=True)
    return rec

def run_cell(C_words):
    kappa = float(np.sqrt(N) / C_words)
    print(f"\n===== CELL C={C_words} (kappa={kappa:.1f}) =====", flush=True)
    cell = {"C": C_words, "kappa": kappa}
    dft, df, tw, lam, lc = generate_documents(num_docs=N, num_topics=K, vocab_size=VOCAB, num_covs=C_COV,
        doc_topic_prior="logistic_normal", min_words=C_words, max_words=C_words, lambda_=lambda_fixed,
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

    cell["oracle"] = ols_report(true_theta, y, list(range(K)), "oracle")

    # ---- joint fit ----
    t0 = time.time()
    model = GTM(train_data=corpus, n_topics=K, ae_type="vae", vi_type="mixture_of_gaussians",
                mixture_components=COMP, doc_topic_prior="logistic_normal", update_prior=True, w_prior=1.0,
                learn_prior_cov=True, labels_in_encoder=True, predictor_args=PRED_ARGS,
                encoder_args={"text_bow": {"hidden_dims": HIDDEN}}, decoder_args={"text_bow": {"hidden_dims": []}},
                batch_size=256, num_steps=STEPS, num_workers=0, print_every_n_steps=10**9,
                optim_args=OPTIM, return_best_model=False, ckpt_folder=tempfile.mkdtemp(), seed=SEED, device=device)
    print(f"  joint fit done ({time.time()-t0:.0f}s)", flush=True)
    try:
        model.save_model(f"audit/postfit_model_C{C_words}.ckpt")
    except Exception as exc:
        print(f"  WARNING: save_model failed ({exc})", flush=True)

    th_y0 = model.get_doc_topic_distribution(corpus, num_samples=ENC_S).astype(np.float64)
    perm_j = [align(true_theta, th_y0)[t] for t in range(K)]
    Wh = model.predictor.predictors["y"].neural_net["pred_0"].weight.detach().cpu().numpy().astype(np.float64)
    head_b = center(np.array([Wh[0, perm_j[t]] for t in range(K)]))
    sig2_y = float(torch.exp(model.predictor.noise_log_var["y"]).detach().cpu())
    cell["head"] = {"b_hat": head_b.tolist(), "slope": float((head_b @ true_b) / (true_b @ true_b)),
                    "mab": float(np.abs(head_b - true_b).mean()), "sigma2_hat": sig2_y}
    with torch.no_grad():
        probe = torch.zeros(1, C_COV + 1, device=device); probe[0, 0] = 1.0
        _, Sig_f = model.prior.get_prior_params(probe, return_full_cov=True)
    Sig_np = Sig_f.cpu().numpy().astype(np.float64)
    cell["prior_sigma_diag"] = np.diag(Sig_np).tolist()
    cell["prior_sigma"] = Sig_np.tolist()          # full matrix (contrast space in v0.2.0)
    d = np.sqrt(np.diag(Sig_np))
    corr_off = (Sig_np / np.outer(d, d))[np.triu_indices(Sig_np.shape[0], 1)]
    cell["prior_sigma_offdiag_corr"] = {"mean_abs": float(np.abs(corr_off).mean()),
                                        "max_abs": float(np.abs(corr_off).max())}
    print(f"    head c={cell['head']['slope']:.3f} sig2={sig2_y:.3f} | "
          f"Sigma diag={np.round(cell['prior_sigma_diag'],2).tolist()} (true 1.0) | "
          f"offdiag corr mean|.|={cell['prior_sigma_offdiag_corr']['mean_abs']:.3f} "
          f"max|.|={cell['prior_sigma_offdiag_corr']['max_abs']:.3f}", flush=True)

    cell["encoder_y0"] = ols_report(th_y0, y, perm_j, "encoder_y0")

    # ---- SNIS E[theta|x] readout ----
    for module in (model.encoder, model.decoders, model.prior, model.predictor):
        for p in module.parameters():
            p.requires_grad_(False)
    enc = model.encoder
    enc_key = list(enc.encoders.keys())[0]
    loader = torch.utils.data.DataLoader(corpus, batch_size=256, shuffle=False, num_workers=0)
    out, esss = [], []
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
            mu_p, Sig = model.prior.get_prior_params(prevalence, return_full_cov=True)
            prior_dist = MultivariateNormal(mu_p.detach(), covariance_matrix=Sig.detach().unsqueeze(0).expand(B, -1, -1))
            _, _, info = enc({enc_key: x}, prevalence_covariates=prevalence)
            _, means, logvars, pi = info[-1]
            means, pi = means.detach(), pi.detach()
            lv_q = torch.clamp(logvars.detach(), -8.0, 8.0) + 2.0 * np.log(INFLATE)
            pi_t = ("mog", means, lv_q, pi)
            lws, ths = [], []
            for _ in range(S):
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
    th_snis = np.concatenate(out).astype(np.float64)
    ess = np.concatenate(esss)
    print(f"    [snis_x] ESS/S mean={ess.mean()/S:.2f} min={ess.min():.1f}/{S} ({time.time()-t1:.0f}s)", flush=True)
    rec = ols_report(th_snis, y, perm_j, "snis_x_only")
    rec["ess_mean"] = float(ess.mean()); rec["ess_min"] = float(ess.min()); rec["S"] = S
    cell["snis_x"] = rec

    del model, loader
    gc.collect(); torch.cuda.empty_cache()

    # ---- two-step fit ----
    t0 = time.time()
    model2 = GTM(train_data=corpus_u, n_topics=K, ae_type="vae", vi_type="mixture_of_gaussians",
                 mixture_components=COMP, doc_topic_prior="logistic_normal", update_prior=False, w_prior=1.0,
                 learn_prior_cov=False,
                 encoder_args={"text_bow": {"hidden_dims": HIDDEN}}, decoder_args={"text_bow": {"hidden_dims": []}},
                 batch_size=256, num_steps=STEPS, num_workers=0, print_every_n_steps=10**9,
                 optim_args=OPTIM, return_best_model=False, ckpt_folder=tempfile.mkdtemp(), seed=SEED, device=device)
    print(f"  two-step fit done ({time.time()-t0:.0f}s)", flush=True)
    th_2s = model2.get_doc_topic_distribution(corpus_u, num_samples=ENC_S).astype(np.float64)
    perm_2s = [align(true_theta, th_2s)[t] for t in range(K)]
    cell["two_step"] = ols_report(th_2s, y, perm_2s, "two_step")

    del model2, corpus, corpus_u, df, dft
    gc.collect(); torch.cuda.empty_cache()
    return cell

results = {"smoke": SMOKE, "config": {"N": N, "steps": STEPS, "sigma": SIGMA, "S": S,
           "cells": CELLS, "note": "C=10 cell lives in results_postfit_ols.json (same config/seed)"},
           "true_b": true_b.tolist(), "cells": []}
print(f"{'[SMOKE] ' if SMOKE else ''}C-sweep {CELLS} | N={N}, steps={STEPS}", flush=True)
for cw in CELLS:
    results["cells"].append(run_cell(cw))
    json.dump(results, open(OUT, "w"), indent=2)   # crash-safe partial dump
    print(f"  cell C={cw} saved -> {OUT}", flush=True)

print("\n=== C-SWEEP SUMMARY (c / max|t|) ===")
print(f"{'C':>5} {'kappa':>6} {'SigDiag~':>8} {'head':>7} {'snis_x':>16} {'enc_y0':>16} {'two_step':>16}")
for cell in results["cells"]:
    sd = float(np.mean(cell["prior_sigma_diag"]))
    print(f"{cell['C']:>5} {cell['kappa']:>6.1f} {sd:>8.2f} {cell['head']['slope']:>7.3f} "
          f"{cell['snis_x']['slope']:>7.3f}/{cell['snis_x']['max_t']:>7.1f} "
          f"{cell['encoder_y0']['slope']:>7.3f}/{cell['encoder_y0']['max_t']:>7.1f} "
          f"{cell['two_step']['slope']:>7.3f}/{cell['two_step']['max_t']:>7.1f}")
print(f"saved -> {OUT}", flush=True)
