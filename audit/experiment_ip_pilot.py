"""Pilot + MC harness: consistency & CI-coverage for IdealPointNN (1D votes-only IRT).

Design doc: IDEALPOINT_CONSISTENCY.md (root). Interview decisions 2026-08-20:
1D latent; votes-only 2PL measurement; y = c*theta + eps; LEARNED prior Sigma in the
joint arm (practitioner setting) -> raw coefficients are NOT identified, so we evaluate
identified functionals:
    PSI   = c * sd(theta)     truth C_TRUE*SIG_TH   (y per 1 SD of theta)
    B1STD = beta1 / sigma_u   truth B1/SIG_U        (party gap in residual SDs)
    RF    = c * beta1         truth C_TRUE*B1       (reduced form; fully gauge-invariant)
Sign fixed per rep by corr(posterior means, theta_true); theta-coverage gauge = affine
map matching model-implied marginal moments of theta to the true marginal.

ARMS per rep:
  JOINT    : labels_in_encoder, linear head, learned sigma_eps^2, update_prior=True
             (mean_net on x + learned Cholesky Sigma).
  TWO-STEP : unsupervised, fixed N(0,1) prior, then OLS y~theta_hat and theta_hat~x
             (naive OLS SEs -> the CI whose coverage should die as n grows).
  ORACLE-PM: OLS y on the EXACT posterior mean E[theta|V] under the true DGP
             (Gauss-Hermite). Theory: per-unit slope UNBIASED (regression calibration),
             standardized slope attenuated by corr(theta, pm), exact 90% credible
             intervals ~90% average coverage (the VI-calibration benchmark).
  ORACLE   : OLS y ~ theta_true (finite-n floor).

MODES (env, mirrors experiment_mc_calibrate.py):
  IP_REPS=1 (default) CALIBRATION: one rep, trajectory over IP_CKPTS (U-shape search).
  IP_REPS>1           MONTE CARLO: fresh seed per rep, records EVERY checkpoint per rep
                      (U-shape in expectation), incremental JSON dump after each rep.
"""
import json, os, time, gc, tempfile
import numpy as np
import pandas as pd
import torch
from numpy.polynomial.hermite_e import hermegauss
from deeplatent import Corpus, IdealPointNN

# ---- knobs ----
REPS      = int(os.environ.get("IP_REPS", "1"))
N         = int(os.environ.get("IP_N", "2000"))
J         = int(os.environ.get("IP_J", "25"))
# Measurement channel: "vote" = 2PL IRT (main study); "gauss" = LINEAR-GAUSSIAN RUNG
# (positive control: exact posterior is Gaussian and in the variational family ->
# ELBO = exact MLE; if consistency/coverage fails here it's optimization, not VI).
MODALITY  = os.environ.get("IP_MODALITY", "vote")
# Linear-Gaussian rung: w_ij = lam_j*theta_i + delta_j + N(0, SIG_W2). SIG_W2 is
# pinned to 0.5 because the embedding decoder's Gaussian NLL is plain SSE, i.e. an
# IMPLICIT FIXED sigma^2=0.5 (AUDIT_REPORT LOW-3) -> with the DGP matched, the
# model is correctly specified with KNOWN measurement variance (textbook case).
# LAM_SD=0.25 with J=25 features gives posterior var ~1/(1+2*J*LAM_SD^2)~0.24,
# i.e. reliability ~0.8 -- matched to the J=25 votes design.
SIG_W2    = 0.5
LAM_SD    = float(os.environ.get("IP_LAM_SD", "0.25"))
# FIXED measurement design (proper Monte Carlo): item parameters are drawn ONCE
# from their own generator and held fixed across reps and across n-cells, so the
# across-rep spread is pure sampling noise under one DGP (the classical frame, and
# BCHS's). Set IP_ITEM_SEED to study a different design; pre-2026-08-20-evening
# runs (results_run1) redrew items per rep = a random-design (across-study) MC.
ITEM_SEED = int(os.environ.get("IP_ITEM_SEED", "777"))
CKPTS     = json.loads(os.environ.get("IP_CKPTS", "[2000, 4000, 8000, 16000, 24000]"))
CKPTS_2S  = json.loads(os.environ.get("IP_CKPTS_2STEP", os.environ.get("IP_CKPTS", "[2000, 4000, 8000, 16000, 24000]")))
BATCH     = int(os.environ.get("IP_BATCH", "256"))
HIDDEN    = json.loads(os.environ.get("IP_HIDDEN", "[64, 64]"))
BASE_SEED = int(os.environ.get("IP_SEED", "3000"))
NS_POST   = int(os.environ.get("IP_NS_POST", "200"))   # posterior draws for mean/std readout
OUT       = os.environ.get("IP_OUT", "audit/results_ip_pilot.json")

# ---- fixed estimands (population truth, shared across reps) ----
B0, B1, SIG_U, C_TRUE, SIG_EPS, P_X = 0.0, 1.0, 1.0, 1.0, 1.0, 0.5
SIG_TH = float(np.sqrt(P_X * (1 - P_X) * B1**2 + SIG_U**2))   # marginal sd(theta) ~ 1.118
MU_TH  = B0 + B1 * P_X                                         # marginal mean(theta)
PSI_TRUE, B1STD_TRUE, RF_TRUE = C_TRUE * SIG_TH, B1 / SIG_U, C_TRUE * B1

OPTIM = {"main": {"lr": 1e-3, "weight_decay": 0.0}, "prior": {"lr": 1e-4, "weight_decay": 0.0}}
PRED_ARGS = {"y": {"hidden_dims": [], "loss_weight": 1.0}}     # linear head, natural ELBO weight
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
VOTE_COLS = [f"vote_{j+1}" for j in range(J)]


def ols(cols, y):
    """OLS with intercept; returns (coefs incl. intercept, SEs, residual SD)."""
    X = np.column_stack([np.ones(len(y))] + cols)
    XtX_inv = np.linalg.inv(X.T @ X)
    bh = XtX_inv @ (X.T @ y)
    resid = y - X @ bh
    s2 = float(resid @ resid) / (len(y) - X.shape[1])
    return bh, np.sqrt(np.diag(XtX_inv) * s2), float(np.sqrt(s2))


def gen_rep(seed):
    """One DGP draw. Returns (corpus_j, corpus_u, x, theta, b, d, V, y) where for
    MODALITY="vote" (b, d, V) = (discriminations, difficulties, votes) and for
    MODALITY="gauss" (b, d, V) = (loadings, intercepts, features)."""
    rng = np.random.default_rng(seed)
    irng = np.random.default_rng(ITEM_SEED)          # fixed design: same items every rep
    x = rng.binomial(1, P_X, N).astype(np.float64)
    theta = B0 + B1 * x + rng.normal(0.0, SIG_U, N)
    y = C_TRUE * theta + rng.normal(0.0, SIG_EPS, N)
    if MODALITY == "vote":
        b = irng.normal(0.0, 1.0, J)
        d = irng.normal(0.0, 0.5, J)
        V = rng.binomial(1, 1.0 / (1.0 + np.exp(-(theta[:, None] * b[None, :] - d[None, :])))).astype(np.float64)
        df = pd.DataFrame(V, columns=VOTE_COLS)
        df["x"] = x
        df["label"] = y
        mods = {"vote": {"column": VOTE_COLS, "views": {"responses": {"type": "vote"}}}}
    else:
        b = irng.normal(0.0, LAM_SD, J)              # loadings
        d = irng.normal(0.0, 0.5, J)                 # feature intercepts
        V = theta[:, None] * b[None, :] + d[None, :] + rng.normal(0.0, np.sqrt(SIG_W2), (N, J))
        df = pd.DataFrame({"x": x, "label": y})
        mods = {"emb": {"column": "label",           # placeholder; matrix is passed directly
                        "views": {"w": {"type": "embedding",
                                        "matrix": V.astype(np.float32)}}}}
    corpus_j = Corpus(df, modalities=mods, prevalence="~ x",
                      labels={"y": {"column": "label", "type": "regression"}})
    corpus_u = Corpus(df, modalities=mods)           # two-step: no covariates, no labels
    return corpus_j, corpus_u, x, theta, b, d, V, y


def exact_posterior(x, V, b, d, y=None, quad=61, chunk=4096, mu=None, sig_u=None):
    """Exact E[theta|.] and Var[theta|.] under the TRUE DGP.

    y=None -> p(theta|V) (the two-step measurement posterior);
    y given -> p(theta|V,y) (the supervised posterior the joint encoder targets).
    MODALITY="vote": Gauss-Hermite quadrature. MODALITY="gauss": closed form
    (Normal-Normal conjugacy; posterior variance is constant across i).
    mu/sig_u override the TRUE prior with FITTED values (post-fit readout).
    """
    if mu is None:
        mu = B0 + B1 * x
    if sig_u is None:
        sig_u = SIG_U
    if MODALITY == "gauss":
        tau = 1.0 / sig_u**2 + float(b @ b) / SIG_W2
        num = mu / sig_u**2 + ((V - d[None, :]) @ b) / SIG_W2
        if y is not None:
            tau = tau + C_TRUE**2 / SIG_EPS**2
            num = num + C_TRUE * y / SIG_EPS**2
        pm = num / tau
        pv = np.full(len(x), 1.0 / tau)
        return pm, pv
    nodes, wts = hermegauss(quad)                    # int f(t) exp(-t^2/2) dt = sum w f(node)
    logw = np.log(wts)
    pm, pv = np.empty(len(x)), np.empty(len(x))
    for lo in range(0, len(x), chunk):
        sl = slice(lo, lo + chunk)
        TH = mu[sl, None] + sig_u * nodes[None, :]                       # (m, Q)
        Z = TH[:, :, None] * b[None, None, :] - d[None, None, :]         # (m, Q, J)
        Vs = V[sl, None, :]
        ll = -(np.logaddexp(0, -Z) * Vs + np.logaddexp(0, Z) * (1 - Vs)).sum(-1)
        if y is not None:
            ll = ll - 0.5 * (y[sl, None] - C_TRUE * TH) ** 2 / SIG_EPS**2
        lw = ll + logw[None, :]
        lw -= lw.max(1, keepdims=True)
        w = np.exp(lw)
        w /= w.sum(1, keepdims=True)
        pm[sl] = (w * TH).sum(1)
        pv[sl] = (w * TH**2).sum(1) - pm[sl] ** 2
    return pm, pv


def coverage90(m, s, g, a, theta):
    """Coverage of nominal-90% intervals mean+-1.645*sd after affine gauge map theta ~ a + g*m."""
    e1, e2 = a + g * (m - 1.645 * s), a + g * (m + 1.645 * s)
    return float(((theta >= np.minimum(e1, e2)) & (theta <= np.maximum(e1, e2))).mean())


def posterior_read(model, corpus):
    m, s = model.get_ideal_points(corpus, num_samples=NS_POST, return_std=True)
    return m[:, 0].astype(np.float64), s[:, 0].astype(np.float64)


def joint_read(model, corpus, theta, x, pm, ps, pms, pss):
    m, s = posterior_read(model, corpus)             # NOTE: y zeroed in encoder at readout
    corr = float(np.corrcoef(m, theta)[0, 1])
    sgn = float(np.sign(corr)) if corr != 0 else 1.0
    c_hat = float(model.predictor.predictors["y"].neural_net["pred_0"]
                  .weight.detach().cpu().reshape(-1)[0])
    sig2_eps = float(torch.exp(model.predictor.noise_log_var["y"]).detach().cpu())
    pdev = next(model.prior.parameters()).device
    M = torch.tensor(corpus.M_prevalence_covariates, dtype=torch.float32, device=pdev)
    mu = model.prior.mean_net(M).detach().cpu().numpy()[:, 0].astype(np.float64)
    b1 = float(mu[x == 1].mean() - mu[x == 0].mean())
    sig_u = float(torch.sqrt(model.prior.sigma[0, 0]).detach().cpu())
    sig_th = float(np.sqrt(mu.var() + sig_u**2))     # model-implied marginal sd(theta)
    g = sgn * SIG_TH / sig_th
    a = MU_TH - g * float(mu.mean())
    # VI-vs-exact diagnostics (in true units after the gauge map). The y-zeroed readout
    # should track p(theta|V) [pm]; if VI itself is calibrated, sd_ratio_* ~ 1.
    mg = a + g * m
    return {"psi": sgn * c_hat * sig_th, "b1_std": sgn * b1 / sig_u, "rf": c_hat * b1,
            "c_raw": c_hat, "b1_raw": b1, "sig_u_hat": sig_u, "sig_th_hat": sig_th,
            "sig2_eps_hat": sig2_eps, "corr": abs(corr),
            "cov90_theta_yzero": coverage90(m, s, g, a, theta),
            "corr_pm": float(np.corrcoef(m, pm)[0, 1]),
            "corr_pm_sup": float(np.corrcoef(m, pms)[0, 1]),
            "center_rmse_pm": float(np.sqrt(((mg - pm) ** 2).mean())),
            "sd_ratio_pm": float((abs(g) * s).mean() / ps.mean()),
            "sd_ratio_pm_sup": float((abs(g) * s).mean() / pss.mean())}


def twostep_read(model, corpus, theta, x, y, pm, ps):
    m, s = posterior_read(model, corpus)
    corr = float(np.corrcoef(m, theta)[0, 1])
    sgn = float(np.sign(corr)) if corr != 0 else 1.0
    bh, se, _ = ols([m], y)                          # y ~ 1 + theta_hat, naive OLS SEs
    c2, c2_se = float(bh[1]), float(se[1])
    sd_m = float(m.std())
    ci = sorted([sgn * (c2 - 1.96 * c2_se) * sd_m, sgn * (c2 + 1.96 * c2_se) * sd_m])
    bp, sep, resid_sd = ols([x], m)                  # theta_hat ~ 1 + x
    b1_2s, b1_2s_se = float(bp[1]), float(sep[1])
    b1ci = sorted([sgn * (b1_2s - 1.96 * b1_2s_se) / resid_sd,
                   sgn * (b1_2s + 1.96 * b1_2s_se) / resid_sd])
    g = sgn * SIG_TH / 1.0                           # model marginal = fixed prior N(0,1)
    a = MU_TH - g * 0.0
    mg = a + g * m
    return {"psi": sgn * c2 * sd_m, "psi_cover": int(ci[0] <= PSI_TRUE <= ci[1]),
            "b1_std": sgn * b1_2s / resid_sd,
            "b1_std_cover": int(b1ci[0] <= B1STD_TRUE <= b1ci[1]),
            "rf": c2 * b1_2s, "c2_raw": c2, "c2_se": c2_se, "sd_theta_hat": sd_m,
            "corr": abs(corr), "cov90_theta": coverage90(m, s, g, a, theta),
            "corr_pm": float(np.corrcoef(m, pm)[0, 1]),
            "center_rmse_pm": float(np.sqrt(((mg - pm) ** 2).mean())),
            "sd_ratio_pm": float((abs(g) * s).mean() / ps.mean())}


def postfit_read(model, corpus, theta, x, y, V):
    """Post-fit OLS test (BCHS location-shift logic): regress y on the FITTED-
    parameter, Y-FREE posterior mean E_hat[theta|V] (regression calibration with
    estimated params). Records naive and HC1 CIs for psi to test whether the
    'cheap' regression width is valid once the centering is fixed by joint
    estimation. Items come from the linear decoder; prior from mean_net/Sigma."""
    dec = model.decoders[MKEY].decoder["dec_0"]
    w = dec.weight.detach().cpu().numpy().astype(np.float64)[:, 0]
    bias = dec.bias.detach().cpu().numpy().astype(np.float64)
    b_fit = w
    # vote: model logit = w*z + bias vs DGP theta*b - d  ->  d_fit = -bias
    # gauss: model recon = w*z + bias vs DGP lam*theta + delta -> d_fit = bias
    d_fit = -bias if MODALITY == "vote" else bias
    pdev = next(model.prior.parameters()).device
    M = torch.tensor(corpus.M_prevalence_covariates, dtype=torch.float32, device=pdev)
    mu = model.prior.mean_net(M).detach().cpu().numpy()[:, 0].astype(np.float64)
    sig_u = float(torch.sqrt(model.prior.sigma[0, 0]).detach().cpu())
    pm, _ = exact_posterior(x, V, b_fit, d_fit, mu=mu, sig_u=sig_u)
    bh, se, _ = ols([pm], y)
    c_pf, c_pf_se = float(bh[1]), float(se[1])
    X = np.column_stack([np.ones(len(y)), pm])
    resid = y - X @ bh
    XtX_inv = np.linalg.inv(X.T @ X)
    meat = (X * (resid**2)[:, None]).T @ X * (len(y) / (len(y) - 2.0))
    hc_se = float(np.sqrt((XtX_inv @ meat @ XtX_inv)[1, 1]))
    sig_th = float(np.sqrt(mu.var() + sig_u**2))
    b1 = float(mu[x == 1].mean() - mu[x == 0].mean())
    corr = float(np.corrcoef(pm, theta)[0, 1])
    sgn = float(np.sign(corr)) if corr != 0 else 1.0
    psi_hat = sgn * c_pf * sig_th
    # Delta-method correction for the sigma_theta estimation noise in psi = c*sd:
    # Var(psi_hat) += psi^2*(kappa-1)/(4n), kappa = fitted-marginal kurtosis
    # (theta | fitted model ~ mixture over i of N(mu_i, sig_u^2); moments in
    # closed form). Verified on run 3: closes coverage from ~.90-.93 to ~.95.
    dev2 = (mu - mu.mean()) ** 2
    mu4_hat = float(np.mean(3 * sig_u**4 + 6 * sig_u**2 * dev2 + dev2**2))
    kappa_hat = mu4_hat / sig_th**4
    se_corr = float(np.sqrt((c_pf_se * sig_th) ** 2
                            + psi_hat**2 * (kappa_hat - 1) / (4 * len(y))))
    lo, hi = sorted([sgn * (c_pf - 1.96 * c_pf_se) * sig_th,
                     sgn * (c_pf + 1.96 * c_pf_se) * sig_th])
    lo_h, hi_h = sorted([sgn * (c_pf - 1.96 * hc_se) * sig_th,
                         sgn * (c_pf + 1.96 * hc_se) * sig_th])
    return {"pf_psi": psi_hat, "pf_rf": c_pf * b1,
            "pf_se_psi": c_pf_se * sig_th, "pf_hc_se_psi": hc_se * sig_th,
            "pf_se_psi_corr": se_corr, "pf_kappa_hat": kappa_hat,
            "pf_psi_cover": int(lo <= PSI_TRUE <= hi),
            "pf_psi_cover_hc": int(lo_h <= PSI_TRUE <= hi_h),
            "pf_psi_cover_corr": int(abs(psi_hat - PSI_TRUE) <= 1.96 * se_corr),
            "pf_corr": abs(corr)}


def oracle_reads(x, V, b, d, theta, y):
    ob, ose, _ = ols([theta], y)
    pm, pv = exact_posterior(x, V, b, d)                 # p(theta|V)
    pms, pvs = exact_posterior(x, V, b, d, y=y)          # p(theta|V,y)
    pb, pse, _ = ols([pm], y)
    c_pm, c_pm_se = float(pb[1]), float(pse[1])
    bx, _, _ = ols([x], pm)
    b1_pm = float(bx[1])
    ps = np.sqrt(np.maximum(pv, 0.0))
    pss = np.sqrt(np.maximum(pvs, 0.0))
    cov = float(((theta >= pm - 1.645 * ps) & (theta <= pm + 1.645 * ps)).mean())
    cov_s = float(((theta >= pms - 1.645 * pss) & (theta <= pms + 1.645 * pss)).mean())
    out = {"oracle_ols": {"c": float(ob[1]), "se": float(ose[1])},
           "oracle_pm": {"c_perunit": c_pm, "se": c_pm_se,
                         "perunit_cover": int(c_pm - 1.96 * c_pm_se <= C_TRUE
                                              <= c_pm + 1.96 * c_pm_se),
                         "psi": c_pm * float(pm.std()),
                         "b1": b1_pm, "rf": c_pm * b1_pm,
                         "corr": float(np.corrcoef(pm, theta)[0, 1]),
                         "reliability": float(pm.var() / SIG_TH**2),
                         "cov90_theta": cov},
           "oracle_pm_sup": {"reliability": float(pms.var() / SIG_TH**2),
                             "cov90_theta": cov_s}}
    return out, pm, ps, pms, pss


MKEY = "vote_responses" if MODALITY == "vote" else "emb_w"


def build_model(corpus, with_labels, n_steps, seed):
    kw = dict(predictor_args=PRED_ARGS, labels_in_encoder=True) if with_labels else {}
    return IdealPointNN(
        train_data=corpus, n_ideal_points=1, ae_type="vae", vi_type="mean_field",
        update_prior=with_labels, w_prior=1.0,
        encoder_args={MKEY: {"hidden_dims": HIDDEN}},
        # bias=True = per-item intercept (difficulty d_j / feature intercept delta_j);
        # the default bias=False would misspecify the measurement model
        decoder_args={MKEY: {"hidden_dims": [], "bias": True}},
        batch_size=BATCH, num_steps=n_steps, num_workers=0, print_every_n_steps=10**9,
        optim_args=OPTIM, return_best_model=False, ckpt_folder=tempfile.mkdtemp(),
        seed=seed, device=device, **kw)


def run_rep(seed, verbose):
    corpus_j, corpus_u, x, theta, b, d, V, y = gen_rep(seed)
    oracle, pm, ps, pms, pss = oracle_reads(x, V, b, d, theta, y)
    out = {"seed": seed, **oracle, "joint": [], "two_step": []}
    if verbose:
        o, p = out["oracle_ols"], out["oracle_pm"]
        print(f"  oracle OLS(y~theta): c={o['c']:.3f} (se {o['se']:.3f})   "
              f"oracle-PM: c_perunit={p['c_perunit']:.3f} psi={p['psi']:.3f} "
              f"rf={p['rf']:.3f} rel={p['reliability']:.3f} "
              f"cov90={p['cov90_theta']:.3f}", flush=True)
    for tag, corpus, with_labels, read, cks in (
            ("JOINT", corpus_j, True,
             lambda mm: joint_read(mm, corpus_j, theta, x, pm, ps, pms, pss), CKPTS),
            ("2STEP", corpus_u, False,
             lambda mm: twostep_read(mm, corpus_u, theta, x, y, pm, ps), CKPTS_2S)):
        t0 = time.time()
        model = build_model(corpus, with_labels, cks[0], seed)
        for i, cp in enumerate(cks):
            if i > 0:
                model.num_steps = cp
                model.train(corpus)
            rec = {"step": cp, **read(model)}
            if with_labels and i == len(cks) - 1:
                rec.update(postfit_read(model, corpus, theta, x, y, V))
            out["joint" if with_labels else "two_step"].append(rec)
            if verbose:
                extra = (f" sig2eps={rec['sig2_eps_hat']:.3f} sigu={rec['sig_u_hat']:.3f}"
                         if with_labels else f" psi_cover={rec['psi_cover']}")
                print(f"  [{tag}|{cp:>6}] psi={rec['psi']:.3f} b1_std={rec['b1_std']:.3f} "
                      f"rf={rec['rf']:.3f} corr={rec['corr']:.3f} "
                      f"cov90={rec.get('cov90_theta', rec.get('cov90_theta_yzero')):.3f}"
                      f"{extra}  ({time.time()-t0:.0f}s)", flush=True)
        del model
        gc.collect(); torch.cuda.empty_cache()
    del corpus_j, corpus_u
    gc.collect(); torch.cuda.empty_cache()
    return out


cfg = {"N": N, "J": J, "modality": MODALITY, "item_seed": ITEM_SEED, "lam_sd": LAM_SD, "sig_w2": SIG_W2,
       "ckpts": CKPTS, "ckpts_2step": CKPTS_2S, "batch": BATCH, "hidden": HIDDEN, "reps": REPS,
       "ns_post": NS_POST, "base_seed": BASE_SEED,
       "truth": {"psi": PSI_TRUE, "b1_std": B1STD_TRUE, "rf": RF_TRUE,
                 "c": C_TRUE, "b1": B1, "sig_u": SIG_U, "sig_eps": SIG_EPS}}
print(f"IP [{MODALITY}] {'CALIBRATION (1 rep, trajectory)' if REPS == 1 else f'MC ({REPS} reps)'} | "
      f"N={N} J={J} ckpts={CKPTS} | truth: psi={PSI_TRUE:.3f} b1_std={B1STD_TRUE:.3f} "
      f"rf={RF_TRUE:.3f}", flush=True)

if REPS == 1:
    res = run_rep(BASE_SEED, verbose=True)
    json.dump({"mode": "calibrate", "config": cfg, "rep": res}, open(OUT, "w"), indent=2)
    jf, tf = res["joint"][-1], res["two_step"][-1]
    print(f"\n=== verdict @ {CKPTS[-1]} steps (truth psi={PSI_TRUE:.3f}, rf=1, b1_std=1) ===")
    print(f"  JOINT    psi={jf['psi']:.3f} rf={jf['rf']:.3f} b1_std={jf['b1_std']:.3f} "
          f"cov90(yzero)={jf['cov90_theta_yzero']:.3f}")
    print(f"  JOINT psi-trajectory: {[round(float(r['psi']), 3) for r in res['joint']]}")
    print(f"  2STEP    psi={tf['psi']:.3f} rf={tf['rf']:.3f} b1_std={tf['b1_std']:.3f} "
          f"cov90={tf['cov90_theta']:.3f} psi_cover={tf['psi_cover']}")
    print(f"  ORACLE-PM psi={res['oracle_pm']['psi']:.3f} "
          f"c_perunit={res['oracle_pm']['c_perunit']:.3f} "
          f"cov90={res['oracle_pm']['cov90_theta']:.3f}")
    print(f"saved -> {OUT}", flush=True)
else:
    agg = {"mode": "mc", "config": cfg, "reps": []}
    for r in range(REPS):
        seed = BASE_SEED + r
        t0 = time.time()
        res = run_rep(seed, verbose=False)
        agg["reps"].append(res)
        json.dump(agg, open(OUT, "w"), indent=2)
        jl = " ".join(f"{cp//1000}k:{rec['psi']:.3f}" for cp, rec in zip(CKPTS, res["joint"]))
        print(f"rep {r+1:>2}/{REPS} seed={seed} | joint psi[{jl}] | "
              f"2step psi={res['two_step'][-1]['psi']:.3f} "
              f"({time.time()-t0:.0f}s)", flush=True)
    print("\n=== MC summary (per checkpoint; truth in config) ===")
    for arm, cks in (("joint", CKPTS), ("two_step", CKPTS_2S)):
        for i, cp in enumerate(cks):
            for key, truth in (("psi", PSI_TRUE), ("b1_std", B1STD_TRUE), ("rf", RF_TRUE)):
                v = np.array([rep[arm][i][key] for rep in agg["reps"]])
                print(f"  {arm:>8} @{cp:>6} {key:>6}: E={v.mean():+.3f} "
                      f"bias={v.mean()-truth:+.3f} SD={v.std(ddof=1):.3f} "
                      f"RMSE={np.sqrt(((v-truth)**2).mean()):.3f}")
    for key in ("psi_cover", "b1_std_cover"):
        v = np.array([rep["two_step"][-1][key] for rep in agg["reps"]])
        print(f"  two-step naive-CI coverage ({key}) @final: {v.mean():.3f}")
    for arm, key in (("joint", "cov90_theta_yzero"), ("two_step", "cov90_theta")):
        v = np.array([rep[arm][-1][key] for rep in agg["reps"]])
        print(f"  {arm} avg theta-coverage@90 @final: {v.mean():.3f}")
    v = np.array([rep["oracle_pm"]["cov90_theta"] for rep in agg["reps"]])
    print(f"  oracle-PM avg theta-coverage@90: {v.mean():.3f}")
    print(f"saved -> {OUT}", flush=True)
