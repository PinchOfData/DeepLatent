"""Monte-Carlo calibration for the supervised topic->outcome coefficient.

GOAL: find the CHEAPEST per-rep config (N, encoder depth, MoG components, num_steps) at which the
JOINT estimator stays ~unbiased AND clearly beats the two-step baseline, so that many MC reps run in
reasonable wall-clock. The full-scale config (N=100k, [256,256] MoG-20, 60k steps) costs ~1600s/sigma
-- far too slow for an MC with dozens of reps. Here we shrink everything and verify one rep still works.

Per rep the DGP is drawn fresh (generate_documents reseeds np.random from `seed`, so docs, topic-word
matrix, theta_true, covariates and y-noise all change), while the ESTIMAND is fixed:
    y = theta_true @ label_coeffs + N(0, sigma^2),   true_b = center(label_coeffs).
Topics are Hungarian-aligned to truth and centered (simplex additive-constant identification) each rep.

JOINT  : GTM, labels_in_encoder=True, linear head, learned sigma^2 (noise_log_var), learned Sigma.
TWO-STEP: unsupervised GTM (no y) -> theta_hat, then OLS of y on theta_hat (errors-in-variables -> attenuated).
ORACLE : OLS(y ~ theta_true) -- the best any estimator on this draw can do (finite-N floor).

MODES (env):
  REPS=1 (default) CALIBRATION: one rep, joint+two-step over a checkpoint TRAJECTORY vs the oracle floor.
                   Read it to pick num_steps and confirm the small encoder/MoG keeps joint mab << two-step.
  REPS>1           MONTE CARLO: fixed num_steps=NUM_STEPS, loop fresh seeds, aggregate across reps:
                   mean bias, SD, RMSE of b_hat for joint vs two-step (+ oracle as the floor).
"""
import json, os, time, gc, numpy as np, scipy.sparse, tempfile, torch
from scipy.optimize import linear_sum_assignment
from sklearn.feature_extraction.text import CountVectorizer
from deeplatent import Corpus, GTM, generate_documents

# ---- knobs (calibration defaults; override via env for the real MC) ----
REPS      = int(os.environ.get("MC_REPS", "1"))
N         = int(os.environ.get("MC_N", "10000"))
SIGMA     = float(os.environ.get("MC_SIGMA", "1.0"))
HIDDEN    = json.loads(os.environ.get("MC_HIDDEN", "[128, 128]"))
COMP      = int(os.environ.get("MC_COMP", "10"))
NUM_STEPS = int(os.environ.get("MC_STEPS", "12000"))            # used in MC mode (REPS>1)
# calibration trajectory; in MC mode we only keep the final step
CHECKPOINTS = json.loads(os.environ.get("MC_CKPTS", "[4000, 8000, 12000]"))
K, C, VOCAB, L = 5, 3, 200, 10
ANCHOR = int(os.environ.get("MC_ANCHOR", "10"))   # per-topic exclusive anchor words (topic separability)
BASE_SEED = int(os.environ.get("MC_BASE_SEED", "1000"))
LR        = float(os.environ.get("MC_LR", "1e-3"))             # main optimizer lr (convergence-speed lever)
PRIOR_LR  = float(os.environ.get("MC_PRIOR_LR", "1e-4"))
BATCH     = int(os.environ.get("MC_BATCH", "256"))
OPTIM = {"main": {"lr": LR, "weight_decay": 0.0}, "prior": {"lr": PRIOR_LR, "weight_decay": 0.0}}
PRED_ARGS = {"y": {"hidden_dims": [], "loss_weight": 1.0}}      # linear head, natural ELBO weight
OUT = os.environ.get("MC_OUT", "audit/results_mc_calibrate.json")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---- fixed ground truth (shared across reps) ----
lambda_fixed = np.random.default_rng(7).standard_normal((C + 1, K)) * 0.5
label_coeffs = np.array([2.0, 1.0, 0.0, -1.0, -2.0])
true_b = label_coeffs - label_coeffs.mean()

def align(tt, et):
    r, c = linear_sum_assignment(-(tt.T @ et)); return {int(i): int(j) for i, j in zip(r, c)}
def center(v): return v - v.mean()
def fit_stats(b):
    return {"mab": float(np.abs(b - true_b).mean()),
            "slope": float((b @ true_b) / (true_b @ true_b)),
            "corr": float(np.corrcoef(b, true_b)[0, 1]),
            "b_hat": b.tolist()}

def gen_rep(seed):
    """Fresh DGP draw; returns the two corpora, theta_true, y, and the oracle OLS fit."""
    dft, df, tw, lam, lc = generate_documents(
        num_docs=N, num_topics=K, vocab_size=VOCAB, num_covs=C, doc_topic_prior="logistic_normal",
        min_words=L, max_words=L, lambda_=lambda_fixed, anchor_words=ANCHOR,
        label_type="regression", label_coeffs=label_coeffs, random_seed=seed)
    true_theta = dft[[f"Topic{i}" for i in range(K)]].values.astype(np.float64)
    # generate_documents HARDCODES outcome noise at 0.05 (simulations.py:171) and has no knob for it,
    # so we overwrite the label here to honor MC_SIGMA: y = theta_true . b + N(0, SIGMA^2).
    mu_y = true_theta @ label_coeffs                       # = simulations.py "logits" (noiseless mean)
    y = mu_y + np.random.default_rng(seed + 999).normal(0, SIGMA, len(df))
    df = df.copy(); df["label"] = y
    y = y.astype(np.float64)
    vec = CountVectorizer(); vec.fit(df["doc_clean_0"])
    mods = {"text": {"column": "doc_clean_0", "views": {"bow": {"type": "bow", "vectorizer": vec}}}}

    def mk(with_labels):
        kw = {"labels": {"y": {"column": "label", "type": "regression"}}} if with_labels else {}
        c = Corpus(df, modalities=mods, prevalence="~ cov_1 + cov_2 + cov_3", **kw)
        mm = c.processed_modalities["text"]["bow"]["matrix"]
        if scipy.sparse.issparse(mm):
            c.processed_modalities["text"]["bow"]["matrix"] = np.asarray(mm.todense(), np.float32)
        return c

    ob, *_ = np.linalg.lstsq(true_theta, y, rcond=None)
    oracle = fit_stats(center(ob))
    return mk(True), mk(False), true_theta, y, oracle

def build_model(corpus, with_labels, n_steps, seed):
    # JOINT (with_labels): covariate-driven prior mean + LEARNED Sigma (its supervised machinery).
    # TWO-STEP: a plain logistic STANDARD normal prior N(0, I) -- update_prior=False, learn_prior_cov=False --
    #   so there is no learned-Sigma scale drift/overshoot masking its true errors-in-variables attenuation.
    kw = dict(predictor_args=PRED_ARGS, labels_in_encoder=True) if with_labels else {}
    return GTM(train_data=corpus, n_topics=K, ae_type="vae", vi_type="mixture_of_gaussians",
               mixture_components=COMP, doc_topic_prior="logistic_normal", update_prior=with_labels, w_prior=1.0,
               learn_prior_cov=with_labels,
               encoder_args={"text_bow": {"hidden_dims": HIDDEN}}, decoder_args={"text_bow": {"hidden_dims": []}},
               batch_size=BATCH, num_steps=n_steps, num_workers=0, print_every_n_steps=10**9,
               optim_args=OPTIM, return_best_model=False, ckpt_folder=tempfile.mkdtemp(),
               seed=seed, device=device, **kw)

def perm_of(model, corpus, true_theta):
    th = model.get_doc_topic_distribution(corpus, num_samples=10).astype(np.float64)
    mp = align(true_theta, th)
    return [mp[t] for t in range(K)], th

def joint_b(model, corpus, true_theta):
    perm, _ = perm_of(model, corpus, true_theta)
    W = model.predictor.predictors["y"].neural_net["pred_0"].weight.detach().cpu().numpy().astype(np.float64)
    bh = center(np.array([W[0, perm[t]] for t in range(K)]))
    nv = float(torch.exp(model.predictor.noise_log_var["y"]).detach().cpu())
    return fit_stats(bh), nv

def twostep_b(model, corpus, true_theta, y):
    perm, th = perm_of(model, corpus, true_theta)
    th_aln = np.column_stack([th[:, perm[t]] for t in range(K)])
    ob, *_ = np.linalg.lstsq(th_aln, y, rcond=None)
    return fit_stats(center(ob))

def run_rep(seed, checkpoints, verbose):
    """Train joint + two-step on one DGP draw; measure b_hat at each checkpoint."""
    corpus_j, corpus_u, true_theta, y, oracle = gen_rep(seed)
    if verbose:
        print(f"  oracle OLS(y~theta_true): b={np.round(oracle['b_hat'],3).tolist()} "
              f"c={oracle['slope']:.3f} mab={oracle['mab']:.3f}  (finite-N floor)", flush=True)
    out = {"oracle": oracle, "joint": [], "two_step": []}
    arms = [("JOINT", corpus_j, True, lambda m: joint_b(m, corpus_j, true_theta))]
    if os.environ.get("MC_JOINT_ONLY", "") != "1":
        arms.append(("2STEP", corpus_u, False, lambda m: twostep_b(m, corpus_u, true_theta, y)))
    for tag, corpus, with_labels, meas in arms:
        t0 = time.time()
        model = build_model(corpus, with_labels, checkpoints[0], seed)
        for i, cp in enumerate(checkpoints):
            if i > 0:
                model.num_steps = cp; model.train(corpus)
            r = meas(model)
            if with_labels:
                fs, nv = r; rec = {"step": cp, **fs, "sigma2_hat": nv}
            else:
                rec = {"step": cp, **r}
            out["joint" if with_labels else "two_step"].append(rec)
            if verbose:
                extra = f" sig2={rec.get('sigma2_hat', float('nan')):.3f}" if with_labels else ""
                print(f"  [{tag}|{cp:>6}] mab={rec['mab']:.3f} c={rec['slope']:.3f} "
                      f"corr={rec['corr']:.3f}{extra}  ({time.time()-t0:.0f}s)", flush=True)
        del model                                  # free the arm's model before the next arm/rep
        gc.collect(); torch.cuda.empty_cache()
    del corpus_j, corpus_u
    gc.collect(); torch.cuda.empty_cache()
    return out

print(f"MC {'CALIBRATION (1 rep, trajectory)' if REPS == 1 else f'RUN ({REPS} reps)'} | "
      f"N={N} sigma={SIGMA} HIDDEN={HIDDEN} MoG-{COMP} | true_b={true_b.round(2).tolist()}")
print(f"per-rep: joint(labels_in_encoder, learned sig2, learned Sigma) vs two-step(unsup+OLS); "
      f"oracle=OLS(y~theta_true)\n", flush=True)

if REPS == 1:
    # CALIBRATION: one rep, full checkpoint trajectory, verbose
    res = run_rep(BASE_SEED, CHECKPOINTS, verbose=True)
    json.dump({"mode": "calibrate", "config": {"N": N, "sigma": SIGMA, "hidden": HIDDEN, "comp": COMP,
               "checkpoints": CHECKPOINTS}, "true_b": true_b.tolist(), "rep": res}, open(OUT, "w"), indent=2)
    jf = res["joint"][-1]
    print(f"\n=== calibration verdict @ {CHECKPOINTS[-1]} steps ===")
    print(f"  oracle   mab={res['oracle']['mab']:.3f} c={res['oracle']['slope']:.3f}")
    print(f"  JOINT    mab={jf['mab']:.3f} c={jf['slope']:.3f} corr={jf['corr']:.3f} (sig2={jf['sigma2_hat']:.3f})")
    print(f"  JOINT c-trajectory: {[round(r['slope'],3) for r in res['joint']]} over {CHECKPOINTS}")
    if res["two_step"]:
        tf = res["two_step"][-1]
        print(f"  TWO-STEP mab={tf['mab']:.3f} c={tf['slope']:.3f} corr={tf['corr']:.3f}")
        print(f"  -> joint {'BEATS' if jf['mab'] < tf['mab'] else 'does NOT beat'} two-step; "
              f"joint c {'~unbiased' if jf['slope'] > 0.9 else 'still attenuated'}")
    print(f"saved -> {OUT}", flush=True)
else:
    # MONTE CARLO: many reps; record b_hat at EVERY checkpoint so E[c] reveals the U-shape in
    # expectation (the coef bias is U-shaped in training time: attenuated early, c~1 at the sweet
    # spot, overshoot c>1 late). CHECKPOINTS should bracket the sweet spot (batch=1024: ~[16k..28k]).
    agg = {"mode": "mc", "config": {"N": N, "sigma": SIGMA, "hidden": HIDDEN, "comp": COMP,
           "lr": LR, "batch": BATCH, "checkpoints": CHECKPOINTS, "reps": REPS},
           "true_b": true_b.tolist(), "reps": []}
    joint_by_cp = {cp: [] for cp in CHECKPOINTS}
    twostep_by_cp = {cp: [] for cp in CHECKPOINTS}
    for r in range(REPS):
        seed = BASE_SEED + r
        t0 = time.time()
        res = run_rep(seed, CHECKPOINTS, verbose=False)
        for rec in res["joint"]:
            joint_by_cp[rec["step"]].append(rec["b_hat"])
        for rec in res["two_step"]:
            twostep_by_cp[rec["step"]].append(rec["b_hat"])
        agg["reps"].append({"seed": seed, "joint": res["joint"], "two_step": res["two_step"],
                            "oracle": res["oracle"]})
        json.dump(agg, open(OUT, "w"), indent=2)
        jc = " ".join(f"{cp//1000}k:{r0['slope']:.3f}" for cp in CHECKPOINTS
                      for r0 in res["joint"] if r0["step"] == cp)
        print(f"rep {r+1:>2}/{REPS} seed={seed} | joint c[{jc}] | "
              f"oracle mab={res['oracle']['mab']:.3f} ({time.time()-t0:.0f}s)", flush=True)
    def agg_report(name, by_cp):
        print(f"  {name} -- E[b_hat] across reps, per checkpoint:")
        for cp in CHECKPOINTS:
            B = np.array(by_cp[cp])
            if not len(B):
                continue
            meanb = B.mean(0); bias = meanb - true_b
            c = float((meanb @ true_b) / (true_b @ true_b))
            rmse = float(np.sqrt(((B - true_b) ** 2).mean(0)).mean())
            print(f"    {cp:>7}: E[c]={c:.3f}  mean|bias of E[b]|={np.abs(bias).mean():.3f}  "
                  f"meanSD={B.std(0).mean():.3f}  RMSE={rmse:.3f}")
    print(f"\n=== Monte Carlo ({REPS} reps) === true_b={true_b.round(3).tolist()}")
    print("E[c]=1 marks the population sweet spot; watch it cross 1 across checkpoints (U-shape).")
    agg_report("JOINT", joint_by_cp)
    agg_report("TWO-STEP", twostep_by_cp)
    print(f"saved -> {OUT}", flush=True)
