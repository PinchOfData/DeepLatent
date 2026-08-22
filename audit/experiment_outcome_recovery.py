"""Joint (y in encoder) vs two-step recovery of the topic->outcome coefficients.

Outcome generated from the TRUE topic proportions (simulations.py):
    y = theta_true @ label_coeffs + N(0, 0.05)
so label_coeffs[k] is the ground-truth coefficient of topic k on y.

JOINT estimator:
    GTM with labels_in_encoder=True (y concatenated to the encoder input so q(theta|x,y) can
    reach the true posterior p(theta|x,y)) and a LINEAR predictor head (hidden_dims=[]). The
    regression term is a proper Gaussian log-likelihood with a LEARNED noise variance sigma^2,
    so at loss_weight=1 the ELBO is the correct joint likelihood log p(x,y). Recovered b_hat =
    the head's weight. Per Hansen, a tight ELBO of the correct likelihood => b_hat unbiased.

TWO-STEP estimator (the standard baseline):
    Stage 1: fit an UNSUPERVISED topic model (no outcome at all) -> theta_hat.
    Stage 2: OLS of y on theta_hat. Suffers errors-in-variables attenuation (regressing on a
    noisy theta_hat), which the joint estimator is supposed to remove.

Identification: theta is on the simplex (sums to 1) and the head/OLS have an implicit constant,
so b is identified up to an additive constant -> center across topics. Model topics are
Hungarian-aligned to the true topics via theta.
"""
import json, os, time, numpy as np, scipy.sparse, tempfile, torch
from scipy.optimize import linear_sum_assignment
from sklearn.feature_extraction.text import CountVectorizer
from deeplatent import Corpus, GTM, generate_documents

SMOKE = os.environ.get("OUTCOME_SMOKE", "") == "1"
N, K, C, VOCAB, L = (3000 if SMOKE else 100000), 5, 3, 200, 10
HIDDEN, COMP = [256, 256], 20
CHECKPOINTS = [400, 800] if SMOKE else [20000, 40000, 60000, 80000, 100000]
SEED = 1000
OPTIM = {"main": {"lr": 1e-3, "weight_decay": 0.0},
         "prior": {"lr": 1e-4, "weight_decay": 0.0}}
PRED_ARGS = {"y": {"hidden_dims": [], "loss_weight": 1.0}}   # linear head, natural ELBO weight
PIN_SIGMA = True   # pin prior Sigma = I (= Sigma_true) to kill the logit-scale/temperature degeneracy
OUT = "audit/results_outcome_recovery.json"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---- fixed ground truth ----
lambda_fixed = np.random.default_rng(7).standard_normal((C + 1, K)) * 0.5
label_coeffs = np.array([2.0, 1.0, 0.0, -1.0, -2.0])         # true topic -> y coefficients
true_b = label_coeffs - label_coeffs.mean()                  # centered (simplex identifiability)
Lc = lambda_fixed.T - lambda_fixed.T.mean(0, keepdims=True)  # [K, C+1]
true_lam = Lc[:, 1:]                                         # [K, C]

def align(tt, et):
    r, c = linear_sum_assignment(-(tt.T @ et)); return {int(i): int(j) for i, j in zip(r, c)}
def center(v): return v - v.mean()
def center_rows(m): return m - m.mean(0, keepdims=True)
def fit_stats(b):
    mab = float(np.abs(b - true_b).mean())
    slope = float((b @ true_b) / (true_b @ true_b))
    corr = float(np.corrcoef(b, true_b)[0, 1])
    unexpl = float(np.sum((b - slope * true_b) ** 2) / np.sum(b ** 2))
    return {"mab": mab, "slope": slope, "corr": corr, "unexpl": unexpl, "b_hat": b.tolist()}

# ---- data + regression outcome ----
dft, df, tw, lam, lc = generate_documents(
    num_docs=N, num_topics=K, vocab_size=VOCAB, num_covs=C, doc_topic_prior="logistic_normal",
    min_words=L, max_words=L, lambda_=lambda_fixed,
    label_type="regression", label_coeffs=label_coeffs, random_seed=SEED)
true_theta = dft[[f"Topic{i}" for i in range(K)]].values.astype(np.float64)
y = df["label"].values.astype(np.float64)
vec = CountVectorizer(); vec.fit(df["doc_clean_0"])
mods = {"text": {"column": "doc_clean_0", "views": {"bow": {"type": "bow", "vectorizer": vec}}}}

def make_corpus(with_labels):
    kw = {"labels": {"y": {"column": "label", "type": "regression"}}} if with_labels else {}
    c = Corpus(df, modalities=mods, prevalence="~ cov_1 + cov_2 + cov_3", **kw)
    mm = c.processed_modalities["text"]["bow"]["matrix"]
    if scipy.sparse.issparse(mm):
        c.processed_modalities["text"]["bow"]["matrix"] = np.asarray(mm.todense(), np.float32)
    return c

def ols_b(th):  # centered LS coefficients of y on theta (no intercept; full-rank on simplex)
    b, *_ = np.linalg.lstsq(th, y, rcond=None)
    return center(b)

oracle = fit_stats(ols_b(true_theta))
print(f"outcome recovery | N={N}, {L} w/doc, {HIDDEN} MoG-20 | y = theta_true.b + N(0,0.05^2) | "
      f"Sigma{'=I (pinned)' if PIN_SIGMA else ' learned'}")
print(f"  true_b (centered) = {true_b.round(3).tolist()}")
print(f"  oracle OLS(y~theta_true): b={np.round(oracle['b_hat'],3).tolist()} c={oracle['slope']:.3f} "
      f"(sanity: estimand recoverable)\n", flush=True)

corpus_j = make_corpus(with_labels=True)
corpus_ns = make_corpus(with_labels=False)
prev0 = torch.tensor(corpus_j.M_prevalence_covariates[:4], dtype=torch.float32, device=device)

def aligned_perm(model, corpus):
    th = model.get_doc_topic_distribution(corpus, num_samples=10).astype(np.float64)
    mp = align(true_theta, th)
    return [mp[t] for t in range(K)], th

def measure_joint(model, step, t0):
    perm, _ = aligned_perm(model, corpus_j)
    W = model.predictor.predictors["y"].neural_net["pred_0"].weight.detach().cpu().numpy().astype(np.float64)
    jb = fit_stats(center(np.array([W[0, perm[t]] for t in range(K)])))
    # secondary: prevalence lambda-bias, learned sigma^2_y, prior Sigma_hat diag
    Wl = model.prior.mean_net.weight.detach().cpu().numpy().astype(np.float64).copy()
    Wl[:, 0] += model.prior.mean_net.bias.detach().cpu().numpy().astype(np.float64)
    Wlc = center_rows(np.stack([Wl[perm[t]] for t in range(K)]))
    lam_mab = float(np.abs(Wlc[:, 1:] - true_lam).mean())
    _, Sig = model.prior.get_prior_params(prev0, return_full_cov=True)
    sig_diag = float(np.mean(np.diag(Sig.detach().cpu().numpy().astype(np.float64)[np.ix_(perm, perm)])))
    nv = float(torch.exp(model.predictor.noise_log_var["y"]).detach().cpu())
    print(f"  [JOINT  |{step:>6}] mab={jb['mab']:.3f} c={jb['slope']:.3f} corr={jb['corr']:.3f} "
          f"unexpl={jb['unexpl']:.3f} | sig2_y={nv:.4f} lam_mab={lam_mab:.3f} Sig_diag={sig_diag:.3f} "
          f"({time.time()-t0:.0f}s)", flush=True)
    return {"step": step, **jb, "noise_var": nv, "lambda_mab": lam_mab, "sigma_diag": sig_diag}

def measure_twostep(model, step, t0):
    perm, th = aligned_perm(model, corpus_ns)
    th_aln = np.column_stack([th[:, perm[t]] for t in range(K)])
    tb = fit_stats(ols_b(th_aln))
    print(f"  [2STEP  |{step:>6}] mab={tb['mab']:.3f} c={tb['slope']:.3f} corr={tb['corr']:.3f} "
          f"unexpl={tb['unexpl']:.3f}  ({time.time()-t0:.0f}s)", flush=True)
    return {"step": step, **tb}

results = {"meta": {"true_b": true_b.tolist(), "oracle": oracle, "checkpoints": CHECKPOINTS},
           "joint": [], "two_step": []}

def run(tag, corpus, with_labels, measure):
    print(f"=== {tag} ===", flush=True)
    t0 = time.time()
    kw = dict(predictor_args=PRED_ARGS, labels_in_encoder=True) if with_labels else {}
    model = GTM(train_data=corpus, n_topics=K, ae_type="vae", vi_type="mixture_of_gaussians",
                mixture_components=COMP, doc_topic_prior="logistic_normal", update_prior=True, w_prior=1.0,
                learn_prior_cov=not PIN_SIGMA,
                encoder_args={"text_bow": {"hidden_dims": HIDDEN}}, decoder_args={"text_bow": {"hidden_dims": []}},
                batch_size=256, num_steps=CHECKPOINTS[0], num_workers=4, print_every_n_steps=10**9,
                optim_args=OPTIM, return_best_model=False, ckpt_folder=tempfile.mkdtemp(),
                seed=SEED, device=device, **kw)
    out = [measure(model, CHECKPOINTS[0], t0)]
    for cp in CHECKPOINTS[1:]:
        model.num_steps = cp; model.train(corpus)
        out.append(measure(model, cp, t0))
    print(flush=True)
    return out

results["joint"] = run("JOINT (y in encoder, learned sigma^2, loss_weight=1)", corpus_j, True, measure_joint)
json.dump(results, open(OUT, "w"), indent=2)
results["two_step"] = run("TWO-STEP (unsupervised topics, then OLS y~theta_hat)", corpus_ns, False, measure_twostep)
json.dump(results, open(OUT, "w"), indent=2)

# ---- report ----
print("=== joint vs two-step: topic->y coefficient recovery ===")
print(f"true_b={true_b.round(3).tolist()}  oracle_c={oracle['slope']:.3f}")
print(f"{'step':>7} | {'joint_mab':>9} {'joint_c':>8} {'joint_corr':>10} {'sig2_y':>7} | {'2step_mab':>9} {'2step_c':>8}")
for j, t in zip(results["joint"], results["two_step"]):
    print(f"{j['step']:>7} | {j['mab']:>9.3f} {j['slope']:>8.3f} {j['corr']:>10.3f} {j['noise_var']:>7.4f} | "
          f"{t['mab']:>9.3f} {t['slope']:>8.3f}", flush=True)
jbm = min(results["joint"], key=lambda r: r["mab"]); tbm = min(results["two_step"], key=lambda r: r["mab"])
print(f"\njoint    bias-min={jbm['mab']:.3f} @ {jbm['step']} (c={jbm['slope']:.3f})")
print(f"two-step bias-min={tbm['mab']:.3f} @ {tbm['step']} (c={tbm['slope']:.3f})")
print(f"saved -> {OUT}", flush=True)
