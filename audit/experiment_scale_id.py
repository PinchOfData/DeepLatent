"""Is the residual joint-coefficient bias a logistic-normal SCALE-identifiability effect?

Measured fact (audit/experiment_gap_decomp.py): the variational bound is tight (KL~0.076 nats),
so the ~0.026 coef bias is NOT the variational gap. Leading suspect: the prevalence effect x'lambda
is the prior MEAN in logit space, and its scale couples to the learned prior covariance Sigma via the
softmax. Generative truth is Sigma_true = I (simulations.py: sigma=np.eye(K)) and z ~ N(x'lambda, I),
theta=softmax(z). If the model learns Sigma_hat != I, the recovered lambda is a RESCALED true lambda.

Predictions if the scale hypothesis is right:
  (a) the attenuation is a single scalar: recovered W ~ c * Lc, i.e. corr(vec W, vec Lc) ~ 1, slope c<1;
  (b) Sigma_hat departs from I (mean diagonal != 1);
  (c) over training the slope c and Sigma_hat move together -- as Sigma_hat inflates, c shrinks
      (this would explain the U-shape: bias min ~ where Sigma_hat ~ I, then drifts).

Train [256,256] MoG-20 JOINT (same fixed lambda/seed/data as single100k) across checkpoints spanning
the U-shape; at each, read W (Hungarian-aligned + per-covariate centered) and the learned prior Sigma_hat
(prior.get_prior_params(..., return_full_cov=True), topic-permuted to align)."""
import json, time, numpy as np, scipy.sparse, tempfile, torch
from scipy.optimize import linear_sum_assignment
from sklearn.feature_extraction.text import CountVectorizer
from deeplatent import Corpus, GTM, generate_documents

N, K, C, VOCAB, L = 100000, 5, 3, 200, 10
HIDDEN, COMP = [256, 256], 20
CHECKPOINTS = [20000, 40000, 60000, 80000, 100000]
SEED = 1000
OPTIM = {"main": {"lr": 1e-3, "weight_decay": 0.0},
         "prior": {"lr": 1e-4, "weight_decay": 0.0}}
OUT = "audit/results_scale_id.json"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

lambda_fixed = (np.random.default_rng(7).standard_normal((C + 1, K)) * 0.5)
Lc = (lambda_fixed.T - lambda_fixed.T.mean(0, keepdims=True))   # [K, C+1]
true_c = Lc[:, 1:]                                              # [K, C]

def align(tt, et):
    r, c = linear_sum_assignment(-(tt.T @ et)); return {int(i): int(j) for i, j in zip(r, c)}
def center_rows(m): return m - m.mean(0, keepdims=True)

# ---- data (same as single100k) ----
dft, df, tw, lam, _ = generate_documents(num_docs=N, num_topics=K, vocab_size=VOCAB, num_covs=C,
    doc_topic_prior="logistic_normal", min_words=L, max_words=L, lambda_=lambda_fixed, random_seed=SEED)
true_theta = dft[[f"Topic{i}" for i in range(K)]].values.astype(np.float64)
vec = CountVectorizer(); vec.fit(df["doc_clean_0"])
mods = {"text": {"column": "doc_clean_0", "views": {"bow": {"type": "bow", "vectorizer": vec}}}}
corpus = Corpus(df, modalities=mods, prevalence="~ cov_1 + cov_2 + cov_3")
m = corpus.processed_modalities["text"]["bow"]["matrix"]
if scipy.sparse.issparse(m):
    corpus.processed_modalities["text"]["bow"]["matrix"] = np.asarray(m.todense(), np.float32)
prev0 = torch.tensor(corpus.M_prevalence_covariates[:4], dtype=torch.float32, device=device)
print(f"scale-id diagnostic | N={N}, {L} w/doc, {HIDDEN} MoG-20 | Sigma_true = I_{K}\n", flush=True)

def measure(model, step, t0):
    th = model.get_doc_topic_distribution(corpus, num_samples=10).astype(np.float64)
    mp = align(true_theta, th)
    perm = [mp[t] for t in range(K)]                       # true topic t -> model topic perm[t]
    # recovered coefficients (aligned + centered), and bias
    W = model.get_prevalence_coefficients().astype(np.float64)  # [K, C+1] lifted+centered (v0.2.0)
    Wc = center_rows(np.stack([W[perm[t]] for t in range(K)]))
    w = Wc[:, 1:].ravel(); t = true_c.ravel()
    mab = float(np.abs(Wc[:, 1:] - true_c).mean())
    slope = float((w @ t) / (t @ t))                       # single-scale fit recovered ~ c * true
    corr = float(np.corrcoef(w, t)[0, 1])                  # 1 => pure scale; <1 => heterogeneous
    resid_frac = float(np.sum((w - slope * t) ** 2) / np.sum(w ** 2))  # variance NOT explained by c*true
    # learned prior covariance Sigma_hat (covariate-independent), topic-permuted to align.
    # v0.2.0: Sigma lives in (K-1)-dim contrast space; topic permutations only make
    # sense in theta-logit space, so lift via V first (V Sig V^T = centered K x K).
    mu_p, Sig = model.prior.get_prior_params(prev0, return_full_cov=True)
    Vb = model.prior.V.detach().cpu().numpy().astype(np.float64)
    Sig = Vb @ Sig.detach().cpu().numpy().astype(np.float64) @ Vb.T
    Sig = Sig[np.ix_(perm, perm)]
    diag = float(np.mean(np.diag(Sig)))
    offdiag = float(np.mean(np.abs(Sig - np.diag(np.diag(Sig)))))
    evals = np.linalg.eigvalsh(Sig)
    print(f"  [{step:>6}] mean|bias|={mab:.3f} | slope c={slope:.3f} corr={corr:.3f} "
          f"unexpl={resid_frac:.3f} | Sigma_hat diag={diag:.3f} off={offdiag:.3f} "
          f"eig[{evals.min():.2f},{evals.max():.2f}]  ({time.time()-t0:.0f}s)", flush=True)
    return {"step": step, "mean_abs_bias": mab, "slope": slope, "corr": corr,
            "unexplained_frac": resid_frac, "sigma_diag_mean": diag, "sigma_offdiag_mean": offdiag,
            "sigma_eig_min": float(evals.min()), "sigma_eig_max": float(evals.max()),
            "W_aligned": Wc[:, 1:].tolist(), "Sigma_hat": Sig.tolist()}

t0 = time.time()
model = GTM(train_data=corpus, n_topics=K, ae_type="vae", vi_type="mixture_of_gaussians",
            mixture_components=COMP, doc_topic_prior="logistic_normal", update_prior=True, w_prior=1.0,
            encoder_args={"text_bow": {"hidden_dims": HIDDEN}}, decoder_args={"text_bow": {"hidden_dims": []}},
            batch_size=256, num_steps=CHECKPOINTS[0], num_workers=4, print_every_n_steps=10**9,
            optim_args=OPTIM, return_best_model=False, ckpt_folder=tempfile.mkdtemp(), seed=SEED, device=device)
traj = [measure(model, CHECKPOINTS[0], t0)]
json.dump(traj, open(OUT, "w"), indent=2)
for cp in CHECKPOINTS[1:]:
    model.num_steps = cp; model.train(corpus)
    traj.append(measure(model, cp, t0))
    json.dump(traj, open(OUT, "w"), indent=2)

# ---- report ----
print(f"\n=== scale-identifiability diagnostic (Sigma_true = I_{K}) ===")
print(f"{'step':>7} {'mean|bias|':>10} {'slope c':>8} {'corr':>6} {'unexpl':>7} {'Sig diag':>9} {'Sig off':>8}")
for r in traj:
    print(f"{r['step']:>7} {r['mean_abs_bias']:>10.3f} {r['slope']:>8.3f} {r['corr']:>6.3f} "
          f"{r['unexplained_frac']:>7.3f} {r['sigma_diag_mean']:>9.3f} {r['sigma_offdiag_mean']:>8.3f}", flush=True)
bm = min(traj, key=lambda r: r["mean_abs_bias"])
print(f"\nbias-min @ step {bm['step']}: slope c={bm['slope']:.3f}, Sigma_hat diag={bm['sigma_diag_mean']:.3f} (vs 1.0)")
print("interpretation:")
print(" - corr~1 & unexpl~0  => attenuation is a SINGLE SCALE (recovered ~ c*true), not heterogeneous")
print(" - Sigma_hat diag drifting >1 as bias rises => U-shape is prior-variance inflation (scale identifiability)")
print(f"saved -> {OUT}", flush=True)
