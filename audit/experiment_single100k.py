"""Single-rep, LARGE-N version of experiment_mc20: joint vs two-step coefficient
recovery at SHORT documents, but one replication and 100,000 docs instead of an MC.

Rationale: a Monte Carlo over reps maps each estimator's *sampling distribution*.
A single rep at N=100k instead shrinks that sampling distribution directly -- with
10x the documents of mc20 (N=10k), the one estimate is already close to the
estimator's mean, so joint-vs-two-step bias is readable from a single fit.

Setup (matches mc20 except N, REPS and STEPS): 10 words/doc, 100000 docs, MoG-20,
100000 steps (eval every 5000), K=5, C=3, encoder [256,256], prior lr=1e-4 wd=0.0.
TRUE prevalence coefficients lambda are the SAME fixed truth as mc20 (rng(7)),
and seed=1000 == mc20's rep 0, so this run is directly comparable to that rep.

Two estimators of the prevalence coefficients (both identified: Hungarian
topic-alignment + per-covariate centering across topics):
  joint    = prior.mean_net of a JOINT model (covariates in the prior, update_prior=True)
  two_step = OLS of clogit(theta-hat) on X, theta-hat from a covariate-NAIVE stage-1 model
beta is estimated jointly in both.

Resumable across the joint checkpoints + the two-step stage via audit/results_single100k.json."""
import json, os, time, numpy as np, scipy.sparse, tempfile, torch
from scipy.optimize import linear_sum_assignment
from sklearn.feature_extraction.text import CountVectorizer
from deeplatent import Corpus, GTM, generate_documents

N, K, C, VOCAB, L, STEPS = 100000, 5, 3, 200, 10, 100000
CHECKPOINTS = list(range(5000, STEPS + 1, 5000))   # eval/print every 5000 steps
HIDDEN, COMP = [256, 256], 20
SEED = 1000                                  # == mc20 rep 0
OPTIM = {"main": {"lr": 1e-3, "weight_decay": 0.0},
         "prior": {"lr": 1e-4, "weight_decay": 0.0}}
OUT = "audit/results_single100k.json"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# FIXED true coefficients lambda (shape [C+1, K]); identical to mc20.
lambda_fixed = (np.random.default_rng(7).standard_normal((C + 1, K)) * 0.5)
Lc = (lambda_fixed.T - lambda_fixed.T.mean(0, keepdims=True))   # identified true coeffs [K, C+1]

def align(tt, et):
    r, c = linear_sum_assignment(-(tt.T @ et)); return {int(i): int(j) for i, j in zip(r, c)}
def clogit(theta):
    lg = np.log(np.clip(theta, 1e-8, 1.0)); return lg - lg.mean(1, keepdims=True)
def ols(X, Y): return np.linalg.lstsq(X, Y, rcond=None)[0]
def center_rows(M): return M - M.mean(0, keepdims=True)

def fit(corpus, prevalence, num_steps):
    return GTM(train_data=corpus, n_topics=K, ae_type="vae", vi_type="mixture_of_gaussians",
               mixture_components=COMP, doc_topic_prior="logistic_normal",
               update_prior=prevalence, w_prior=1.0,
               encoder_args={"text_bow": {"hidden_dims": HIDDEN}},
               decoder_args={"text_bow": {"hidden_dims": []}},
               batch_size=256, num_steps=num_steps, num_workers=4, print_every_n_steps=10**9,
               optim_args=OPTIM, return_best_model=False, ckpt_folder=tempfile.mkdtemp(),
               seed=SEED, device=device)

def aligned_theta(model, corpus, true_theta):
    th = model.get_doc_topic_distribution(corpus, num_samples=10).astype(np.float64)
    mp = align(true_theta, th)
    return np.stack([th[:, mp[t]] for t in range(K)], axis=1), mp

# ---------------- data (single rep) ----------------
print(f"single rep | {L} words/doc, N={N}, MoG-{COMP}, enc {HIDDEN}, {STEPS} steps, seed={SEED}", flush=True)
t_data = time.time()
dft, df, tw, lam, _ = generate_documents(num_docs=N, num_topics=K, vocab_size=VOCAB, num_covs=C,
    doc_topic_prior="logistic_normal", min_words=L, max_words=L, lambda_=lambda_fixed, random_seed=SEED)
true_theta = dft[[f"Topic{i}" for i in range(K)]].values.astype(np.float64)
vec = CountVectorizer(); vec.fit(df["doc_clean_0"])
mods = {"text": {"column": "doc_clean_0", "views": {"bow": {"type": "bow", "vectorizer": vec}}}}
joint_corpus = Corpus(df, modalities=mods, prevalence="~ cov_1 + cov_2 + cov_3")
naive_corpus = Corpus(df, modalities=mods)
for cp in (joint_corpus, naive_corpus):
    m = cp.processed_modalities["text"]["bow"]["matrix"]
    if scipy.sparse.issparse(m):
        cp.processed_modalities["text"]["bow"]["matrix"] = np.asarray(m.todense(), np.float32)
X = joint_corpus.M_prevalence_covariates.astype(np.float64)
print(f"data ready ({time.time()-t_data:.0f}s)", flush=True)

results = json.load(open(OUT)) if os.path.exists(OUT) else {"joint": [], "two_step": None}

def joint_coef(model, mp):
    W = model.prior.mean_net.weight.detach().cpu().numpy().astype(np.float64).copy()
    W[:, 0] += model.prior.mean_net.bias.detach().cpu().numpy().astype(np.float64)
    return center_rows(np.stack([W[mp[t]] for t in range(K)]))

# ---------------- JOINT model (covariates in the prior) ----------------
done_steps = {r["step"] for r in results["joint"]}
t0 = time.time()
jm = fit(joint_corpus, True, CHECKPOINTS[0])
for i, step in enumerate(CHECKPOINTS):
    if i > 0:
        jm.num_steps = step; jm.train(joint_corpus)
    if step in done_steps:
        print(f"[joint {step:>6}] already recorded, skipping eval", flush=True); continue
    th_j, mp = aligned_theta(jm, joint_corpus, true_theta)
    theta_corr = float(np.corrcoef(true_theta.ravel(), th_j.ravel())[0, 1])
    Wc_joint = joint_coef(jm, mp)
    jb = float(np.abs((Wc_joint - Lc)[:, 1:]).mean())
    recon = float(np.mean(jm.train_recon_losses[-100:])); kl = float(np.mean(jm.train_div_losses[-100:]))
    results["joint"].append({"step": step, "theta_corr": theta_corr, "mean_abs_bias": jb,
                             "max_abs_bias": float(np.abs((Wc_joint - Lc)[:, 1:]).max()),
                             "coef": Wc_joint[:, 1:].tolist()})
    json.dump(results, open(OUT, "w"), indent=2)
    print(f"[joint {step:>6}] recon={recon:6.2f} KL={kl:5.2f} | theta_corr={theta_corr:.3f} | "
          f"mean|bias|={jb:.3f} (max={np.abs((Wc_joint-Lc)[:,1:]).max():.3f})  ({time.time()-t0:.0f}s)", flush=True)
del jm; torch.cuda.empty_cache()

# ---------------- TWO-STEP (naive stage-1 + OLS) ----------------
if results["two_step"] is None:
    t0 = time.time()
    nm = fit(naive_corpus, False, STEPS)
    th_n, _ = aligned_theta(nm, naive_corpus, true_theta)
    theta_corr_n = float(np.corrcoef(true_theta.ravel(), th_n.ravel())[0, 1])
    Wc_two = center_rows(ols(X, clogit(th_n)).T)
    tb = float(np.abs((Wc_two - Lc)[:, 1:]).mean())
    results["two_step"] = {"step": STEPS, "theta_corr": theta_corr_n, "mean_abs_bias": tb,
                           "max_abs_bias": float(np.abs((Wc_two - Lc)[:, 1:]).max()),
                           "coef": Wc_two[:, 1:].tolist()}
    json.dump(results, open(OUT, "w"), indent=2)
    print(f"[two_step {STEPS:>4}] theta_corr={theta_corr_n:.3f} | mean|bias|={tb:.3f} "
          f"(max={np.abs((Wc_two-Lc)[:,1:]).max():.3f})  ({time.time()-t0:.0f}s)", flush=True)
    del nm; torch.cuda.empty_cache()

# ---------------- report ----------------
true_c = Lc[:, 1:]                                # [K, C] identified true coeffs
jr = results["joint"][-1]; tr = results["two_step"]
jc = np.array(jr["coef"]); tc = np.array(tr["coef"])
print(f"\n=== single-rep coefficient recovery, N={N}, 10 words/doc ({STEPS} steps) ===")
print(f"{'metric':>22} {'joint':>9} {'two_step':>9}")
print(f"{'theta_corr':>22} {jr['theta_corr']:>9.3f} {tr['theta_corr']:>9.3f}")
print(f"{'mean|bias|':>22} {jr['mean_abs_bias']:>9.3f} {tr['mean_abs_bias']:>9.3f}")
print(f"{'max|bias|':>22} {jr['max_abs_bias']:>9.3f} {tr['max_abs_bias']:>9.3f}")
print(f"\n=== per-coefficient bias (est - true) ===")
print(f"{'coef':>10} {'true':>8} {'joint_bias':>11} {'twostep_bias':>13}")
for k in range(K):
    for c in range(C):
        print(f"{'T'+str(k)+'.cov'+str(c+1):>10} {true_c[k,c]:>8.3f} "
              f"{jc[k,c]-true_c[k,c]:>+11.3f} {tc[k,c]-true_c[k,c]:>+13.3f}", flush=True)
print("\nsaved -> " + OUT, flush=True)
