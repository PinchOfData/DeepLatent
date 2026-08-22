"""Monte Carlo: two-step vs joint coefficient recovery at SHORT documents.
Setup (per user): 10 words/doc, 10000 docs, MoG-20, 50000 steps, 20 replications.
TRUE prevalence coefficients lambda are FIXED across reps; only the data (covariates,
topic draws, words, and the topic-word matrix beta) is resampled per rep -> a proper
Monte Carlo of each estimator's sampling distribution around the same truth.

Per rep, two estimators of the prevalence coefficients (both identified: Hungarian
topic-alignment + per-covariate centering across topics):
  joint    = prior.mean_net of a JOINT model (covariates in the prior, update_prior=True)
  two_step = OLS of clogit(theta-hat) on X, theta-hat from a covariate-NAIVE stage-1 model
beta is estimated jointly in both. Encoder [256,256], prior lr=1e-4 wd=0.0.

Resumable: skips reps already in audit/results_mc20.json."""
import json, os, time, numpy as np, scipy.sparse, tempfile, torch
from scipy.optimize import linear_sum_assignment
from sklearn.feature_extraction.text import CountVectorizer
from deeplatent import Corpus, GTM, generate_documents

N, K, C, VOCAB, L, STEPS, REPS = 10000, 5, 3, 200, 10, 50000, 20
HIDDEN, COMP = [256, 256], 20
OPTIM = {"main": {"lr": 1e-3, "weight_decay": 0.0},
         "prior": {"lr": 1e-4, "weight_decay": 0.0}}
OUT = "audit/results_mc20.json"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# FIXED true coefficients lambda (shape [C+1, K]); same truth in every rep.
lambda_fixed = (np.random.default_rng(7).standard_normal((C + 1, K)) * 0.5)
Lc = (lambda_fixed.T - lambda_fixed.T.mean(0, keepdims=True))   # identified true coeffs [K, C+1]

def align(tt, et):
    r, c = linear_sum_assignment(-(tt.T @ et)); return {int(i): int(j) for i, j in zip(r, c)}
def clogit(theta):
    lg = np.log(np.clip(theta, 1e-8, 1.0)); return lg - lg.mean(1, keepdims=True)
def ols(X, Y): return np.linalg.lstsq(X, Y, rcond=None)[0]
def center_rows(M): return M - M.mean(0, keepdims=True)

def fit(corpus, prevalence, seed):
    return GTM(train_data=corpus, n_topics=K, ae_type="vae", vi_type="mixture_of_gaussians",
               mixture_components=COMP, doc_topic_prior="logistic_normal",
               update_prior=prevalence, w_prior=1.0,
               encoder_args={"text_bow": {"hidden_dims": HIDDEN}},
               decoder_args={"text_bow": {"hidden_dims": []}},
               batch_size=256, num_steps=STEPS, num_workers=4, print_every_n_steps=10**9,
               optim_args=OPTIM, return_best_model=False, ckpt_folder=tempfile.mkdtemp(),
               seed=seed, device=device)

def aligned_theta(model, corpus, true_theta):
    th = model.get_doc_topic_distribution(corpus, num_samples=10).astype(np.float64)
    mp = align(true_theta, th)
    return np.stack([th[:, mp[t]] for t in range(K)], axis=1), mp

results = json.load(open(OUT)) if os.path.exists(OUT) else []
done = {r["rep"] for r in results}
print(f"true lambda fixed | {L} words/doc, N={N}, MoG-{COMP}, enc {HIDDEN}, {STEPS} steps, {REPS} reps", flush=True)

for rep in range(REPS):
    if rep in done:
        print(f"[rep {rep}] already done, skipping", flush=True); continue
    seed = 1000 + rep
    dft, df, tw, lam, _ = generate_documents(num_docs=N, num_topics=K, vocab_size=VOCAB, num_covs=C,
        doc_topic_prior="logistic_normal", min_words=L, max_words=L, lambda_=lambda_fixed, random_seed=seed)
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
    t0 = time.time()

    jm = fit(joint_corpus, True, seed)
    th_j, mp = aligned_theta(jm, joint_corpus, true_theta)
    theta_corr = float(np.corrcoef(true_theta.ravel(), th_j.ravel())[0, 1])
    W = jm.prior.mean_net.weight.detach().cpu().numpy().astype(np.float64).copy()
    W[:, 0] += jm.prior.mean_net.bias.detach().cpu().numpy().astype(np.float64)
    Wc_joint = center_rows(np.stack([W[mp[t]] for t in range(K)]))

    nm = fit(naive_corpus, False, seed)
    th_n, _ = aligned_theta(nm, naive_corpus, true_theta)
    Wc_two = center_rows(ols(X, clogit(th_n)).T)

    jb = float(np.abs((Wc_joint - Lc)[:, 1:]).mean())
    tb = float(np.abs((Wc_two - Lc)[:, 1:]).mean())
    results.append({"rep": rep, "theta_corr": theta_corr,
                    "joint_mean_abs_bias": jb, "two_step_mean_abs_bias": tb,
                    "joint_coef": Wc_joint[:, 1:].tolist(), "two_step_coef": Wc_two[:, 1:].tolist()})
    json.dump(results, open(OUT, "w"), indent=2)
    print(f"[rep {rep:>2}] theta_corr={theta_corr:.3f} | joint |bias|={jb:.3f} | "
          f"two_step |bias|={tb:.3f} | gain={tb-jb:+.3f}  ({time.time()-t0:.0f}s)", flush=True)
    del jm, nm, joint_corpus, naive_corpus; torch.cuda.empty_cache()

# ---- aggregate ----
jbs = np.array([r["joint_mean_abs_bias"] for r in results])
tbs = np.array([r["two_step_mean_abs_bias"] for r in results])
tcs = np.array([r["theta_corr"] for r in results])
true_c = Lc[:, 1:]                                   # [K, C] identified true coeffs
jc = np.array([r["joint_coef"] for r in results])    # [reps, K, C]
tc = np.array([r["two_step_coef"] for r in results])
print(f"\n=== Monte Carlo ({len(results)} reps): coefficient recovery, 10 words/doc ===")
print(f"{'metric':>26} {'mean':>8} {'sd':>8}")
print(f"{'theta_corr':>26} {tcs.mean():>8.3f} {tcs.std():>8.3f}")
print(f"{'JOINT  mean|bias|':>26} {jbs.mean():>8.3f} {jbs.std():>8.3f}")
print(f"{'TWO-STEP mean|bias|':>26} {tbs.mean():>8.3f} {tbs.std():>8.3f}")
print(f"{'paired gain (two-joint)':>26} {(tbs-jbs).mean():>8.3f} {(tbs-jbs).std():>8.3f}")
print(f"joint beats two-step in {int((jbs<tbs).sum())}/{len(results)} reps")
print(f"\n=== per-coefficient MC bias (mean est - true), averaged over reps ===")
print(f"{'coef':>10} {'true':>8} {'joint_bias':>11} {'twostep_bias':>13}")
for k in range(K):
    for c in range(C):
        jbias = jc[:, k, c].mean() - true_c[k, c]; tbias = tc[:, k, c].mean() - true_c[k, c]
        print(f"{'T'+str(k)+'.cov'+str(c+1):>10} {true_c[k,c]:>8.3f} {jbias:>+11.3f} {tbias:>+13.3f}")
print(f"\nmean |MC bias|: joint={np.abs(jc.mean(0)-true_c).mean():.3f}  "
      f"two_step={np.abs(tc.mean(0)-true_c).mean():.3f}", flush=True)
print("saved -> " + OUT, flush=True)
