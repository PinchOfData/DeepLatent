"""Detailed per-coefficient reporting (replaces aggregate corr/slope).
For L in {10, 80} words/doc, N=10000, 20000 steps, prior lr=1e-4 wd=0.0:
  - JOINT model (covariates in prior, update_prior=True) -> prior coefficients W (mean_net)
  - NAIVE model (no covariates, update_prior=False)      -> covariate-naive theta for the
    TEXTBOOK two-step (OLS of centered-log topic shares on X)
Outputs:
  1. correlation(true topic shares, estimated topic shares): overall + per-topic
  2. per-coefficient table for the PRIOR (mean_net): true / estimate / bias (centered, aligned)
  3. same table for the two-step OLS coefficients (covariate-naive theta)
VAE + mixture_of_gaussians."""
import json, time, numpy as np, scipy.sparse, tempfile, torch
from scipy.optimize import linear_sum_assignment
from sklearn.feature_extraction.text import CountVectorizer
from deeplatent import Corpus, GTM, generate_documents

N, K, C, VOCAB, SEED, STEPS = 10000, 5, 3, 200, 100, 20000
LENGTHS = [10, 80]
OPTIM = {"main": {"lr": 1e-3, "weight_decay": 0.0},
         "prior": {"lr": 1e-4, "weight_decay": 0.0}}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def align(tt, et):
    r, c = linear_sum_assignment(-(tt.T @ et)); return {int(i): int(j) for i, j in zip(r, c)}
def clogit(theta):
    lg = np.log(np.clip(theta, 1e-8, 1.0)); return lg - lg.mean(1, keepdims=True)
def ols(X, Y): return np.linalg.lstsq(X, Y, rcond=None)[0]
def center_rows(M): return M - M.mean(0, keepdims=True)   # center each covariate (column) across topics (rows)

def fit(corpus, prevalence):
    return GTM(train_data=corpus, n_topics=K, ae_type="vae", vi_type="mixture_of_gaussians",
               mixture_components=10, doc_topic_prior="logistic_normal",
               update_prior=prevalence, w_prior=1.0,
               encoder_args={"text_bow": {"hidden_dims": [128]}},
               decoder_args={"text_bow": {"hidden_dims": []}},
               batch_size=256, num_steps=STEPS, num_workers=0, print_every_n_steps=10**9,
               optim_args=OPTIM, return_best_model=False, ckpt_folder=tempfile.mkdtemp(),
               seed=SEED, device=device)

COVS = [f"cov_{j}" for j in range(1, C + 1)]
out = {}
for L in LENGTHS:
    dft, df, tw, lam, _ = generate_documents(num_docs=N, num_topics=K, vocab_size=VOCAB, num_covs=C,
        doc_topic_prior="logistic_normal", min_words=L, max_words=L, random_seed=SEED)
    true_theta = dft[[f"Topic{i}" for i in range(K)]].values.astype(np.float64)
    lamT = lam.T.astype(np.float64)                         # [K, C+1], col 0 = intercept
    vec = CountVectorizer(); vec.fit(df["doc_clean_0"])
    mods = {"text": {"column": "doc_clean_0", "views": {"bow": {"type": "bow", "vectorizer": vec}}}}
    joint_corpus = Corpus(df, modalities=mods, prevalence="~ cov_1 + cov_2 + cov_3")
    naive_corpus = Corpus(df, modalities=mods)
    for cp in (joint_corpus, naive_corpus):
        m = cp.processed_modalities["text"]["bow"]["matrix"]
        if scipy.sparse.issparse(m):
            cp.processed_modalities["text"]["bow"]["matrix"] = np.asarray(m.todense(), np.float32)
    X = joint_corpus.M_prevalence_covariates.astype(np.float64)
    Lc = center_rows(lamT)                                  # identified TRUE coefficients
    t0 = time.time()

    # --- JOINT model: prior coefficients + its topic shares ---
    jm = fit(joint_corpus, prevalence=True)
    th = jm.get_doc_topic_distribution(joint_corpus, num_samples=10).astype(np.float64)
    mp = align(true_theta, th)
    th_al = np.stack([th[:, mp[t]] for t in range(K)], axis=1)
    overall = float(np.corrcoef(true_theta.ravel(), th_al.ravel())[0, 1])
    per_topic = [float(np.corrcoef(true_theta[:, k], th_al[:, k])[0, 1]) for k in range(K)]

    W = jm.prior.mean_net.weight.detach().cpu().numpy().astype(np.float64).copy()
    W[:, 0] += jm.prior.mean_net.bias.detach().cpu().numpy().astype(np.float64)
    Wal = np.stack([W[mp[t]] for t in range(K)])           # align rows to true topic order
    Wc = center_rows(Wal)                                  # identified PRIOR estimate

    # --- NAIVE model: covariate-naive theta -> textbook two-step ---
    nm = fit(naive_corpus, prevalence=False)
    thn = nm.get_doc_topic_distribution(naive_corpus, num_samples=10).astype(np.float64)
    mpn = align(true_theta, thn)
    thn_al = np.stack([thn[:, mpn[t]] for t in range(K)], axis=1)
    Bc = center_rows(ols(X, clogit(thn_al)).T)             # two-step OLS estimate (centered)

    out[str(L)] = {"theta_corr_overall": overall, "theta_corr_per_topic": per_topic,
                   "true": Lc.tolist(), "prior_est": Wc.tolist(), "twostep_est": Bc.tolist(),
                   "prior_bias": (Wc - Lc).tolist(), "twostep_bias": (Bc - Lc).tolist()}

    print(f"\n{'='*78}\nL={L} words/doc | N={N} | {STEPS} steps | prior lr=1e-4 wd=0.0  ({time.time()-t0:.0f}s)")
    print(f"topic-share correlation (true vs estimated): overall = {overall:.3f}")
    print("  per topic: " + "  ".join(f"T{k}={per_topic[k]:.3f}" for k in range(K)))
    print(f"\nPRIOR coefficients (mean_net), identified & topic-aligned -- true / est / bias")
    print(f"{'':>8}" + "".join(f"{c:>26}" for c in COVS))
    print(f"{'topic':>8}" + "".join(f"{'true':>8}{'est':>9}{'bias':>9}" for _ in COVS))
    for k in range(K):
        row = f"{('T'+str(k)):>8}"
        for j in range(1, C + 1):
            row += f"{Lc[k,j]:>8.3f}{Wc[k,j]:>9.3f}{(Wc[k,j]-Lc[k,j]):>+9.3f}"
        print(row)
    mab_p = float(np.abs((Wc - Lc)[:, 1:]).mean())
    mab_t = float(np.abs((Bc - Lc)[:, 1:]).mean())
    print(f"  mean |bias|  prior(mean_net) = {mab_p:.3f}   |   two-step(naive theta) = {mab_t:.3f}")
    print(f"\nTWO-STEP coefficients (OLS of covariate-naive theta on X) -- true / est / bias")
    print(f"{'':>8}" + "".join(f"{c:>26}" for c in COVS))
    print(f"{'topic':>8}" + "".join(f"{'true':>8}{'est':>9}{'bias':>9}" for _ in COVS))
    for k in range(K):
        row = f"{('T'+str(k)):>8}"
        for j in range(1, C + 1):
            row += f"{Lc[k,j]:>8.3f}{Bc[k,j]:>9.3f}{(Bc[k,j]-Lc[k,j]):>+9.3f}"
        print(row)
    json.dump(out, open("audit/coef_bias_results.json", "w"), indent=2)

print("\nsaved -> audit/coef_bias_results.json", flush=True)
