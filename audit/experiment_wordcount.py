"""Why prevalence-coefficient recovery is great with many words/doc and poor with few:
controlled sweep over document length, EVERYTHING ELSE FIXED, trained to convergence.
Reports theta recovery + mean_net + two-step + oracle at 6k and 12k steps (the 6k/12k
pair separates 'word count' from 'under-training'). Prevalence setup (theta ~ softmax(X lambda)),
VAE + mixture_of_gaussians, update_prior=True, new default optimizer (prior lr=1e-3, wd=0).
"""
import json, time, numpy as np, scipy.sparse, tempfile, torch
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
from sklearn.feature_extraction.text import CountVectorizer
from deeplatent import Corpus, GTM, generate_documents

LENGTHS = [10, 20, 40, 80, 160]
N, K, C, VOCAB, SEED = 3000, 5, 3, 200, 100
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def align(tt, et):
    r, c = linear_sum_assignment(-(tt.T @ et)); return {int(i): int(j) for i, j in zip(r, c)}
def cmetrics(estKC, trueKC):
    cr = lambda M: M - M.mean(0, keepdims=True)
    t = cr(trueKC)[:, 1:].ravel(); e = cr(estKC)[:, 1:].ravel()
    return float(np.corrcoef(t, e)[0, 1]), float(np.polyfit(t, e, 1)[0])
def clogit(theta):
    lg = np.log(np.clip(theta, 1e-8, 1.0)); return lg - lg.mean(1, keepdims=True)
def ols(X, Y): return np.linalg.lstsq(X, Y, rcond=None)[0]

def evaluate(model, corpus, X, true_theta, lamT):
    th = model.get_doc_topic_distribution(corpus, num_samples=20).astype(np.float64)
    mp = align(true_theta, th)
    th_al = np.stack([th[:, mp[t]] for t in range(K)], axis=1)
    theta_corr = float(np.corrcoef(true_theta.ravel(), th_al.ravel())[0, 1])
    W = model.prior.mean_net.weight.detach().cpu().numpy().astype(np.float64).copy()
    W[:, 0] += model.prior.mean_net.bias.detach().cpu().numpy().astype(np.float64)
    c_mn, s_mn = cmetrics(np.stack([W[mp[t]] for t in range(K)]), lamT)
    c_ts, s_ts = cmetrics(ols(X, clogit(th_al)).T, lamT)
    return theta_corr, c_mn, s_mn, c_ts, s_ts

rows = []
for L in LENGTHS:
    dft, df, tw, lam, _ = generate_documents(num_docs=N, num_topics=K, vocab_size=VOCAB, num_covs=C,
        doc_topic_prior="logistic_normal", min_words=L, max_words=L, random_seed=SEED)
    true_theta = dft[[f"Topic{i}" for i in range(K)]].values.astype(np.float64)
    lamT = lam.T.astype(np.float64)
    vec = CountVectorizer(); vec.fit(df["doc_clean_0"])
    corpus = Corpus(df, modalities={"text": {"column": "doc_clean_0",
                    "views": {"bow": {"type": "bow", "vectorizer": vec}}}}, prevalence="~ cov_1 + cov_2 + cov_3")
    m = corpus.processed_modalities["text"]["bow"]["matrix"]
    if scipy.sparse.issparse(m):
        corpus.processed_modalities["text"]["bow"]["matrix"] = np.asarray(m.todense(), np.float32)
    X = corpus.M_prevalence_covariates.astype(np.float64)
    c_or, _ = cmetrics(ols(X, clogit(true_theta)).T, lamT)
    avg_nonzero = float((corpus.processed_modalities["text"]["bow"]["matrix"] > 0).sum(1).mean())
    t0 = time.time()

    model = GTM(train_data=corpus, n_topics=K, ae_type="vae", vi_type="mixture_of_gaussians",
                mixture_components=10, doc_topic_prior="logistic_normal", update_prior=True, w_prior=1.0,
                encoder_args={"text_bow": {"hidden_dims": [128]}}, decoder_args={"text_bow": {"hidden_dims": []}},
                batch_size=256, num_steps=6000, num_workers=0, print_every_n_steps=10**9,
                return_best_model=False, ckpt_folder=tempfile.mkdtemp(), seed=SEED, device=device)
    th6, mn6, smn6, ts6, sts6 = evaluate(model, corpus, X, true_theta, lamT)
    model.num_steps = 12000; model.train(corpus)
    th12, mn12, smn12, ts12, sts12 = evaluate(model, corpus, X, true_theta, lamT)

    rows.append({"L": L, "avg_distinct_words": avg_nonzero, "oracle": c_or,
                 "theta6": th6, "theta12": th12,
                 "mean_net6": mn6, "mean_net12": mn12, "mn_slope12": smn12,
                 "two_step6": ts6, "two_step12": ts12, "ts_slope12": sts12})
    print(f"L={L:>3} (~{avg_nonzero:.1f} distinct words/doc) | "
          f"theta: {th6:.2f}->{th12:.2f} | "
          f"mean_net: {mn6:+.2f}->{mn12:+.2f}(slope {smn12:.2f}) | "
          f"two_step: {ts6:+.2f}->{ts12:+.2f}(slope {sts12:.2f}) | oracle={c_or:.3f} "
          f"({time.time()-t0:.0f}s)", flush=True)

print("\n=== recovery vs words/doc (converged, 12k steps) ===")
print(f"{'words':>6} {'theta':>7} {'mean_net':>9} {'mn_slope':>9} {'two_step':>9} {'oracle':>7}")
for r in rows:
    print(f"{r['L']:>6} {r['theta12']:>7.2f} {r['mean_net12']:>+9.2f} {r['mn_slope12']:>9.2f} "
          f"{r['two_step12']:>+9.2f} {r['oracle']:>7.3f}", flush=True)
json.dump(rows, open("audit/wordcount_results.json", "w"), indent=2)

Ls = [r["L"] for r in rows]
fig, ax = plt.subplots(figsize=(7.5, 5))
ax.plot(Ls, [r["theta12"] for r in rows], "d-", label="theta recovery (corr)")
ax.plot(Ls, [r["mean_net12"] for r in rows], "o-", label="mean_net coef (corr)")
ax.plot(Ls, [r["two_step12"] for r in rows], "s--", label="two-step coef (corr)")
ax.plot(Ls, [r["oracle"] for r in rows], "k:", label="oracle (true theta)")
ax.set_xscale("log"); ax.set_xticks(Ls); ax.set_xticklabels(Ls)
ax.set_xlabel("words per document"); ax.set_ylabel("recovery (correlation)")
ax.set_title("Recovery degrades as documents get shorter (measurement error in theta)")
ax.legend(); ax.grid(alpha=0.3); fig.tight_layout()
fig.savefig("audit/wordcount_recovery.png", dpi=150)
print("\nsaved -> audit/wordcount_results.json, audit/wordcount_recovery.png", flush=True)
