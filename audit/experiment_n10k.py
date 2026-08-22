"""N=10000 docs, 10 words/doc, 20000 steps. Prevalence-coefficient recovery.
VAE + mixture_of_gaussians, update_prior=True, new default optimizer (prior lr=1e-3, wd=0).
Reports theta + mean_net + two-step + oracle every 2000 steps."""
import json, time, numpy as np, scipy.sparse, tempfile, torch
from scipy.optimize import linear_sum_assignment
from sklearn.feature_extraction.text import CountVectorizer
from deeplatent import Corpus, GTM, generate_documents

N, K, C, VOCAB, L, SEED = 10000, 5, 3, 200, 80, 100
CHECKPOINTS = list(range(2000, 20001, 2000))   # 2000,4000,...,20000
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
avg = float((corpus.processed_modalities["text"]["bow"]["matrix"] > 0).sum(1).mean())
c_or, _ = cmetrics(ols(X, clogit(true_theta)).T, lamT)
print(f"N={N}, {L} words/doc (~{avg:.1f} distinct), K={K}, C={C}; oracle coef corr={c_or:.3f}\n", flush=True)

def evaluate(model, step, t0):
    th = model.get_doc_topic_distribution(corpus, num_samples=10).astype(np.float64)
    mp = align(true_theta, th)
    th_al = np.stack([th[:, mp[t]] for t in range(K)], axis=1)
    theta_corr = float(np.corrcoef(true_theta.ravel(), th_al.ravel())[0, 1])
    W = model.prior.mean_net.weight.detach().cpu().numpy().astype(np.float64).copy()
    W[:, 0] += model.prior.mean_net.bias.detach().cpu().numpy().astype(np.float64)
    c_mn, s_mn = cmetrics(np.stack([W[mp[t]] for t in range(K)]), lamT)
    c_ts, s_ts = cmetrics(ols(X, clogit(th_al)).T, lamT)
    recon = np.mean(model.train_recon_losses[-100:]); kl = np.mean(model.train_div_losses[-100:])
    print(f"[{step:>6} steps] recon={recon:6.2f} KL={kl:5.2f} | theta={theta_corr:.3f} | "
          f"mean_net corr={c_mn:+.3f}(slope {s_mn:.2f}) | two_step corr={c_ts:+.3f}(slope {s_ts:.2f}) "
          f"| oracle={c_or:.3f}  ({time.time()-t0:.0f}s)", flush=True)
    return {"step": step, "theta": theta_corr, "mean_net": c_mn, "mn_slope": s_mn,
            "two_step": c_ts, "ts_slope": s_ts, "oracle": c_or}

t0 = time.time()
traj = []
model = GTM(train_data=corpus, n_topics=K, ae_type="vae", vi_type="mixture_of_gaussians",
            mixture_components=10, doc_topic_prior="logistic_normal", update_prior=True, w_prior=1.0,
            encoder_args={"text_bow": {"hidden_dims": [128]}}, decoder_args={"text_bow": {"hidden_dims": []}},
            batch_size=256, num_steps=CHECKPOINTS[0], num_workers=0, print_every_n_steps=10**9,
            optim_args={"main": {"lr": 1e-3, "weight_decay": 0.0},
                        "prior": {"lr": 1e-4, "weight_decay": 0.01}},  # OLD optimizer
            return_best_model=False, ckpt_folder=tempfile.mkdtemp(), seed=SEED, device=device)
traj.append(evaluate(model, CHECKPOINTS[0], t0))
for cp in CHECKPOINTS[1:]:
    model.num_steps = cp; model.train(corpus)
    traj.append(evaluate(model, cp, t0))
    json.dump(traj, open("audit/n10k_80w_oldopt_results.json", "w"), indent=2)
print("\nsaved -> audit/n10k_results.json", flush=True)
