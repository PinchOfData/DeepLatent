"""Does the JOINT model's prior-coefficient bias really vanish, or floor?
10 words/doc only, N=10000, train to 120000 steps, checkpoint every 20k.
Same architecture as before (VAE + MoG-10, encoder [128], linear BoW decoder),
prior lr=1e-4 wd=0.0, update_prior=True. Reports prior coefficient mean|bias|,
the per-coefficient bias, and topic-share correlation at each checkpoint."""
import json, time, numpy as np, scipy.sparse, tempfile, torch
from scipy.optimize import linear_sum_assignment
from sklearn.feature_extraction.text import CountVectorizer
from deeplatent import Corpus, GTM, generate_documents

N, K, C, VOCAB, SEED, L = 10000, 5, 3, 200, 100, 10
CHECKPOINTS = [20000, 40000, 60000, 80000, 100000, 120000]
OPTIM = {"main": {"lr": 1e-3, "weight_decay": 0.0},
         "prior": {"lr": 1e-4, "weight_decay": 0.0}}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def align(tt, et):
    r, c = linear_sum_assignment(-(tt.T @ et)); return {int(i): int(j) for i, j in zip(r, c)}
def clogit(theta):
    lg = np.log(np.clip(theta, 1e-8, 1.0)); return lg - lg.mean(1, keepdims=True)
def ols(X, Y): return np.linalg.lstsq(X, Y, rcond=None)[0]
def center_rows(M): return M - M.mean(0, keepdims=True)

dft, df, tw, lam, _ = generate_documents(num_docs=N, num_topics=K, vocab_size=VOCAB, num_covs=C,
    doc_topic_prior="logistic_normal", min_words=L, max_words=L, random_seed=SEED)
true_theta = dft[[f"Topic{i}" for i in range(K)]].values.astype(np.float64)
lamT = lam.T.astype(np.float64); Lc = center_rows(lamT)
vec = CountVectorizer(); vec.fit(df["doc_clean_0"])
corpus = Corpus(df, modalities={"text": {"column": "doc_clean_0",
                "views": {"bow": {"type": "bow", "vectorizer": vec}}}}, prevalence="~ cov_1 + cov_2 + cov_3")
m = corpus.processed_modalities["text"]["bow"]["matrix"]
if scipy.sparse.issparse(m):
    corpus.processed_modalities["text"]["bow"]["matrix"] = np.asarray(m.todense(), np.float32)
X = corpus.M_prevalence_covariates.astype(np.float64)
avg = float((corpus.processed_modalities["text"]["bow"]["matrix"] > 0).sum(1).mean())
print(f"L={L} words/doc (~{avg:.1f} distinct), N={N}, K={K} | training JOINT model to "
      f"{CHECKPOINTS[-1]} steps | prior lr=1e-4 wd=0.0\n", flush=True)

def evaluate(model, step, t0):
    th = model.get_doc_topic_distribution(corpus, num_samples=20).astype(np.float64)
    mp = align(true_theta, th)
    th_al = np.stack([th[:, mp[t]] for t in range(K)], axis=1)
    overall = float(np.corrcoef(true_theta.ravel(), th_al.ravel())[0, 1])
    W = model.prior.mean_net.weight.detach().cpu().numpy().astype(np.float64).copy()
    W[:, 0] += model.prior.mean_net.bias.detach().cpu().numpy().astype(np.float64)
    Wc = center_rows(np.stack([W[mp[t]] for t in range(K)]))
    bias = (Wc - Lc)[:, 1:]                       # drop intercept
    mab = float(np.abs(bias).mean())
    recon = np.mean(model.train_recon_losses[-100:]); kl = np.mean(model.train_div_losses[-100:])
    print(f"  [{step:>6}] recon={recon:6.2f} KL={kl:5.2f} | theta_corr={overall:.3f} | "
          f"prior mean|bias|={mab:.3f} (max|bias|={np.abs(bias).max():.3f})  ({time.time()-t0:.0f}s)", flush=True)
    return {"step": step, "theta_corr": overall, "mean_abs_bias": mab,
            "max_abs_bias": float(np.abs(bias).max()), "bias": bias.tolist()}

t0 = time.time(); traj = []
model = GTM(train_data=corpus, n_topics=K, ae_type="vae", vi_type="mixture_of_gaussians",
            mixture_components=10, doc_topic_prior="logistic_normal", update_prior=True, w_prior=1.0,
            encoder_args={"text_bow": {"hidden_dims": [128]}}, decoder_args={"text_bow": {"hidden_dims": []}},
            batch_size=256, num_steps=CHECKPOINTS[0], num_workers=0, print_every_n_steps=10**9,
            optim_args=OPTIM, return_best_model=False, ckpt_folder=tempfile.mkdtemp(), seed=SEED, device=device)
traj.append(evaluate(model, CHECKPOINTS[0], t0))
for cp in CHECKPOINTS[1:]:
    model.num_steps = cp; model.train(corpus)
    traj.append(evaluate(model, cp, t0))
    json.dump(traj, open("audit/results_joint_long.json", "w"), indent=2)

print("\n=== prior coefficient mean|bias| vs steps (10 words/doc) ===")
print(f"{'step':>8} {'theta_corr':>11} {'mean|bias|':>11} {'max|bias|':>10}")
for r in traj:
    print(f"{r['step']:>8} {r['theta_corr']:>11.3f} {r['mean_abs_bias']:>11.3f} {r['max_abs_bias']:>10.3f}", flush=True)
print("\nsaved -> audit/results_joint_long.json", flush=True)
