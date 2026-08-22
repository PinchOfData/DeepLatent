"""Is the imperfect topic-share recovery just under-training? Train to 40000 steps,
checkpoint at 10k/20k/30k/40k, and watch the correlation between true and estimated
topic shares. Same architecture as before (VAE + MoG-10, encoder [128], linear BoW
decoder), prior lr=1e-4 wd=0.0. N=10000. Runs L=10 and L=80 words/doc."""
import json, time, numpy as np, scipy.sparse, tempfile, torch
from scipy.optimize import linear_sum_assignment
from sklearn.feature_extraction.text import CountVectorizer
from deeplatent import Corpus, GTM, generate_documents

N, K, C, VOCAB, SEED = 10000, 5, 3, 200, 100
LENGTHS = [10, 80]
CHECKPOINTS = [10000, 20000, 30000, 40000]
OPTIM = {"main": {"lr": 1e-3, "weight_decay": 0.0},
         "prior": {"lr": 1e-4, "weight_decay": 0.0}}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def align(tt, et):
    r, c = linear_sum_assignment(-(tt.T @ et)); return {int(i): int(j) for i, j in zip(r, c)}
def clogit(theta):
    lg = np.log(np.clip(theta, 1e-8, 1.0)); return lg - lg.mean(1, keepdims=True)
def ols(X, Y): return np.linalg.lstsq(X, Y, rcond=None)[0]
def center_rows(M): return M - M.mean(0, keepdims=True)

def build(L):
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
    return corpus, X, true_theta, lamT, avg

def evaluate(model, corpus, X, true_theta, lamT, step, t0):
    th = model.get_doc_topic_distribution(corpus, num_samples=20).astype(np.float64)
    mp = align(true_theta, th)
    th_al = np.stack([th[:, mp[t]] for t in range(K)], axis=1)
    overall = float(np.corrcoef(true_theta.ravel(), th_al.ravel())[0, 1])
    per_topic = [float(np.corrcoef(true_theta[:, k], th_al[:, k])[0, 1]) for k in range(K)]
    W = model.prior.mean_net.weight.detach().cpu().numpy().astype(np.float64).copy()
    W[:, 0] += model.prior.mean_net.bias.detach().cpu().numpy().astype(np.float64)
    Wc = center_rows(np.stack([W[mp[t]] for t in range(K)]))
    mab = float(np.abs((Wc - center_rows(lamT))[:, 1:].ravel()).mean())
    recon = np.mean(model.train_recon_losses[-100:]); kl = np.mean(model.train_div_losses[-100:])
    print(f"  [{step:>6}] recon={recon:7.2f} KL={kl:5.2f} | theta_corr={overall:.3f} "
          f"[{' '.join(f'{p:.2f}' for p in per_topic)}] | prior mean|bias|={mab:.3f}  ({time.time()-t0:.0f}s)", flush=True)
    return {"step": step, "theta_corr": overall, "per_topic": per_topic, "prior_mean_abs_bias": mab}

out = {}
for L in LENGTHS:
    corpus, X, true_theta, lamT, avg = build(L)
    print(f"\n=== L={L} words/doc (~{avg:.1f} distinct), N={N}, K={K} | encoder [128], MoG-10, "
          f"prior lr=1e-4 wd=0.0 | training to 40k ===", flush=True)
    t0 = time.time(); traj = []
    model = GTM(train_data=corpus, n_topics=K, ae_type="vae", vi_type="mixture_of_gaussians",
                mixture_components=10, doc_topic_prior="logistic_normal", update_prior=True, w_prior=1.0,
                encoder_args={"text_bow": {"hidden_dims": [128]}}, decoder_args={"text_bow": {"hidden_dims": []}},
                batch_size=256, num_steps=CHECKPOINTS[0], num_workers=0, print_every_n_steps=10**9,
                optim_args=OPTIM, return_best_model=False, ckpt_folder=tempfile.mkdtemp(), seed=SEED, device=device)
    traj.append(evaluate(model, corpus, X, true_theta, lamT, CHECKPOINTS[0], t0))
    for cp in CHECKPOINTS[1:]:
        model.num_steps = cp; model.train(corpus)
        traj.append(evaluate(model, corpus, X, true_theta, lamT, cp, t0))
        out[str(L)] = traj
        json.dump(out, open("audit/results_40k.json", "w"), indent=2)

print("\n=== topic-share correlation vs training steps ===")
print(f"{'words':>6} {'10k':>8} {'20k':>8} {'30k':>8} {'40k':>8}")
for L in LENGTHS:
    tr = out[str(L)]
    print(f"{L:>6} " + " ".join(f"{r['theta_corr']:>8.3f}" for r in tr), flush=True)
print("\nsaved -> audit/results_40k.json", flush=True)
