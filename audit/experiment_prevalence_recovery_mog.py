"""VAE + mixture_of_gaussians, update_prior=True, ONE replication, trained in
segments with LIVE loss printing; recovery (coefficients + topic shares) evaluated
at step checkpoints so we can see both the loss and the recovery converge."""
import numpy as np, scipy.sparse, tempfile, torch
from scipy.optimize import linear_sum_assignment
from sklearn.feature_extraction.text import CountVectorizer
from deeplatent import Corpus, GTM, generate_documents

N, K, C, VOCAB, WORDS, SEED = 2500, 5, 3, 200, 80, 100
CHECKPOINTS = [2000, 5000, 10000, 20000]
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
    doc_topic_prior="logistic_normal", min_words=WORDS, max_words=WORDS, random_seed=SEED)
true_theta = dft[[f"Topic{i}" for i in range(K)]].values.astype(np.float64)
lamT = lam.T.astype(np.float64)
vec = CountVectorizer(); vec.fit(df["doc_clean_0"])
corpus = Corpus(df, modalities={"text": {"column": "doc_clean_0",
                "views": {"bow": {"type": "bow", "vectorizer": vec}}}}, prevalence="~ cov_1 + cov_2 + cov_3")
m = corpus.processed_modalities["text"]["bow"]["matrix"]
if scipy.sparse.issparse(m):
    corpus.processed_modalities["text"]["bow"]["matrix"] = np.asarray(m.todense(), np.float32)
X = corpus.M_prevalence_covariates.astype(np.float64)

def evaluate(model, steps):
    theta_hat = model.get_doc_topic_distribution(corpus, num_samples=20).astype(np.float64)
    mp = align(true_theta, theta_hat)
    theta_al = np.stack([theta_hat[:, mp[t]] for t in range(K)], axis=1)
    W = model.get_prevalence_coefficients().astype(np.float64)  # [K, C+1] lifted+centered (v0.2.0)
    W_al = np.stack([W[mp[t]] for t in range(K)])
    c_mn, s_mn = cmetrics(W_al, lamT)
    c_ts, s_ts = cmetrics(ols(X, clogit(theta_al)).T, lamT)
    overall = float(np.corrcoef(true_theta.ravel(), theta_al.ravel())[0, 1])
    recon = np.mean(model.train_recon_losses[-100:]); kl = np.mean(model.train_div_losses[-100:])
    print(f"[{steps:>6} steps] recon={recon:7.2f} KL={kl:6.3f} || "
          f"coef: mean_net corr={c_mn:+.3f}(slope {s_mn:.2f}) two-step corr={c_ts:+.3f} || "
          f"theta corr={overall:.3f}", flush=True)

c_or, s_or = cmetrics(ols(X, clogit(true_theta)).T, lamT)
print(f"oracle coefficient corr={c_or:.3f} (slope {s_or:.2f}) -- ceiling\n", flush=True)

model = GTM(train_data=corpus, n_topics=K, ae_type="vae", vi_type="mixture_of_gaussians",
            mixture_components=10, doc_topic_prior="logistic_normal", update_prior=True, w_prior=1.0,
            encoder_args={"text_bow": {"hidden_dims": [128]}}, decoder_args={"text_bow": {"hidden_dims": []}},
            batch_size=256, num_steps=CHECKPOINTS[0], num_workers=0, print_every_n_steps=2000,
            return_best_model=False, ckpt_folder=tempfile.mkdtemp(), seed=SEED, device=device)
evaluate(model, CHECKPOINTS[0])
for cp in CHECKPOINTS[1:]:
    model.num_steps = cp
    model.train(corpus)
    evaluate(model, cp)

# plateau check on smoothed loss
loss = np.array(model.train_losses, dtype=np.float64)
sm = np.convolve(loss, np.ones(200)/200, mode="valid")
last = sm[int(0.9*len(sm)):].mean(); prev = sm[int(0.8*len(sm)):int(0.9*len(sm))].mean()
print(f"\nfinal smoothed loss change (last 10% vs prev 10%): {abs(last-prev)/abs(prev)*100:.3f}%  "
      f"({'CONVERGED' if abs(last-prev)/abs(prev) < 0.005 else 'still moving'})", flush=True)
