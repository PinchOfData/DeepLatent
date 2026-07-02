"""Confirm: the default prior optimizer (lr=1e-4, weight_decay=0.01) shrinks the
recovered prevalence coefficients. Compare default vs tuned (lr=1e-3, wd=0)."""
import numpy as np, scipy.sparse, tempfile, torch
from scipy.optimize import linear_sum_assignment
from sklearn.feature_extraction.text import CountVectorizer
from deeplatent import Corpus, GTM, generate_documents
R, N, K, C, VOCAB, WORDS, STEPS = 6, 2500, 5, 3, 200, 80, 2500
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def align(tt, et):
    r, c = linear_sum_assignment(-(tt.T @ et)); return {int(i): int(j) for i, j in zip(r, c)}
def cov_metrics(estKC, trueKC):
    cr = lambda M: M - M.mean(0, keepdims=True)
    t = cr(trueKC)[:, 1:].ravel(); e = cr(estKC)[:, 1:].ravel()
    return float(np.corrcoef(t, e)[0, 1]), float(np.polyfit(t, e, 1)[0])

settings = {
    "default (prior lr=1e-4, wd=0.01)": None,
    "tuned (prior lr=1e-3, wd=0)": {"main": {"lr": 1e-3, "weight_decay": 0.0},
                                    "prior": {"lr": 1e-3, "weight_decay": 0.0}},
}
acc = {k: {"corr": [], "slope": []} for k in settings}
for r in range(R):
    seed = 100 + r
    dft, df, tw, lam, _ = generate_documents(num_docs=N, num_topics=K, vocab_size=VOCAB,
        num_covs=C, doc_topic_prior="logistic_normal", min_words=WORDS, max_words=WORDS, random_seed=seed)
    true_theta = dft[[f"Topic{i}" for i in range(K)]].values.astype(np.float64)
    lamT = lam.T.astype(np.float64)
    vec = CountVectorizer(); vec.fit(df["doc_clean_0"])
    corpus = Corpus(df, modalities={"text": {"column": "doc_clean_0",
                    "views": {"bow": {"type": "bow", "vectorizer": vec}}}}, prevalence="~ cov_1 + cov_2 + cov_3")
    m = corpus.processed_modalities["text"]["bow"]["matrix"]
    if scipy.sparse.issparse(m):
        corpus.processed_modalities["text"]["bow"]["matrix"] = np.asarray(m.todense(), np.float32)
    for label, oa in settings.items():
        kw = {"optim_args": oa} if oa is not None else {}
        model = GTM(train_data=corpus, n_topics=K, ae_type="vae", vi_type="mean_field",
                    doc_topic_prior="logistic_normal", update_prior=True, w_prior=1.0,
                    encoder_args={"text_bow": {"hidden_dims": [128]}},
                    decoder_args={"text_bow": {"hidden_dims": []}}, batch_size=256, num_steps=STEPS,
                    num_workers=0, print_every_n_steps=10**9, return_best_model=False,
                    ckpt_folder=tempfile.mkdtemp(), seed=seed, device=device, **kw)
        theta_hat = model.get_doc_topic_distribution(corpus, num_samples=20).astype(np.float64)
        mp = align(true_theta, theta_hat)
        W = model.get_prevalence_coefficients().astype(np.float64)  # [K, C+1] lifted+centered (v0.2.0)
        W_al = np.stack([W[mp[t]] for t in range(K)])
        cr, sl = cov_metrics(W_al, lamT)
        acc[label]["corr"].append(cr); acc[label]["slope"].append(sl)
        print(f"rep {r} | {label:34s} mean_net corr={cr:.3f} slope={sl:.2f}", flush=True)

print("\n=== mean_net recovery: default vs tuned prior optimizer (VAE, R=6) ===")
for k, v in acc.items():
    print(f"{k:34s} corr={np.mean(v['corr']):.3f}±{np.std(v['corr']):.3f}  slope={np.mean(v['slope']):.2f}")
