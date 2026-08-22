"""Real-data test: do expressive variational families (full-rank, IAF, mixture-of-
Gaussians) tighten the held-out marginal likelihood vs mean-field on US congressional
speeches? Same data / vocab / prior / decoder; only vi_type changes. Metric: held-out
IWAE estimate of log p(x) (and the 1-sample ELBO), in nats per document.

Vocab mirrors scripts/04_topic_model.py: the curated frozen (1,2)-gram vectorizer
(utils.dict_filter.make_frozen_vectorizer), capped to the top-V terms by document
frequency for tractability.
"""
import sys, time, json, numpy as np, scipy.sparse, tempfile, torch
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

DATA = "/mnt/c/Users/Gauthier/Dropbox/Projet platforms&policies/Data/us_congress/us_congress_speeches_clean.csv"
PYDIR = "/mnt/c/Users/Gauthier/Dropbox/Projet platforms&policies/Codes/python"
N_SAMPLE, N_TEST, VOCAB, K = 45000, 9000, 5000, 20
NUM_STEPS, BATCH, IWAE_S, SEED = 2500, 512, 50, 0
rng = np.random.default_rng(SEED); torch.manual_seed(SEED); np.random.seed(SEED)

# ---- load a subsample (chunked, to bound memory) ----------------------------
print("Loading subsample from US congress speeches ...", flush=True)
keep = []
frac = N_SAMPLE / 900000 * 1.3
for chunk in pd.read_csv(DATA, usecols=["doc_clean", "word_count"], chunksize=200000):
    chunk = chunk[chunk["word_count"] >= 25].dropna(subset=["doc_clean"])
    take = chunk.sample(frac=min(1.0, frac), random_state=SEED)
    keep.append(take[["doc_clean"]])
    if sum(len(k) for k in keep) >= N_SAMPLE:
        break
df = pd.concat(keep, ignore_index=True).iloc[:N_SAMPLE].reset_index(drop=True)
print(f"  {len(df):,} docs", flush=True)
idx = rng.permutation(len(df))
test_df = df.iloc[idx[:N_TEST]].reset_index(drop=True)
train_df = df.iloc[idx[N_TEST:]].reset_index(drop=True)

# ---- vectorizer: curated frozen vocab capped to top-V by doc frequency -------
sys.path.insert(0, PYDIR)
try:
    from utils.dict_filter import make_frozen_vectorizer
    full = make_frozen_vectorizer("us")
    Xtr = full.transform(train_df["doc_clean"].astype(str))
    dfreq = np.asarray((Xtr > 0).sum(0)).ravel()
    vocab_full = full.get_feature_names_out()
    top = sorted(np.argsort(dfreq)[::-1][:VOCAB])
    top_terms = [vocab_full[i] for i in top]
    vectorizer = make_frozen_vectorizer("us", vocab=top_terms)
    print(f"  frozen curated vocab capped to {len(top_terms):,} terms", flush=True)
except Exception as e:
    print(f"  frozen vectorizer unavailable ({e}); CountVectorizer fallback", flush=True)
    vectorizer = CountVectorizer(ngram_range=(1, 2), min_df=20, max_features=VOCAB, stop_words="english")
vectorizer.fit(train_df["doc_clean"].astype(str))

from deeplatent import Corpus, GTM

def make_corpus(d):
    c = Corpus(d, modalities={"text": {"column": "doc_clean",
              "views": {"bow": {"type": "bow", "vectorizer": vectorizer}}}})
    m = c.processed_modalities["text"]["bow"]["matrix"]
    if scipy.sparse.issparse(m):
        c.processed_modalities["text"]["bow"]["matrix"] = np.asarray(m.todense(), np.float32)
    return c

print("Building corpora ...", flush=True)
train_corpus, test_corpus = make_corpus(train_df), make_corpus(test_df)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device={device}", flush=True)

def fit_eval(vi, **kw):
    t0 = time.time()
    m = GTM(train_data=train_corpus, n_topics=K, ae_type="vae", vi_type=vi,
            update_prior=False, doc_topic_prior="logistic_normal",
            encoder_args={"text_bow": {"hidden_dims": [256, 128], "dropout": 0.1}},
            decoder_args={"text_bow": {"hidden_dims": []}},
            w_prior=1.0, batch_size=BATCH, num_steps=NUM_STEPS, num_workers=4,
            print_every_n_steps=10**9, return_best_model=False,
            ckpt_folder=tempfile.mkdtemp(), seed=SEED, device=device, **kw)
    train_t = time.time() - t0
    iwae, elbo = m.estimate_marginal_log_likelihood(test_corpus, n_samples=IWAE_S)
    tr_elbo = -(np.mean(m.train_recon_losses[-100:]) + np.mean(m.train_div_losses[-100:]))
    name = vi + (f"(C={kw.get('mixture_components')})" if vi == "mixture_of_gaussians" else "")
    print(f"{name:26s} train_ELBO={tr_elbo:9.2f}  test_ELBO={float(elbo):9.2f}  "
          f"test_IWAE={float(iwae):9.2f}  gap={float(iwae-elbo):5.2f}  ({train_t:.0f}s)", flush=True)
    return {"family": name, "train_ELBO": float(tr_elbo), "test_ELBO": float(elbo),
            "test_IWAE": float(iwae), "gap": float(iwae - elbo), "train_s": train_t}

print(f"\n=== US Congress: {len(train_df):,} train / {len(test_df):,} test, V={VOCAB}, K={K}, "
      f"S(IWAE)={IWAE_S} ===\n(higher test_IWAE = tighter bound on log p(x); nats/doc)\n", flush=True)
results = []
for vi, kw in [("mean_field", {}), ("full_rank", {}), ("iaf", {}),
               ("mixture_of_gaussians", {"mixture_components": 10}),
               ("mixture_of_gaussians", {"mixture_components": 20})]:
    try:
        results.append(fit_eval(vi, **kw))
    except Exception as e:
        print(f"{vi} {kw} FAILED: {e}", flush=True)

results.sort(key=lambda r: -r["test_IWAE"])
print("\n=== ranked by held-out IWAE (best first) ===", flush=True)
for r in results:
    print(f"  {r['family']:26s}  test_IWAE={r['test_IWAE']:9.2f}", flush=True)
json.dump(results, open("audit/us_congress_results.json", "w"), indent=2)
print("\nsaved -> audit/us_congress_results.json", flush=True)
