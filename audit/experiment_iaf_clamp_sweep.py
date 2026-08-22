"""Does widening the IAF per-step log-scale clamp help on real data?
Sweep flow_logscale_bound in {2 (old default), 4 (new), 6, 8} on the SAME US-congress
split/vocab as experiment_us_congress.py. Metric: held-out IWAE (nats/doc)."""
import sys, time, json, numpy as np, scipy.sparse, tempfile, torch
import pandas as pd

DATA = "/mnt/c/Users/Gauthier/Dropbox/Projet platforms&policies/Data/us_congress/us_congress_speeches_clean.csv"
PYDIR = "/mnt/c/Users/Gauthier/Dropbox/Projet platforms&policies/Codes/python"
N_SAMPLE, N_TEST, VOCAB, K = 45000, 9000, 5000, 20
NUM_STEPS, BATCH, IWAE_S, SEED = 2500, 512, 50, 0
rng = np.random.default_rng(SEED); torch.manual_seed(SEED); np.random.seed(SEED)

keep = []; frac = N_SAMPLE / 900000 * 1.3
for chunk in pd.read_csv(DATA, usecols=["doc_clean", "word_count"], chunksize=200000):
    chunk = chunk[chunk["word_count"] >= 25].dropna(subset=["doc_clean"])
    keep.append(chunk.sample(frac=min(1.0, frac), random_state=SEED)[["doc_clean"]])
    if sum(len(k) for k in keep) >= N_SAMPLE:
        break
df = pd.concat(keep, ignore_index=True).iloc[:N_SAMPLE].reset_index(drop=True)
idx = rng.permutation(len(df))
test_df = df.iloc[idx[:N_TEST]].reset_index(drop=True)
train_df = df.iloc[idx[N_TEST:]].reset_index(drop=True)

sys.path.insert(0, PYDIR)
from utils.dict_filter import make_frozen_vectorizer
full = make_frozen_vectorizer("us")
Xtr = full.transform(train_df["doc_clean"].astype(str))
dfreq = np.asarray((Xtr > 0).sum(0)).ravel()
vocab_full = full.get_feature_names_out()
top_terms = [vocab_full[i] for i in sorted(np.argsort(dfreq)[::-1][:VOCAB])]
vectorizer = make_frozen_vectorizer("us", vocab=top_terms)
vectorizer.fit(train_df["doc_clean"].astype(str))

from deeplatent import Corpus, GTM
def make_corpus(d):
    c = Corpus(d, modalities={"text": {"column": "doc_clean",
              "views": {"bow": {"type": "bow", "vectorizer": vectorizer}}}})
    m = c.processed_modalities["text"]["bow"]["matrix"]
    if scipy.sparse.issparse(m):
        c.processed_modalities["text"]["bow"]["matrix"] = np.asarray(m.todense(), np.float32)
    return c
train_corpus, test_corpus = make_corpus(train_df), make_corpus(test_df)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device={device}; sweeping IAF flow_logscale_bound\n", flush=True)

def run(bound):
    t0 = time.time()
    m = GTM(train_data=train_corpus, n_topics=K, ae_type="vae", vi_type="iaf",
            flow_logscale_bound=bound, update_prior=False, doc_topic_prior="logistic_normal",
            encoder_args={"text_bow": {"hidden_dims": [256, 128], "dropout": 0.1}},
            decoder_args={"text_bow": {"hidden_dims": []}},
            w_prior=1.0, batch_size=BATCH, num_steps=NUM_STEPS, num_workers=4,
            print_every_n_steps=10**9, return_best_model=False,
            ckpt_folder=tempfile.mkdtemp(), seed=SEED, device=device)
    iwae, elbo = m.estimate_marginal_log_likelihood(test_corpus, n_samples=IWAE_S)
    kl = float(np.mean(m.train_div_losses[-100:]))
    print(f"  b={bound:>4}: test_IWAE={float(iwae):9.2f}  test_ELBO={float(elbo):9.2f}  "
          f"trainKL={kl:6.2f}  ({time.time()-t0:.0f}s)", flush=True)
    return {"flow_logscale_bound": bound, "test_IWAE": float(iwae),
            "test_ELBO": float(elbo), "train_KL": kl}

res = []
for b in [2.0, 4.0, 6.0, 8.0]:
    try:
        res.append(run(b))
    except Exception as e:
        print(f"  b={b} FAILED: {repr(e)[:120]}", flush=True)
print("\nbaselines from main run: mean_field IWAE=-772.44, MoG(C=10) IWAE=-768.84", flush=True)
json.dump(res, open("audit/iaf_clamp_sweep_results.json", "w"), indent=2)
print("saved -> audit/iaf_clamp_sweep_results.json", flush=True)
