"""Step 03 (baseline): LDA by collapsed Gibbs sampling (tomotopy) on the training records, the first stage of the
two-step approach. Runs in the `gtm` conda env (tomotopy).

Fits LDA with K topics (tomotopy's defaults: alpha, one per topic, re-estimated every 10 iterations) on the training
months' records of the document-term matrix (02), then infers the topic shares of every record (training ones
included, so all months are treated alike) and averages them by month. Writes results/<tag>/: config.json,
theta_<which>.npy (months x K attention), beta_<which>.npy (K x V topic-word distributions), months.csv and
topic_words_<which>.csv for which in {selected, final} (the same: LDA has one fit), in the layout of 03_fit_joint.py
so that 04 treats both alike.

Usage: python 03_fit_lda.py --tag lda_k50 [--n-topics 50 --iterations 1000 --train-docs 0]
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
import tomotopy as tp

import common


def docs_of(X, rows, vocab):
    """Token lists (each term repeated by its count) of the rows of X."""
    for i in rows:
        lo, hi = X.indptr[i], X.indptr[i + 1]
        yield [vocab[j] for j, c in zip(X.indices[lo:hi], X.data[lo:hi]) for _ in range(int(c))]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--n-topics", type=int, default=50)
    ap.add_argument("--train-end", default="1999-12")
    ap.add_argument("--iterations", type=int, default=1000)
    ap.add_argument("--infer-iterations", type=int, default=100)
    ap.add_argument("--train-docs", type=int, default=0, help="random training records to fit on; 0 = all")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dtm-dir", default=None, help="document-term matrix (default <data>/dtm, 1984-2011)")
    a = ap.parse_args()
    out = common.RESULTS / a.tag
    out.mkdir(parents=True, exist_ok=True)
    dtm = Path(a.dtm_dir) if a.dtm_dir else common.DATA / "dtm"
    built = json.loads((dtm / "settings.json").read_text())
    vocab_end = built.get("train_end", built.get("vocab_end"))  # 02: the training period; w2_build_dtm: --vocab-end
    assert vocab_end <= a.train_end, f"the DTM's vocabulary uses records up to {vocab_end}, after the training period"
    X = sp.load_npz(dtm / "X.npz").tocsr()
    rec = pd.read_csv(dtm / "records.csv.gz", dtype={"month": str})
    vocab = (dtm / "vocab.txt").read_text().split("\n")[:-1]
    months = sorted(rec.month.unique())
    train_rows = np.flatnonzero((rec.month <= a.train_end).to_numpy())
    if a.train_docs:
        train_rows = np.sort(np.random.default_rng(a.seed).choice(train_rows, a.train_docs, replace=False))

    t0 = time.time()
    mdl = tp.LDAModel(k=a.n_topics, seed=a.seed)
    for words in docs_of(X, train_rows, vocab):
        mdl.add_doc(words)
    print(f"LDA: {len(train_rows):,} training records, K = {a.n_topics} ({time.time() - t0:.0f}s)", flush=True)
    for it in range(0, a.iterations, 50):
        mdl.train(50, workers=a.workers)
        print(f"  iteration {it + 50}: log-likelihood per word {mdl.ll_per_word:.4f} ({time.time() - t0:.0f}s)",
              flush=True)

    # Topic shares of every record, averaged by month
    month_of = rec.month.map({m: i for i, m in enumerate(months)}).to_numpy()
    sums = np.zeros((len(months), a.n_topics))
    bs = 50_000
    for s in range(0, X.shape[0], bs):
        rows = range(s, min(s + bs, X.shape[0]))
        batch = [mdl.make_doc(w) for w in docs_of(X, rows, vocab)]
        dists, _ = mdl.infer(batch, iter=a.infer_iterations, workers=a.workers)
        np.add.at(sums, month_of[s:s + len(batch)], np.asarray(dists))
        print(f"  inferred {s + len(batch):,} records ({time.time() - t0:.0f}s)", flush=True)
    theta = sums / np.bincount(month_of, minlength=len(months))[:, None]

    # Topic-word distributions over the full vocabulary (tomotopy drops terms absent from the training docs)
    beta = np.full((a.n_topics, len(vocab)), 1e-12)
    col = {w: j for j, w in enumerate(vocab)}
    used = [col[w] for w in mdl.used_vocabs]
    for k in range(a.n_topics):
        beta[k, used] = mdl.get_topic_word_dist(k)
    beta /= beta.sum(axis=1, keepdims=True)
    for which in ["selected", "final"]:
        np.save(out / f"theta_{which}.npy", theta)
        np.save(out / f"beta_{which}.npy", beta.astype(np.float32))
    pd.DataFrame({"month": months}).to_csv(out / "months.csv", index=False)
    background = np.asarray(X[train_rows].sum(axis=0)).ravel() + 1.0
    background /= background.sum()
    lift = np.log(beta) - np.log(background)[None, :]
    relevance = 0.6 * np.log(beta) + 0.4 * lift
    words = pd.DataFrame({"topic": range(a.n_topics),
                  "share": theta[np.array(months) <= a.train_end].mean(0).round(4),
                  "probable": [", ".join(vocab[j] for j in np.argsort(-beta[k])[:15]) for k in range(a.n_topics)],
                  "relevant": [", ".join(vocab[j] for j in np.argsort(-relevance[k])[:15])
                               for k in range(a.n_topics)]})
    for which in ["selected", "final"]:
        words.to_csv(out / f"topic_words_{which}.csv", index=False)
    (out / "config.json").write_text(json.dumps({**vars(a), "alpha": [float(x) for x in mdl.alpha],
                                                 "eta": float(mdl.eta), "ll_per_word": float(mdl.ll_per_word)},
                                                indent=1))
    print(f"done ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
