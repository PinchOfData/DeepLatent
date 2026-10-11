"""Wave 2, step 3 (baseline): LDA (tomotopy, collapsed Gibbs sampling) for one window of the expanding-window
evaluation: fit on the records of the window's training months (up to E - val_years, as the joint model), then infer
the topic shares of every record of the sample and average them by month. Runs in the `gtm` conda env.

Writes <results>/w2/<grid>/lda_e<E>_s<seed>/: config.json, months.csv, theta_<which>.npy (months x K),
beta_<which>.npy, topic_words_<which>.csv for which in {selected, final} (the same; LDA has one fit).

--seeds fits several seeds one after the other (one job per window: the cluster caps submitted jobs).

Usage: python w2_fit_lda.py --dtm-dir <scratch>/data/dtm_w2 --grid k50 --window-end 1979 --seeds 0 1 2
"""
import argparse
import importlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
import tomotopy as tp

import common

lda03 = importlib.import_module("03_fit_lda")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dtm-dir", required=True)
    ap.add_argument("--grid", required=True)
    ap.add_argument("--window-end", type=int, required=True)
    ap.add_argument("--val-years", type=int, default=5)
    ap.add_argument("--n-topics", type=int, default=50)
    ap.add_argument("--iterations", type=int, default=1000)
    ap.add_argument("--infer-iterations", type=int, default=100)
    ap.add_argument("--train-docs", type=int, default=0, help="random training records to fit on; 0 = all")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    args = ap.parse_args()
    for seed in args.seeds:
        a = argparse.Namespace(**{**vars(args), "seed": seed})
        fit(a)


def fit(a) -> None:
    tag = f"lda_e{a.window_end}_s{a.seed}"
    out = common.RESULTS / "w2" / a.grid / tag
    out.mkdir(parents=True, exist_ok=True)
    dtm = Path(a.dtm_dir)
    X = sp.load_npz(dtm / "X.npz").tocsr()
    rec = pd.read_csv(dtm / "records.csv.gz", dtype={"month": str})
    vocab = (dtm / "vocab.txt").read_text().split("\n")[:-1]
    months = sorted(rec.month.unique())
    train_end, val_end = f"{a.window_end - a.val_years}-12", f"{a.window_end}-12"
    common.month_targets(months, train_end, val_end, 0, 3).to_csv(out / "months.csv", index=False)
    train_rows = np.flatnonzero((rec.month <= train_end).to_numpy())
    if a.train_docs and len(train_rows) > a.train_docs:
        train_rows = np.sort(np.random.default_rng(a.seed).choice(train_rows, a.train_docs, replace=False))

    t0 = time.time()
    mdl = tp.LDAModel(k=a.n_topics, seed=a.seed)
    for words in lda03.docs_of(X, train_rows, vocab):
        mdl.add_doc(words)
    print(f"LDA, window ending {a.window_end}: {len(train_rows):,} training records (to {train_end}), "
          f"K = {a.n_topics}", flush=True)
    for it in range(0, a.iterations, 50):
        mdl.train(50, workers=a.workers)
        print(f"  iteration {it + 50}: log-likelihood per word {mdl.ll_per_word:.4f} ({time.time() - t0:.0f}s)",
              flush=True)

    month_of = rec.month.map({m: i for i, m in enumerate(months)}).to_numpy()
    sums = np.zeros((len(months), a.n_topics))
    for s in range(0, X.shape[0], 50_000):
        batch = [mdl.make_doc(w) for w in lda03.docs_of(X, range(s, min(s + 50_000, X.shape[0])), vocab)]
        dists, _ = mdl.infer(batch, iter=a.infer_iterations, workers=a.workers)
        np.add.at(sums, month_of[s:s + len(batch)], np.asarray(dists))
        if (s // 50_000) % 10 == 0:
            print(f"  inferred {s + len(batch):,} records ({time.time() - t0:.0f}s)", flush=True)
    theta = sums / np.bincount(month_of, minlength=len(months))[:, None]

    beta = np.full((a.n_topics, len(vocab)), 1e-12)
    col = {w: j for j, w in enumerate(vocab)}
    used = [col[w] for w in mdl.used_vocabs]
    for k in range(a.n_topics):
        beta[k, used] = mdl.get_topic_word_dist(k)
    beta /= beta.sum(axis=1, keepdims=True)
    words = pd.DataFrame({"topic": range(a.n_topics),
                          "share": theta[np.array(months) <= train_end].mean(0).round(4),
                          "probable": [", ".join(vocab[j] for j in np.argsort(-beta[k])[:15])
                                       for k in range(a.n_topics)]})
    for which in ["selected", "final"]:
        np.save(out / f"theta_{which}.npy", theta)
        np.save(out / f"beta_{which}.npy", beta.astype(np.float32))
        words.to_csv(out / f"topic_words_{which}.csv", index=False)
    (out / "config.json").write_text(json.dumps({**{k: v for k, v in vars(a).items() if k != "seeds"}, "tag": tag, "train_end": train_end, "val_end": val_end,
                                                 "ll_per_word": float(mdl.ll_per_word)}, indent=1))
    print(f"done ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
