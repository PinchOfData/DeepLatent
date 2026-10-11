"""Step 08: review the checkpoints of a fit (05_fit_gtm.py --checkpoint-every) to choose one by its topics.

For every checkpoint in <common.CHECKPOINTS>/<tag>/ from --min-step on (e.g. after the KL annealing, so
that the model was trained at w_prior = 1), on a fixed sample of --per-cell speeches per congress x
party: the posterior-mean topic shares, each topic's top words (probable, FREX and observed, as in
topic_words.py), and a t-SNE of the topic shares of --tsne-docs of them (colored by top topic, as in
06_exhibits.py). Also the learned prior at each checkpoint: its covariance eigenvalues and the party
coefficients of the topic prevalence (K x congresses, prior_party_by_step.npz), to see whether the prior
is still moving at the chosen checkpoint. And where the speeches on a subject go: for each --probe-words term,
the sampled speeches using it at least --probe-min times, the topics they have as top topic (shares of
those speeches) and their median largest topic share. Writes results/<tag>/checkpoint_review/:
topic_words_by_step.csv, summary_by_step.csv, probes_by_step.csv, prior_party_by_step.npz and
tsne_step_<step>.png.

Usage: python 08_checkpoint_topics.py --tag anneal5k_prior1e-4 [--min-step 5000] [--max-step 20000] [--out-name ...]
"""
import argparse
import importlib
import json
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.manifold import TSNE

import common
import topic_words

exhibits = importlib.import_module("06_exhibits")
fit = importlib.import_module("05_fit_gtm")


PROBES = ["farmer", "crop", "breast cancer", "disease", "tariff", "railroad", "school", "worker", "immigration",
          "veteran", "tax", "budget", "silver", "missile", "civil right", "abortion", "court", "precinct",
          "housing", "energy"]


def probe_rows(step: int, theta: np.ndarray, X, vocab: list, words: list, min_count: int) -> list:
    col = {v: j for j, v in enumerate(vocab)}
    Xc = X.tocsc()
    rows = []
    for w in words:
        if w not in col:
            continue
        use = np.asarray(Xc[:, col[w]].todense()).ravel() >= min_count
        if not use.any():
            continue
        top = np.bincount(theta[use].argmax(axis=1), minlength=theta.shape[1]) / use.sum()
        order = np.argsort(-top)[:3]
        rows.append({"step": step, "word": w, "speeches": int(use.sum()),
                     "median_max_share": round(float(np.median(theta[use].max(axis=1))), 3),
                     **{f"top{i + 1}": f"T{k} {top[k]:.2f}" for i, k in enumerate(order)}})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--min-step", type=int, default=5000)
    ap.add_argument("--max-step", type=int, default=10**9)
    ap.add_argument("--out-name", default="checkpoint_review", help="output folder in results/<tag>/")
    ap.add_argument("--per-cell", type=int, default=200, help="speeches per congress x party in the sample")
    ap.add_argument("--tsne-docs", type=int, default=5000)
    ap.add_argument("--posterior-samples", type=int, default=10)
    ap.add_argument("--n-words", type=int, default=8)
    ap.add_argument("--probe-words", nargs="+", default=PROBES)
    ap.add_argument("--probe-min", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    res = common.RESULTS / a.tag
    out = res / a.out_name
    out.mkdir(exist_ok=True)
    device = torch.device(a.device)

    cfg = json.load(open(res / "config.json"))
    vocab = common.load_vocab(cfg["vocab"])
    K = cfg["n_topics"]
    docs = pd.read_csv(res / "docs.csv.gz", dtype={"speaker_session": str, "person_id": str})
    rng = np.random.default_rng(a.seed)
    cells = docs.groupby(["congress", "party"], sort=True).indices.values()
    idx = np.concatenate([rng.choice(ix, size=min(a.per_cell, len(ix)), replace=False) for ix in cells])
    sample = docs.iloc[np.sort(idx)].reset_index(drop=True)
    corpus = exhibits.design_corpus(sample, vocab, common.terms_dir(cfg["vocab"]), cfg["formula"])
    X = corpus.processed_modalities["text"]["bow"]["matrix"]
    tsne_idx = rng.choice(len(sample), size=min(a.tsne_docs, len(sample)), replace=False)
    print(f"{len(sample):,} speeches in the review sample", flush=True)

    ckpts = sorted(((int(re.search(r"_(\d+)\.ckpt$", p.name).group(1)), p) for p in (common.CHECKPOINTS / a.tag).glob("*.ckpt")))
    ckpts = [(s, p) for s, p in ckpts if a.min_step <= s <= a.max_step]
    tm = None
    words, summary, probes, prior_party = [], [], [], {}
    party_cols = [j for j, n in enumerate(cfg["prevalence_colnames"]) if n.endswith(":party_signed")]
    for step, path in ckpts:
        if tm is None:
            tm = exhibits.build_model(cfg, corpus, device, path)
        else:
            exhibits.load_checkpoint(tm, path, device)
        theta = np.asarray(tm.get_doc_topic_distribution(corpus, num_samples=a.posterior_samples))
        beta = topic_words.model_distribution(tm)
        scores = {"probable": beta, "frex": topic_words.frex(beta),
                  "observed": topic_words.frex(topic_words.observed_distribution(theta, X))}
        shares = theta.mean(axis=0)
        for k in range(K):
            words.append({"step": step, "topic": k, "share": round(float(shares[k]), 3),
                          **{n: ", ".join(vocab[j] for j in np.argsort(-sc[k])[:a.n_words]) for n, sc in scores.items()}})
        probes += probe_rows(step, theta, X, vocab, a.probe_words, a.probe_min)
        eig = np.linalg.eigvalsh(tm.prior.sigma.detach().cpu().numpy())
        prior_party[f"step_{step}"] = tm.get_prevalence_coefficients(fold_bias=False)[:, party_cols]
        summary.append({"step": step, "median_max_share": float(np.median(theta.max(axis=1))),
                        "median_entropy": float(np.median(-(theta * np.log(theta + 1e-12)).sum(axis=1))),
                        "sigma_eig_min": float(eig.min()), "sigma_eig_max": float(eig.max()),
                        **fit.party_summaries(tm, K)})
        print(" | ".join(f"{k} {v:.4g}" if isinstance(v, float) else f"{k} {v}" for k, v in summary[-1].items()),
              flush=True)

        th = theta[tsne_idx]
        emb = TSNE(n_components=2, init="pca", random_state=a.seed, perplexity=30).fit_transform(th)
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.scatter(emb[:, 0], emb[:, 1], c=th.argmax(axis=1), cmap="plasma", s=2 + 30 * th.max(axis=1) ** 2,
                   alpha=0.6, linewidths=0)
        ax.set_title(f"{a.tag}, step {step}")
        ax.set_axis_off()
        fig.savefig(out / f"tsne_step_{step:06d}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        pd.DataFrame(words).to_csv(out / "topic_words_by_step.csv", index=False)
        pd.DataFrame(summary).to_csv(out / "summary_by_step.csv", index=False)
        pd.DataFrame(probes).to_csv(out / "probes_by_step.csv", index=False)
        np.savez(out / "prior_party_by_step.npz", **prior_party)
    print(f"reviewed {len(ckpts)} checkpoints", flush=True)


if __name__ == "__main__":
    main()
