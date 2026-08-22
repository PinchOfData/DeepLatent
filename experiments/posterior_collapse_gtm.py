#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
Posterior collapse in the GTM VAE: diagnosis and fix.
=====================================================

This script reproduces the posterior-collapse problem in the GTM variational
autoencoder and demonstrates the fix.

The problem
-----------
The GTM ELBO (for a single document) is

    ELBO = E_q[ log p(doc | theta) ]  -  KL( q(z|doc) || p(z) )
         = sum_w  n_w * log p(w | theta)   -   KL_doc                 (*)

i.e. the reconstruction term is a SUM of the per-token log-likelihoods over the
~L words of the document. In the original implementation the BoW reconstruction
was instead divided by the number of tokens (a *per-token average*), while the
KL was a *per-document average*:

    loss = mean_token[ -log p(w|theta) ]  +  w_prior * mean_doc[ KL_doc ]

Because the reconstruction is divided by ~L (=document length) but the KL is
not, the KL is effectively over-weighted by a factor of ~L. At w_prior=1 -- the
value for which the objective is supposed to be the ELBO and the theoretical
guarantees hold -- the KL is ~L times too strong, the encoder is driven to
q(z|x)=p(z), and the posterior collapses. A tiny w_prior ~ 1/L rebalances the
two terms (this is what the original notebook does with w_prior=0.01), but then
the objective is no longer a valid lower bound on the log-likelihood.

The fix
-------
Reconstruction is now always the per-document log-likelihood (the multinomial
NLL summed over tokens, averaged over the batch -- exactly equation (*)), the
same per-document scale as the KL. So ``w_prior=1`` is the genuine (negative)
ELBO, a valid lower bound, and the model does NOT collapse.

This script demonstrates that, and reproduces the old failure mode by *over-
weighting* the KL: setting ``w_prior = L`` (the document length) makes the KL
exactly as dominant as it was under the old per-token reconstruction at
``w_prior=1``, and the posterior collapses again.

Run
---
    python experiments/posterior_collapse_gtm.py                 # default sweep
    python experiments/posterior_collapse_gtm.py --num_steps 30000 --figures
"""
import argparse
import json
import time

import numpy as np
import pandas as pd
import scipy.sparse
from scipy.optimize import linear_sum_assignment
from sklearn.feature_extraction.text import CountVectorizer

from deeplatent import Corpus, GTM, generate_documents


def build_corpus(num_docs, num_topics, vocab_size, words, seed=42):
    """Simulate a logistic-normal GTM corpus (matches 03_gtm_basic_simulation)."""
    true_df, df, *_ = generate_documents(
        num_docs, num_topics, vocab_size,
        doc_topic_prior="logistic_normal",
        min_words=words, max_words=words, random_seed=seed,
    )
    vectorizer = CountVectorizer()
    vectorizer.fit(df["doc_clean_0"])
    modalities = {
        "text": {"column": "doc_clean_0",
                 "views": {"bow": {"type": "bow", "vectorizer": vectorizer}}}
    }
    corpus = Corpus(df, modalities=modalities)
    # Densify the BoW matrix so the (tiny) model is not DataLoader-bound.
    m = corpus.processed_modalities["text"]["bow"]["matrix"]
    if scipy.sparse.issparse(m):
        corpus.processed_modalities["text"]["bow"]["matrix"] = np.asarray(
            m.todense(), dtype=np.float32)
    return true_df, corpus


def aligned_topic_correlation(true_df, theta, num_topics):
    """Match estimated to true topics (Hungarian on dot products) and correlate."""
    est = pd.DataFrame(theta, columns=[f"Topic{i}" for i in range(num_topics)])
    score = np.array([[np.dot(true_df[f"Topic{t}"], est[f"Topic{e}"])
                       for e in range(num_topics)] for t in range(num_topics)])
    true_idx, est_idx = linear_sum_assignment(-score)
    mapping = {int(t): int(e) for t, e in zip(true_idx, est_idx)}
    corrs = np.array([
        np.corrcoef(est[f"Topic{mapping[t]}"], true_df[f"Topic{t}"].values)[0, 1]
        for t in range(num_topics)
    ])
    return corrs, mapping


def run_config(true_df, corpus, num_topics, label, w_prior,
               num_steps, batch_size):
    encoder_args = {"text_bow": {"hidden_dims": [200, 200], "activation": "relu",
                                 "bias": True, "dropout": 0.0}}
    decoder_args = {"text_bow": {"hidden_dims": [200, 200], "activation": "relu",
                                 "bias": True, "dropout": 0.0}}
    t0 = time.time()
    tm = GTM(
        train_data=corpus, n_topics=num_topics, ae_type="vae", vi_type="iaf",
        update_prior=False, doc_topic_prior="logistic_normal",
        optim_args={"main": {"lr": 1e-3, "weight_decay": 0.0}},
        encoder_args=encoder_args, decoder_args=decoder_args,
        w_prior=w_prior,
        batch_size=batch_size, num_steps=num_steps, num_workers=0,
        print_every_n_steps=10_000_000, seed=42,
    )
    dt = time.time() - t0

    theta, theta_std = tm.get_doc_topic_distribution(
        corpus, num_samples=30, return_std=True)
    corrs, mapping = aligned_topic_correlation(true_df, theta, num_topics)

    # collapse signal: how much do per-document topic proportions actually vary?
    est_std = float(theta.std(axis=0).mean())
    true_std = float(true_df.values.std(axis=0).mean())

    # 95% CI coverage of the true proportions by the posterior (notebook metric)
    coverage = []
    for t in range(num_topics):
        e = mapping[t]
        lo = theta[:, e] - 1.96 * theta_std[:, e]
        hi = theta[:, e] + 1.96 * theta_std[:, e]
        tv = true_df[f"Topic{t}"].values
        coverage.append(float(np.mean((tv >= lo) & (tv <= hi))))

    tail = slice(-200, None)
    div = float(np.mean(tm.train_div_losses[tail]))     # = w_prior * KL at end
    recon = float(np.mean(tm.train_recon_losses[tail]))
    raw_kl = div / w_prior if w_prior else float("nan")

    return {
        "label": label, "w_prior": w_prior,
        "steps": num_steps, "time_s": round(dt, 1),
        "raw_kl_per_doc": round(raw_kl, 4),
        "recon_term": round(recon, 4),
        "theta_std_est": round(est_std, 4), "theta_std_true": round(true_std, 4),
        "collapse_ratio": round(est_std / true_std, 3),
        "topic_corr_mean": round(float(corrs.mean()), 3),
        "topic_corr_min": round(float(corrs.min()), 3),
        "ci_coverage_mean": round(float(np.mean(coverage)), 3),
        "_theta": theta, "_mapping": mapping,
    }


def maybe_plot(true_df, result, num_topics, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    theta, mapping = result["_theta"], result["_mapping"]
    ncols = 3
    nrows = (num_topics + ncols - 1) // ncols
    fig, axs = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
    axs = np.atleast_1d(axs).flatten()
    for t in range(num_topics):
        e = mapping[t]
        x, y = theta[:, e], true_df[f"Topic{t}"].values
        axs[t].scatter(x, y, s=2, alpha=0.2)
        axs[t].plot([0, 1], [0, 1], "k--", lw=1)
        axs[t].set_title(f"Topic {t}  (r={np.corrcoef(x, y)[0,1]:.2f})")
        axs[t].set_xlabel("estimated"); axs[t].set_ylabel("true")
    for j in range(num_topics, len(axs)):
        fig.delaxes(axs[j])
    fig.suptitle(f"{result['label']}  (w_prior={result['w_prior']})")
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    print(f"  saved figure -> {path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--num_docs", type=int, default=6000)
    ap.add_argument("--num_topics", type=int, default=6)
    ap.add_argument("--vocab_size", type=int, default=500)
    ap.add_argument("--words", type=int, default=200)
    ap.add_argument("--num_steps", type=int, default=8000)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--figures", action="store_true",
                    help="save estimate-vs-truth scatter plots to figures/")
    args = ap.parse_args()

    true_df, corpus = build_corpus(
        args.num_docs, args.num_topics, args.vocab_size, args.words)

    # Reconstruction is always per-document now, so w_prior is the genuine ELBO
    # weight. w_prior=1 is the true ELBO; w_prior=L reproduces the old per-token
    # collapse by over-weighting the KL.
    configs = [
        # (label, w_prior)
        ("w_prior=1 (true ELBO)",            1.0),
        (f"w_prior=L={args.words} (KL over-weighted)", float(args.words)),
        ("w_prior=0.01 (works, not a bound)", 0.01),
    ]

    print(f"# GTM VAE posterior-collapse comparison "
          f"(num_docs={args.num_docs}, steps={args.num_steps})\n")
    header = (f"{'config':>34} {'w_prior':>8} {'raw_KL':>8} "
              f"{'collapse_ratio':>14} {'topic_corr':>11} {'CI_cov':>7}")
    print(header)
    print("-" * len(header))

    results = []
    for label, w_prior in configs:
        r = run_config(true_df, corpus, args.num_topics, label, w_prior,
                       num_steps=args.num_steps, batch_size=args.batch_size)
        results.append(r)
        print(f"{r['label']:>34} {r['w_prior']:>8} "
              f"{r['raw_kl_per_doc']:>8.3f} {r['collapse_ratio']:>14.3f} "
              f"{r['topic_corr_mean']:>11.3f} {r['ci_coverage_mean']:>7.3f}")
        if args.figures and w_prior == 1.0:
            maybe_plot(true_df, r, args.num_topics,
                       "figures/gtm_per_document_w1_recovery.png")

    print("\nInterpretation (reconstruction is per-document => w_prior is the ELBO weight):")
    print("  w_prior=1   -> KL>0, topics recovered: the true ELBO, no collapse.")
    print("  w_prior=L   -> KL over-weighted ~Lx (old per-token w=1): posterior collapses.")
    print("  w_prior=0.01-> also fine here, but not a valid lower bound on the likelihood.")

    dump = [{k: v for k, v in r.items() if not k.startswith("_")} for r in results]
    with open("experiments/posterior_collapse_results.json", "w") as f:
        json.dump(dump, f, indent=2)


if __name__ == "__main__":
    main()
