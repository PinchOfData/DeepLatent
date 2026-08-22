#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
Posterior collapse in the IdealPointNN VAE: same diagnosis, same fix.
=====================================================================

This is the IdealPointNN analogue of ``posterior_collapse_gtm.py``. The neural
ideal-point model has exactly the same reconstruction/KL scale mismatch: the
vote reconstruction (Bernoulli NLL) was divided by the number of observed votes
(a *per-vote average*) and the BoW text reconstruction by the number of tokens,
while the KL is a *per-document average*. At ``w_prior=1`` the KL is therefore
over-weighted by ~(votes or tokens per document) and the posterior collapses;
the notebook works around it with ``w_prior=0.01``.

Reconstruction is now always per-document, so the likelihood and the KL share a
per-document scale, ``w_prior=1`` is the true ELBO, and the ideal points are
recovered without collapse. Over-weighting the KL (large ``w_prior``) reproduces
the old collapse.

Setup mirrors notebooks/06_idealpointnn_simulations.ipynb (text BoW + votes,
IAF, corrected PoE, Gaussian prior).

Run
---
    python experiments/posterior_collapse_idealpoint.py
    python experiments/posterior_collapse_idealpoint.py --num_steps 8000 --figures
"""
import argparse
import json
import time

import numpy as np
import scipy.sparse
from sklearn.feature_extraction.text import CountVectorizer

from deeplatent import Corpus, IdealPointNN, generate_ideal_points


def build_corpus(num_politicians, num_bills, doc_length, vocab_size, seed=42):
    ideal_points, df, *_ = generate_ideal_points(
        num_politicians=num_politicians, dim_ideal_points=1, num_bills=num_bills,
        num_survey_questions=2, doc_length=doc_length, vocab_size=vocab_size,
        seed=seed, progress_bar=False)
    vectorizer = CountVectorizer()
    vectorizer.fit(df["doc_clean"])
    modalities = {
        "text": {"column": "doc_clean",
                 "views": {"bow": {"type": "bow", "vectorizer": vectorizer}}},
        "vote": {"column": [f"vote_{i+1}" for i in range(num_bills)],
                 "views": {"responses": {"type": "vote"}}},
    }
    corpus = Corpus(df, modalities=modalities)
    m = corpus.processed_modalities["text"]["bow"]["matrix"]
    if scipy.sparse.issparse(m):
        corpus.processed_modalities["text"]["bow"]["matrix"] = np.asarray(
            m.todense(), dtype=np.float32)
    return ideal_points[:, 0], corpus


def recovery_corr(true, est):
    true, est = true.flatten(), np.asarray(est).flatten()
    return abs(float(np.corrcoef(est, true)[0, 1]))


def run_config(true_pts, corpus, label, w_prior, num_steps, batch_size):
    encoder_args, decoder_args = {}, {}
    for key in ["text_bow", "vote_responses"]:
        encoder_args[key] = {"hidden_dims": [128, 64], "activation": "relu",
                             "bias": True, "dropout": 0.0}
        decoder_args[key] = {"hidden_dims": [], "activation": "relu",
                             "bias": True, "dropout": 0.0}
    t0 = time.time()
    m = IdealPointNN(
        ae_type="vae", vi_type="iaf", update_prior=False, n_ideal_points=1,
        train_data=corpus, encoder_args=encoder_args, decoder_args=decoder_args,
        w_prior=w_prior, fusion="corrected_poe",
        batch_size=batch_size, num_steps=num_steps, num_workers=0,
        print_every_n_steps=10_000_000, return_best_model=True, seed=12345)
    dt = time.time() - t0

    z, zstd = m.get_ideal_points(corpus, num_samples=50, return_std=True)
    corr = recovery_corr(true_pts, z)
    tail = slice(-200, None)
    div = float(np.mean(m.train_div_losses[tail]))
    recon = float(np.mean(m.train_recon_losses[tail]))
    raw_kl = div / w_prior if w_prior else float("nan")
    return {
        "label": label, "w_prior": w_prior, "steps": num_steps,
        "time_s": round(dt, 1), "raw_kl_per_doc": round(raw_kl, 4),
        "recon_term": round(recon, 4),
        "ideal_pt_std_est": round(float(np.asarray(z).std()), 4),
        "idealpoint_corr": round(corr, 3), "_z": np.asarray(z).flatten(),
    }


def maybe_plot(true_pts, result, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    z = result["_z"]
    # affine-align estimated points to truth for display
    c = np.cov(true_pts, z, bias=True)[0, 1] / np.var(z)
    d = true_pts.mean() - c * z.mean()
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(c * z + d, true_pts, s=4, alpha=0.25)
    lim = [min(true_pts.min(), (c * z + d).min()), max(true_pts.max(), (c * z + d).max())]
    ax.plot(lim, lim, "k--", lw=1)
    ax.set_xlabel("estimated ideal point (aligned)")
    ax.set_ylabel("true ideal point")
    ax.set_title(f"IdealPointNN  {result['label']}, w_prior={result['w_prior']} "
                 f"(|r|={result['idealpoint_corr']})")
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    print(f"  saved figure -> {path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--num_politicians", type=int, default=4000)
    ap.add_argument("--num_bills", type=int, default=300)
    ap.add_argument("--doc_length", type=int, default=100)
    ap.add_argument("--vocab_size", type=int, default=500)
    ap.add_argument("--num_steps", type=int, default=5000)
    ap.add_argument("--batch_size", type=int, default=100)
    ap.add_argument("--figures", action="store_true")
    args = ap.parse_args()

    true_pts, corpus = build_corpus(
        args.num_politicians, args.num_bills, args.doc_length, args.vocab_size)

    # Reconstruction is always per-document, so w_prior is the genuine ELBO weight.
    # A large w_prior over-weights the KL (as the old per-token scaling did) and collapses.
    big_w = float(args.num_bills)
    configs = [
        ("w_prior=1 (true ELBO)",                      1.0),
        (f"w_prior={int(big_w)} (KL over-weighted)",   big_w),
        ("w_prior=0.01 (works, not a bound)",          0.01),
    ]

    print(f"# IdealPointNN VAE posterior-collapse comparison "
          f"(N={args.num_politicians}, bills={args.num_bills}, steps={args.num_steps})\n")
    header = f"{'config':>34} {'w_prior':>8} {'raw_KL':>8} {'idealpoint_corr':>16}"
    print(header)
    print("-" * len(header))

    results = []
    for label, w_prior in configs:
        r = run_config(true_pts, corpus, label, w_prior,
                       num_steps=args.num_steps, batch_size=args.batch_size)
        results.append(r)
        print(f"{r['label']:>34} {r['w_prior']:>8} "
              f"{r['raw_kl_per_doc']:>8.3f} {r['idealpoint_corr']:>16.3f}")
        if args.figures and w_prior == 1.0:
            maybe_plot(true_pts, r, "figures/idealpoint_per_document_w1_recovery.png")

    print("\nInterpretation (reconstruction is per-document => w_prior is the ELBO weight):")
    print("  w_prior=1     -> KL>0, ideal points recovered: the true ELBO, no collapse.")
    print("  w_prior=large -> KL over-weighted (old per-token w=1): posterior collapses.")
    print("  w_prior=0.01  -> also fine here, but not a valid lower bound on the likelihood.")

    dump = [{k: v for k, v in r.items() if not k.startswith("_")} for r in results]
    with open("experiments/posterior_collapse_idealpoint_results.json", "w") as f:
        json.dump(dump, f, indent=2)


if __name__ == "__main__":
    main()
