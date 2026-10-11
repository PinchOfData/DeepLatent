"""Step 06: the words at each end of a fit's ideal-point spectrum.

    python 06_words_by_ideal_point.py --tag votes_text_103_118_cpu --step 40000 --estimate theta_text

The step-05 runs saved no model, so their word loadings cannot be read directly. Instead: the
words that most distinguish the legislator-congresses at the two ends of an estimate (bottom vs
top quintile among units with speeches), by the log-odds ratio with an informative Dirichlet prior
of Monroe, Colaresi & Quinn (2008): z-score of the difference in a word's log-odds between the two
groups, the prior being the word's overall frequency (10 pseudo-counts per vocabulary word in
total). Two groupings: quintiles over all congresses pooled (time included), and quintiles within
each congress, pooled afterwards (time removed).

For each word, its counts in the two groups and its partisan lean: the Republican share of its
uses minus the Republican share of all words (positive: used more by Republicans).
Writes results/<tag>/words_<estimate>_<step>.csv and prints the top words.
"""

import argparse

import numpy as np
import pandas as pd
import scipy.sparse as sp

import common as c


def log_odds(low, high, prior):
    """Monroe-Colaresi-Quinn z-scores, positive for words of the high group."""
    a0 = prior.sum()
    n_hi, n_lo = high.sum(), low.sum()
    d = np.log((high + prior) / (n_hi + a0 - high - prior)) - np.log(
        (low + prior) / (n_lo + a0 - low - prior))
    return d / np.sqrt(1 / (high + prior) + 1 / (low + prior))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="votes_text_103_118_cpu")
    ap.add_argument("--step", type=int, default=40000)
    ap.add_argument("--estimate", default="theta_text")
    ap.add_argument("--top", type=int, default=30)
    args = ap.parse_args()
    e = pd.read_csv(c.MODULE / "results" / args.tag / f"estimates_{args.step}.csv",
                    dtype={"bioguide": str})
    house = c.DATA / "house"
    S = sp.load_npz(house / "speech_counts.npz").tocsr()[e["unit"].to_numpy()]
    vocab = np.array([w for w in (house / "vocab.txt").read_text().split("\n") if w])
    ok = e[args.estimate].notna().to_numpy() & (np.asarray(S.sum(1)).ravel() > 0)

    corpus = np.asarray(S[ok].sum(0)).ravel().astype(float)
    prior = 10 * len(vocab) * corpus / corpus.sum()
    rep = (e["party_code"] == 200).to_numpy() & ok
    dem = (e["party_code"] == 100).to_numpy() & ok
    r_words = np.asarray(S[rep].sum(0)).ravel().astype(float)
    d_words = np.asarray(S[dem].sum(0)).ravel().astype(float)
    lean = r_words / np.maximum(r_words + d_words, 1) - r_words.sum() / (r_words + d_words).sum()

    x = e[args.estimate]
    groups = {
        "pooled": (ok & (x <= x[ok].quantile(0.2)).to_numpy(), ok & (x >= x[ok].quantile(0.8)).to_numpy()),
    }
    q = e[ok].groupby("congress")[args.estimate].quantile([0.2, 0.8]).unstack()
    lo_cut, hi_cut = e["congress"].map(q[0.2]), e["congress"].map(q[0.8])
    groups["within congress"] = (ok & (x <= lo_cut).to_numpy(), ok & (x >= hi_cut).to_numpy())

    rows = []
    for name, (lo, hi) in groups.items():
        low = np.asarray(S[lo].sum(0)).ravel().astype(float)
        high = np.asarray(S[hi].sum(0)).ravel().astype(float)
        z = log_odds(low, high, prior)
        mean_lo, mean_hi = e.loc[lo, "congress"].mean(), e.loc[hi, "congress"].mean()
        r_lo, r_hi = (e.loc[lo, "party_code"] == 200).mean(), (e.loc[hi, "party_code"] == 200).mean()
        print(f"\n=== {args.estimate}, {name}: low end n={lo.sum()} (mean congress {mean_lo:.1f}, "
              f"{r_lo:.0%} Republican) vs high end n={hi.sum()} (mean congress {mean_hi:.1f}, "
              f"{r_hi:.0%} Republican)")
        for end, order in (("low", np.argsort(z)), ("high", np.argsort(-z))):
            top = order[: args.top]
            print(f"  {end} end: " + ", ".join(f"{vocab[j]} ({lean[j]:+.2f})" for j in top))
            print(f"    mean partisan lean of these words: {lean[top].mean():+.3f}")
            for rank, j in enumerate(top, 1):
                rows.append({"grouping": name, "end": end, "rank": rank, "word": vocab[j],
                             "z": z[j], "count_low": low[j], "count_high": high[j],
                             "partisan_lean": lean[j]})
    out = c.MODULE / "results" / args.tag / f"words_{args.estimate}_{args.step}.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
