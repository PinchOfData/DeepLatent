"""Step 09: party-specific language within topics, by era, for a fitted model in results/<tag>/.

For each topic and each era (a range of congresses), the terms with the largest average (over the era's
congresses) Republican-minus-Democratic difference in the topic's word distribution, and the reverse, as in
exhibit 7 of 06_exhibits.py (which averages over all congresses). The topic's word distribution at congress t and
party p is the decoder's at the design row of (t, p) (GTM.get_topic_word_distribution), so a model whose
party deviations differ by topic and congress (SAGE --interactions congress_party) can show how each topic's
partisan vocabulary changes over time. Also each era's baseline terms (FREX of the era-average distribution
with party at 0). Writes results/<tag>/exhibits/party_language_by_era.csv.

Usage: python 09_party_language_by_era.py --tag sage_congress_party_l1-0_fixed_kmeans_step20000
"""
import argparse
import importlib

import numpy as np
import pandas as pd
import torch

import common
import topic_words

exhibits = importlib.import_module("06_exhibits")
ERAS = [(43, 61), (62, 81), (82, 101), (102, 118)]  # 1873-1910, 1911-1950, 1951-1990, 1991-2024


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--n-words", type=int, default=6)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    res = common.RESULTS / a.tag
    cfg, docs, theta, vocab, tm = exhibits.load_fit(res, torch.device(a.device))
    K = cfg["n_topics"]
    congresses = sorted(docs["congress"].unique().tolist())
    colnames = cfg["content_colnames"]

    def dist(t, ps):
        x = exhibits.design_row(t, ps, congresses, colnames, cfg["formula"])
        return tm.get_topic_word_distribution(content_covariates=x).astype(np.float64)

    rows = []
    for lo, hi in ERAS:
        ts = [t for t in congresses if lo <= t <= hi]
        gap, base = np.zeros((K, len(vocab))), np.zeros((K, len(vocab)))
        for t in ts:
            gap += dist(t, 1) - dist(t, -1)
            base += dist(t, 0)
        gap /= len(ts)
        frex = topic_words.frex(base / len(ts))
        era = f"{exhibits.start_year(lo)}-{exhibits.start_year(hi) + 1}"
        for k in range(K):
            rows.append({"topic": k, "era": era,
                         "baseline": exhibits.top_terms(frex[k], vocab, a.n_words),
                         "republicans": exhibits.top_terms(gap[k], vocab, a.n_words),
                         "democrats": exhibits.top_terms(-gap[k], vocab, a.n_words)})
        print(f"  era {era} done", flush=True)
    out = res / "exhibits" / "party_language_by_era.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
