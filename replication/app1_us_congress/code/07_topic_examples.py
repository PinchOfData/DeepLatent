"""Step 07: example speeches per topic, to label the topics (run in the P&P environment: it reads
the raw text in data/speeches).

For each topic, the speeches with the largest posterior share of the topic among those with at least
--min-terms vocabulary terms, at most two per congress. Writes results/<tag>/exhibits/topic_examples.csv
and topic_examples.txt (top terms, then the opening of each example speech).

Usage: python 07_topic_examples.py --tag pilot_k20 [--per-topic 8] [--chars 700]
"""
import argparse
import textwrap

import numpy as np
import pandas as pd

import common


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--per-topic", type=int, default=8)
    ap.add_argument("--min-terms", type=int, default=60)
    ap.add_argument("--chars", type=int, default=700)
    a = ap.parse_args()
    res = common.RESULTS / a.tag
    out = res / "exhibits"
    out.mkdir(parents=True, exist_ok=True)
    docs = pd.read_csv(res / "docs.csv.gz", dtype={"speaker_session": str, "person_id": str, "speech_id": str})
    theta = np.load(res / "theta.npy")
    topics = pd.read_csv(out / "topic_words.csv") if (out / "topic_words.csv").exists() else None
    eligible = docs["n_terms"].to_numpy() >= a.min_terms
    picks = []
    for k in range(theta.shape[1]):
        order = np.argsort(-np.where(eligible, theta[:, k], -1))
        per_congress = {}
        for i in order:
            c = docs.at[i, "congress"]
            if per_congress.get(c, 0) < 2:
                per_congress[c] = per_congress.get(c, 0) + 1
                picks.append({"topic": k, "row": i, "share": float(theta[i, k])})
            if sum(per_congress.values()) >= a.per_topic:
                break
    picks = pd.DataFrame(picks)
    picks = picks.join(docs[["speech_id", "source", "congress", "date", "party", "chamber"]], on="row")
    texts = {}
    for (source, congress), g in picks.groupby(["source", "congress"]):
        sp = pd.read_parquet(common.SPEECHES / f"{common.stem(source, congress)}.parquet", columns=["speech_id", "text"])
        sp["speech_id"] = sp["speech_id"].astype(str)  # integer ids (hein) are read as numbers elsewhere
        texts.update(sp[sp["speech_id"].isin(g["speech_id"])].set_index("speech_id")["text"].to_dict())
    picks["text"] = picks["speech_id"].map(texts).str.slice(0, a.chars)
    picks.drop(columns=["row"]).to_csv(out / "topic_examples.csv", index=False)
    with open(out / "topic_examples.txt", "w") as f:
        for k, g in picks.groupby("topic"):
            f.write(f"\n{'=' * 100}\nTOPIC {k}")
            if topics is not None:
                f.write(f" (share {topics.at[k, 'share']:.3f})\n  observed: {topics.at[k, 'observed']}"
                        f"\n  FREX: {topics.at[k, 'frex']}")
            f.write("\n")
            for _, r in g.iterrows():
                f.write(f"\n-- {r['date']} | congress {r['congress']} | {r['party']} | {r['chamber']} | share {r['share']:.2f}\n")
                f.write(textwrap.fill(str(r["text"]), 110) + "\n")
    print(f"wrote {out / 'topic_examples.txt'}", flush=True)


if __name__ == "__main__":
    main()
