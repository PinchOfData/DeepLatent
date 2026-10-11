"""Step 03: term counts per speech for a given (1,2)-gram vocabulary (cheap; rerun per vocabulary).

Python port of build_counts() in P&P Codes/R/02_party_classifier_gst.R (vendored copy in
vendor_pp/R/; check_counts_vs_R.py verifies identical counts): lowercase the cleaned text,
split on runs of non-word characters, keep tokens of at least 3 characters, then walk left
to right and emit a vocabulary bigram when the current adjacent pair is one (consuming both
tokens, so a bigram is not also counted as its two unigrams), else the single token; tokens
outside the vocabulary are dropped. Columns follow the order of the vocabulary file.

Every cleaned speech is counted. Writes terms/<vocabulary>/<source>/<congress>.npz (CSR speeches x
vocabulary) and <congress>_meta.csv.gz, one row per speech in the same order: speech_id, n_terms and
the step-01 metadata and sample flags (CSV so that the model environment needs no parquet reader).

Usage: python 03_count_terms.py --task I [--vocab inputs/vocab_us.txt]
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

import common

SPLIT = re.compile(r"\W+")
TOKEN_MIN_LEN = 3
META_COLS = ["speech_id", "source", "congress", "date", "chamber", "extension", "identified", "speaker_session",
             "person_id", "party", "state", "gender", "voting", "raw_word_count"]


def term_ids(doc: str, col: dict, bigrams: set) -> list:
    toks = [t for t in SPLIT.split(doc.lower()) if len(t) >= TOKEN_MIN_LEN]
    out, i, n = [], 0, len(toks)
    while i < n:
        if i < n - 1 and f"{toks[i]} {toks[i + 1]}" in bigrams:
            out.append(col[f"{toks[i]} {toks[i + 1]}"])
            i += 2
        else:
            j = col.get(toks[i])
            if j is not None:
                out.append(j)
            i += 1
    return out


def count_matrix(docs, vocab: list) -> sp.csr_matrix:
    col = {v: j for j, v in enumerate(vocab)}
    bigrams = {v for v in vocab if " " in v}
    indptr, indices = [0], []
    for doc in docs:
        indices.extend(term_ids(doc if isinstance(doc, str) else "", col, bigrams))
        indptr.append(len(indices))
    data = np.ones(len(indices), dtype=np.int32)
    m = sp.csr_matrix((data, np.asarray(indices, dtype=np.int32), np.asarray(indptr, dtype=np.int64)),
                      shape=(len(indptr) - 1, len(vocab)))
    m.sum_duplicates()
    return m


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", type=int, required=True)
    ap.add_argument("--vocab", default=str(common.VOCAB_PATH))
    a = ap.parse_args()
    source, congress = common.task(a.task)
    clean = pd.read_parquet(common.CLEAN / f"{common.stem(source, congress)}.parquet",
                            columns=["speech_id", "doc_clean"])
    vocab = common.load_vocab(a.vocab)
    m = count_matrix(clean["doc_clean"], vocab)
    out = common.terms_dir(a.vocab) / f"{common.stem(source, congress)}.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    sp.save_npz(out, m)
    n_terms = np.asarray(m.sum(axis=1)).ravel()
    meta = pd.read_parquet(common.SPEECHES / f"{common.stem(source, congress)}.parquet", columns=META_COLS)
    meta = pd.DataFrame({"speech_id": clean["speech_id"], "n_terms": n_terms}).merge(
        meta, on="speech_id", how="left", validate="one_to_one")
    meta.to_csv(out.with_name(f"{congress:03d}_meta.csv.gz"), index=False)
    gst = common.gst_sample(meta).to_numpy()
    stats = {"source": source, "congress": congress, "vocab": Path(a.vocab).name, "speeches": m.shape[0],
             "terms": int(m.sum()), "vocab_terms_used": int((np.asarray(m.sum(axis=0)).ravel() > 0).sum()),
             "n_terms_quantiles": {q: float(np.quantile(n_terms, q)) for q in (0.1, 0.25, 0.5, 0.75, 0.9)},
             "gst_sample_speeches": int(gst.sum()), "gst_sample_terms": int(n_terms[gst].sum())}
    with open(out.with_suffix(".json"), "w") as f:
        json.dump(stats, f, indent=1)
    print(json.dumps(stats), flush=True)


if __name__ == "__main__":
    main()
