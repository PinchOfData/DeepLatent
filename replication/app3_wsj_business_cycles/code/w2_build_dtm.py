"""Wave 2, step 2: the document-term matrix for the expanding-window evaluation (a longer sample).

As 02_build_dtm.py (the same filters, the Bybee et al. lemma vocabulary minus DROP_TERMS, spaCy lemma mapping), but
from step 01's flags of all years (--flags-dir), for --min-year .. --max-year, into --out-dir, with the vocabulary
restricted to the terms of the records up to --vocab-end (the first window's training and validation text). Text from
other years still enters through the Bybee list itself (built on 1984-2017 news) and step 01's corpus-wide known-word
counts and lemma set (text only, no outcomes). Optionally --extra-terms N adds the N most frequent lemmas that are not
in the Bybee list (words of older text the list lacks), ranked by step 01's counts over all years (not used so far).

Writes <out-dir>/: X.npz, records.csv.gz, vocab.txt, filters.csv, settings.json, monthly_ip.csv (as 02).

Usage (deeplatentfactormodels env, for spaCy's model):
  python w2_build_dtm.py --flags-dir <scratch>/data/all_years --out-dir <scratch>/data/dtm_w2 --min-year 1947
"""
import argparse
import importlib
import json
import multiprocessing as mp
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

import common

build = importlib.import_module("02_build_dtm")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--flags-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--min-year", type=int, default=1947)
    ap.add_argument("--max-year", type=int, default=2011)
    ap.add_argument("--vocab-end", default="1979-12", help="last month whose records choose the vocabulary")
    ap.add_argument("--extra-terms", type=int, default=0)
    ap.add_argument("--min-known", type=float, default=0.9)
    ap.add_argument("--max-tokens", type=int, default=2500)
    ap.add_argument("--max-numeric", type=float, default=0.4)
    ap.add_argument("--min-terms", type=int, default=30)
    ap.add_argument("--lemma-min-df", type=int, default=20)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    flags, out = Path(a.flags_dir), Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    rec = pd.read_csv(flags / "records.csv.gz", dtype={"goid": str, "md5": str, "month": str})
    rec = rec[(rec.year >= a.min_year) & (rec.year <= a.max_year)].reset_index(drop=True)
    known = rec.n_known / rec.n_alpha.clip(lower=1)
    numeric = rec.n_numeric / rec.n_tokens.clip(lower=1)
    filt = pd.DataFrame({"year": rec.year, "duplicate": rec.dup_rank > 0, "scan_quality": known < a.min_known,
                         "numeric": numeric > a.max_numeric, "page": rec.n_tokens > a.max_tokens})
    pre = ~(filt.duplicate | filt.scan_quality | filt.numeric | filt.page)
    print(f"{len(rec):,} records {a.min_year}-{a.max_year}; {pre.sum():,} pass the duplicate, scan-quality, "
          f"numeric and page filters", flush=True)

    import spacy

    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
    vocab = list(dict.fromkeys(w for w in common.read_words(common.BYBEE_VOCAB) if w not in build.DROP_TERMS))
    tdf = pd.read_csv(flags / "token_df.csv.gz", keep_default_na=False)
    tdf = tdf[tdf.df >= a.lemma_min_df]
    lemma = {w: (d[0].lemma_.lower() if len(d) == 1 else w) for w, d in
             zip(tdf.token, nlp.pipe(tdf.token.tolist(), batch_size=4096))}
    if a.extra_terms:
        stop = nlp.Defaults.stop_words
        ldf = tdf.assign(lemma=tdf.token.map(lemma)).groupby("lemma")["df"].sum().sort_values(ascending=False)
        bybee = set(vocab)
        extra = [w for w in ldf.index if w not in bybee and w.isalpha() and len(w) >= 3 and w not in stop
                 and w not in build.DROP_TERMS][:a.extra_terms]
        print(f"extra terms ({len(extra)}): {', '.join(extra[:60])} ...", flush=True)
        vocab += extra
    col = {w: j for j, w in enumerate(vocab)}
    form = dict(col)
    for w in tdf.token:
        if w not in col and lemma[w] in col:
            form[w] = col[lemma[w]]
    print(f"vocabulary: {len(vocab):,} terms; {len(form) - len(col):,} token types map to a term through a lemma",
          flush=True)

    assert rec.raw_row.is_unique
    keep_rows = set(rec.raw_row[pre])

    def chunks():
        for c in pd.read_csv(common.RAW_WSJ, header=0, names=["raw_row", "goid", "date", "doc"],
                             dtype={"doc": str, "date": str, "goid": str}, chunksize=100_000):
            c = c[c.raw_row.isin(keep_rows)]
            yield c.raw_row.to_numpy(), c.doc.fillna("").tolist()

    parts = []
    with mp.Pool(a.workers, initializer=build._init, initargs=(form,)) as pool:
        for i, part in enumerate(pool.imap(build.count_chunk, chunks())):
            parts.append(part)
            if i % 10 == 0:
                print(f"  chunk {i}: {sum(len(p[0]) for p in parts):,} records counted", flush=True)
    rows = np.concatenate([p[0] for p in parts])
    indptr = np.concatenate([[0], np.cumsum(np.concatenate([p[1] for p in parts]))]).astype(np.int64)
    X = sp.csr_matrix((np.concatenate([p[3] for p in parts]), np.concatenate([p[2] for p in parts]), indptr),
                      shape=(len(rows), len(vocab)))
    order = pd.Series(np.arange(len(rows)), index=rows)
    sub = rec[pre].copy()
    X = X[order.loc[sub.raw_row].to_numpy()]  # date order, as rec

    # Vocabulary: the terms of the records up to --vocab-end; then the minimum number of terms
    used = np.asarray(X[(sub.month <= a.vocab_end).to_numpy()].sum(axis=0)).ravel() > 0
    X = X[:, np.flatnonzero(used)].tocsr()
    vocab = [w for w, u in zip(vocab, used) if u]
    sub["n_terms"] = np.asarray(X.sum(axis=1)).ravel()
    keep = sub.n_terms.to_numpy() >= a.min_terms
    filt["few_terms"] = False
    filt.loc[sub.index[~keep], "few_terms"] = True
    X, sub = X[keep], sub[keep].reset_index(drop=True)

    sp.save_npz(out / "X.npz", X)
    sub[["raw_row", "goid", "date", "month", "n_terms"]].to_csv(out / "records.csv.gz", index=False)
    (out / "vocab.txt").write_text("\n".join(vocab) + "\n")
    by_year = filt.groupby("year").agg(records=("year", "size"), duplicate=("duplicate", "sum"),
                                       scan_quality=("scan_quality", "sum"), numeric=("numeric", "sum"),
                                       page=("page", "sum"), few_terms=("few_terms", "sum"))
    by_year["kept"] = sub.groupby(sub.month.str[:4].astype(int)).size()
    by_year.to_csv(out / "filters.csv")
    (out / "settings.json").write_text(json.dumps(vars(a), indent=1))
    ip = pd.read_csv(common.INDPRO, parse_dates=["observation_date"])
    ip["month"] = ip.observation_date.dt.strftime("%Y-%m")
    ip["growth"] = 100 * np.log(ip.INDPRO).diff()
    ip[["month", "INDPRO", "growth"]].to_csv(out / "monthly_ip.csv", index=False)

    print("\nRecords by year (few_terms counts those passing the other filters):")
    print(by_year.to_string())
    m = sub.groupby("month").size()
    print(f"\nkept {len(sub):,} records, {X.nnz:,} nonzeros, {int(X.sum()):,} tokens, vocabulary {len(vocab):,}")
    print(f"records per month: min {m.min()}, median {int(m.median())}, max {m.max()}; terms per record: median "
          f"{int(sub.n_terms.median())}, 10% {int(sub.n_terms.quantile(.1))}, 90% {int(sub.n_terms.quantile(.9))}")


if __name__ == "__main__":
    main()
