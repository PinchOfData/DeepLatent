"""Step 02: filter the flagged records (01), build the vocabulary and the document-term matrix, and the monthly
industrial-production series.

Records are kept when they fall in [--min-year, --max-year], are not an exact copy of an earlier record (same bag
of tokens: recurring ads and notices), have at least --min-known of their alphabetic tokens among the words known to
the corpus (step 01; scanning quality), at most --max-numeric of their tokens with a digit (tables, listings), at
most --max-tokens tokens (whole scanned pages) and at least --min-terms vocabulary terms. The vocabulary is the Bybee et al. unigram list minus DROP_TERMS (stock-table abbreviations, month
abbreviations, OCR fragments and web/print boilerplate that the list contains), restricted to the terms that occur in
the training records (the records up to --train-end, the only ones the topic models see).

The Bybee list holds lemmas ("company", "share", "analyst") while the records are not lemmatized ("companies",
"shares", "analysts", "said"). A token counts for the term it equals; otherwise, for the term its spaCy lemma
(en_core_web_sm, the word on its own: the records are unordered bags) equals. The lemmas are computed once per
token type, for the types in at least --lemma-min-df records (step 01's token_df).

Writes to <scratch>/data/dtm/: X.npz (records x terms, CSR, int32 counts, records in date order), records.csv.gz
(raw_row, goid, date, month, n_terms of the kept records), vocab.txt, filters.csv (records dropped by each filter,
by year), and monthly_ip.csv (month, INDPRO, growth = 100 x monthly log change).

Runs in the `deeplatentfactormodels` conda env (spaCy with en_core_web_sm; the deeplatent env lacks the model):
  sbatch --export=ALL,STEP=02_build_dtm.py,PY=/home/Gauthier/.conda/envs/deeplatentfactormodels/bin/python \
      slurm/cpu.sbatch
Usage: python 02_build_dtm.py [--min-year 1984] [--max-year 2011] [--train-end 1999-12] ...
"""
import argparse
import json
import multiprocessing as mp

import numpy as np
import pandas as pd
import scipy.sparse as sp

import common

DROP_TERMS = {
    # stock tables and listings
    "chg", "vol", "est", "yld", "div", "pe", "hi", "lo", "sym", "cls", "pct", "wk", "yr", "qtr", "ytd", "nav",
    # month abbreviations (not weekdays: the list's "sun" is mostly Sun Microsystems)
    "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "sept", "oct", "nov", "dec",
    # OCR fragments
    "ing", "tion", "ment", "ness", "ly",
    # web and print boilerplate
    "wsj", "copyright", "www", "com", "http", "html", "reprints", "reprint",
}


_COL = {}


def _init(col):
    global _COL
    _COL = col


def count_chunk(args):
    """Term counts of a chunk of records: (raw rows, nonzeros per record, term indices, counts)."""
    raw_rows, docs = args
    lengths, indices, data = [], [], []
    for doc in docs:
        counts = {}
        for t in doc.split():
            j = _COL.get(t)
            if j is not None:
                counts[j] = counts.get(j, 0) + 1
        js = sorted(counts)
        lengths.append(len(js))
        indices.extend(js)
        data.extend(counts[j] for j in js)
    return raw_rows, np.array(lengths, np.int64), np.array(indices, np.int32), np.array(data, np.int32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-year", type=int, default=1984)
    ap.add_argument("--max-year", type=int, default=2011)
    ap.add_argument("--train-end", default="1999-12", help="last month whose records train the topic models")
    ap.add_argument("--min-known", type=float, default=0.9)
    ap.add_argument("--max-tokens", type=int, default=2500)
    ap.add_argument("--max-numeric", type=float, default=0.4)
    ap.add_argument("--min-terms", type=int, default=30)
    ap.add_argument("--keep-duplicates", action="store_true")
    ap.add_argument("--lemma-min-df", type=int, default=20)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    out = common.DATA / "dtm"
    out.mkdir(parents=True, exist_ok=True)

    rec = pd.read_csv(common.DATA / "records.csv.gz", dtype={"goid": str, "md5": str, "month": str})
    rec = rec[(rec.year >= a.min_year) & (rec.year <= a.max_year)].reset_index(drop=True)
    known = rec.n_known / rec.n_alpha.clip(lower=1)
    numeric = rec.n_numeric / rec.n_tokens.clip(lower=1)
    filt = pd.DataFrame({"year": rec.year,
                         "duplicate": (rec.dup_rank > 0) & ~a.keep_duplicates,
                         "scan_quality": known < a.min_known,
                         "numeric": numeric > a.max_numeric,
                         "page": rec.n_tokens > a.max_tokens})
    pre = ~(filt.duplicate | filt.scan_quality | filt.numeric | filt.page)
    print(f"{len(rec):,} records {a.min_year}-{a.max_year}; {pre.sum():,} pass the duplicate, scan-quality, "
          f"numeric and page filters", flush=True)

    vocab = [w for w in common.read_words(common.BYBEE_VOCAB) if w not in DROP_TERMS]
    vocab = list(dict.fromkeys(vocab))
    col = {w: j for j, w in enumerate(vocab)}
    print(f"vocabulary: {len(vocab):,} Bybee terms after dropping {len(DROP_TERMS)} listed terms", flush=True)

    # Token type -> term: the term itself, else the term its lemma equals
    import spacy

    tdf = pd.read_csv(common.DATA / "token_df.csv.gz", keep_default_na=False)
    tdf = tdf[tdf.df >= a.lemma_min_df]
    other = [w for w in tdf.token if w not in col]
    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
    form = dict(col)
    mapped = []
    for w, d in zip(other, nlp.pipe(other, batch_size=4096)):
        j = col.get(d[0].lemma_.lower()) if len(d) == 1 else None
        if j is not None:
            form[w] = j
            mapped.append(w)
    dfs = tdf.set_index("token")["df"]
    exact = dfs.reindex([w for w in vocab]).fillna(0).sum()
    print(f"lemmatized {len(other):,} token types (df >= {a.lemma_min_df}); {len(mapped):,} map to a term; record "
          f"occurrences: {exact:,.0f} exact, {dfs.loc[mapped].sum():,.0f} through a lemma", flush=True)
    top = dfs.loc[mapped].sort_values(ascending=False).head(40)
    print("  most frequent:", ", ".join(f"{w}->{vocab[form[w]]}" for w in top.index), flush=True)

    # Count the terms of the records that pass so far (one pass over the raw file, in its own row order)
    assert rec.raw_row.is_unique
    keep_rows = set(rec.raw_row[pre])

    def chunks():
        for c in pd.read_csv(common.RAW_WSJ, header=0, names=["raw_row", "goid", "date", "doc"],
                             dtype={"doc": str, "date": str, "goid": str}, chunksize=100_000):
            c = c[c.raw_row.isin(keep_rows)]
            yield c.raw_row.to_numpy(), c.doc.fillna("").tolist()

    parts = []
    with mp.Pool(a.workers, initializer=_init, initargs=(form,)) as pool:
        for i, part in enumerate(pool.imap(count_chunk, chunks())):
            parts.append(part)
            if i % 10 == 0:
                print(f"  chunk {i}: {sum(len(p[0]) for p in parts):,} records counted", flush=True)
    rows = np.concatenate([p[0] for p in parts])
    lengths = np.concatenate([p[1] for p in parts])
    indptr = np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64)
    X = sp.csr_matrix((np.concatenate([p[3] for p in parts]), np.concatenate([p[2] for p in parts]), indptr),
                      shape=(len(rows), len(vocab)))
    order = pd.Series(np.arange(len(rows)), index=rows)
    sub = rec[pre].copy()
    X = X[order.loc[sub.raw_row].to_numpy()]  # date order, as rec
    sub["n_terms"] = np.asarray(X.sum(axis=1)).ravel()

    filt["few_terms"] = False
    filt.loc[sub.index, "few_terms"] = sub.n_terms.to_numpy() < a.min_terms
    keep = sub.n_terms.to_numpy() >= a.min_terms
    X, sub = X[keep], sub[keep].reset_index(drop=True)

    # Vocabulary: terms that occur in the training records
    train = (sub.month <= a.train_end).to_numpy()
    used = np.asarray(X[train].sum(axis=0)).ravel() > 0
    X = X[:, np.flatnonzero(used)].tocsr()
    vocab = [w for w, u in zip(vocab, used) if u]
    sub["n_terms"] = np.asarray(X.sum(axis=1)).ravel()
    # A few validation/test records can fall below --min-terms once the terms absent from training are gone
    keep = sub.n_terms.to_numpy() >= a.min_terms
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

    print("\nRecords by year (a record can fail several filters; few_terms counts those passing the others):")
    print(by_year.to_string())
    m = sub.groupby("month").size()
    print(f"\nkept {len(sub):,} records, {X.nnz:,} nonzeros, {int(X.sum()):,} tokens, vocabulary {len(vocab):,}")
    print(f"records per month: min {m.min()}, median {int(m.median())}, max {m.max()}; terms per record: median "
          f"{int(sub.n_terms.median())}, 10% {int(sub.n_terms.quantile(.1))}, 90% {int(sub.n_terms.quantile(.9))}")
    print(f"training records (to {a.train_end}): {train.sum():,}")
    print("dropped listed terms present in the Bybee list:", sorted(DROP_TERMS & set(common.read_words(common.BYBEE_VOCAB))))


if __name__ == "__main__":
    main()
