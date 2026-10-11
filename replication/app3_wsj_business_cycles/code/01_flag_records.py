"""Step 01: one row of flags per WSJ record, no record dropped (filters and the vocabulary come later, in 02).

The raw file (~/deeplatent/data/WSJ_articles_full.csv: an index, the ProQuest GOID, the date and `doc`, the record's
tokens as an alphabetically sorted bag of lowercased unigrams) mixes news articles with advertisements, notices,
stock tables, web boilerplate and badly scanned pages. For every record from --min-year on this step counts

    n_tokens   all tokens (whole scanned pages and tables run to thousands)
    n_numeric  tokens with a digit (tables, prices, addresses)
    n_alpha    purely alphabetic tokens
    n_known    alphabetic tokens that occur in at least --known-df records of the period (scan quality: OCR
               garbage is mostly unique strings; an English word list misses inflections, names and terms)
    n_short    alphabetic tokens of 1-2 letters (OCR fragments)
    n_bybee    tokens in the Bybee et al. vocabulary
    n_table    stock-table abbreviations      n_web      web/print boilerplate words

and an md5 of the bag; dup_rank numbers the records with the same bag in date order (0 = first) and dup_size is the
group size. Writes <scratch>/data/records.csv.gz, <scratch>/data/token_df.csv.gz (the number of records in which
each alphabetic token occurs, for tokens in at least 5; step 02 lemmatizes them) and prints the distributions plus a
few example records per flag region, to choose the thresholds of step 02.

--out-dir writes elsewhere than <scratch>/data (e.g. data/all_years for --min-year 1889, keeping the 1984-2011 files);
the report then also prints, for every decade, a few records that pass the default filters of step 02 and a few that
fail them.

Usage: python 01_flag_records.py [--min-year 1984] [--workers 8] [--out-dir <scratch>/data/all_years]
"""
import argparse
import collections
import hashlib
import multiprocessing as mp
import re

from pathlib import Path

import numpy as np
import pandas as pd

import common

TABLE = {"chg", "vol", "est", "yld", "div", "pe", "hi", "lo", "nyse", "amex", "nasdaq", "sym", "cls", "pct", "wk",
         "yr", "qtr", "ytd", "nav", "bid", "ask", "mo"}
WEB = {"wsj", "online", "copyright", "www", "com", "http", "html", "visit", "subscribe", "subscriber", "subscribers",
       "subscription", "email", "reprints", "reprint", "rights", "reserved"}
DIGIT = re.compile(r"\d")
ALPHA = re.compile(r"^[a-z]+$")
_KNOWN, _BYBEE = set(), set()


def _init(known, bybee):
    global _KNOWN, _BYBEE
    _KNOWN, _BYBEE = known, bybee


def df_chunk(chunk: pd.DataFrame) -> collections.Counter:
    """Number of records of the chunk in which each alphabetic token occurs."""
    c = collections.Counter()
    for doc in chunk["doc"].fillna("").astype(str):
        c.update({t for t in doc.split() if ALPHA.match(t)})
    return c


def flag_chunk(chunk: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for doc in chunk["doc"].fillna("").astype(str):
        toks = doc.split()
        n_num = n_alpha = n_kn = n_short = n_byb = n_tab = n_web = 0
        for t in toks:
            if t in _BYBEE:
                n_byb += 1
            if DIGIT.search(t):
                n_num += 1
            elif ALPHA.match(t):
                n_alpha += 1
                if t in _KNOWN:
                    n_kn += 1
                if len(t) <= 2:
                    n_short += 1
            if t in TABLE:
                n_tab += 1
            if t in WEB:
                n_web += 1
        rows.append((len(toks), n_num, n_alpha, n_kn, n_short, n_byb, n_tab, n_web,
                     hashlib.md5(doc.encode()).hexdigest()))
    out = pd.DataFrame(rows, columns=["n_tokens", "n_numeric", "n_alpha", "n_known", "n_short", "n_bybee",
                                      "n_table", "n_web", "md5"], index=chunk.index)
    return pd.concat([chunk[["raw_row", "goid", "date"]], out], axis=1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-year", type=int, default=1984)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--chunksize", type=int, default=50_000)
    ap.add_argument("--known-df", type=int, default=100)
    ap.add_argument("--out-dir", default=None, help="default: <scratch>/data")
    a = ap.parse_args()
    out_dir = Path(a.out_dir) if a.out_dir else common.DATA
    bybee = set(common.read_words(common.BYBEE_VOCAB))

    years_all, parts = [], []

    def chunks(count_years=False):
        reader = pd.read_csv(common.RAW_WSJ, header=0, names=["raw_row", "goid", "date", "doc"],
                             dtype={"raw_row": np.int64, "goid": str, "date": str, "doc": str},
                             chunksize=a.chunksize)
        for c in reader:
            y = pd.to_numeric(c["date"].str[:4], errors="coerce")
            if count_years:
                years_all.append(y.value_counts(dropna=False))
            yield c[y >= a.min_year]

    df_all = collections.Counter()
    with mp.Pool(a.workers) as pool:
        for part in pool.imap_unordered(df_chunk, chunks()):
            df_all.update(part)
    known = {t for t, n in df_all.items() if n >= a.known_df}
    print(f"{len(df_all):,} distinct alphabetic tokens; {len(known):,} occur in at least {a.known_df} records "
          f"(the known words); Bybee vocabulary {len(bybee):,}", flush=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    tdf = pd.DataFrame([(t, n) for t, n in df_all.items() if n >= 5], columns=["token", "df"])
    tdf.sort_values("df", ascending=False).to_csv(out_dir / "token_df.csv.gz", index=False)
    del df_all, tdf

    with mp.Pool(a.workers, initializer=_init, initargs=(known, bybee)) as pool:
        for i, part in enumerate(pool.imap(flag_chunk, chunks(count_years=True))):
            parts.append(part)
            if i % 20 == 0:
                print(f"  chunk {i}: {sum(len(p) for p in parts):,} records flagged", flush=True)

    years = pd.concat(years_all, axis=1).sum(axis=1).sort_index()
    print("\nRaw records by year (all years in the file):")
    print(years.astype(int).to_string())

    df = pd.concat(parts, ignore_index=True)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    bad = df["date"].isna().sum()
    df = df.dropna(subset=["date"]).sort_values(["date", "raw_row"]).reset_index(drop=True)
    df["year"] = df.date.dt.year
    df["month"] = df.date.dt.strftime("%Y-%m")
    df["dup_rank"] = df.groupby("md5").cumcount()
    df["dup_size"] = df.groupby("md5")["md5"].transform("size")
    print(f"\n{len(df):,} records from {a.min_year} ({bad} with unparseable dates left out of the file below)")

    out = out_dir / "records.csv.gz"
    df.to_csv(out, index=False)
    print(f"wrote {out}", flush=True)

    # Distributions, to choose the thresholds of step 02
    s = pd.DataFrame({
        "tokens": df.n_tokens, "bybee": df.n_bybee,
        "known_share": df.n_known / df.n_alpha.clip(lower=1),
        "numeric_share": df.n_numeric / df.n_tokens.clip(lower=1),
        "short_share": df.n_short / df.n_alpha.clip(lower=1),
        "table": df.n_table, "web": df.n_web,
    })
    q = [.01, .05, .1, .25, .5, .75, .9, .95, .99]
    print("\nQuantiles:")
    print(s.quantile(q).round(3).to_string())
    print("\nBy year: records, median bybee tokens, shares with bybee<30, known<0.9, numeric>0.5, tokens>3000, "
          "web>=3, duplicate of an earlier record")
    g = pd.concat([df[["year"]], s, (df.dup_rank > 0).rename("dup")], axis=1).groupby("year")
    print(pd.DataFrame({
        "records": g.size(), "med_bybee": g.bybee.median(),
        "bybee<30": g.bybee.apply(lambda x: (x < 30).mean()), "known<.9": g.known_share.apply(lambda x: (x < .9).mean()),
        "num>.5": g.numeric_share.apply(lambda x: (x > .5).mean()), "tok>3000": g.tokens.apply(lambda x: (x > 3000).mean()),
        "web>=3": g.web.apply(lambda x: (x >= 3).mean()),
        "dup": g.dup.mean(),
    }).round(3).to_string())
    print("\nLargest duplicate groups (size, median bybee tokens, years):")
    big = df[df.dup_rank == 0].nlargest(15, "dup_size")
    for r in big.itertuples():
        yrs = df.loc[df.md5 == r.md5, "year"]
        print(f"  {r.dup_size:6d} {r.n_bybee:5d}  {yrs.min()}-{yrs.max()}")

    # Example records per flag region (first 50 tokens of the bag)
    raw_rows = {}
    rng = np.random.default_rng(0)
    regions = {
        "known_share < 0.8": s.known_share < 0.8,
        "0.8 <= known_share < 0.9": (s.known_share >= 0.8) & (s.known_share < 0.9),
        "0.9 <= known_share < 0.95": (s.known_share >= 0.9) & (s.known_share < 0.95),
        "tokens > 3000": s.tokens > 3000,
        "1500 < tokens <= 3000": (s.tokens > 1500) & (s.tokens <= 3000),
        "numeric_share > 0.6": s.numeric_share > 0.6,
        "0.4 < numeric_share <= 0.6": (s.numeric_share > 0.4) & (s.numeric_share <= 0.6),
        "table >= 5": s.table >= 5,
        "web >= 3": s.web >= 3,
        "bybee < 30": s.bybee < 30,
        "dup_rank > 0 and bybee >= 30": (df.dup_rank > 0) & (s.bybee >= 30),
        "clean (known >= .95, numeric <= .3, bybee >= 30, no dup)":
            (s.known_share >= .95) & (s.numeric_share <= .3) & (s.bybee >= 30) & (df.dup_rank == 0),
    }
    # By decade: records passing step 02's default filters and records failing them
    passing = ((s.known_share >= 0.9) & (s.numeric_share <= 0.4) & (s.tokens <= 2500) & (s.bybee >= 30)
               & (df.dup_rank == 0))
    decade = (df.year // 10) * 10
    print("\nBy decade: records, share passing step 02's default filters, median Bybee tokens of the passing ones")
    print(pd.DataFrame({"records": decade.value_counts().sort_index(),
                        "pass": passing.groupby(decade).mean().round(3),
                        "med_bybee_pass": s.bybee[passing].groupby(decade[passing]).median()}).to_string())
    for d in sorted(decade.unique()):
        for ok in [True, False]:
            regions[f"decade {d}, {'passing' if ok else 'failing'}"] = (decade == d) & (passing == ok)
    for name, m in regions.items():
        idx = np.flatnonzero(m.to_numpy())
        pick = rng.choice(idx, size=min(4 if name.startswith("decade") else 6, len(idx)), replace=False) \
            if len(idx) else []
        raw_rows[name] = df.loc[pick, "raw_row"].tolist()
        print(f"\n[{name}] {m.mean():.3f} of records")
    wanted = {r for v in raw_rows.values() for r in v}
    found = {}
    for c in pd.read_csv(common.RAW_WSJ, header=0, names=["raw_row", "goid", "date", "doc"],
                         dtype={"doc": str, "date": str, "goid": str}, chunksize=200_000):
        hit = c[c.raw_row.isin(wanted)]
        found.update(zip(hit.raw_row, zip(hit.date, hit.doc.fillna(""))))
        if len(found) == len(wanted):
            break
    for name, rows in raw_rows.items():
        print(f"\n== {name}")
        for r in rows:
            d, doc = found.get(r, ("?", ""))
            toks = doc.split()
            print(f"  {d} ({len(toks)} tokens): {' '.join(toks[:50])[:330]}")


if __name__ == "__main__":
    main()
