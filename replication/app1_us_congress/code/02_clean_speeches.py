"""Step 02: clean the speeches with the Platforms & Policies cleaner, unchanged (run once).

Calls the vendored P&P Codes/python/scripts/01_clean_texts.py exactly as its us_congress run does
(slurm/run_01_clean_texts.sh): filter_by_language_raw keeps the speeches whose raw text fasttext
(lid.176) classifies as English, then TextCleaner(language="english", stopwords=STOPWORDS_US_CONGRESS)
.clean_df with batch_size=10 and min_tokens=0 (no length threshold). TextCleaner runs spaCy
en_core_web_lg: drops punctuation, tokens containing digits and PERSON/GPE/LOC entities, keeps
VERB/NOUN/PROPN/ADJ, lowercases and lemmatizes, then drops tokens shorter than 3 characters and the
P&P stopwords.

Writes clean/<source>/<congress>.parquet: speech_id, doc_clean, word_count, one row per English
speech of speeches/<source>/<congress>.parquet, in the same order.

Usage: python 02_clean_speeches.py --task I [--n-process N]   (I indexes common.ALL_TASKS)
"""
import argparse
import json
import time

import pandas as pd

import common

CHUNK = 50_000  # as P&P --chunksize


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", type=int, required=True)
    ap.add_argument("--n-process", type=int, default=4)
    ap.add_argument("--force", action="store_true", help="re-clean even if the output exists")
    a = ap.parse_args()
    source, congress = common.task(a.task)
    src = common.SPEECHES / f"{common.stem(source, congress)}.parquet"
    out = common.CLEAN / f"{common.stem(source, congress)}.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists() and out.with_suffix(".json").exists() and not a.force:
        print(f"{source} {congress}: already cleaned, skipped", flush=True)
        return

    pp = common.load_pp_cleaning()
    cleaner = pp.TextCleaner(language="english", stopwords=pp.STOPWORDS_US_CONGRESS)
    df = pd.read_parquet(src, columns=["speech_id", "text"])
    t0 = time.time()
    parts, n_english = [], 0
    for start in range(0, len(df), CHUNK):
        chunk = pp.filter_by_language_raw(df.iloc[start:start + CHUNK], "text", "en")
        n_english += len(chunk)
        if chunk.empty:
            continue
        cleaned = cleaner.clean_df(chunk, text_col="text", batch_size=10, n_process=a.n_process,
                                   min_tokens=0, show_progress=False)
        assert len(cleaned) == len(chunk), "the cleaner dropped speeches"
        parts.append(cleaned[["speech_id", "doc_clean", "word_count"]])
        print(f"{source} {congress}: {min(start + CHUNK, len(df)):,}/{len(df):,} speeches "
              f"({time.time() - t0:.0f}s)", flush=True)
    res = pd.concat(parts, ignore_index=True)
    res.to_parquet(out, index=False)
    stats = {"source": source, "congress": congress, "speeches": len(df), "english": n_english,
             "tokens": int(res["word_count"].sum()),
             "seconds": time.time() - t0, "n_process": a.n_process}
    with open(out.with_suffix(".json"), "w") as f:
        json.dump(stats, f, indent=1)
    print(json.dumps(stats), flush=True)


if __name__ == "__main__":
    main()
