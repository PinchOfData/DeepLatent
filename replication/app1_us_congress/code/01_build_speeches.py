"""Step 01: raw speeches -> one parquet per (source, congress), every speech kept.

No speech is dropped here: sample restrictions are flags, applied downstream (common.gst_sample),
so that changing them never requires re-cleaning.

hein-bound / hein-daily (Gentzkow-Shapiro-Taddy release; speeches_NNN.txt, descr_NNN.txt,
NNN_SpeakerMap.txt): every speech of speeches_NNN.txt, with descr (date, chamber, speaker string)
and, for speeches by identified legislators, the SpeakerMap (speakerid, party, state, gender,
district, nonvoting). descr chamber "E" marks Extensions of Remarks (absent from the SpeakerMap).
speakerid = congress followed by a six-digit person id that is stable across congresses and editions.

govinfo (us_congress_speeches_1994_2024.csv: speeches downloaded from govinfo.gov and split into
speaking turns by the congressional-record parser, one row per turn): every
turn, with its Extensions-of-Remarks flag and bioguide id; party and state from the Voteview record
of (congress, chamber, bioguide) (a member with two records in a congress, i.e. a party switch,
gets the last one). The congress is assigned from the date with the vendored P&P session table.

Columns: speech_id, source, congress, date, chamber (H/S/E/None), extension, speaker_raw, identified,
speaker_session, person_id, party (R/D/other codes, missing if unidentified), state, gender, voting
(False for non-voting delegates, missing if unidentified), raw_word_count, text.

Usage:
    python 01_build_speeches.py --hein bound --congresses 43-111
    python 01_build_speeches.py --hein daily --congresses 97-114
    python 01_build_speeches.py --govinfo --congresses 103-118
"""
import argparse
import csv
import json
import time

import pandas as pd

import common

TERRITORIES = {"DC", "PR", "GU", "VI", "AS", "MP"}  # delegates and resident commissioners do not vote
COLUMNS = ["speech_id", "source", "congress", "date", "chamber", "extension", "speaker_raw", "identified",
           "speaker_session", "person_id", "party", "state", "gender", "voting", "raw_word_count", "text"]


def congress_range(s: str) -> list:
    lo, _, hi = s.partition("-")
    return list(range(int(lo), int(hi or lo) + 1))


def write(df: pd.DataFrame, source: str, congress: int, stats: dict) -> None:
    out = common.SPEECHES / f"{common.stem(source, congress)}.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    df = df[COLUMNS].reset_index(drop=True)
    df.to_parquet(out, index=False)
    def counts(col):
        return {str(k): int(v) for k, v in df[col].value_counts(dropna=False).items()}

    stats.update({
        "speeches": len(df),
        "identified": int(df["identified"].sum()),
        "extension": int(df["extension"].sum()),
        "party": counts("party"),
        "chamber": counts("chamber"),
        "voting": counts("voting"),
        "gst_sample": int(common.gst_sample(df).sum()),
    })
    with open(out.with_suffix(".json"), "w") as f:
        json.dump(stats, f, indent=1, default=str)
    print(f"{source} {congress}: {json.dumps(stats, default=str)}", flush=True)


def build_hein(edition: str, congress: int) -> None:
    d = common.RAW / f"hein-{edition}"
    tag = f"{congress:03d}"
    ids, texts = [], []
    # Windows-1252 text, one speech per line: speech_id|speech (split on the first bar only).
    with open(d / f"speeches_{tag}.txt", encoding="cp1252", errors="replace") as f:
        next(f)
        for line in f:
            sid, _, txt = line.rstrip("\r\n").partition("|")
            ids.append(sid)
            texts.append(txt)
    df = pd.DataFrame({"speech_id": ids, "text": texts})
    read = dict(sep="|", dtype=str, quoting=csv.QUOTE_NONE, encoding="cp1252", encoding_errors="replace")
    descr = pd.read_csv(d / f"descr_{tag}.txt", usecols=["speech_id", "chamber", "date", "speaker"], **read)
    smap = pd.read_csv(d / f"{tag}_SpeakerMap.txt", **read)
    stats = {"speakermap_duplicate_speech_ids": int(smap["speech_id"].duplicated().sum()),
             "descr_duplicate_speech_ids": int(descr["speech_id"].duplicated().sum())}
    descr = descr.drop_duplicates("speech_id").rename(columns={"chamber": "chamber_descr", "speaker": "speaker_raw"})
    smap = smap.drop_duplicates("speech_id", keep=False)
    df = df.merge(descr, on="speech_id", how="left").merge(
        smap[["speech_id", "speakerid", "chamber", "state", "gender", "party", "nonvoting"]], on="speech_id", how="left")
    identified = df["speakerid"].notna()
    df = df.assign(
        source=edition,
        congress=congress,
        date=pd.to_datetime(df["date"], format="%Y%m%d", errors="coerce"),
        chamber=df["chamber"].where(identified, df["chamber_descr"]),
        extension=df["chamber_descr"].eq("E"),
        identified=identified,
        speaker_session=df["speakerid"],
        person_id=("h" + df["speakerid"].str[-6:]).where(identified),
        voting=df["nonvoting"].map({"voting": True, "nonvoting": False}).astype("boolean"),
        raw_word_count=df["text"].str.split().str.len().astype("int32"),
    )
    write(df, edition, congress, stats)


def voteview_lookup() -> pd.DataFrame:
    vv = pd.read_csv(common.RAW / "HSall_members.csv",
                     usecols=["congress", "chamber", "bioguide_id", "party_code", "state_abbrev"])
    vv = vv[vv["chamber"].isin(["House", "Senate"]) & vv["bioguide_id"].notna()]
    vv["chamber"] = vv["chamber"].str[0]
    # Party switchers have two records in a congress: keep the last (the new party).
    return vv.drop_duplicates(["congress", "chamber", "bioguide_id"], keep="last")


def build_govinfo(congresses: list) -> None:
    pp = common.load_pp_cleaning()
    sessions = pp.build_sessions()
    vv = voteview_lookup()
    keep = set(congresses)
    parts = {c: [] for c in congresses}
    cols = ["date", "text", "speaker", "speaker_bioguide", "chamber", "extension"]
    t0 = time.time()
    # The first column (blank header) is the row index of the govinfo file: kept as raw_idx.
    reader = pd.read_csv(common.RAW / "us_congress_speeches_1994_2024.csv",
                         usecols=lambda name: name in cols or name == "" or name.startswith("Unnamed"),
                         chunksize=200_000, dtype={"speaker_bioguide": str})
    n_undated = 0
    for chunk in reader:
        chunk = chunk.rename(columns={next(n for n in chunk.columns if n not in cols): "raw_idx"})
        date = pd.to_datetime(chunk["date"], errors="coerce")
        chunk["congress"] = pp.assign_congress(date, sessions)
        chunk["date"] = date
        n_undated += int(chunk["congress"].isna().sum())
        chunk = chunk[chunk["congress"].isin(keep)]
        if chunk.empty:
            continue
        chunk["congress"] = chunk["congress"].astype(int)
        for c, g in chunk.groupby("congress"):
            parts[c].append(g)
        print(f"  govinfo rows through index {int(chunk['raw_idx'].max()):,} ({time.time() - t0:.0f}s)", flush=True)
    print(f"  govinfo turns without a congress (undated or outside the session table): {n_undated:,}", flush=True)
    for c in congresses:
        df = pd.concat(parts[c], ignore_index=True)
        df["chamber"] = df["chamber"].astype(str).str[0]
        df = df.merge(vv, left_on=["congress", "chamber", "speaker_bioguide"],
                      right_on=["congress", "chamber", "bioguide_id"], how="left")
        identified = df["party_code"].notna()
        has_bioguide = df["speaker_bioguide"].notna()
        df = df.assign(
            speech_id="g" + df["raw_idx"].astype(str),
            source="govinfo",
            extension=df["extension"].astype(str).str.lower().eq("true"),
            speaker_raw=df["speaker"],
            identified=identified,
            speaker_session=(df["speaker_bioguide"] + "_" + df["congress"].astype(str) + "_" + df["chamber"]).where(has_bioguide),
            person_id=("b" + df["speaker_bioguide"]).where(has_bioguide),
            party=df["party_code"].map(lambda x: {100: "D", 200: "R"}.get(x, str(int(x))) if pd.notna(x) else pd.NA),
            state=df["state_abbrev"],
            gender=pd.NA,
            voting=(~df["state_abbrev"].isin(TERRITORIES)).astype("boolean").where(identified),
            raw_word_count=df["text"].fillna("").str.split().str.len().astype("int32"),
            text=df["text"].fillna(""),
        )
        write(df, "govinfo", c, {"bioguide": int(has_bioguide.sum())})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hein", choices=["bound", "daily"])
    ap.add_argument("--govinfo", action="store_true")
    ap.add_argument("--congresses", required=True, help="e.g. 43-111")
    a = ap.parse_args()
    congresses = congress_range(a.congresses)
    if a.govinfo:
        build_govinfo(congresses)
    else:
        for c in congresses:
            build_hein(a.hein, c)


if __name__ == "__main__":
    main()
