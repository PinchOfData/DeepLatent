"""Step 01: crosswalk from Gentzkow-Shapiro-Taddy (GST) speaker ids to Voteview members.

    python 01_crosswalk.py

The Hein speeches (bound 43rd-111th, daily 97th-114th) identify legislators by a GST speakerid
(congress followed by a six-digit person id that is stable across congresses and editions); the
Voteview roll calls identify them by icpsr, and HSall_members.csv maps (congress, chamber, icpsr)
to bioguide ids. The govinfo speeches (103rd-118th) already carry bioguide ids and need no
crosswalk.

Matching, within each (congress, chamber, state) cell of Voteview members, for every GST speaker:
  1. last name, normalized (upper case, accents, spaces and punctuation removed);
     if no exact match, a partial match (one name contains the other, >= 4 letters), then a close
     match (difflib ratio >= 0.85);
  2. if several candidates remain, narrowed by district, then first name (or nickname, or the same
     first three letters), then party; a narrowing step is skipped if it would leave none.
  3. person level: a GST person id should be one Voteview member across congresses. Rows whose
     match differs from the person's most frequent icpsr are flagged (person_conflict); unmatched
     or ambiguous rows of a person whose usual icpsr sits in the same cell are filled
     (rule person_propagated).
Nothing is dropped: every speaker of both editions and both chambers is kept, with its rule and
flags; estimation steps filter downstream.

Checks written to summary.json (main build, House): share of speakers and of speeches matched,
agreement of party and district between the two sources among matches, consistency of the two
editions where they overlap (97th-111th), and overlap with the bioguide ids of the govinfo speeches
(103rd-114th).

Output: data/crosswalk/gst_to_voteview.csv.gz (one row per edition x speakerid), summary.json.
"""

import difflib
import json
import re
import time
import unicodedata

import numpy as np
import pandas as pd

import common as c

PARTY = {100: "D", 200: "R"}


def norm(s):
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode()
    return re.sub(r"[^A-Z]", "", s.upper())


def gst_speakers():
    """One row per (edition, congress, speakerid), with its attributes and speech count."""
    parts = []
    for edition, congresses in c.HEIN.items():
        for cong in congresses:
            f = c.RAW / f"hein-{edition}" / f"{cong:03d}_SpeakerMap.txt"
            df = pd.read_csv(f, sep="|", dtype=str, keep_default_na=False, encoding="latin-1",
                             quoting=3)
            attrs = ["lastname", "firstname", "chamber", "state", "gender", "party", "district",
                     "nonvoting"]
            g = df.groupby("speakerid").agg(
                n_speeches=("speech_id", "size"),
                **{a: (a, "first") for a in attrs},
            )
            # a speakerid whose attributes vary across its speeches is flagged, not split
            varies = df.groupby("speakerid")[attrs].nunique().max(axis=1) > 1
            g["attrs_vary"] = varies.reindex(g.index).values
            g = g.reset_index()
            g.insert(0, "congress", cong)
            g.insert(0, "edition", edition)
            parts.append(g)
    sp = pd.concat(parts, ignore_index=True)
    sp["person"] = sp["speakerid"].str[-6:]
    return sp


def voteview_members():
    vv = pd.read_csv(c.RAW / "HSall_members.csv", dtype={"bioguide_id": str})
    vv = vv[vv["chamber"].isin(["House", "Senate"]) & (vv["congress"] >= 43)].copy()
    vv["chamber_code"] = vv["chamber"].map({"House": "H", "Senate": "S"})
    name = vv["bioname"].fillna("")
    vv["last"] = name.str.split(",").str[0].map(norm)
    given = name.str.split(",", n=1).str[1].fillna("")
    vv["first"] = given.str.replace(r"\(.*?\)", " ", regex=True).str.split().str[0].fillna("").map(norm)
    vv["nick"] = given.str.extract(r"\((.*?)\)")[0].fillna("").map(norm)
    vv["party_letter"] = vv["party_code"].map(PARTY).fillna("O")
    return vv


def match_one(s, cands):
    """(icpsr or None, rule, n_candidates) for one GST speaker within its Voteview cell."""
    L = norm(s.lastname)
    hits = cands[cands["last"] == L]
    rule = "last"
    if hits.empty:
        hits = cands[cands["last"].map(lambda v: min(len(v), len(L)) >= 4 and (L in v or v in L))]
        rule = "last_partial"
    if hits.empty:
        close = difflib.get_close_matches(L, cands["last"].tolist(), n=3, cutoff=0.85)
        hits = cands[cands["last"].isin(close)]
        rule = "last_fuzzy"
    if hits.empty:
        return None, "unmatched", 0
    F = norm(s.firstname)
    narrowers = [
        ("district", lambda h: h["district_code"].astype("Int64").astype(str) == s.district
         if s.district.isdigit() else None),
        ("first", lambda h: ((h["first"] == F) | (h["nick"] == F)
                             | ((h["first"].str[:3] == F[:3]) & (len(F) >= 3))) if F else None),
        ("party", lambda h: h["party_letter"] == s.party if s.party in ("D", "R") else None),
    ]
    for label, cond in narrowers:
        if len(hits) == 1:
            break
        keep = cond(hits)
        if keep is not None and keep.any():
            hits = hits[keep.values]
            rule += "+" + label
    if len(hits) > 1 and hits["icpsr"].nunique() == 1:  # same member listed twice (e.g. party switch)
        hits = hits.iloc[:1]
    if len(hits) == 1:
        return int(hits["icpsr"].iloc[0]), rule, 1
    return None, "ambiguous", len(hits)


def main():
    t0 = time.time()
    sp = gst_speakers()
    vv = voteview_members()
    print(f"{len(sp):,} GST speaker rows; {len(vv):,} Voteview member rows "
          f"({time.time() - t0:.0f}s)", flush=True)

    cells = {k: g for k, g in vv.groupby(["congress", "chamber_code", "state_abbrev"])}
    out = np.empty(len(sp), dtype=object)
    rules, ncand = [], []
    for i, s in enumerate(sp.itertuples(index=False)):
        cands = cells.get((s.congress, s.chamber, s.state))
        if cands is None or s.chamber not in ("H", "S"):
            out[i], r, n = None, "no_cell", 0
        else:
            out[i], r, n = match_one(s, cands)
        rules.append(r)
        ncand.append(n)
    sp["icpsr"] = pd.array([x if x is not None else pd.NA for x in out], dtype="Int64")
    sp["rule"], sp["n_candidates"] = rules, ncand

    # Person level: one Voteview member per GST person id.
    mode = sp.dropna(subset=["icpsr"]).groupby("person")["icpsr"].agg(lambda s: s.mode().iloc[0])
    sp["person_icpsr"] = sp["person"].map(mode).astype("Int64")
    sp["person_conflict"] = sp["icpsr"].notna() & sp["person_icpsr"].notna() & (
        sp["icpsr"] != sp["person_icpsr"])
    in_cell = vv.set_index(["congress", "chamber_code", "icpsr"]).index
    fill = sp["icpsr"].isna() & sp["person_icpsr"].notna() & pd.Series(
        [(cg, ch, ic) in in_cell if pd.notna(ic) else False
         for cg, ch, ic in zip(sp["congress"], sp["chamber"], sp["person_icpsr"])], index=sp.index)
    sp.loc[fill, "icpsr"] = sp.loc[fill, "person_icpsr"]
    sp.loc[fill, "rule"] = "person_propagated"
    dup = sp.dropna(subset=["icpsr"]).duplicated(["edition", "congress", "chamber", "icpsr"], keep=False)
    sp["duplicate_icpsr"] = dup.reindex(sp.index, fill_value=False)

    vv_key = vv.drop_duplicates(["congress", "chamber_code", "icpsr"], keep="last")[
        ["congress", "chamber_code", "icpsr", "bioguide_id", "bioname", "party_letter",
         "district_code", "nominate_dim1"]].astype({"icpsr": "Int64"})
    sp = sp.merge(vv_key, how="left", left_on=["congress", "chamber", "icpsr"],
                  right_on=["congress", "chamber_code", "icpsr"]).drop(columns="chamber_code")

    c.CROSSWALK.mkdir(parents=True, exist_ok=True)
    sp.to_csv(c.CROSSWALK / "gst_to_voteview.csv.gz", index=False)

    # ---- checks (House) ----
    house = sp[sp["chamber"] == "H"]
    main = house[house.apply(lambda r: c.MAIN_EDITION.get(r.congress) == r.edition, axis=1)]
    voting = main[main["nonvoting"] == "voting"]
    m = voting["icpsr"].notna()
    both_party = m & voting["party"].isin(["D", "R"]) & voting["party_letter"].isin(["D", "R"])
    both_dist = m & voting["district"].str.isdigit()
    per_congress = voting.assign(matched=m).groupby("congress").apply(lambda g: pd.Series({
        "speakers": len(g),
        "matched_share": g["matched"].mean(),
        "speech_share": g.loc[g["matched"], "n_speeches"].sum() / g["n_speeches"].sum(),
    }))
    # editions overlap (97th-111th): the same person-congress should get the same member
    ov = house.assign(ic=house["icpsr"].astype(float)).pivot_table(
        index=["congress", "person"], columns="edition", values="ic", aggfunc="first").dropna()
    # govinfo (bioguide in the speeches): House speakers per congress vs our matched members
    gov = []
    for cong in range(103, 115):
        g = pd.read_parquet(c.APP1_DATA / "speeches" / "govinfo" / f"{cong}.parquet",
                            columns=["chamber", "person_id"])
        gb = set(g.loc[(g["chamber"] == "H") & g["person_id"].notna(), "person_id"].str[1:])
        ours = set(main.loc[(main["congress"] == cong) & main["bioguide_id"].notna(), "bioguide_id"])
        gov.append({"congress": cong, "govinfo_speakers": len(gb), "gst_matched": len(ours),
                    "jaccard": len(gb & ours) / max(1, len(gb | ours)),
                    "govinfo_found_in_gst": len(gb & ours) / max(1, len(gb))})
    summary = {
        "rows": len(sp),
        "rules_all": sp["rule"].value_counts().to_dict(),
        "house_main_voting": {
            "speakers": int(len(voting)),
            "matched_share": float(m.mean()),
            "speech_share": float(voting.loc[m, "n_speeches"].sum() / voting["n_speeches"].sum()),
            "rules": voting["rule"].value_counts().to_dict(),
            "party_agreement": float((voting.loc[both_party, "party"]
                                      == voting.loc[both_party, "party_letter"]).mean()),
            "district_agreement": float((voting.loc[both_dist, "district"]
                                         == voting.loc[both_dist, "district_code"].astype("Int64")
                                         .astype(str)).mean()),
            "person_conflicts": int(voting["person_conflict"].sum()),
            "duplicate_icpsr_rows": int(voting["duplicate_icpsr"].sum()),
            "worst_congresses": per_congress.sort_values("matched_share").head(8)
            .round(4).reset_index().to_dict("records"),
        },
        "edition_overlap_97_111": {"person_congresses": int(len(ov)),
                                   "same_member": float((ov["bound"] == ov["daily"]).mean())},
        "govinfo_check_103_114": gov,
        "seconds": time.time() - t0,
    }
    (c.CROSSWALK / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
    per_congress.round(4).to_csv(c.CROSSWALK / "house_match_by_congress.csv")
    print(json.dumps(summary, indent=1, default=str))


if __name__ == "__main__":
    main()
