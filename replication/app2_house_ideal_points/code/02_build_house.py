"""Step 02: House legislator-congress units with their votes and speech term counts.

    python 02_build_house.py

Units: one per (congress, bioguide) for every House member in Voteview, 43rd-118th Congress
(1873-2024). Legislators are identified by their bioguide id (Voteview gives a member a new icpsr
when they switch parties; 34 House members have two). Nothing is dropped: members without votes
or speeches are kept with zero counts; estimation steps choose the sample.

Votes (HSall_votes.csv): cast codes 1-3 = yea (+1), 4-6 = nay (-1), anything else (present, not
voting, not a member) = not observed (0). A roll call is (congress, rollnumber). A member whose
two icpsr records fall in the same congress has both merged into one unit.

Speeches: the main build of Application 1 (Hein bound 43rd-111th, daily 112th-114th, govinfo
115th-118th; term counts over its 32,668-term vocabulary), House floor speeches only (no
Extensions of Remarks) by identified speakers; Hein speakers are mapped to bioguide through the
step-01 crosswalk, govinfo speakers carry their bioguide. Counts are summed per unit.

Outputs (data/house/): units.csv (attributes, DW-NOMINATE and Nokken-Poole scores, career,
counts), votes.npz (units x roll calls, int8 in {-1, 0, +1}), rollcalls.csv, speech_counts.npz
(units x vocabulary, int32), vocab.txt, build.json (summary).
"""

import json
import shutil
import time

import numpy as np
import pandas as pd
import scipy.sparse as sp

import common as c

TERMS = c.APP1_DATA / "terms" / "vocab_us"
VOCAB = c.APP1_DATA.parent / "inputs" / "vocab_us.txt"
CONGRESSES = range(43, 119)
SPEECH_EDITION = {**c.MAIN_EDITION, **{k: "govinfo" for k in range(115, 119)}}
YEA, NAY = {1, 2, 3}, {4, 5, 6}


def as_bool(s):
    return s.astype(str).str.lower().isin(["true", "1"])


def members():
    vv = pd.read_csv(c.RAW / "HSall_members.csv", dtype={"bioguide_id": str})
    house = vv[(vv["chamber"] == "House") & vv["bioguide_id"].notna()]
    career = house.groupby("bioguide_id")["congress"].agg(
        house_first_congress="min", house_last_congress="max", house_n_congresses="nunique")
    m = house[house["congress"].isin(CONGRESSES)].sort_values(["congress", "bioguide_id"])
    units = m.groupby(["congress", "bioguide_id"], as_index=False).agg(
        icpsrs=("icpsr", lambda s: "|".join(map(str, sorted(set(s))))),
        bioname=("bioname", "last"), party_code=("party_code", "last"),
        state_abbrev=("state_abbrev", "last"), district_code=("district_code", "last"),
        nominate_dim1=("nominate_dim1", "last"), nominate_dim2=("nominate_dim2", "last"),
        nokken_poole_dim1=("nokken_poole_dim1", "last"),
    ).rename(columns={"bioguide_id": "bioguide"})
    units = units.merge(career, left_on="bioguide", right_index=True)
    units.insert(0, "unit", np.arange(len(units)))
    icpsr_to_unit = m.merge(units[["unit", "congress", "bioguide"]],
                            left_on=["congress", "bioguide_id"], right_on=["congress", "bioguide"])
    icpsr_to_unit = icpsr_to_unit.drop_duplicates(["congress", "icpsr"])
    return units, icpsr_to_unit.set_index(["congress", "icpsr"])["unit"]


def votes(units, icpsr_to_unit):
    v = pd.read_csv(c.RAW / "HSall_votes.csv", usecols=["congress", "chamber", "rollnumber",
                                                         "icpsr", "cast_code"])
    v = v[(v["chamber"] == "House") & v["congress"].isin(CONGRESSES)]
    rc = v[["congress", "rollnumber"]].drop_duplicates().sort_values(["congress", "rollnumber"])
    rc.insert(0, "rollcall", np.arange(len(rc)))
    v = v[v["cast_code"].isin(YEA | NAY)]
    v = v.assign(value=np.where(v["cast_code"].isin(YEA), 1, -1).astype(np.int8))
    v = v.merge(rc, on=["congress", "rollnumber"])
    key = pd.MultiIndex.from_frame(v[["congress", "icpsr"]])
    v["unit"] = icpsr_to_unit.reindex(key).to_numpy()
    no_unit = int(v["unit"].isna().sum())  # votes by icpsrs without a bioguide (none expected)
    v = v.dropna(subset=["unit"]).astype({"unit": int})
    dup = int(v.duplicated(["unit", "rollcall"]).sum())  # two records of one member, one roll call
    v = v.drop_duplicates(["unit", "rollcall"], keep="last")
    M = sp.csr_matrix((v["value"].to_numpy(), (v["unit"].to_numpy(), v["rollcall"].to_numpy())),
                      shape=(len(units), len(rc)), dtype=np.int8)
    rc["n_yea"] = np.asarray((M == 1).sum(0)).ravel()
    rc["n_nay"] = np.asarray((M == -1).sum(0)).ravel()
    return M, rc, {"votes_without_unit": no_unit, "duplicate_unit_rollcall": dup}


def speeches(units, n_vocab):
    xw = pd.read_csv(c.CROSSWALK / "gst_to_voteview.csv.gz", dtype={"speakerid": str})
    xw = xw[xw["chamber"] == "H"].dropna(subset=["bioguide_id"])
    xw_key = xw.set_index(["edition", "congress", "speakerid"])["bioguide_id"]
    unit_of = units.set_index(["congress", "bioguide"])["unit"]
    blocks, stats = [], []
    for cong in CONGRESSES:
        ed = SPEECH_EDITION[cong]
        X = sp.load_npz(TERMS / ed / f"{cong:03d}.npz").tocsr()
        meta = pd.read_csv(TERMS / ed / f"{cong:03d}_meta.csv.gz",
                           dtype={"speaker_session": str, "person_id": str})
        assert len(meta) == X.shape[0] and X.shape[1] == n_vocab
        floor = (meta["chamber"] == "H") & ~as_bool(meta["extension"]) & as_bool(meta["identified"])
        if ed == "govinfo":
            bio = meta["person_id"].str[1:]
        else:
            k = pd.MultiIndex.from_arrays([[ed] * len(meta), [cong] * len(meta),
                                           meta["speaker_session"]])
            bio = pd.Series(xw_key.reindex(k).to_numpy(), index=meta.index)
        u = unit_of.reindex(pd.MultiIndex.from_arrays([[cong] * len(meta), bio])).to_numpy()
        keep = floor.to_numpy() & ~pd.isna(u)
        rows = u[keep].astype(int)
        A = sp.csr_matrix((np.ones(keep.sum()), (rows, np.arange(keep.sum()))),
                          shape=(len(units), keep.sum()))
        blocks.append(A @ X[keep])
        stats.append({"congress": cong, "edition": ed, "floor_speeches": int(floor.sum()),
                      "matched": int(keep.sum()),
                      "matched_terms_share": float(X[keep].sum() / max(1, X[floor.to_numpy()].sum())),
                      "speeches_per_unit": np.bincount(rows, minlength=len(units))})
    S = sum(blocks).tocsr().astype(np.int32)
    n_speeches = sum(s.pop("speeches_per_unit") for s in stats)
    return S, n_speeches, stats


def main():
    t0 = time.time()
    out = c.DATA / "house"
    out.mkdir(parents=True, exist_ok=True)
    vocab = VOCAB.read_text().split("\n")
    vocab = [w for w in vocab if w]
    units, icpsr_to_unit = members()
    print(f"{len(units):,} units ({time.time() - t0:.0f}s)", flush=True)
    V, rc, vstats = votes(units, icpsr_to_unit)
    print(f"votes: {V.nnz:,} cast on {len(rc):,} roll calls ({time.time() - t0:.0f}s)", flush=True)
    S, n_sp, sstats = speeches(units, len(vocab))
    print(f"speeches: {int(n_sp.sum()):,} matched ({time.time() - t0:.0f}s)", flush=True)
    units["n_votes"] = np.diff(V.indptr)
    units["n_speeches"] = n_sp
    units["n_terms"] = np.asarray(S.sum(1)).ravel()

    units.to_csv(out / "units.csv", index=False)
    rc.to_csv(out / "rollcalls.csv", index=False)
    sp.save_npz(out / "votes.npz", V)
    sp.save_npz(out / "speech_counts.npz", S)
    shutil.copy(VOCAB, out / "vocab.txt")
    by_cong = units.groupby("congress").agg(
        units=("unit", "size"), with_votes=("n_votes", lambda s: int((s > 0).sum())),
        with_speech=("n_speeches", lambda s: int((s > 0).sum())),
        median_votes=("n_votes", "median"), median_terms=("n_terms", "median"))
    summary = {
        "units": len(units), "legislators": int(units["bioguide"].nunique()),
        "rollcalls": len(rc), "votes_cast": int(V.nnz), "vote_density": V.nnz / np.prod(V.shape),
        **vstats,
        "units_with_votes": int((units["n_votes"] > 0).sum()),
        "units_with_speeches": int((units["n_speeches"] > 0).sum()),
        "terms_total": int(units["n_terms"].sum()),
        "speech_matching": sstats,
        "by_congress": by_cong.reset_index().to_dict("records"),
        "seconds": time.time() - t0,
    }
    (out / "build.json").write_text(json.dumps(summary, indent=1, default=float))
    print(json.dumps({k: v for k, v in summary.items()
                      if k not in ("speech_matching", "by_congress")}, indent=1, default=float))
    print("lowest matched term shares:", sorted(
        (round(s["matched_terms_share"], 4), s["congress"]) for s in sstats)[:6])


if __name__ == "__main__":
    main()
