"""Step 10: the House roles of each legislator-congress of the 103rd-118th Congresses: committee
chairs and ranking members, party leaders, and floor-manager phrases in their floor speeches.

    python 10_house_roles.py     (needs pandas, pyarrow, pyyaml: Application 1's environment)

Sources (data/raw/roles/):
  house_assignments_103-115-3.csv  Stewart & Woon, House committee assignments, 103rd-115th (the
      MIT .xls saved as csv, unchanged). Senior party member codes 11-16 = chair (only, 1st, 2nd,
      3rd, acting), 21-24 = ranking minority member, on the committees CRS R47284 covers:
      standing committees (codes 102-196), Homeland Security (251, select in the 107th-108th)
      and the Permanent Select Committee on Intelligence (242); committee 661 = House
      leadership: 31-33 Speaker, 41-44 majority leader, 51-53 majority whip, 61-63 minority
      leader, 64-66 minority whip.
  crs_R47284_2026-09-03.html  CRS report R47284, chairs and ranking members of the House standing
      committees, 104th-119th: the holder at the end of each Congress (mid-Congress changes are
      footnotes, not used).
  legislators-{current,historical}.yaml  unitedstates/congress-legislators, leadership_roles:
      Speaker, majority and minority leaders and whips (dated; 108th-118th).
A unit holds a role if any source records it (the sources differ on mid-Congress changes); the
CRS-Stewart agreement on the 104th-115th is reported.
Floor managers: in each unit's House floor speeches (Application 1's speech files, by identified
speakers, mapped to units as in step 02), the counts of "yield myself" (the member controlling
debate time on a measure yields time to themself) and "reserve the balance of my time" (only a
member controlling time reserves it), with the unit's floor speeches and words.

Writes data/house/roles.csv (one row per unit of the 103rd-118th) and data/house/roles.json (match
counts and the CRS-Stewart agreement).
"""

import html
import json
import re
import unicodedata

import numpy as np
import pandas as pd
import yaml

import common as c

CONGRESSES = range(103, 119)
EDITION = {**c.MAIN_EDITION, **{k: "govinfo" for k in range(115, 119)}}
ROLES = c.RAW / "roles"
LEADERS = {31: "speaker", 32: "speaker", 33: "speaker",
           41: "majority_leader", 42: "majority_leader", 43: "majority_leader",
           44: "majority_leader", 51: "majority_whip", 52: "majority_whip", 53: "majority_whip",
           61: "minority_leader", 62: "minority_leader", 63: "minority_leader",
           64: "minority_whip", 65: "minority_whip", 66: "minority_whip"}
YAML_LEADERS = {"Speaker of the House": "speaker", "House Majority Leader": "majority_leader",
                "House Minority Leader": "minority_leader", "House Majority Whip": "majority_whip",
                "House Minority Whip": "minority_whip"}
PHRASES = {"yield_myself": r"\byield myself\b",
           "reserve_balance": r"\breserve the balance of my time\b"}


def as_bool(s):
    return s.astype(str).str.lower().isin(["true", "1"])


def congress_of(date):
    """Congress in session on a date (the t-th starts on January 3 of 1787 + 2t)."""
    d = pd.Timestamp(date)
    return (d.year - ((d.month, d.day) < (1, 3)) - 1787) // 2


def stewart(units):
    s = pd.read_csv(ROLES / "house_assignments_103-115-3.csv")
    s = s[s["Congress"].between(CONGRESSES.start, 115) & s["ID #"].notna()].copy()
    s["congress"] = s["Congress"].astype(int)
    s["icpsr"] = s["ID #"].astype(int)
    ic = units.assign(icpsr=units["icpsrs"].astype(str).str.split("|")).explode("icpsr")
    ic["icpsr"] = ic["icpsr"].astype(int)
    s = s.merge(ic[["congress", "icpsr", "bioguide"]], on=["congress", "icpsr"], how="left")
    code, spm = s["Committee code"], s["Senior Party Member"]
    standing = code.between(102, 196) | code.isin([242, 251])
    rows = []
    for role, mask in (("chair", standing & spm.between(11, 16)),
                       ("ranking", standing & spm.between(21, 24))):
        r = s[mask]
        rows.append(pd.DataFrame({"congress": r["congress"], "bioguide": r["bioguide"],
                                  "role": role, "committee": r["Committee Name"].str.strip(),
                                  "name": r["Name"], "source": "stewart"}))
    lead = s[(code == 661) & spm.isin(LEADERS)]
    rows.append(pd.DataFrame({"congress": lead["congress"], "bioguide": lead["bioguide"],
                              "role": lead["Senior Party Member"].map(LEADERS),
                              "committee": "", "name": lead["Name"], "source": "stewart"}))
    return pd.concat(rows, ignore_index=True)


def crs(units):
    page = (ROLES / "crs_R47284_2026-09-03.html").read_text(encoding="utf-8")
    tables = re.findall(r"Table (\d+)\. ([^<]+)</p>\s*</div>\s*<table>(.*?)</table>", page, re.S)
    rows = []
    for _, committee, body in tables:
        for tr in re.findall(r"<tr>(.*?)</tr>", body, re.S)[1:]:
            cells = [html.unescape(re.sub(r"<[^>]+>", "", td)).strip()
                     for td in re.findall(r"<td>(.*?)</td>", tr, re.S)]
            congress = int(re.match(r"\d+", cells[0]).group())
            for role, cell in zip(("chair", "ranking"), cells[1:3]):
                m = re.match(r"(.*?)\s*\(([DRI])-([A-Z]{2})\)", cell)
                if m is None:  # no committee yet ("—")
                    continue
                rows.append({"congress": congress, "role": role, "committee": committee.strip(),
                             "name": m.group(1), "party": m.group(2), "state": m.group(3)})
    d = pd.DataFrame(rows)
    d = d[d["congress"].isin(CONGRESSES)]
    def norm(x):
        x = unicodedata.normalize("NFKD", x).encode("ascii", "ignore").decode().upper()
        return re.sub(r"[^A-Z]", "", x)

    key = units.assign(
        surname=units["bioname"].str.split(",").str[0].map(norm),
        given=units["bioname"].str.split(",").str[1].fillna("").map(lambda g: g.upper().split()),
        party=units["party_code"].map({100: "D", 200: "R"}))

    def match(r):
        toks = [t for t in re.sub(r"[.,()]", " ", r["name"]).split()
                if t.upper() not in {"JR", "SR", "II", "III", "IV"}]
        here = key[(key["congress"] == r["congress"]) & (key["state_abbrev"] == r["state"])]
        for cands in (here[here["party"] == r["party"]], here):  # party switchers: any party
            for k in (1, 2, 3):
                hit = cands[cands["surname"] == norm("".join(toks[-k:]))]
                if len(hit) > 1:  # same surname, state and party: the given names decide
                    hit = hit[hit["given"].map(lambda g: any(norm(t) in map(norm, g)
                                                             for t in toks[:-k] if len(t) > 2))]
                if len(hit) == 1:
                    return hit["bioguide"].iloc[0]
        return np.nan

    d["bioguide"] = d.apply(match, axis=1)
    d["source"] = "crs"
    return d


def yaml_leaders():
    rows = []
    loader = getattr(yaml, "CSafeLoader", yaml.SafeLoader)
    for f in ("legislators-historical.yaml", "legislators-current.yaml"):
        for person in yaml.load((ROLES / f).read_text(), Loader=loader):
            for r in person.get("leadership_roles", []):
                if r.get("chamber") != "house" or r["title"] not in YAML_LEADERS:
                    continue
                end = r.get("end", "2027-01-03")
                for t in range(congress_of(r["start"]), congress_of(pd.Timestamp(end)
                                                                    - pd.Timedelta(days=1)) + 1):
                    name = person["name"].get("official_full", person["name"]["last"])
                    rows.append({"congress": t, "bioguide": person["id"]["bioguide"],
                                 "role": YAML_LEADERS[r["title"]], "name": name,
                                 "committee": "", "source": "congress-legislators"})
    d = pd.DataFrame(rows)
    return d[d["congress"].isin(CONGRESSES)]


def floor_phrases(units):
    xw = pd.read_csv(c.CROSSWALK / "gst_to_voteview.csv.gz", dtype={"speakerid": str})
    xw = xw[xw["chamber"] == "H"].dropna(subset=["bioguide_id"])
    xw_key = xw.set_index(["edition", "congress", "speakerid"])["bioguide_id"]
    unit_of = units.set_index(["congress", "bioguide"])["unit"]
    out = []
    for t in CONGRESSES:
        ed = EDITION[t]
        d = pd.read_parquet(c.APP1_DATA / "speeches" / ed / f"{t:03d}.parquet",
                            columns=["chamber", "extension", "identified", "speaker_session",
                                     "person_id", "text"])
        d = d[(d["chamber"] == "H") & ~as_bool(d["extension"]) & as_bool(d["identified"])]
        if ed == "govinfo":
            bio = d["person_id"].str[1:]
        else:
            k = pd.MultiIndex.from_arrays([[ed] * len(d), [t] * len(d), d["speaker_session"]])
            bio = pd.Series(xw_key.reindex(k).to_numpy(), index=d.index)
        d["unit"] = unit_of.reindex(pd.MultiIndex.from_arrays([[t] * len(d), bio])).to_numpy()
        d = d.dropna(subset=["unit"])
        text = d["text"].fillna("").str.lower()
        agg = pd.DataFrame({"unit": d["unit"].astype(int), "floor_speeches": 1,
                            "floor_words": text.str.split().str.len(),
                            **{k: text.str.count(p) for k, p in PHRASES.items()}})
        out.append(agg.groupby("unit").sum())
        print(f"{t}: {len(d):,} floor speeches matched to units", flush=True)
    return pd.concat(out)


def main():
    units = pd.read_csv(c.DATA / "house" / "units.csv", dtype={"bioguide": str})
    units = units[units["congress"].isin(CONGRESSES)]
    st, cr, yl = stewart(units), crs(units), yaml_leaders()
    stats = {"stewart_unmatched": int(st["bioguide"].isna().sum()),
             "crs_unmatched": cr[cr["bioguide"].isna()][["congress", "role", "name"]]
             .to_dict("records")}
    both = cr[cr["congress"] <= 115].merge(st[["congress", "bioguide", "role"]].drop_duplicates(),
                                           on=["congress", "bioguide", "role"], how="left",
                                           indicator=True)
    stats["crs_104_115_found_in_stewart"] = float((both["_merge"] == "both").mean())
    yl_check = yl[yl["congress"] <= 115].merge(
        st[["congress", "bioguide", "role"]], how="left", indicator=True)
    stats["yaml_leaders_103_115_found_in_stewart"] = float((yl_check["_merge"] == "both").mean())
    # a role any source records (the sources differ on mid-Congress changes and the CRS lists
    # the end-of-Congress holder): Stewart 103rd-115th, CRS 104th-118th, leaders 103rd-118th
    roles = pd.concat([st, cr, yl], ignore_index=True).dropna(subset=["bioguide"])
    stats["roles_by_source"] = roles.groupby(["source", "role"]).size().astype(int).to_dict()
    stats["roles_by_source"] = {f"{k[0]}:{k[1]}": v for k, v in stats["roles_by_source"].items()}

    r = units[["unit", "congress", "bioguide", "bioname", "party_code"]].copy()
    for role in ("chair", "ranking", *sorted(set(LEADERS.values()))):
        hit = roles[roles["role"] == role].groupby(["congress", "bioguide"])["committee"].agg(
            lambda s: "; ".join(sorted(set(x for x in s if x))))
        idx = pd.MultiIndex.from_frame(r[["congress", "bioguide"]])
        r[role] = idx.isin(hit.index)
        if role in ("chair", "ranking"):
            r[f"{role}_of"] = hit.reindex(idx).fillna("").to_numpy()
    r["leader"] = r[sorted(set(LEADERS.values()))].any(axis=1)
    fp = floor_phrases(units)
    r = r.merge(fp, left_on="unit", right_index=True, how="left")
    r[["floor_speeches", "floor_words", *PHRASES]] = r[["floor_speeches", "floor_words",
                                                        *PHRASES]].fillna(0).astype(int)
    out = c.DATA / "house"
    r.to_csv(out / "roles.csv", index=False)
    stats["units"] = len(r)
    stats["per_congress"] = r.groupby("congress")[["chair", "ranking", "leader"]].sum() \
        .astype(int).to_dict("index")
    (out / "roles.json").write_text(json.dumps(stats, indent=1, default=str))
    print(json.dumps({k: v for k, v in stats.items() if k != "per_congress"}, indent=1,
                     default=str))
    print(r.groupby("congress")[["chair", "ranking", "leader"]].sum().to_string())
    print(f"wrote {out / 'roles.csv'}")


if __name__ == "__main__":
    main()
