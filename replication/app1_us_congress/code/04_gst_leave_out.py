"""Step 04: Gentzkow-Shapiro-Taddy (2019) leave-out estimator of average partisanship.

Sample: the GST speeches (common.gst_sample) of the main build (common.MAIN_SOURCES). For each
congress t, with c_i the term counts of speaker-session i (all of the speaker's speeches in t,
House and Senate pooled) and m_i = sum_j c_ij:

    pi_t = 1/2 * mean_{i in R_t} q_i . rho_{-i}  +  1/2 * mean_{i in D_t} q_i . (1 - rho_{-i})

where q_i = c_i / m_i, rho_{-ij} = q^R_{-ij} / (q^R_{-ij} + q^D_{-ij}) and q^P_{-ij} pools the
counts of the other speakers of party P (GST eq. 8 and footnote 14). Terms spoken by a single
speaker in the congress are excluded (GST footnote 14). Only the nonzero entries of q_i enter,
so the computation is sparse.

"random" series (GST Figure 2 notes): each person gets a random party, Republican with
probability equal to the average Republican share of speakers in the congresses in which the
person is active. Pointwise 90% intervals by subsampling (GST Figure 2 notes): 100 subsamples,
each with one tenth of the persons drawn without replacement; with pi^k the estimate on
subsample k, tau_k its number of speakers, tau the full number, Q^k = sqrt(tau_k)(pi^k - mean_l
pi^l), the interval is (pi - Q^(90)/sqrt(tau), pi - Q^(11)/sqrt(tau)) with Q^(b) the b-th
order statistic of the 100 Q^k.

Outputs results/gst/<vocabulary>/: leave_out.csv (main build, both chambers and by chamber),
leave_out_overlap.csv (the same congresses from two sources: bound vs daily edition, 97th-111th, as
GST's alternative build; daily edition vs govinfo, 104th-114th, for the 2017 splice) and leave_out.png.

Usage: python 04_gst_leave_out.py [--vocab inputs/vocab_us.txt] [--subsamples 100] [--seed 42]
"""
import argparse
import json

import numpy as np
import pandas as pd
import scipy.sparse as sp

import common

CONGRESS_START_YEAR = {c: 1789 + 2 * (c - 1) for c in range(43, 120)}


def load_speaker_sessions(source: str, congress: int, terms):
    """Speaker-session x term counts of the GST sample for one (source, congress), with metadata."""
    counts = sp.load_npz(terms / f"{common.stem(source, congress)}.npz")
    meta = pd.read_csv(terms / source / f"{congress:03d}_meta.csv.gz",
                       dtype={"speaker_session": str, "person_id": str})
    keep = common.gst_sample(meta).to_numpy()
    counts, meta = counts[keep], meta[keep].reset_index(drop=True)
    ss, row = np.unique(meta["speaker_session"].to_numpy(), return_inverse=True)
    agg = sp.csr_matrix((np.ones(len(row)), (row, np.arange(len(row)))), shape=(len(ss), len(row)))
    C = (agg @ counts).tocsr()
    info = meta.drop_duplicates("speaker_session").set_index("speaker_session").loc[ss].reset_index()
    return C, info


def leave_out(C: sp.csr_matrix, rep: np.ndarray) -> float:
    """GST leave-out partisanship for one congress (C: speakers x terms, rep: boolean)."""
    C = C.tocsc()
    multi = np.diff(C.indptr) >= 2  # terms spoken by at least two speakers
    C = C[:, multi].tocsr()
    m = np.asarray(C.sum(axis=1)).ravel()
    keep = m > 0
    C, m, rep = C[keep], m[keep], rep[keep]
    if rep.sum() < 2 or (~rep).sum() < 2:
        return np.nan
    S_R = np.asarray(C[rep].sum(axis=0)).ravel()
    S_D = np.asarray(C[~rep].sum(axis=0)).ravel()
    M_R, M_D = S_R.sum(), S_D.sum()
    C = C.tocoo()
    i, j, c = C.row, C.col, C.data.astype(float)
    r_i = rep[i]
    # Leave speaker i out of its own party's pooled frequencies.
    qR = np.where(r_i, (S_R[j] - c) / (M_R - m[i]), S_R[j] / M_R)
    qD = np.where(r_i, S_D[j] / M_D, (S_D[j] - c) / (M_D - m[i]))
    den = qR + qD
    rho = np.divide(qR, den, out=np.full_like(den, 0.5), where=den > 0)
    own = np.where(r_i, rho, 1.0 - rho)  # posterior on the speaker's true party
    per_speaker = np.bincount(i, weights=c / m[i] * own, minlength=len(m))
    return 0.5 * per_speaker[rep].mean() + 0.5 * per_speaker[~rep].mean()


def series(panels: dict, label_col: str, persons_subset=None) -> pd.Series:
    out = {}
    for t, (C, info) in panels.items():
        mask = np.ones(len(info), dtype=bool) if persons_subset is None else info["person_id"].isin(persons_subset).to_numpy()
        out[t] = leave_out(C[mask], info.loc[mask, label_col].eq("R").to_numpy()) if mask.sum() > 3 else np.nan
    return pd.Series(out)


def random_party(panels: dict, rng) -> None:
    """GST placebo: a person's party is R with prob. = mean R share of speakers in the person's congresses."""
    share = {t: info["party"].eq("R").mean() for t, (_, info) in panels.items()}
    rows = pd.concat([info[["person_id"]].assign(share=share[t]) for t, (_, info) in panels.items()])
    p = rows.groupby("person_id")["share"].mean()
    draw = pd.Series(np.where(rng.random(len(p)) < p.to_numpy(), "R", "D"), index=p.index)
    for _, (_, info) in panels.items():
        info["party_random"] = info["person_id"].map(draw).to_numpy()


def subsample_ci(panels: dict, label_col: str, full: pd.Series, persons: np.ndarray, rng, K: int) -> pd.DataFrame:
    """GST pointwise 90% interval from K subsamples of one tenth of the persons."""
    tau = {t: len(info) for t, (_, info) in panels.items()}
    est, tau_k = [], []
    for _ in range(K):
        sub = rng.choice(persons, size=len(persons) // 10, replace=False)
        est.append(series(panels, label_col, set(sub)))
        tau_k.append({t: int(info["person_id"].isin(set(sub)).sum()) for t, (_, info) in panels.items()})
    est, tau_k = pd.DataFrame(est), pd.DataFrame(tau_k)
    Q = np.sqrt(tau_k) * (est - est.mean(axis=0))
    q90, q11 = Q.apply(lambda s: np.sort(s.dropna().to_numpy())[89] if s.notna().sum() >= 90 else np.nan), \
        Q.apply(lambda s: np.sort(s.dropna().to_numpy())[10] if s.notna().sum() >= 90 else np.nan)
    root_tau = np.sqrt(pd.Series(tau))
    return pd.DataFrame({"lo": full - q90 / root_tau, "hi": full - q11 / root_tau})


def estimate(tasks: list, terms, chamber=None, K: int = 100, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    panels = {}
    for source, t in tasks:
        C, info = load_speaker_sessions(source, t, terms)
        if chamber is not None:
            mask = info["chamber"].eq(chamber).to_numpy()
            C, info = C[mask], info[mask].reset_index(drop=True)
        panels[t] = (C, info)
    random_party(panels, rng)
    persons = np.unique(np.concatenate([info["person_id"].to_numpy() for _, info in panels.values()]))
    real = series(panels, "party")
    rand = series(panels, "party_random")
    ci_real = subsample_ci(panels, "party", real, persons, rng, K)
    ci_rand = subsample_ci(panels, "party_random", rand, persons, rng, K)
    return pd.DataFrame({
        "congress": real.index, "year": [CONGRESS_START_YEAR[t] for t in real.index],
        "pi_real": real.values, "lo_real": ci_real["lo"].values, "hi_real": ci_real["hi"].values,
        "pi_random": rand.values, "lo_random": ci_rand["lo"].values, "hi_random": ci_rand["hi"].values,
        "speakers": [len(panels[t][1]) for t in real.index],
        "republicans": [int(panels[t][1]["party"].eq("R").sum()) for t in real.index],
    })


def plot(df: pd.DataFrame, path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.fill_between(df["year"], df["lo_real"], df["hi_real"], color="#9e2a5e", alpha=0.35, lw=0)
    ax.plot(df["year"], df["pi_real"], color="#9e2a5e", label="real")
    ax.fill_between(df["year"], df["lo_random"], df["hi_random"], color="grey", alpha=0.35, lw=0)
    ax.plot(df["year"], df["pi_random"], color="black", ls="--", label="random")
    ax.set_ylabel("Average partisanship")
    ax.legend(frameon=False, loc="upper center", ncol=2)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    fig.savefig(path.with_suffix(".pdf"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vocab", default=str(common.VOCAB_PATH))
    ap.add_argument("--subsamples", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    terms = common.terms_dir(a.vocab)
    out = common.RESULTS / "gst" / terms.name
    out.mkdir(parents=True, exist_ok=True)
    frames = []
    for chamber in (None, "H", "S"):
        df = estimate(common.MAIN_SOURCES, terms, chamber, a.subsamples, a.seed)
        frames.append(df.assign(chamber=chamber or "both"))
        print(chamber or "both", "done", flush=True)
    res = pd.concat(frames, ignore_index=True)
    res.to_csv(out / "leave_out.csv", index=False)
    plot(res[res["chamber"] == "both"], out / "leave_out.png")
    # The same congresses from two sources.
    overlap = pd.concat([
        estimate([(s, c) for c in cs], terms, None, a.subsamples, a.seed).assign(source=s, comparison=name)
        for name, s, cs in [("bound_vs_daily", "bound", range(97, 112)), ("bound_vs_daily", "daily", range(97, 112)),
                            ("daily_vs_govinfo", "daily", range(104, 115)), ("daily_vs_govinfo", "govinfo", range(104, 115))]
    ], ignore_index=True)
    overlap.to_csv(out / "leave_out_overlap.csv", index=False)
    print(overlap.pivot_table(index=["comparison", "congress"], columns="source", values="pi_real").round(4).to_string())
    with open(out / "leave_out_config.json", "w") as f:
        json.dump(vars(a), f)


if __name__ == "__main__":
    main()
