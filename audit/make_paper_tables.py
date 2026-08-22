"""Generate publication-ready LaTeX table fragments from the production MC reps.

Reads the per-rep JSONs (audit/hpc_results_prod/ by default), computes bias/SD/
RMSE and CI coverage with MC standard errors, and writes booktabs fragments into
tables/ (no table environment -- \\input the fragments and add captions in the
paper):

  tables/sim_point.tex      -- Panel A/B: bias, SD, RMSE of psi-hat by estimator x n
  tables/sim_coverage.tex   -- 95% CI coverage: two-step naive, post-fit naive,
                               post-fit delta-corrected
  tables/sim_vi_appendix.tex-- VI diagnostics: corr vs sqrt(rel) ceiling,
                               theta-interval coverage vs exact, oracle-PM benchmark

Usage: python audit/make_paper_tables.py [results_dir] [--out tables/]
"""
import argparse
import glob
import json
import os
import re
from collections import defaultdict

import numpy as np

PSI_TRUE = float(np.sqrt(1.25))
CHANNELS = {"ip": "Roll-call votes (2PL IRT)", "lg": "Linear--Gaussian"}


def load(results_dir):
    cells = defaultdict(list)
    for f in glob.glob(os.path.join(results_dir, "*_rep*.json")):
        m = re.match(r"((?:ip|lg)_.+)_rep\d+\.json", os.path.basename(f))
        if m:
            d = json.load(open(f))
            cells[m.group(1)].append(d)
    return cells


def stats_psi(vals, truth=PSI_TRUE):
    v = np.asarray(vals)
    r = len(v)
    return {"bias": v.mean() - truth, "bias_se": v.std(ddof=1) / np.sqrt(r),
            "sd": v.std(ddof=1), "rmse": float(np.sqrt(((v - truth) ** 2).mean())),
            "reps": r}


def cov_stat(ind):
    v = np.asarray(ind, dtype=float)
    p = v.mean()
    return p, np.sqrt(p * (1 - p) / len(v))


def core_cells(cells, chan):
    """Return {n: reps} for the channel's core (J=25) cells, sorted by n."""
    out = {}
    for cell, docs in cells.items():
        if not cell.startswith(chan + "_"):
            continue
        cfg = docs[0]["config"]
        if cfg["J"] != 25:
            continue
        out[cfg["N"]] = [d["rep"] for d in docs]
    return dict(sorted(out.items()))


def fmt(x, d=3):
    return f"{x:.{d}f}"


def write_point_table(cells, path):
    lines = [r"\begin{tabular}{lrrrrrrrrr}", r"\toprule",
             r" & \multicolumn{3}{c}{Two-step} & \multicolumn{3}{c}{Joint} & \multicolumn{3}{c}{Post-fit OLS} \\",
             r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}\cmidrule(lr){8-10}",
             r"$n$ & Bias & SD & RMSE & Bias & SD & RMSE & Bias & SD & RMSE \\"]
    for chan, label in CHANNELS.items():
        core = core_cells(cells, chan)
        if not core:
            continue
        reps_note = len(next(iter(core.values())))
        lines += [r"\midrule",
                  rf"\multicolumn{{10}}{{l}}{{\emph{{Panel: {label} ($J=25$, {reps_note} replications)}}}} \\"]
        for n, reps in core.items():
            row = [f"{n:,}"]
            for arm_get in (
                    lambda r: r["two_step"][-1]["psi"],
                    lambda r: r["joint"][-1]["psi"],
                    lambda r: r["joint"][-1]["pf_psi"]):
                s = stats_psi([arm_get(r) for r in reps])
                row += [fmt(s["bias"]), fmt(s["sd"]), fmt(s["rmse"])]
            lines.append(" & ".join(row) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    open(path, "w").write("\n".join(lines) + "\n")


def write_coverage_table(cells, path):
    lines = [r"\begin{tabular}{lccc}", r"\toprule",
             r"$n$ & Two-step (naive) & Post-fit (naive) & Post-fit (corrected) \\"]
    for chan, label in CHANNELS.items():
        core = core_cells(cells, chan)
        if not core:
            continue
        lines += [r"\midrule",
                  rf"\multicolumn{{4}}{{l}}{{\emph{{Panel: {label}}}}} \\"]
        for n, reps in core.items():
            row = [f"{n:,}"]
            for key, arm in (("psi_cover", "two_step"), ("pf_psi_cover", "joint"),
                             ("pf_psi_cover_corr", "joint")):
                vals = [r[arm][-1][key] for r in reps if key in r[arm][-1]]
                if vals:
                    p, se = cov_stat(vals)
                    row.append(f"{p:.3f} ({se:.3f})")
                else:
                    row.append("--")
            lines.append(" & ".join(row) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    open(path, "w").write("\n".join(lines) + "\n")


def write_vi_table(cells, path):
    lines = [r"\begin{tabular}{lccccc}", r"\toprule",
             r"$n$ & $\mathrm{corr}(\hat\theta,\theta)$ & $\sqrt{\mathrm{rel}}$ ceiling & "
             r"$\theta$-cov.\ (VI) & $\theta$-cov.\ (exact) & Oracle-PM $\hat c$ (per unit) \\"]
    for chan, label in CHANNELS.items():
        core = core_cells(cells, chan)
        if not core:
            continue
        lines += [r"\midrule",
                  rf"\multicolumn{{6}}{{l}}{{\emph{{Panel: {label}}}}} \\"]
        for n, reps in core.items():
            corr = np.mean([r["two_step"][-1]["corr"] for r in reps])
            ceil = np.mean([np.sqrt(r["oracle_pm"]["reliability"]) for r in reps])
            cvi = np.mean([r["two_step"][-1]["cov90_theta"] for r in reps])
            cex = np.mean([r["oracle_pm"]["cov90_theta"] for r in reps])
            cpm = np.mean([r["oracle_pm"]["c_perunit"] for r in reps])
            lines.append(f"{n:,} & {corr:.3f} & {ceil:.3f} & {cvi:.3f} & "
                         f"{cex:.3f} & {cpm:.3f} " + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    open(path, "w").write("\n".join(lines) + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("results_dir", nargs="?", default="audit/hpc_results_prod")
    ap.add_argument("--out", default="tables")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    cells = load(args.results_dir)
    write_point_table(cells, os.path.join(args.out, "sim_point.tex"))
    write_coverage_table(cells, os.path.join(args.out, "sim_coverage.tex"))
    write_vi_table(cells, os.path.join(args.out, "sim_vi_appendix.tex"))
    print(f"wrote sim_point.tex, sim_coverage.tex, sim_vi_appendix.tex -> {args.out}/ "
          f"({sum(len(v) for v in cells.values())} reps across {len(cells)} cells)")
