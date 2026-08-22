"""Merge per-rep JSONs from the HPC job arrays into MC summaries.

Each SLURM array task runs audit/experiment_ip_pilot.py with IP_REPS=1 (calibrate
mode) and writes results/ip_<cell>_rep<k>.json. This script aggregates them per
cell: E/bias/SD/RMSE for the identified functionals (psi, b1_std, rf) per arm and
checkpoint, naive-CI coverage rates, theta-coverage means, and oracle stats.

Usage:
    python audit/merge_ip_mc.py audit/hpc_results/           # dir of ip_*_rep*.json
    python audit/merge_ip_mc.py audit/hpc_results/ --out audit/results_ip_mc_merged.json
"""
import argparse
import glob
import json
import os
import re
from collections import defaultdict

import numpy as np

TRUTH = {"psi": None, "b1_std": None, "rf": None}  # filled from config of first file


def summarize(cell, files):
    reps = []
    for f in sorted(files):
        d = json.load(open(f))
        reps.append(d["rep"])
        cfg = d["config"]
    truth = {"psi": cfg["truth"]["psi"], "b1_std": cfg["truth"]["b1_std"],
             "rf": cfg["truth"]["rf"]}
    out = {"cell": cell, "n": cfg["N"], "J": cfg["J"], "reps": len(reps),
           "truth": truth, "arms": {}}
    print(f"\n=== {cell}: n={cfg['N']} J={cfg['J']} reps={len(reps)} "
          f"(truth psi={truth['psi']:.3f}, b1_std=1, rf=1) ===")
    for arm in ("joint", "two_step"):
        by_cp = defaultdict(list)
        for rep in reps:
            for rec in rep[arm]:
                by_cp[rec["step"]].append(rec)
        out["arms"][arm] = {}
        for cp in sorted(by_cp):
            recs = by_cp[cp]
            row = {}
            for key in ("psi", "b1_std", "rf"):
                v = np.array([r[key] for r in recs])
                t = truth[key]
                row[key] = {"E": float(v.mean()), "bias": float(v.mean() - t),
                            "SD": float(v.std(ddof=1)),
                            "RMSE": float(np.sqrt(((v - t) ** 2).mean()))}
                print(f"  {arm:>8} @{cp:>6} {key:>6}: E={v.mean():+.3f} "
                      f"bias={v.mean()-t:+.3f} SD={v.std(ddof=1):.3f} "
                      f"RMSE={row[key]['RMSE']:.3f}")
            for key in ("psi_cover", "b1_std_cover", "cov90_theta",
                        "cov90_theta_yzero", "corr", "sd_ratio_pm", "sd_ratio_pm_sup",
                        "sig2_eps_hat", "sig_u_hat"):
                vals = [r[key] for r in recs if key in r]
                if vals:
                    row[key] = float(np.mean(vals))
            out["arms"][arm][str(cp)] = row
        cover_keys = [k for k in ("psi_cover", "b1_std_cover") if k in recs[0]]
        for k in cover_keys:
            print(f"  {arm:>8} naive-CI coverage ({k}) @final: {row[k]:.3f}")
        covk = "cov90_theta_yzero" if arm == "joint" else "cov90_theta"
        print(f"  {arm:>8} avg theta-coverage@90 @final: {row[covk]:.3f}")
    # post-fit OLS arm (joint final checkpoint only; present in run2+ results)
    pf_recs = [rep["joint"][-1] for rep in reps if "pf_psi" in rep["joint"][-1]]
    if pf_recs:
        v = np.array([r["pf_psi"] for r in pf_recs])
        t = truth["psi"]
        out["postfit"] = {
            "psi": {"E": float(v.mean()), "bias": float(v.mean() - t),
                    "SD": float(v.std(ddof=1))},
            "rf_E": float(np.mean([r["pf_rf"] for r in pf_recs])),
            "se_psi_E": float(np.mean([r["pf_se_psi"] for r in pf_recs])),
            "hc_se_psi_E": float(np.mean([r["pf_hc_se_psi"] for r in pf_recs])),
            "psi_cover": float(np.mean([r["pf_psi_cover"] for r in pf_recs])),
            "psi_cover_hc": float(np.mean([r["pf_psi_cover_hc"] for r in pf_recs])),
        }
        if "pf_psi_cover_corr" in pf_recs[0]:
            out["postfit"]["se_psi_corr_E"] = float(np.mean([r["pf_se_psi_corr"] for r in pf_recs]))
            out["postfit"]["psi_cover_corr"] = float(np.mean([r["pf_psi_cover_corr"] for r in pf_recs]))
        p = out["postfit"]
        corr_str = (f" | CORR SE={p['se_psi_corr_E']:.4f} cover={p['psi_cover_corr']:.3f}"
                    if "psi_cover_corr" in p else "")
        print(f"  POST-FIT OLS: psi E={p['psi']['E']:+.3f} bias={p['psi']['bias']:+.3f} "
              f"SD={p['psi']['SD']:.3f} | E[SE]={p['se_psi_E']:.4f} "
              f"(HC {p['hc_se_psi_E']:.4f}) | 95%CI cover={p['psi_cover']:.3f} "
              f"(HC {p['psi_cover_hc']:.3f}){corr_str} | rf E={p['rf_E']:+.3f}")
    # oracle summaries across reps
    o = {"oracle_ols_c": [r["oracle_ols"]["c"] for r in reps],
         "oracle_pm_c_perunit": [r["oracle_pm"]["c_perunit"] for r in reps],
         "oracle_pm_perunit_cover": [r["oracle_pm"]["perunit_cover"] for r in reps],
         "oracle_pm_psi": [r["oracle_pm"]["psi"] for r in reps],
         "oracle_pm_rf": [r["oracle_pm"].get("rf") for r in reps],
         "oracle_pm_cov90": [r["oracle_pm"]["cov90_theta"] for r in reps],
         "oracle_pm_reliability": [r["oracle_pm"]["reliability"] for r in reps]}
    out["oracle"] = {k: {"E": float(np.mean([x for x in v if x is not None])),
                         "SD": float(np.std([x for x in v if x is not None], ddof=1))}
                     for k, v in o.items()}
    op = out["oracle"]
    print(f"  oracle-PM: c_perunit E={op['oracle_pm_c_perunit']['E']:+.3f} "
          f"(perunit CI cover {op['oracle_pm_perunit_cover']['E']:.3f}) "
          f"psi E={op['oracle_pm_psi']['E']:+.3f} cov90 E={op['oracle_pm_cov90']['E']:.3f} "
          f"rel E={op['oracle_pm_reliability']['E']:.3f}")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("results_dir")
    ap.add_argument("--out", default="audit/results_ip_mc_merged.json")
    args = ap.parse_args()
    cells = defaultdict(list)
    for f in glob.glob(os.path.join(args.results_dir, "*_rep*.json")):
        m = re.match(r"((?:ip|lg)_.+)_rep\d+\.json", os.path.basename(f))
        if m:
            cells[m.group(1)].append(f)
    merged = {cell: summarize(cell, files) for cell, files in sorted(cells.items())}
    json.dump(merged, open(args.out, "w"), indent=2)
    print(f"\nsaved -> {args.out}")
