"""Aggregate completed GTM pilot-format replications without mixing designs.

Naive OLS coverage is measured as a diagnostic; no CI for the joint head is
implied. Reject duplicate seeds and incompatible DGP/training specifications.
"""

import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np


def summarize(estimates, truth):
    estimates = np.asarray(estimates, dtype=np.float64)
    error = estimates - truth
    n = len(estimates)
    sd = estimates.std(axis=0, ddof=1) if n > 1 else None
    return {"mean": estimates.mean(axis=0).tolist(),
            "bias": error.mean(axis=0).tolist(),
            "sd": None if sd is None else sd.tolist(),
            "bias_mc_se": None if sd is None else (sd / np.sqrt(n)).tolist(),
            "rmse": np.sqrt((error ** 2).mean(axis=0)).tolist()}


def ols_summary(records, truth):
    summary = summarize([r["beta"] for r in records], truth)
    covered = np.array([(truth >= r["naive_ci_low"]) & (truth <= r["naive_ci_high"])
                        for r in records])
    rates = covered.mean(axis=0)
    summary["naive_hc1_coverage_95"] = rates.tolist()
    summary["coverage_binomial_se"] = np.sqrt(rates * (1 - rates) / len(records)).tolist()
    summary["mean_naive_hc1_se"] = np.mean([r["naive_hc1_se"] for r in records], axis=0).tolist()
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    cells = defaultdict(list)
    seen = set()
    reference = None
    skipped = []
    for path in sorted(args.results.glob("gtm_n*_w*_rep*.json")):
        r = json.loads(path.read_text())
        if r.get("status") != "complete":
            skipped.append(str(path))
            continue
        cfg = r["config"]
        config_fixed = {k: v for k, v in cfg.items()
                        if k not in {"n", "words", "seed", "out", "device", "threads"}}
        fingerprint = {"design": r["design"], "configuration": config_fixed,
                       "package_sha256": r.get("package_sha256"),
                       "script_sha256": r["script_sha256"]}
        if reference is None:
            reference = fingerprint
        elif fingerprint != reference:
            raise ValueError(f"Different DGP, source, or training settings in {path}; aggregate separately.")
        key = (cfg["n"], cfg["words"])
        identity = (*key, cfg["seed"])
        if identity in seen:
            raise ValueError(f"Duplicate replication {identity}: {path}")
        seen.add(identity)
        cells[key].append(r)
    if not cells:
        raise ValueError("No completed GTM replications found.")
    result = {"notes": ["Coverage columns concern naive HC1 intervals, omitting first-stage uncertainty.",
                        "One replication cannot estimate a sampling SD or consistency."],
              "design_and_settings": reference, "skipped_incomplete": skipped, "cells": {}}
    truth = np.asarray(reference["design"]["beta"])
    for (n, words), reps in sorted(cells.items()):
        cell = {"n": n, "words": words, "replications": len(reps),
                "seeds": [r["config"]["seed"] for r in reps],
                "oracle_true_shares": ols_summary([r["oracle_true_shares"] for r in reps], truth),
                "arms": {}}
        for name in ("joint", "two_step"):
            arm = {"checkpoints": {}}
            cps = reps[0]["config"]["checkpoints"]
            for i, cp in enumerate(cps):
                records = [r["arms"][name]["checkpoints"][i] for r in reps]
                effects = [rec["head" if name == "joint" else "ols_encoder"] for rec in records]
                stats = summarize([eff["beta"] for eff in effects], truth)
                if name == "two_step":
                    stats.update(ols_summary(effects, truth))
                stats["mean_prevalence_rmse"] = float(np.mean([rec["prevalence_rmse"] for rec in records])) if "prevalence_rmse" in records[0] else None
                stats["mean_covariance_relative_error"] = float(np.mean([rec["covariance_relative_frobenius_error"] for rec in records])) if "prevalence_rmse" in records[0] else None
                arm["checkpoints"][str(cp)] = stats
            records = [r["arms"][name]["ols_outcome_free_posterior"] for r in reps]
            arm["ols_outcome_free_posterior"] = ols_summary(records, truth)
            arm["mean_posterior_ess_median"] = float(np.mean([rec["posterior_diagnostics"]["ess_median"] for rec in records]))
            arm["max_half_sample_beta_difference"] = float(max(rec["half_sample_beta_max_difference"] for rec in records))
            cell["arms"][name] = arm
        result["cells"][f"n{n}_w{words}"] = cell
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(f"Saved {len(seen)} replications in {len(cells)} cells to {args.out}; {len(skipped)} incomplete files excluded.")


if __name__ == "__main__":
    main()
