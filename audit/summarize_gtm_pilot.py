"""Make a compact report and figure from one completed GTM pilot replication."""

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/deeplatent_mpl")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import softmax


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("result", type=Path)
    p.add_argument("--report", type=Path, required=True)
    p.add_argument("--figure", type=Path, required=True, help="Filename stem, without extension")
    p.add_argument("--diagnostics", type=Path)
    p.add_argument("--compare", type=Path, help="Earlier completed run with only n changed")
    p.add_argument("--overfit", type=Path, help="Completed exact-replay generalization audit")
    a = p.parse_args()
    r = json.loads(a.result.read_text())
    if r["status"] != "complete":
        raise ValueError("Wait for the replication to complete before reporting it.")
    cfg, design = r["config"], {k: np.asarray(v) for k, v in r["design"].items()}
    truth = design["beta"]
    joint, two_step = r["arms"]["joint"], r["arms"]["two_step"]
    jf, tf = joint["checkpoints"][-1], two_step["checkpoints"][-1]
    series = {
        "Oracle true shares": r["oracle_true_shares"],
        "Joint head": jf["head"],
        "Joint post-fit OLS": joint["ols_outcome_free_posterior"],
        "Two-step encoder OLS": tf["ols_encoder"],
        "Two-step importance OLS": two_step["ols_outcome_free_posterior"],
    }

    # Reconstruct the same replication's latent draws for oracle prior benchmarks.
    rng = np.random.default_rng(cfg["seed"])
    x = np.column_stack([np.ones(cfg["n"]), rng.binomial(1, 0.5, (cfg["n"], cfg["covariates"]))])
    z = x @ design["prevalence"].T @ design["basis"]
    z += rng.multivariate_normal(np.zeros(cfg["topics"] - 1), design["covariance"], cfg["n"])
    theta = softmax(z @ design["basis"].T, axis=1)
    coef = np.linalg.lstsq(x, z, rcond=None)[0]
    residual = z - x @ coef
    oracle_cov = residual.T @ residual / (cfg["n"] - x.shape[1])
    oracle_prevalence_rmse = np.sqrt(np.mean((design["basis"] @ coef.T - design["prevalence"]) ** 2))
    oracle_cov_error = np.linalg.norm(oracle_cov - design["covariance"]) / np.linalg.norm(design["covariance"])

    lines = [f"# GTM laptop pilot — n = {cfg['n']:,}", "",
             f"One replication, seed {cfg['seed']}; {cfg['n']:,} documents, {cfg['words']} words/document, "
             f"{cfg['topics']} topics, {cfg['covariates']} prevalence covariates, {cfg['components']} Gaussian components.", "",
             "Both fitted arms learn a covariate-informed logistic-normal prior with full covariance. "
             f"The data use the package's linear-softmax multinomial decoder. Outcome noise variance is {cfg['sigma_y'] ** 2:g}.", "",
             f"Encoder hidden-layer widths: {cfg['hidden']}. Training checkpoints: {cfg['checkpoints']} optimizer updates.", "",
             f"Runtime: {r['elapsed_seconds'] / 60:.1f} minutes on {cfg['device']} ({cfg['threads']} thread).", "",
             "## Outcome effects", "",
             "Coefficients are topic-aligned and centered. Errors below concern one draw, not Monte Carlo bias.", "",
             "| Topic | Truth | Oracle | Joint head | Joint post-fit | Two-step encoder | Two-step importance |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for k, b in enumerate(truth):
        values = " | ".join(f"{v['beta'][k]:+.3f}" for v in series.values())
        lines.append(f"| {k + 1} | {b:+.3f} | {values} |")
    lines += ["", "| Estimator | Mean absolute error | Overall effect scale (truth = 1) |",
              "|---|---:|---:|"]
    for name, rec in series.items():
        lines.append(f"| {name} | {rec['mean_absolute_error']:.3f} | {rec['scale_ratio']:.3f} |")
    if a.compare:
        previous = json.loads(a.compare.read_text())
        if previous["status"] != "complete" or previous["design"] != r["design"]:
            raise ValueError("Comparison requires a completed run under the same DGP.")
        changes = {key for key in set(previous["config"]) | set(cfg)
                   if previous["config"].get(key) != cfg.get(key)}
        if changes - {"n", "out"}:
            raise ValueError(f"Settings other than n differ: {changes - {'n', 'out'}}")
        if previous["package_sha256"] != r["package_sha256"] or previous["script_sha256"] != r["script_sha256"]:
            raise ValueError("Comparison requires the same simulation and package source.")
        old_joint, old_two = previous["arms"]["joint"], previous["arms"]["two_step"]
        old_series = {"Oracle true shares": previous["oracle_true_shares"],
                      "Joint head": old_joint["checkpoints"][-1]["head"],
                      "Joint post-fit OLS": old_joint["ols_outcome_free_posterior"],
                      "Two-step encoder OLS": old_two["checkpoints"][-1]["ols_encoder"],
                      "Two-step importance OLS": old_two["ols_outcome_free_posterior"]}
        old_n = previous["config"]["n"]
        lines += ["", "## Comparison with the earlier sample size", "",
                  f"Only n changed ({old_n:,} to {cfg['n']:,}); the DGP, source code, seed, "
                  "architecture, optimization budget, and posterior readout settings match.", "",
                  f"| Estimator | Mean absolute error, n={old_n:,} | Mean absolute error, n={cfg['n']:,} |",
                  "|---|---:|---:|"]
        for name, rec in series.items():
            lines.append(f"| {name} | {old_series[name]['mean_absolute_error']:.3f} | {rec['mean_absolute_error']:.3f} |")
        lines += ["", "Each sample size has one replication. The larger dataset is regenerated under the same DGP, "
                  "rather than appending observations to the earlier dataset. Keeping optimizer updates fixed also "
                  "reduces the number of passes through the data at larger n."]
    lines += ["", "## Prior and topic recovery", "",
              "| Fit | Prevalence coefficient RMSE | Relative covariance Frobenius error |",
              "|---|---:|---:|",
              f"| Oracle observed contrast logits | {oracle_prevalence_rmse:.3f} | {oracle_cov_error:.3f} |"]
    for name, rec in [("Joint", jf), ("Unsupervised", tf)]:
        lines.append(f"| {name} | {rec.get('prevalence_rmse', float('nan')):.3f} | "
                     f"{rec.get('covariance_relative_frobenius_error', float('nan')):.3f} |")
    lines += ["", "Mean true topic shares: " + ", ".join(f"{v:.3f}" for v in theta.mean(0)) + ".",
              "Joint decoder-profile correlations: " + ", ".join(f"{v:.3f}" for v in jf["decoder_profile_correlations"]) + ".",
              "Unsupervised decoder-profile correlations: " + ", ".join(f"{v:.3f}" for v in tf["decoder_profile_correlations"]) + ".",
              f"Joint estimated outcome variance: {jf['outcome_noise_variance']:.3f} (truth 1).", "",
              "## Posterior computation", "",
              "Outcome-free means use importance sampling with the outcome likelihood excluded. "
              "The supervised encoder's zero-outcome output is only a proposal.", "",
              "| Fit | Median ESS / draws | 5th-percentile ESS | Max coefficient difference between draw halves |",
              "|---|---:|---:|---:|"]
    for name, arm in [("Joint", joint), ("Unsupervised", two_step)]:
        rec = arm["ols_outcome_free_posterior"]
        d = rec["posterior_diagnostics"]
        lines.append(f"| {name} | {d['ess_median']:.0f} / {d['samples']} | {d['ess_p05']:.0f} | "
                     f"{rec['half_sample_beta_max_difference']:.3f} |")
    if a.diagnostics:
        bounds = json.loads(a.diagnostics.read_text())
        lines += ["", "Held-out encoder ELBO versus 1,024-draw IWAE (256 new documents):", "",
                  "| Fit | Encoder ELBO | IWAE | Estimated gap (nats/document) |",
                  "|---|---:|---:|---:|"]
        for name, rec in bounds.items():
            lines.append(f"| {name} | {rec['encoder_elbo']:.3f} | {rec['iwae_1024']:.3f} | "
                         f"{rec['estimated_gap_nats_per_doc']:.3f} |")
        lines += ["", "The joint bound includes the outcome likelihood; the unsupervised bound excludes it, "
                  "so their levels are not directly comparable. IWAE is a finite-sample evidence approximation. "
                  "The readout's Gaussian-mixture log density agrees with the package implementation to within 1e-6."]
    if a.overfit:
        audit = json.loads(a.overfit.read_text())
        if audit["status"] != "complete" or audit["config"]["n"] != cfg["n"]:
            raise ValueError("A completed audit of this sample size is required.")
        n_eval = audit["audit"]["heldout_n"]
        every = audit["audit"]["every"]
        samples = audit["audit"]["importance_samples"]
        lines += ["", "## Training and overfitting audit", "",
                  f"Replayed the same seeded training run and measured likelihood every {every:,} updates "
                  f"on {n_eval:,} fixed held-out documents and a fixed training subset. "
                  f"IWAE uses {samples} draws/document; multinomial constants are included. "
                  "Diagnostics preserve the training RNG. Original coefficient checkpoints reproduce exactly.", "",
                  "Negative log-likelihood (NLL): lower is better. Joint NLL includes the outcome, "
                  "while two-step NLL covers words, so compare trends within each arm.", "",
                  "| Steps | Joint train NLL | Joint held-out NLL | Two-step train NLL | Two-step held-out NLL |",
                  "|---|---:|---:|---:|---:|"]
        curves = {name: {v["step"]: v for v in arm["trajectory"]} for name, arm in audit["arms"].items()}
        for step in cfg["checkpoints"]:
            j, t = curves["joint"][step], curves["two_step"][step]
            lines.append(f"| {step:,} | {j['train']['nll']:.4f} | {j['heldout']['nll']:.4f} | "
                         f"{t['train']['nll']:.4f} | {t['heldout']['nll']:.4f} |")
        lines += ["", "Joint outcome-coefficient trajectory (truth: " + ", ".join(f"{v:g}" for v in truth) + "):", "",
                  "| Steps | Topic 1 | Topic 2 | Topic 3 | Topic 4 | Topic 5 | MAE |",
                  "|---|---:|---:|---:|---:|---:|---:|"]
        for rec in joint["checkpoints"]:
            coefficients = " | ".join(f"{v:+.3f}" for v in rec["head"]["beta"])
            lines.append(f"| {rec['step']:,} | {coefficients} | {rec['head']['mean_absolute_error']:.3f} |")
        for name, arm in audit["arms"].items():
            trace = arm["trajectory"]
            best = min(trace, key=lambda v: v["heldout"]["nll"])
            final = trace[-1]
            reproduced = arm["reproduced_checkpoints"]
            if len(reproduced) != len(cfg["checkpoints"]) or any(v["max_coefficient_difference"] > 1e-6 for v in reproduced):
                raise ValueError("Original checkpoint reproduction was not verified.")
            lines += ["", f"{name}: lowest observed held-out NLL at {best['step']:,} steps; "
                      f"final minus minimum = {final['heldout']['nll'] - best['heldout']['nll']:+.5f} nats/document."]
            increases = sum(trace[i]["heldout"]["nll"] > trace[i - 1]["heldout"]["nll"] for i in range(1, len(trace)))
            lines.append(f"Held-out NLL increases between adjacent measured checkpoints: {increases}.")
        lines += ["", "The originally reported 16k-step estimates were fixed-budget results; parameter convergence "
                  "had not been established. Prevalence-parameter errors can deteriorate while likelihood and "
                  "outcome-coefficient recovery improve. The audit separates those targets.", "",
                  f"Detailed curves and paired per-document likelihoods: `{a.overfit}` and its `.npz` companion."]
        audit_fig, audit_axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
        for axis, name in zip(audit_axes, ("joint", "two_step")):
            trace = [v for v in audit["arms"][name]["trajectory"] if v["step"] >= 4000]
            for split, color in (("train", "#2463a4"), ("heldout", "#d67722")):
                axis.plot([v["step"] for v in trace], [v[split]["nll"] for v in trace], "o-", color=color,
                          label="Training subset" if split == "train" else "Held-out documents", markersize=3)
            axis.set(xlabel="Optimizer steps", ylabel="Negative log-likelihood / document",
                     title="Joint fit" if name == "joint" else "Unsupervised fit")
            axis.legend(frameon=False)
            axis.spines[["top", "right"]].set_visible(False)
            axis.grid(alpha=0.18)
        a.figure.parent.mkdir(parents=True, exist_ok=True)
        for suffix in (".png", ".pdf"):
            audit_fig.savefig(a.figure.with_name(a.figure.name + "_overfit").with_suffix(suffix), dpi=180)
        plt.close(audit_fig)
    lines += ["", "Naive HC1 OLS intervals are saved in the JSON but omit first-stage uncertainty. "
              "This single replication establishes neither consistency nor repeated-sample CI coverage. "
              "The ideal-point study's scalar standard-error correction has not been applied to GTM.", "",
              "## Files", "", f"- Results: `{a.result}`", "- Design and commands: `GTM_CONSISTENCY.md`",
              "- Harness: `audit/experiment_gtm_pilot.py`", "- HPC worker (not submitted): `audit/gtm_mc.sbatch`",
              "- Monte Carlo aggregator: `audit/merge_gtm_mc.py`", ""]
    a.report.parent.mkdir(parents=True, exist_ok=True)
    a.report.write_text("\n".join(lines))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained")
    topics = np.arange(1, len(truth) + 1)
    axes[0].plot(topics, truth, "k--", label="Truth", linewidth=2)
    for name, color, marker in [("Joint head", "#2463a4", "o"),
                                ("Joint post-fit OLS", "#258451", "s"),
                                ("Two-step encoder OLS", "#d67722", "^")]:
        axes[0].plot(topics, series[name]["beta"], marker=marker, color=color, label=name)
    axes[0].set(xlabel="Topic", ylabel="Centered outcome coefficient", xticks=topics,
                title="One replication: outcome coefficient recovery")
    axes[0].legend(frameon=False, fontsize=9)
    for name, arm, key, color in [("Joint head", joint, "head", "#2463a4"),
                                  ("Two-step encoder OLS", two_step, "ols_encoder", "#d67722")]:
        axes[1].plot([v["step"] for v in arm["checkpoints"]],
                     [v[key]["scale_ratio"] for v in arm["checkpoints"]], "o-", color=color, label=name)
    axes[1].axhline(1, color="black", linestyle="--", linewidth=1)
    axes[1].set(xlabel="Optimizer steps", ylabel="Overall effect scale (truth = 1)",
                title="Training trajectory; all planned checkpoints")
    axes[1].legend(frameon=False, fontsize=9)
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(alpha=0.18)
    a.figure.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".png", ".pdf"):
        fig.savefig(a.figure.with_suffix(suffix), dpi=180)
    plt.close(fig)
    print(a.report)


if __name__ == "__main__":
    main()
