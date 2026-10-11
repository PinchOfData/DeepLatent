"""Figures 1-3 from the saved estimates (runs locally after fetch_results.sh; no refitting).

    python 04_figures.py [--precision rkl|var]

Writes figures/fig1_likelihoods, fig2_multimodal, fig3_prior_mc (.png and .pdf) and
figures/figure_numbers.json (every number printed on the figures, plus diagnostics).

Ideal-point panels: units sorted by their true theta; the model's posterior mean and 95% interval
(mean +/- 1.96 sd) after aligning the latent's sign with the truth (the N(0, 1) prior pins its
location and scale). Coverage = share of units whose interval contains the true theta.
Precision (Figure 2) = average over units of the posterior precision 1/sd^2, against the exact
posterior's: by default the precision of the best Gaussian approximation to it (reverse KL, the
target of a Gaussian variational family), or 1/Var(theta | data) with --precision var.
"""

import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.stats import norm  # noqa: E402

import common as c  # noqa: E402

FIG = c.Path(c.os.environ.get("SIMS_FIGURES", c.MODULE / "figures"))
EST, TRUE = "green", "black"
FIG1_PANELS = {
    "embedding": "Continuous (embeddings)",
    "survey": "Discrete choice (survey responses)",
    "bow": "Counts (bag of words)",
    "vote": "Binary (votes)",
}
FIG2_PANELS = {
    "joint": "Joint (votes + survey + text)",
    "vote": "Votes only",
    "survey": "Survey only",
    "bow": "Text only",
}


def load_panel(path):
    d = dict(np.load(path))
    theta, mean, sd = d["theta"], d["mean"], d["sd"]
    mean = mean * np.sign(np.corrcoef(mean, theta)[0, 1])
    return theta, mean, sd, {k[6:]: v for k, v in d.items() if k.startswith("exact_")}


def ideal_point_panel(ax, theta, mean, sd, title):
    order = np.argsort(theta)
    x = np.arange(len(theta))
    m, s = mean[order], sd[order]
    # One sub-pixel sliver per unit: without antialiasing=False the band washes out.
    ax.fill_between(x, m - 1.96 * s, m + 1.96 * s, color=EST, alpha=0.45, lw=0,
                    antialiased=False, label="95% CI", rasterized=True, zorder=1)
    ax.scatter(x, m, s=1.5, color="darkgreen", alpha=0.5, lw=0, label="Posterior mean",
               rasterized=True, zorder=2)
    ax.scatter(x, theta[order], s=2, color=TRUE, alpha=0.8, lw=0, label="True value",
               rasterized=True, zorder=3)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("Units (sorted by true ideal point)")
    ax.set_ylabel("Ideal point")
    ax.grid(alpha=0.3)


def pending(ax, title):
    """Placeholder for a panel whose fit has not finished."""
    ax.text(0.5, 0.5, "still running", ha="center", va="center", transform=ax.transAxes,
            color="grey")
    ax.set_title(title, fontsize=11)


def stats(theta, mean, sd):
    return {
        "corr": float(np.corrcoef(mean, theta)[0, 1]),
        "coverage95": float(np.mean(np.abs(theta - mean) <= 1.96 * sd)),
    }


def save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / f"{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIG / f"{name}.pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)


def figure1(numbers):
    src = c.RESULTS / "fig1_likelihoods"
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharey=True)
    for ax, (kind, label) in zip(axes.flat, FIG1_PANELS.items()):
        if not (src / f"{kind}.npz").exists():
            pending(ax, label)
            continue
        theta, mean, sd, _ = load_panel(src / f"{kind}.npz")
        st = stats(theta, mean, sd)
        numbers["fig1"][kind] = st
        ideal_point_panel(
            ax, theta, mean, sd,
            f"{label}\nCorrelation: {st['corr']:.3f} | Coverage: {100 * st['coverage95']:.1f}%",
        )
    axes.flat[0].legend(loc="upper left", markerscale=3)
    fig.tight_layout()
    save(fig, "fig1_likelihoods")


def figure2(numbers, precision):
    src = c.RESULTS / "fig2_multimodal"
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharey=True)
    for ax, (panel, label) in zip(axes.flat, FIG2_PANELS.items()):
        if not (src / f"{panel}.npz").exists():
            pending(ax, label)
            continue
        theta, mean, sd, ex = load_panel(src / f"{panel}.npz")
        st = stats(theta, mean, sd)
        exact = ex["s2"] if precision == "rkl" else ex["var"]
        st.update({
            "precision_fitted": float(np.mean(1 / sd**2)),
            "precision_exact": float(np.mean(1 / exact)),
            "precision_unit_corr": float(np.corrcoef(1 / sd**2, 1 / exact)[0, 1]),
        })
        numbers["fig2"][panel] = st
        ideal_point_panel(
            ax, theta, mean, sd,
            f"{label}\nCorrelation: {st['corr']:.3f} | Coverage: {100 * st['coverage95']:.1f}%"
            f" | Precision: {st['precision_fitted']:.1f} (exact {st['precision_exact']:.1f})",
        )
    axes.flat[0].legend(loc="upper left", markerscale=3)
    fig.tight_layout()
    save(fig, "fig2_multimodal")


def figure3(numbers):
    src = c.RESULTS / "fig3_prior_mc"
    design = json.loads((src / "design.json").read_text())
    recs = sorted(
        (json.loads(line) for f in sorted(src.glob("task_*.jsonl"))
         for line in f.read_text().splitlines() if line),
        key=lambda r: r["rep"],
    )
    names = list(design["slopes"])
    labels = {"x1": "Binary covariate", "x2": "Continuous covariate 1",
              "x3": "Continuous covariate 2"}
    est = np.array([r["path"][-1]["g"] for r in recs])  # final checkpoint
    mle = np.array([r["g_mle"] for r in recs])
    numbers["fig3"] = {
        "replications": len(recs),
        "final_step": recs[0]["path"][-1]["step"],
        "max_mle_grad": float(max(r["mle_max_grad"] for r in recs)),
        "gap_to_mle_by_step": {  # optimization error: |estimate - exact MLE|, same sample
            str(f["step"]): {
                "mean_abs": np.abs(np.array([r["path"][i]["g"] for r in recs]) - mle)
                .mean(0).tolist(),
                "max_abs": np.abs(np.array([r["path"][i]["g"] for r in recs]) - mle)
                .max(0).tolist(),
            }
            for i, f in enumerate(recs[0]["path"])
        },
    }
    fig, axes = plt.subplots(1, len(names), figsize=(15, 4.5))
    for j, (ax, k) in enumerate(zip(axes, names)):
        truth, e = design["slopes"][k], est[:, j]
        mu, sdv = float(e.mean()), float(e.std(ddof=1))
        numbers["fig3"][k] = {
            "truth": truth, "mean": mu, "sd": sdv, "bias": mu - truth,
            "mc_se_bias": sdv / np.sqrt(len(e)),
            "mle_mean": float(mle[:, j].mean()), "mle_sd": float(mle[:, j].std(ddof=1)),
        }
        half = 1.05 * np.abs(e - truth).max()  # x-axis centered on the true value
        ax.hist(e, bins=np.linspace(truth - half, truth + half, 41), density=True,
                color=EST, alpha=0.5, edgecolor="white")
        grid = np.linspace(truth - half, truth + half, 300)
        ax.plot(grid, norm.pdf(grid, mu, sdv), color=EST, lw=1.5, label="Normal fit")
        ax.axvline(truth, color=TRUE, lw=1.5, label="True value")
        ax.set_xlim(truth - half, truth + half)
        ax.set_title(f"{labels[k]}\nTrue {truth:.2f} | Mean {mu:.3f} | SD {sdv:.3f}",
                     fontsize=11)
        ax.set_xlabel("Estimated prior coefficient")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Density")
    axes[0].legend(loc="upper left")
    fig.tight_layout()
    save(fig, "fig3_prior_mc")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--precision", choices=["rkl", "var"], default="rkl")
    ap.add_argument("--only", choices=["1", "2", "3"], default=None)
    args = ap.parse_args()
    numbers = {"fig1": {}, "fig2": {}, "fig3": {}, "precision_reference": args.precision}
    if args.only in (None, "1"):
        figure1(numbers)
    if args.only in (None, "2"):
        figure2(numbers, args.precision)
    if args.only in (None, "3"):
        figure3(numbers)
    c.write_json(FIG / "figure_numbers.json", numbers)
    print(json.dumps(numbers, indent=1))


if __name__ == "__main__":
    main()
