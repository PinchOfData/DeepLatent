"""Step 05: plot the target and the predictions of a few runs (04's regression_<which>_<features>.csv), and measure
how far each run's monthly topic attention drifts out of its training range.

Drift: for every month, the attention standardized on the training months (z = (thetabar - mean) / sd, training
months with a target); reported by period as the mean over topics of |z| and the largest |z| of any topic.

Writes results/predictions_<which>_<features>.png and prints the drift table.

Usage: python 05_plot_predictions.py --runs joint_k50_lam0_s0:lasso joint_k50_lam1e5_s0:head ... [--which selected]
"""
import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import common


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True, help="tag:column (a column of the regression file)")
    ap.add_argument("--which", default="selected")
    ap.add_argument("--features", default="level", help="the head inputs of 04's regression file")
    a = ap.parse_args()
    fig, ax = plt.subplots(figsize=(12, 4.5))
    first = True
    colors = ["#2a78d6", "#eb6834", "#1baf7a", "#8a5cd6", "#d6a12a"]
    for i, spec in enumerate(a.runs):
        tag, col = spec.split(":")
        reg = pd.read_csv(common.RESULTS / tag / f"regression_{a.which}_{a.features}.csv", dtype={"month": str})
        x = pd.PeriodIndex(reg.month, freq="M").to_timestamp()
        if first:
            ax.plot(x, reg.target, color="black", lw=1.4, label="Target: growth over the month and the next two")
            for s, shade in [("val", "#f1f1f1"), ("test", "#e3e3e3")]:
                m = reg.split == s
                ax.axvspan(x[m].min(), x[m].max(), color=shade, zorder=0)
            first = False
        ax.plot(x, reg[col], color=colors[i % len(colors)], lw=1.2, label=f"{tag} ({col})")
        theta = np.load(common.RESULTS / tag / f"theta_{a.which}.npy")
        tr = ((reg.split == "train") & reg.target.notna()).to_numpy()
        z = (theta - theta[tr].mean(0)) / (theta[tr].std(0) + 1e-12)
        rows = []
        for s in ["train", "val", "test"]:
            m = (reg.split == s).to_numpy()
            rows.append({"period": s, "mean |z|": np.abs(z[m]).mean(), "largest |z|": np.abs(z[m]).max(),
                         "topics with mean |z| > 2": int((np.abs(z[m]).mean(0) > 2).sum())})
        print(f"\n{tag}: drift of the monthly attention (standardized on the training months)")
        print(pd.DataFrame(rows).round(2).to_string(index=False))
    ax.axhline(0, color="#bbbbbb", lw=0.8)
    ax.set_ylabel("Industrial production growth (% per month)")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    ax.set_title("Validation (light) and test (dark) periods shaded", fontsize=9, loc="right")
    out = common.RESULTS / f"predictions_{a.which}_{a.features}.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
