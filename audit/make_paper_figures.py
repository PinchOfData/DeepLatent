"""Generate publication figures (vector PDF, into figures/) from production MC reps.

  sim_distributions.pdf -- sampling densities of psi-hat: rows = channel, cols = n;
                           the "same width, wrong center" visual (two-step vs joint,
                           post-fit overlaid), truth line.
  sim_bias.pdf          -- bias vs n (log x), +-2 MC-SE bars, per channel panel.
  sim_coverage.pdf      -- 95% CI coverage vs n: two-step naive, post-fit naive,
                           post-fit delta-corrected; nominal line.
  sim_mechanism.pdf     -- votes J-grid: (A) two-step attenuation vs measured
                           reliability with sqrt(rel)/rel theory curves;
                           (B) coverage vs J: naive degrades, corrected flat.

Colors are fixed by estimator identity (CVD-validated): two-step #eb6834,
joint #2a78d6, post-fit #1baf7a. Truth/nominal: near-black dashed.

Usage: python audit/make_paper_figures.py [results_dir] [--out figures/]
"""
import argparse
import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats as sps

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_paper_tables import load, core_cells, PSI_TRUE  # noqa: E402

CHANNELS = {"ip": "Roll-call votes (2PL IRT)", "lg": "Linear\u2013Gaussian"}

COL = {"two_step": "#eb6834", "joint": "#2a78d6", "postfit": "#1baf7a"}
LAB = {"two_step": "Two-step", "joint": "Joint", "postfit": "Post-fit OLS"}
TRUTH_C = "#2b2b2b"

plt.rcParams.update({
    "font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "axes.grid": True, "axes.grid.axis": "y", "grid.linewidth": 0.4, "grid.alpha": 0.35,
    "legend.frameon": False, "pdf.fonttype": 42,
})


def get_psi(reps, arm):
    if arm == "postfit":
        return np.array([r["joint"][-1]["pf_psi"] for r in reps])
    return np.array([r[arm][-1]["psi"] for r in reps])


def fig_distributions(cells, out):
    chans = [c for c in CHANNELS if core_cells(cells, c)]
    ncols = max(len(core_cells(cells, c)) for c in chans)
    fig, axes = plt.subplots(len(chans), ncols, figsize=(2.6 * ncols, 2.2 * len(chans)),
                             sharey=False, squeeze=False)
    for i, chan in enumerate(chans):
        core = core_cells(cells, chan)
        for j, (n, reps) in enumerate(core.items()):
            ax = axes[i][j]
            for arm in ("two_step", "joint", "postfit"):
                v = get_psi(reps, arm)
                grid = np.linspace(min(v.min(), PSI_TRUE) - 0.02,
                                   max(v.max(), PSI_TRUE) + 0.02, 300)
                ls = ":" if arm == "postfit" else "-"
                ax.plot(grid, sps.gaussian_kde(v)(grid), color=COL[arm], lw=1.4,
                        ls=ls, label=LAB[arm] if (i == 0 and j == 0) else None)
            ax.axvline(PSI_TRUE, color=TRUTH_C, lw=0.9, ls="--")
            if i == 0 and j == len(core) - 1:
                ax.annotate("truth", xy=(PSI_TRUE, ax.get_ylim()[1]),
                            xytext=(3, -10), textcoords="offset points",
                            fontsize=8, color=TRUTH_C)
            ax.set_title(f"{CHANNELS[chan]}, $n$={n:,}", fontsize=8.5)
            ax.set_yticks([])
            ax.grid(False)
            if i == len(chans) - 1:
                ax.set_xlabel(r"$\hat\psi$")
    axes[0][0].legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(out)
    fig.savefig(out.replace(".pdf", ".png"), dpi=180)
    plt.close(fig)


def fig_bias(cells, out):
    chans = [c for c in CHANNELS if core_cells(cells, c)]
    fig, axes = plt.subplots(1, len(chans), figsize=(3.2 * len(chans), 2.6),
                             sharey=True, squeeze=False)
    for i, chan in enumerate(chans):
        ax = axes[0][i]
        core = core_cells(cells, chan)
        ns = list(core)
        for arm in ("two_step", "joint", "postfit"):
            b, se = [], []
            for n, reps in core.items():
                v = get_psi(reps, arm)
                b.append(v.mean() - PSI_TRUE)
                se.append(v.std(ddof=1) / np.sqrt(len(v)))
            ax.errorbar(ns, b, yerr=2 * np.array(se), color=COL[arm], lw=1.4,
                        marker="o", ms=3.5, capsize=2,
                        label=LAB[arm] if i == 0 else None)
        ax.axhline(0, color=TRUTH_C, lw=0.9, ls="--")
        ax.set_xscale("log")
        ax.set_xticks(ns)
        ax.set_xticklabels([f"{n:,}" for n in ns])
        ax.minorticks_off()
        ax.set_title(CHANNELS[chan], fontsize=9)
        ax.set_xlabel("$n$ (log scale)")
        if i == 0:
            ax.set_ylabel(r"Bias of $\hat\psi$")
    axes[0][0].legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(out)
    fig.savefig(out.replace(".pdf", ".png"), dpi=180)
    plt.close(fig)


def fig_coverage(cells, out):
    variants = [("two_step", "psi_cover", "two_step", "-", "Two-step, naive SE"),
                ("joint", "pf_psi_cover", "postfit", "--", "Post-fit, naive SE"),
                ("joint", "pf_psi_cover_corr", "postfit", "-", "Post-fit, corrected SE")]
    chans = [c for c in CHANNELS if core_cells(cells, c)]
    fig, axes = plt.subplots(1, len(chans), figsize=(3.2 * len(chans), 2.6),
                             sharey=True, squeeze=False)
    for i, chan in enumerate(chans):
        ax = axes[0][i]
        core = core_cells(cells, chan)
        ns = list(core)
        for arm, key, ckey, ls, lab in variants:
            cov, se = [], []
            for n, reps in core.items():
                vals = [r[arm][-1][key] for r in reps if key in r[arm][-1]]
                if not vals:
                    cov.append(np.nan); se.append(0.0); continue
                p = np.mean(vals)
                cov.append(p)
                se.append(np.sqrt(p * (1 - p) / len(vals)))
            ax.errorbar(ns, cov, yerr=2 * np.array(se), color=COL[ckey], ls=ls,
                        lw=1.4, marker="o", ms=3.5, capsize=2,
                        label=lab if i == 0 else None)
        ax.axhline(0.95, color=TRUTH_C, lw=0.9, ls="--")
        ax.set_xscale("log")
        ax.set_xticks(ns)
        ax.set_xticklabels([f"{n:,}" for n in ns])
        ax.minorticks_off()
        ax.set_ylim(-0.03, 1.03)
        ax.set_title(CHANNELS[chan], fontsize=9)
        ax.set_xlabel("$n$ (log scale)")
        if i == 0:
            ax.set_ylabel("95% CI coverage")
    axes[0][0].legend(fontsize=7.5, loc="center right")
    fig.tight_layout()
    fig.savefig(out)
    fig.savefig(out.replace(".pdf", ".png"), dpi=180)
    plt.close(fig)


def fig_mechanism(cells, out):
    """Votes J-grid cells (any 'ip_*' cell regardless of J, at the J-grid n)."""
    jcells = {}
    for cell, docs in cells.items():
        if not cell.startswith("ip_"):
            continue
        cfg = docs[0]["config"]
        if cfg["N"] == 4000:
            jcells[cfg["J"]] = [d["rep"] for d in docs]
    jcells = dict(sorted(jcells.items()))
    if len(jcells) < 2:
        print("mechanism figure skipped (need >=2 J cells at n=4000)")
        return
    fig, (axa, axb) = plt.subplots(1, 2, figsize=(6.4, 2.6))
    rels, psir, psir_se, rfr, rfr_se = [], [], [], [], []
    for jj, reps in jcells.items():
        rels.append(np.mean([r["oracle_pm"]["reliability"] for r in reps]))
        v = np.array([r["two_step"][-1]["psi"] for r in reps]) / PSI_TRUE
        psir.append(v.mean()); psir_se.append(v.std(ddof=1) / np.sqrt(len(v)))
        w = np.array([r["two_step"][-1]["rf"] for r in reps])
        rfr.append(w.mean()); rfr_se.append(w.std(ddof=1) / np.sqrt(len(w)))
    grid = np.linspace(min(rels) - 0.03, min(1.0, max(rels) + 0.03), 100)
    axa.plot(grid, np.sqrt(grid), color="#9a9a9a", lw=1.0)
    axa.plot(grid, grid, color="#9a9a9a", lw=1.0, ls="--")
    axa.annotate(r"$\sqrt{\mathrm{rel}}$", xy=(grid[8], np.sqrt(grid[8])),
                 xytext=(0, 5), textcoords="offset points", fontsize=8, color="#6a6a6a")
    axa.annotate(r"$\mathrm{rel}$", xy=(grid[8], grid[8]),
                 xytext=(0, -11), textcoords="offset points", fontsize=8, color="#6a6a6a")
    axa.errorbar(rels, psir, yerr=2 * np.array(psir_se), color=COL["two_step"],
                 marker="o", ms=4, lw=0, elinewidth=1, capsize=2,
                 label=r"$\hat\psi/\psi$ (std. slope)")
    axa.errorbar(rels, rfr, yerr=2 * np.array(rfr_se), color=COL["two_step"],
                 marker="^", ms=4, lw=0, elinewidth=1, capsize=2, mfc="white",
                 label=r"$\widehat{rf}/rf$ (reduced form)")
    axa.set_xlabel("reliability of the measurement design")
    axa.set_ylabel("Two-step estimate / truth")
    axa.legend(fontsize=7.5, loc="lower right")
    axa.set_title("(a) Attenuation tracks theory", fontsize=9)
    js = list(jcells)
    for key, ls, lab, ckey in (("psi_cover", "-", "Two-step, naive", "two_step"),
                               ("pf_psi_cover", "--", "Post-fit, naive", "postfit"),
                               ("pf_psi_cover_corr", "-", "Post-fit, corrected", "postfit")):
        cov, se = [], []
        for jj, reps in jcells.items():
            arm = "two_step" if key == "psi_cover" else "joint"
            vals = [r[arm][-1][key] for r in reps if key in r[arm][-1]]
            p = np.mean(vals) if vals else np.nan
            cov.append(p)
            se.append(np.sqrt(p * (1 - p) / len(vals)) if vals else 0.0)
        axb.errorbar(js, cov, yerr=2 * np.array(se), color=COL[ckey], ls=ls, lw=1.4,
                     marker="o", ms=3.5, capsize=2, label=lab)
    axb.axhline(0.95, color=TRUTH_C, lw=0.9, ls="--")
    axb.set_xscale("log")
    axb.set_xticks(js)
    axb.set_xticklabels([str(j) for j in js])
    axb.minorticks_off()
    axb.set_ylim(-0.03, 1.03)
    axb.set_xlabel("votes per legislator $J$ (log scale)")
    axb.set_ylabel("95% CI coverage")
    axb.legend(fontsize=7.5, loc="center left")
    axb.set_title("(b) Naive CI worsens as $J$ grows", fontsize=9)
    fig.tight_layout()
    fig.savefig(out)
    fig.savefig(out.replace(".pdf", ".png"), dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("results_dir", nargs="?", default="audit/hpc_results_prod")
    ap.add_argument("--out", default="figures")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    cells = load(args.results_dir)
    fig_distributions(cells, os.path.join(args.out, "sim_distributions.pdf"))
    fig_bias(cells, os.path.join(args.out, "sim_bias.pdf"))
    fig_coverage(cells, os.path.join(args.out, "sim_coverage.pdf"))
    fig_mechanism(cells, os.path.join(args.out, "sim_mechanism.pdf"))
    print(f"figures written -> {args.out}/")
