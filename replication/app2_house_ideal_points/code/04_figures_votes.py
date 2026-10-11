"""Step 04: figures for a fit of step 03 (votes) or 05 (votes and text), from its saved estimates.

    python 04_figures_votes.py --tag votes_103_118_cong --step 40000

Four panels, against Nokken-Poole (one score per legislator-congress, from Voteview):
  (a) our ideal points against Nokken-Poole, every legislator-congress;
  (b) party means by congress, ours mapped onto the Nokken-Poole scale by the pooled linear
      fit theta = a + b * NP (so the two series share units);
  (c) ours minus Nokken-Poole in those units, by congress: a common movement of the whole
      chamber (or of a party) that Nokken-Poole does not show;
  (d) correlation with Nokken-Poole within each congress, overall and within party.
For a votes-and-text fit (step 05) the ideal points are the fused posterior means, and a second
figure, figures/<tag>_<step>_precision.png, compares the two modalities' precision increments
over the prior (prior precision 1; fused precision 1 + votes + text):
  (a) median increment by congress for votes and for text (interquartile band);
  (b) text share of the increment, Delta_text / (Delta_votes + Delta_text), by congress and party;
  (c), (d) each unit's increment against the amount of data: terms spoken, votes cast.
Reads results/<tag>/estimates_<step>.csv (local copy) and writes figures/<tag>_<step>.png.
"""

import argparse

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter
import numpy as np
import pandas as pd

import common as c

INK, MUTED, GRID = "#1f1f1e", "#6b6a66", "#e6e5e1"
PARTY = {100: ("Democrats", "#2a78d6"), 200: ("Republicans", "#e34948")}


def corr(a, b):
    ok = a.notna() & b.notna()
    return np.corrcoef(a[ok], b[ok])[0, 1] if ok.sum() > 2 else np.nan


def style(ax, title):
    ax.set_title(title, loc="left", fontsize=10, color=INK)
    ax.grid(color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="votes_103_118_cong")
    ap.add_argument("--step", type=int, default=40000)
    args = ap.parse_args()
    e = pd.read_csv(c.MODULE / "results" / args.tag / f"estimates_{args.step}.csv")
    ok = e["nokken_poole_dim1"].notna()
    b, a = np.polyfit(e.loc[ok, "nokken_poole_dim1"], e.loc[ok, "theta_mean"], 1)
    e["theta_np"] = (e["theta_mean"] - a) / b  # ours in Nokken-Poole units

    fig, axes = plt.subplots(2, 2, figsize=(11, 8.2), constrained_layout=True)
    (ax_a, ax_b), (ax_c, ax_d) = axes

    # (a) scatter
    for code, (name, col) in PARTY.items():
        g = e[e["party_code"] == code]
        ax_a.scatter(g["nokken_poole_dim1"], g["theta_mean"], s=4, alpha=0.35, color=col,
                     linewidths=0, label=name)
    style(ax_a, f"(a) Ideal points vs Nokken-Poole, all legislator-congresses "
                f"(r = {corr(e['theta_mean'], e['nokken_poole_dim1']):.3f})")
    ax_a.set_xlabel("Nokken-Poole, 1st dimension", fontsize=9, color=MUTED)
    ax_a.set_ylabel("Our ideal point (posterior mean)", fontsize=9, color=MUTED)
    ax_a.legend(frameon=False, fontsize=8, markerscale=3, loc="upper left")

    # party and chamber means by congress
    g = e.groupby(["congress", "party_code"])[["theta_np", "nokken_poole_dim1"]].mean()
    chamber = e.groupby("congress")[["theta_np", "nokken_poole_dim1"]].mean()
    congresses = chamber.index.to_numpy()

    # (b) party means, ours vs Nokken-Poole
    for code, (name, col) in PARTY.items():
        m = g.xs(code, level="party_code")
        ax_b.plot(m.index, m["theta_np"], color=col, linewidth=2, label=f"{name}, ours")
        ax_b.plot(m.index, m["nokken_poole_dim1"], color=col, linewidth=2, linestyle=(0, (4, 3)),
                  label=f"{name}, Nokken-Poole")
    ax_b.plot(congresses, chamber["theta_np"], color=INK, linewidth=2, label="Chamber, ours")
    ax_b.plot(congresses, chamber["nokken_poole_dim1"], color=INK, linewidth=2,
              linestyle=(0, (4, 3)), label="Chamber, Nokken-Poole")
    style(ax_b, "(b) Party and chamber means by congress (Nokken-Poole units)")
    ax_b.set_xlabel("Congress", fontsize=9, color=MUTED)
    ax_b.legend(frameon=False, fontsize=7.5, ncol=3, loc="upper center",
                bbox_to_anchor=(0.5, -0.12))

    # (c) ours minus Nokken-Poole
    for code, (name, col) in PARTY.items():
        m = g.xs(code, level="party_code")
        ax_c.plot(m.index, m["theta_np"] - m["nokken_poole_dim1"], color=col, linewidth=2,
                  marker="o", markersize=4, label=name)
    ax_c.plot(congresses, chamber["theta_np"] - chamber["nokken_poole_dim1"], color=INK,
              linewidth=2, marker="o", markersize=4, label="Chamber")
    ax_c.axhline(0, color=MUTED, linewidth=1)
    style(ax_c, "(c) Ours minus Nokken-Poole, mean by congress (Nokken-Poole units)")
    ax_c.set_xlabel("Congress", fontsize=9, color=MUTED)
    ax_c.legend(frameon=False, fontsize=8, loc="lower left")

    # (d) within-congress correlations
    rows = []
    for t, h in e.groupby("congress"):
        rows.append((t, corr(h["theta_mean"], h["nokken_poole_dim1"]),
                     *(corr(h.loc[h["party_code"] == p, "theta_mean"],
                            h.loc[h["party_code"] == p, "nokken_poole_dim1"]) for p in PARTY)))
    r = pd.DataFrame(rows, columns=["congress", "all", 100, 200]).set_index("congress")
    ax_d.plot(r.index, r["all"], color=INK, linewidth=2, marker="o", markersize=4, label="All")
    for code, (name, col) in PARTY.items():
        ax_d.plot(r.index, r[code], color=col, linewidth=2, marker="o", markersize=4,
                  label=f"Within {name}")
    style(ax_d, "(d) Correlation with Nokken-Poole within each congress")
    ax_d.set_xlabel("Congress", fontsize=9, color=MUTED)
    ax_d.set_ylim(0, 1.02)
    ax_d.legend(frameon=False, fontsize=8, loc="lower left")

    for ax in (ax_b, ax_c, ax_d):
        ax.set_xticks(congresses[::3])
    kind = "votes and text" if "precision_text" in e.columns else "votes only"
    fig.suptitle(f"House, {congresses.min()}th-{congresses.max()}th Congress, {kind}: "
                 f"'{args.tag}', {args.step:,} steps", fontsize=11, color=INK, x=0.01,
                 ha="left")
    out = c.MODULE / "figures" / f"{args.tag}_{args.step}.png"
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=160, facecolor="#fcfcfb")
    print(out)
    if "precision_text" in e.columns:
        precision_figure(e, args, congresses)


VOTES, TEXT = ("Votes", "#4a3aa7"), ("Text", "#eda100")  # distinct from the party colors


def precision_figure(e, args, congresses):
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.2), constrained_layout=True)
    (ax_a, ax_b), (ax_c, ax_d) = axes

    # (a) increments by congress, median and interquartile range
    for col, (name, color) in (("precision_votes", VOTES), ("precision_text", TEXT)):
        q = e.groupby("congress")[col].quantile([0.25, 0.5, 0.75]).unstack()
        ax_a.fill_between(q.index, q[0.25], q[0.75], color=color, alpha=0.18, linewidth=0)
        ax_a.plot(q.index, q[0.5], color=color, linewidth=2, marker="o", markersize=4,
                  label=f"{name} (median, interquartile band)")
    ax_a.set_yscale("log")
    ax_a.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax_a.yaxis.set_minor_formatter(NullFormatter())
    style(ax_a, "(a) Precision added to the prior by each modality, by congress")
    ax_a.set_ylabel("Precision increment (prior precision = 1)", fontsize=9, color=MUTED)
    ax_a.set_xlabel("Congress", fontsize=9, color=MUTED)
    ax_a.legend(frameon=False, fontsize=8, loc="best")

    # (b) text share by congress and party
    share = e.groupby("congress")["text_share"].median()
    ax_b.plot(share.index, share, color=INK, linewidth=2, marker="o", markersize=4, label="All")
    for code, (name, col) in PARTY.items():
        m = e[e["party_code"] == code].groupby("congress")["text_share"].median()
        ax_b.plot(m.index, m, color=col, linewidth=2, marker="o", markersize=4, label=name)
    ax_b.set_ylim(0, 1)
    style(ax_b, "(b) Text share of the precision increment (median), by congress")
    ax_b.set_xlabel("Congress", fontsize=9, color=MUTED)
    ax_b.legend(frameon=False, fontsize=8, loc="best")

    # (c), (d) increment against the amount of data
    for ax, x, col, (name, color), label in (
        (ax_c, "n_terms", "precision_text", TEXT, "Terms spoken in the congress"),
        (ax_d, "n_votes", "precision_votes", VOTES, "Votes cast in the congress"),
    ):
        ok = (e[x] > 0) & (e[col] > 0)
        ax.scatter(e.loc[ok, x], e.loc[ok, col], s=5, alpha=0.45, color=color, linewidths=0)
        ax.set_xscale("log")
        ax.set_yscale("log")
        for axis, values in ((ax.xaxis, e.loc[ok, x]), (ax.yaxis, e.loc[ok, col])):
            decades = np.log10(values.max() / values.min())
            axis.set_major_locator(LogLocator(subs=(1, 2, 5) if decades < 3 else (1,)))
            axis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
            axis.set_minor_formatter(NullFormatter())
        rho = np.corrcoef(np.log(e.loc[ok, x]), np.log(e.loc[ok, col]))[0, 1]
        style(ax, f"({'c' if ax is ax_c else 'd'}) {name} precision vs amount of {name.lower()} "
                  f"(log-log r = {rho:.2f})")
        ax.set_xlabel(label, fontsize=9, color=MUTED)
        ax.set_ylabel(f"{name} precision increment", fontsize=9, color=MUTED)

    for ax in (ax_a, ax_b):
        ax.set_xticks(congresses[::3])
    fig.suptitle(f"Precision by modality: '{args.tag}', {args.step:,} steps", fontsize=11,
                 color=INK, x=0.01, ha="left")
    out = c.MODULE / "figures" / f"{args.tag}_{args.step}_precision.png"
    fig.savefig(out, dpi=160, facecolor="#fcfcfb")
    print(out)


if __name__ == "__main__":
    main()
