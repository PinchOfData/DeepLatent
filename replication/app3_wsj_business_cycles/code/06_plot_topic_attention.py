"""Step 06: the monthly attention of a run's topics, as levels and as changes (common.head_features), against the target.

For the --n-level topics whose attention level correlates most with the target and the --n-change topics whose
attention change does (absolute correlation over the training months only), one row per topic: left, the attention
level; right, its change from the previous --change-window months' average; both with the target. Every series is
standardized on the training months, so the panels share one axis (standard deviations); each panel reports the
correlation with the target in the training, validation and test months.

Writes results/<tag>/topic_attention_<which>.png and topic_attention_<which>.csv (every topic's correlations).

Usage: python 06_plot_topic_attention.py --tag joint_k50chg_lam0_s0 [--which final --n-level 4 --n-change 4]
"""
import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import common


def corr(a, b):
    m = ~(np.isnan(a) | np.isnan(b))
    return float(np.corrcoef(a[m], b[m])[0, 1]) if m.sum() > 2 else np.nan


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--which", default="final")
    ap.add_argument("--n-level", type=int, default=4)
    ap.add_argument("--n-change", type=int, default=4)
    ap.add_argument("--change-window", type=int, default=12)
    a = ap.parse_args()
    res = common.RESULTS / a.tag
    cfg = json.loads((res / "config.json").read_text())
    months = pd.read_csv(res / "months.csv", dtype={"month": str}).month.tolist()
    mt = common.month_targets(months, cfg.get("train_end", "1999-12"), cfg.get("val_end", "2003-12"),
                              cfg.get("target_start", 0), cfg.get("target_window", 3))
    theta = np.load(res / f"theta_{a.which}.npy")
    words = pd.read_csv(res / f"topic_words_{a.which}.csv")
    level, _ = common.head_features(theta, "level")
    change, hist = common.head_features(theta, "change", a.change_window)
    y = mt.target.to_numpy()
    split = mt.split.to_numpy()
    tr = (split == "train") & hist & ~np.isnan(y)  # the same months for both, so the comparison is like for like

    def z(x):
        return (x - np.nanmean(x[tr], axis=0)) / np.nanstd(x[tr], axis=0)

    zl, zc, zy = z(level), z(change), z(y[:, None])[:, 0]
    rows = []
    for k in range(theta.shape[1]):
        r = {"topic": k, "words": ", ".join(words.loc[words.topic == k, "probable"].iloc[0].split(", ")[:6])}
        for name, x in [("level", zl[:, k]), ("change", zc[:, k])]:
            for s in ["train", "val", "test"]:
                m = (split == s) & hist
                r[f"{name}_{s}"] = corr(x[m], y[m])
        rows.append(r)
    tab = pd.DataFrame(rows)
    tab.to_csv(res / f"topic_attention_{a.which}.csv", index=False)
    pick = list(tab.reindex(tab.level_train.abs().sort_values(ascending=False).index).topic[:a.n_level])
    pick += [k for k in tab.reindex(tab.change_train.abs().sort_values(ascending=False).index).topic
             if k not in pick][:a.n_change]

    x = pd.PeriodIndex(months, freq="M").to_timestamp()
    fig, axes = plt.subplots(len(pick), 2, figsize=(13, 2.1 * len(pick)), sharex=True, sharey=True, squeeze=False)
    for i, k in enumerate(pick):
        r = tab.loc[tab.topic == k].iloc[0]
        for j, (name, series) in enumerate([("level", zl[:, k]), ("change", zc[:, k])]):
            ax = axes[i, j]
            for s, shade in [("val", "#f1f1f1"), ("test", "#e3e3e3")]:
                m = split == s
                ax.axvspan(x[m].min(), x[m].max(), color=shade, zorder=0)
            ax.plot(x, zy, color="black", lw=1.0, label="Target")
            sign = np.sign(r[f"{name}_train"]) or 1.0
            ax.plot(x, sign * series, color="#2a78d6" if j == 0 else "#eb6834", lw=1.1,
                    label=f"Topic {k} {'level' if j == 0 else 'change'}" + (" (sign flipped)" if sign < 0 else ""))
            ax.axhline(0, color="#bbbbbb", lw=0.6)
            ax.set_ylim(-5, 5)
            ax.spines[["top", "right"]].set_visible(False)
            ax.text(0.01, 0.97, f"corr. with target: train {r[f'{name}_train']:+.2f}, val {r[f'{name}_val']:+.2f}, "
                                f"test {r[f'{name}_test']:+.2f}", transform=ax.transAxes, fontsize=7.5, va="top")
            if i == 0:
                ax.set_title(["Attention level", f"Attention change (minus previous {a.change_window} months' mean)"][j],
                             fontsize=10)
            if j == 0:
                ax.set_ylabel(f"Topic {k}\n{r.words[:38]}", fontsize=7.5)
            ax.legend(frameon=False, fontsize=7, loc="lower left")
    fig.text(0.5, 0.005, "Standard deviations (standardized on the training months); validation (light) and test (dark) "
                         "shaded", ha="center", fontsize=8)
    fig.tight_layout(rect=(0, 0.015, 1, 1))
    out = res / f"topic_attention_{a.which}.png"
    fig.savefig(out, dpi=150)
    print(tab.round(2).to_string(index=False))
    print("\nmean |corr| over all topics:")
    print(tab[[c for c in tab if c.startswith(("level_", "change_"))]].abs().mean().round(3).to_string())
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
