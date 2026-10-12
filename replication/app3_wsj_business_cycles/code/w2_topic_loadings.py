"""Wave 2, exhibit: the topics of a window's models and their loadings in the forecasting equation.

For the window ending in year E and each model given (a run folder prefix: lda, joint_lam0, joint_lam10000, ...; one
seed, since topic numbers differ across seeds), the forecasting layer of w2_forecast (ridge or LASSO on the
literature inputs, months t .. t-6) is fit as at the window's end: penalty chosen on the validation years after
fitting on the training months, then refit on every month whose target is published by the window's end (the
equation of the block's first forecast). Inputs are standardized, so coefficients are comparable across topics; per
topic, "sum" adds the 7 lag coefficients (the effect of a sustained one-standard-deviation rise of the topic's
detrended attention, in percentage points of monthly growth) and "lag0" is the current month's. For joint runs with a
ridge head the head's own coefficients (head_selected.npz) are reported the same way.

Also the usual time-series plot, per model: the truth and the forecasting equation's predictions over the training
months (in-sample: the equation fit on them, penalty chosen on validation), the validation months (out of sample, same
equation) and the window's block (the monthly refits of w2_forecast, from --forecasts-file, seed --seed), with R^2
by period against the training mean.

Writes <results>/w2/<grid>/loadings_e<E>_s<seed>.csv and .png, fits_e<E>_s<seed>.png, and prints, per model, the
topics by |sum|.

Usage: python w2_topic_loadings.py --grid k50 --window-end 1985 --models lda joint_lam0 joint_lam10000 [--seed 0]
"""
import argparse
import importlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import common
import w2_common as w2

fc = importlib.import_module("w2_forecast")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", required=True)
    ap.add_argument("--lda-grid", default=None)
    ap.add_argument("--dtm-dir", required=True)
    ap.add_argument("--window-end", type=int, required=True)
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--which", default="selected")
    ap.add_argument("--layer", choices=["ridge", "lasso"], default="ridge")
    ap.add_argument("--val-years", type=int, default=5)
    ap.add_argument("--smooth", type=int, default=3)
    ap.add_argument("--lags", type=int, default=6)
    ap.add_argument("--n-words", type=int, default=8)
    ap.add_argument("--show", type=int, default=12, help="topics per model in the figure")
    ap.add_argument("--forecasts-file", default=None, help="w2_forecast output with the block's forecasts")
    a = ap.parse_args()
    fc.LAYER = a.layer
    root = common.RESULTS / "w2" / a.grid
    months = sorted(pd.read_csv(Path(a.dtm_dir) / "records.csv.gz", dtype={"month": str}, usecols=["month"]).month.unique())
    T = len(months)
    pm = pd.PeriodIndex(months, freq="M")
    train_end, val_end = f"{a.window_end - a.val_years}-12", f"{a.window_end}-12"
    mt = common.month_targets(months, train_end, val_end, 0, 3)
    y_sel, split = mt.target.to_numpy(), mt.split.to_numpy()
    y_all = common.month_targets(months, months[-1], months[-1], 0, 3).target.to_numpy()
    trm, vam = split == "train", split == "val"
    end_i = np.flatnonzero(np.asarray(pm <= pd.Period(val_end, "M")))[-1]
    A_c = np.ones((T, 1))

    rows, panels, series = [], [], []
    for model in a.models:
        mroot = common.RESULTS / "w2" / (a.lda_grid or a.grid) if model == "lda" else root
        d = mroot / f"{model}_e{a.window_end}_s{a.seed}"
        theta = np.load(d / f"theta_{a.which}.npy")
        K = theta.shape[1]
        words = pd.read_csv(d / f"topic_words_{a.which}.csv")
        F_sel, ok_sel = w2.literature_inputs(theta, trm, a.smooth, a.lags)
        fit_m, val_m = trm & ok_sel & ~np.isnan(y_sel), vam & ok_sel & ~np.isnan(y_sel)
        alpha, _ = fc.choose_alpha(F_sel, A_c, y_sel, fit_m, val_m)
        F, ok = w2.literature_inputs(theta, np.arange(T) <= end_i, a.smooth, a.lags)
        avail = (np.arange(T) <= end_i - 3) & ~np.isnan(y_all) & ok
        sel = fc.FWLLasso(alpha).fit(F_sel[fit_m], A_c[fit_m], y_sel[fit_m])
        pred_sel = np.where(ok_sel, sel.predict(np.nan_to_num(F_sel), A_c), np.nan)
        series.append((model, pred_sel))
        est = fc.FWLLasso(alpha).fit(F[avail], A_c[avail], y_all[avail])
        coef = est.las.coef_.reshape(a.lags + 1, K)  # lag x topic (literature_inputs stacks lag blocks)
        tab = pd.DataFrame({"model": model, "topic": np.arange(K), "share": theta[trm].mean(0),
                            "sum": coef.sum(0), "lag0": coef[0],
                            "words": [", ".join(str(words.loc[words.topic == k, "probable"].iloc[0]).split(", ")[:a.n_words])
                                      for k in range(K)]})
        head = d / f"head_{a.which}.npz"
        cfg = json.loads((d / "config.json").read_text())
        if head.exists() and cfg.get("head") == "ridge":
            c = np.load(head)["c"].reshape(a.lags + 1, K)
            tab["head_sum"], tab["head_lag0"] = c.sum(0), c[0]
        tab["alpha"] = alpha
        rows.append(tab)
        panels.append((model, tab.reindex(tab["sum"].abs().sort_values(ascending=False).index).head(a.show)))
        print(f"\n=== {model} (window ending {a.window_end}, seed {a.seed}, {a.layer} penalty {alpha:.3g}); topics by "
              f"|sum of lag coefficients| (pp of monthly growth per sd of detrended attention)")
        show = tab.reindex(tab["sum"].abs().sort_values(ascending=False).index)
        cols = ["topic", "share", "sum", "lag0"] + (["head_sum"] if "head_sum" in tab else []) + ["words"]
        print(show[cols].round(4).to_string(index=False, max_colwidth=70))
    out = pd.concat(rows)
    out.to_csv(root / f"loadings_e{a.window_end}_s{a.seed}.csv", index=False)

    fig, axes = plt.subplots(1, len(panels), figsize=(7.6 * len(panels), 0.42 * a.show + 1.6), squeeze=False)
    for ax, (model, t) in zip(axes[0], panels):
        t = t.iloc[::-1]
        ax.barh(range(len(t)), t["sum"], color=np.where(t["sum"] > 0, "#1baf7a", "#eb6834"))
        ax.set_yticks(range(len(t)))
        ax.set_yticklabels([f"{k}: {w[:46]}" for k, w in zip(t.topic, t.words)], fontsize=7.5)
        ax.axvline(0, color="#999999", lw=0.7)
        ax.set_title(model, fontsize=10)
        ax.set_xlabel("Sum of lag coefficients (pp of growth per sd)", fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(f"Forecasting-equation loadings at the end of the window ending {a.window_end} ({a.layer}; seed {a.seed})",
                 fontsize=10)
    fig.tight_layout(w_pad=2.5)
    fig.savefig(root / f"loadings_e{a.window_end}_s{a.seed}.png", dpi=150, bbox_inches="tight")
    print(f"\nwrote {root}/loadings_e{a.window_end}_s{a.seed}.*")

    # The usual time-series plot: training (in-sample), validation (out of sample), the block (monthly refits)
    block_fc = pd.read_csv(root / a.forecasts_file, dtype={"month": str}).set_index("month") if a.forecasts_file else None
    blk = np.asarray((pm > pd.Period(val_end, "M")) & (pm <= pd.Period(val_end, "M") + 36))
    x = pm.to_timestamp()
    show_from = np.asarray((pm >= pd.Period(f"{a.window_end - 25}-01", "M")) & (pm <= pd.Period(val_end, "M") + 36))
    fig, axes = plt.subplots(len(series), 1, figsize=(13, 2.6 * len(series)), sharex=True, sharey=True, squeeze=False)
    bench = np.nanmean(y_sel[trm])
    for ax, (model, pred) in zip(axes[:, 0], series):
        name = {"lda": "lda"}.get(model, model.replace("joint_", ""))
        f = pred.copy()
        f[blk] = np.nan
        if block_fc is not None and f"{name} + layer [seed {a.seed}]" in block_fc:
            b = block_fc[f"{name} + layer [seed {a.seed}]"].reindex(months).to_numpy()
            f[blk] = b[blk]
        for per, shade in [(vam, "#f1f1f1"), (blk, "#e3e3e3")]:
            ax.axvspan(x[per].min(), x[per].max(), color=shade, zorder=0)
        m = show_from
        ax.plot(x[m], y_all[m], color="black", lw=1.1, label="Truth")
        ax.plot(x[m], f[m], color="#eb6834" if model != "lda" else "#2a78d6", lw=1.1, label="Prediction")
        txt = []
        for lab, per in [("train", trm), ("validation", vam), ("block", blk)]:
            mm = per & ~np.isnan(y_all) & ~np.isnan(f)
            if mm.sum() > 2:
                txt.append(f"{lab} {1 - ((y_all[mm] - f[mm]) ** 2).sum() / ((y_all[mm] - bench) ** 2).sum():.2f}")
        ax.text(0.01, 0.04, "R²  " + "   ".join(txt), transform=ax.transAxes, fontsize=8.5,
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.8, pad=1.5))
        ax.text(0.01, 0.95, model, transform=ax.transAxes, fontsize=9, va="top", fontweight="bold")
        ax.axhline(0, color="#bbbbbb", lw=0.6)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_ylabel("% per month", fontsize=8)
    axes[0, 0].legend(frameon=False, fontsize=8, loc="upper right", ncol=2)
    fig.text(0.5, 0.005, f"Window ending {a.window_end}: training (white, from {a.window_end - 25}; the equation is fit on "
             f"them), validation (light; out of sample), block (dark; forecasts refit monthly). {a.layer} layer, "
             f"seed {a.seed}. R² against the training mean.", ha="center", fontsize=8)
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    fig.savefig(root / f"fits_e{a.window_end}_s{a.seed}.png", dpi=150)
    print(f"wrote {root}/fits_e{a.window_end}_s{a.seed}.png")


if __name__ == "__main__":
    main()
