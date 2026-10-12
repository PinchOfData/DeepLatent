"""Wave 2, step 4: assemble the expanding-window out-of-sample forecasts and evaluate them.

Topic models are fit once per window (w2_fit_joint.py, w2_fit_lda.py) and frozen over the window's block; the
forecasting equations are re-estimated at every forecast origin, as in the literature (Ellingsen, Larsen & Thorsrud
2022 refit them each period with the topic model fixed). At origin s (end of month s), the information is the text
up to s and industrial production up to s-1, so an equation is fit on the months s' <= s-3 (whose target, growth over
s'..s'+2, is published by then) and forecasts month s's target.

For every window and every topic model (LDA; the joint model at each lam, lam = 0 being our topic model alone), with
the forecasting regression ("layer") a ridge (--layer ridge, the default since 2026-10-12) or the LASSO:
  - inputs: w2_common.literature_inputs of the model's monthly attention (backward 3-month average, linear trend and
    scale estimated on the months up to the origin, months t .. t-6);
  - the layer's penalty is chosen once per window: fit on the training months (inputs detrended and standardized on
    them), scored on the validation months;
  - at every origin of the block the layer is refit with that penalty and forecasts.
Forecasts are averaged over seeds (missing seeds are skipped with a warning). "joint (lam chosen)" takes, in each
window, the lam > 0 with the lowest validation error of that layer (averaged over seeds); the joint models' own heads
(frozen) are reported too. Benchmarks, re-estimated at every origin: the historical mean, and a direct
autoregression on the published monthly growth (lag order by BIC, chosen once per window). "AR + <model>" adds the
text inputs to the autoregression: the AR terms are not penalized (Frisch-Waugh: both the target and the text inputs
are residualized on them, the penalized regression runs on the residuals), so the larger model nests the AR as Clark-West requires.

Evaluation over the concatenated blocks: out-of-sample R^2 against the historical mean, MSE relative to the AR, both
also over NBER recession and expansion months (a target month is a recession month if its middle month, s+1, is);
Diebold-Mariano (joint vs each two-step model; every model vs the AR) and Clark-West (AR vs AR + text) tests,
Newey-West with 3 lags. Industrial production is the current FRED vintage (revised data), not real-time vintages.
Writes <results>/w2/<grid>/: forecasts_<layer><out-name>.csv, summary_..., tests_..., windows_..., forecasts_....png.

Usage: python w2_forecast.py --grid k50 --dtm-dir <scratch>/data/dtm_w2 --lams 0 1e2 1e3 1e4 1e5 --seeds 0 1 2
"""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso, Ridge

import common
import w2_common as w2

GRIDS = {"lasso": np.logspace(-3, 0.5, 36), "ridge": np.logspace(-1, 5, 37)}
LAYER = "ridge"  # set from --layer


class FWLLasso:
    """y = A phi + X beta with phi unpenalized (A includes the constant) and a penalty on beta (the ridge's L2 with
    --layer ridge, the LASSO's L1 with --layer lasso): X and y are residualized on A (OLS), the penalized regression
    runs on the residuals without intercept, then phi is refit by OLS."""

    def __init__(self, alpha):
        self.alpha = alpha

    def fit(self, X, A, y):
        bx = np.linalg.lstsq(A, X, rcond=None)[0]
        by = np.linalg.lstsq(A, y, rcond=None)[0]
        est = Ridge(alpha=self.alpha, fit_intercept=False) if LAYER == "ridge" else \
            Lasso(alpha=self.alpha, fit_intercept=False, max_iter=50_000)
        self.las = est.fit(X - A @ bx, y - A @ by)
        self.phi = np.linalg.lstsq(A, y - X @ self.las.coef_, rcond=None)[0]
        return self

    def predict(self, X, A):
        return A @ self.phi + X @ self.las.coef_


def choose_alpha(F, A, y, fit_m, val_m):
    best = None
    for alpha in GRIDS[LAYER]:
        m = FWLLasso(alpha).fit(F[fit_m], A[fit_m], y[fit_m])
        err = ((y[val_m] - m.predict(F[val_m], A[val_m])) ** 2).mean()
        if best is None or err < best[0]:
            best = (err, alpha)
    return best[1], best[0]


def oos_r2(y, f, bench):
    return float(1 - ((y - f) ** 2).sum() / ((y - bench) ** 2).sum())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", required=True)
    ap.add_argument("--dtm-dir", required=True)
    ap.add_argument("--lams", nargs="+", required=True, help="joint lam values fit (0 = topics alone)")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0])
    ap.add_argument("--which", default="selected")
    ap.add_argument("--first-end", type=int, default=1979)
    ap.add_argument("--step", type=int, default=3)
    ap.add_argument("--val-years", type=int, default=5)
    ap.add_argument("--last-year", type=int, default=2011)
    ap.add_argument("--smooth", type=int, default=3)
    ap.add_argument("--lags", type=int, default=6)
    ap.add_argument("--lda-grid", default=None, help="read the LDA fits from this grid (default: --grid)")
    ap.add_argument("--layer", choices=["ridge", "lasso"], default="ridge", help="the forecasting regression")
    ap.add_argument("--inputs", choices=["literature", "change"], default="literature")
    ap.add_argument("--change-window", type=int, default=12)
    ap.add_argument("--out-name", default="", help="suffix of the output files (e.g. _early)")
    a = ap.parse_args()
    global LAYER
    LAYER = a.layer
    root = common.RESULTS / "w2" / a.grid
    months = sorted(pd.read_csv(Path(a.dtm_dir) / "records.csv.gz", dtype={"month": str}, usecols=["month"]).month.unique())
    T = len(months)
    pm = pd.PeriodIndex(months, freq="M")
    growth = pd.read_csv(Path(a.dtm_dir) / "monthly_ip.csv", dtype={"month": str}).set_index("month")["growth"]
    g1 = pd.read_csv(common.DATA / "dtm" / "monthly_ip.csv", dtype={"month": str}).set_index("month")["growth"]
    common_m = growth.index.intersection(g1.index)
    assert np.allclose(growth[common_m], g1[common_m], equal_nan=True), "the two monthly_ip.csv files differ"
    lda_root = common.RESULTS / "w2" / (a.lda_grid or a.grid)
    models = {"lda": (lda_root, "lda"), **{f"lam{float(l):g}": (root, f"joint_lam{float(l):g}") for l in a.lams}}

    # Target settings of the joint runs (all the same); targets without split masking, for the origins
    cfgs = [json.loads(p.read_text()) for p in root.glob("joint_*/config.json")]
    tw = {(c["target_start"], c["target_window"], c["smooth"], c["lags"], c.get("inputs", "literature"),
           c.get("change_window", 12)) for c in cfgs}
    assert len(tw) == 1, f"joint runs disagree on target/inputs: {tw}"
    t_start, t_win, smooth, lags, kind, cw = tw.pop()
    assert (smooth, lags, kind, cw) == (a.smooth, a.lags, a.inputs, a.change_window), \
        "the forecasting inputs differ from the runs' settings"
    y_all = common.month_targets(months, months[-1], months[-1], t_start, t_win).target.to_numpy()
    publish = t_start + t_win  # the target of s' is known at origin s iff s' <= s - publish

    cols, win_rows = {}, []

    def put(name, idx, values):
        cols.setdefault(name, np.full(T, np.nan))[idx] = values

    for w in w2.windows(a.first_end, a.step, a.last_year, a.val_years):
        mt = common.month_targets(months, w["train_end"], w["val_end"], t_start, t_win)
        y_sel, split = mt.target.to_numpy(), mt.split.to_numpy()
        trm, vam = split == "train", split == "val"
        block = np.flatnonzero(np.asarray((pm >= pd.Period(w["block_start"], "M")) & (pm <= pd.Period(w["block_end"], "M"))))
        info = {"end": w["end"], "block": f"{w['block_start']}..{w['block_end']}"}

        # Autoregression: lag order by BIC on the months known at the window's end, refit at every origin
        last_known = np.flatnonzero(np.asarray(pm <= pd.Period(w["val_end"], "M")))[-1] - publish
        _, p, L = w2.ar_forecast(growth, months, y_all, np.arange(T) <= last_known)
        info["ar_p"] = p
        A_ar = np.column_stack([np.ones(T), L[:, :p]])
        A_c = np.ones((T, 1))
        ok_ar = ~np.isnan(L[:, :p]).any(axis=1)
        for i in block:
            avail = (np.arange(T) <= i - publish) & ~np.isnan(y_all)
            put("historical mean", i, y_all[avail].mean())
            m = avail & ok_ar
            put("AR", i, A_ar[i] @ np.linalg.lstsq(A_ar[m], y_all[m], rcond=None)[0])

        val_err = {}
        for name, (mroot, prefix) in models.items():
            runs = []
            for s in a.seeds:
                d = mroot / f"{prefix}_e{w['end']}_s{s}"
                if not (d / f"theta_{a.which}.npy").exists():
                    print(f"WARNING: {d.name} missing, skipped", flush=True)
                    continue
                runs.append((s, d, np.load(d / f"theta_{a.which}.npy")))
            assert runs, f"no run of {name} for the window ending {w['end']}"
            info[f"seeds {name}"] = len(runs)
            fc, fc_ar, heads, errs = [], [], [], []
            for s, d, theta in runs:
                F_sel, ok_sel = w2.make_inputs(theta, trm, a.inputs, a.smooth, a.change_window, a.lags)
                fit_m = trm & ok_sel & ~np.isnan(y_sel)
                val_m = vam & ok_sel & ~np.isnan(y_sel)
                alpha, err = choose_alpha(F_sel, A_c, y_sel, fit_m, val_m)
                alpha_ar, _ = choose_alpha(F_sel, A_ar, y_sel, fit_m & ok_ar, val_m & ok_ar)
                errs.append(err)
                f, f_ar = np.full(T, np.nan), np.full(T, np.nan)
                for i in block:
                    F, ok = w2.make_inputs(theta, np.arange(T) <= i, a.inputs, a.smooth, a.change_window, a.lags)
                    avail = (np.arange(T) <= i - publish) & ~np.isnan(y_all) & ok
                    f[i] = FWLLasso(alpha).fit(F[avail], A_c[avail], y_all[avail]).predict(F[i:i + 1], A_c[i:i + 1])[0]
                    av = avail & ok_ar
                    f_ar[i] = FWLLasso(alpha_ar).fit(F[av], A_ar[av], y_all[av]).predict(F[i:i + 1], A_ar[i:i + 1])[0]
                fc.append(f)
                fc_ar.append(f_ar)
                put(f"{name} + layer [seed {s}]", block, f[block])
                pred = d / f"predictions_{a.which}.csv"
                if name not in ("lda", "lam0") and pred.exists():
                    heads.append(pd.read_csv(pred)["prediction"].to_numpy())
            put(f"{name} + layer", block, np.mean(fc, 0)[block])
            put(f"AR + {name}", block, np.mean(fc_ar, 0)[block])
            if heads:
                put(f"{name} own head", block, np.mean(heads, 0)[block])
            val_err[name] = float(np.mean(errs))
            info[f"val_mse {name}"] = val_err[name]
        chosen = min((n for n in models if n.startswith("lam") and n != "lam0"), key=lambda n: val_err[n])
        info["lam chosen"] = chosen
        put("joint (lam chosen) + layer", block, cols[f"{chosen} + layer"][block])
        put("AR + joint (lam chosen)", block, cols[f"AR + {chosen}"][block])
        if f"{chosen} own head" in cols:
            put("joint (lam chosen) own head", block, cols[f"{chosen} own head"][block])
        win_rows.append(info)
        print(f"window ending {w['end']}: AR({p}), lam chosen {chosen}", flush=True)

    fc = pd.DataFrame({"month": months, "target": y_all, **cols})
    fc["recession"] = np.r_[w2.recession(months)[1:], False]  # the target's middle month
    fc = fc[fc["historical mean"].notna() & fc.target.notna()].reset_index(drop=True)
    root.mkdir(parents=True, exist_ok=True)
    fc.to_csv(root / f"forecasts_{a.layer}{a.out_name}.csv", index=False)
    pd.DataFrame(win_rows).to_csv(root / f"windows_{a.layer}{a.out_name}.csv", index=False)

    y, hm, ar = fc.target.to_numpy(), fc["historical mean"].to_numpy(), fc["AR"].to_numpy()
    rows = []
    for c in cols:
        f = fc[c].to_numpy()
        r = {"method": c}
        for lab, m in [("all", np.ones(len(fc), bool)), ("recession", fc.recession.to_numpy()),
                       ("expansion", ~fc.recession.to_numpy())]:
            r[f"r2_vs_mean_{lab}"] = oos_r2(y[m], f[m], hm[m])
            r[f"mse_ratio_vs_ar_{lab}"] = float(((y[m] - f[m]) ** 2).mean() / ((y[m] - ar[m]) ** 2).mean())
        rows.append(r)
    summary = pd.DataFrame(rows)
    summary.to_csv(root / f"summary_{a.layer}{a.out_name}.csv", index=False)

    tests = []
    for other in ["lda + layer", "lam0 + layer"]:
        st, pv = w2.diebold_mariano(y - fc[other], y - fc["joint (lam chosen) + layer"])
        tests.append({"test": "DM", "model 1": other, "model 2": "joint (lam chosen) + layer", "stat": st, "p": pv})
    for c in ["lda + layer", "lam0 + layer", "joint (lam chosen) + layer"]:
        st, pv = w2.diebold_mariano(y - ar, y - fc[c])
        tests.append({"test": "DM", "model 1": "AR", "model 2": c, "stat": st, "p": pv})
    for c in ["AR + lda", "AR + lam0", "AR + joint (lam chosen)"]:
        st, pv = w2.clark_west(y, ar, fc[c])
        tests.append({"test": "CW", "model 1": "AR", "model 2": c, "stat": st, "p": pv})
    tests = pd.DataFrame(tests)
    tests.to_csv(root / f"tests_{a.layer}{a.out_name}.csv", index=False)

    show = [c for c in cols if "[seed" not in c]
    print(summary[summary.method.isin(show)].round(3).to_string(index=False))
    print("\n" + tests.round(3).to_string(index=False))
    print("\n" + pd.DataFrame(win_rows)[["end", "block", "ar_p", "lam chosen"]].to_string(index=False))

    x = pd.PeriodIndex(fc.month, freq="M").to_timestamp()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 7.5), sharex=True, gridspec_kw={"height_ratios": [3, 2]})
    for ax in (ax1, ax2):
        for peak, trough in w2.NBER:
            if pd.Period(trough, "M") >= pd.Period(fc.month.iloc[0], "M") and \
                    pd.Period(peak, "M") <= pd.Period(fc.month.iloc[-1], "M"):
                ax.axvspan(pd.Period(peak, "M").to_timestamp(), pd.Period(trough, "M").to_timestamp(),
                           color="#e8e8e8", zorder=0)
        ax.spines[["top", "right"]].set_visible(False)
    ax1.plot(x, y, color="black", lw=1.2, label="Truth")
    lines = [("AR", "#999999"), ("lda + layer", "#2a78d6"), ("lam0 + layer", "#d6a12a"),
             ("joint (lam chosen) + layer", "#eb6834")]
    for c, col in lines:
        ax1.plot(x, fc[c], color=col, lw=1.0, label=c)
        ax2.plot(x, np.cumsum((y - hm) ** 2 - (y - fc[c]) ** 2), color=col, lw=1.1, label=c)
    for ax in (ax1, ax2):
        ax.axhline(0, color="#bbbbbb", lw=0.6)
    ax1.set_ylabel("% per month")
    ax1.legend(frameon=False, fontsize=8, ncol=5, loc="lower left")
    ax2.set_ylabel("Cumulative squared error\nof the historical mean minus the model's")
    fig.text(0.5, 0.005, "Out-of-sample forecasts (topic models refit every few years, forecasting equations every "
             "month); NBER recessions shaded. Rising lines in the lower panel: the model beats the historical mean.",
             ha="center", fontsize=8)
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    fig.savefig(root / f"forecasts_{a.layer}{a.out_name}.png", dpi=150)
    print(f"wrote {root}")


if __name__ == "__main__":
    main()
