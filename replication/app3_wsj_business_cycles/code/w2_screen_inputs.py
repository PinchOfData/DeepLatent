"""Wave 2: choose the forecasting inputs with pre-1980 data only.

The topics of the first window (fit on 1947-1974: LDA and our topic model alone, lam = 0, which never see the
outcome) give each month's attention theta_t (K shares). Every input variant below is scored by a pseudo-out-of-sample
exercise over 1960-1979: at every origin s, the inputs are built from statistics of the months up to s only
(standardization, seasonal means, trends), a ridge regression (penalty by generalized cross-validation) is fit on the
months whose target is published by s, and forecasts month s's target. The score is the R^2 against the expanding
historical mean, for the seed-averaged forecasts of each topic model; variants are ranked by the average over LDA
and lam = 0. Nothing after 1979 is used.

Variants (all with lags 0..6; "dev_w" = theta_t minus its mean over the previous w months; every block standardized
on the estimation months):
  A  dev_12                          C  theta_t - theta_{t-12}
  B  dev_12 of seasonally adjusted   D  dev_12 and dev_60 (change and de-drifted level)
     theta (calendar-month means     E  dev_12 and the linearly detrended level
     removed)                        F  literature: 3-month moving average, linear trend removed (current setup)
                                     G  raw levels
  A_log, D_log: A and D on log shares.
Two horizons: the target of month s is the average growth over s..s+2 (h0, a nowcast of the coming quarter) or over
s+1..s+3 (h1, a pure forecast). The direct autoregression on published growth is the reference.

Writes <results>/w2/<grid>/screen_inputs_<horizons>.csv and prints the ranking.

The last origins are those whose target window ends by December 1979.

Usage: python w2_screen_inputs.py --grid k50 --dtm-dir <scratch>/data/dtm_w2 [--window-end 1979 --seeds 0 1 2]
"""
import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")  # one BLAS thread per worker process (the workers share the job's cores)

import argparse
import multiprocessing as mp
from pathlib import Path

import numpy as np
import pandas as pd

import common
import w2_common as w2

LAGS = 6


def standardize(z, est):
    mu, sd = np.nanmean(z[est], 0), np.nanstd(z[est], 0)
    return (z - mu) / np.where(sd > 0, sd, 1.0)


def dev(theta, w):
    """theta_t minus its mean over months t-w .. t-1 (NaN without w months of history)."""
    T = theta.shape[0]
    cs = np.vstack([np.zeros((1, theta.shape[1])), np.cumsum(theta, 0)])
    out = np.full(theta.shape, np.nan)
    t = np.arange(w, T)
    out[w:] = theta[w:] - (cs[t] - cs[t - w]) / w
    return out


def seasonal_adjust(theta, months_of_year, est):
    out = theta.copy()
    for m in range(1, 13):
        sel = months_of_year == m
        out[sel] -= theta[sel & est].mean(0) - theta[est].mean(0)
    return out


def detrend(theta, est):
    t = np.arange(theta.shape[0], dtype=float)
    A = np.column_stack([np.ones(est.sum()), t[est]])
    coef = np.linalg.lstsq(A, theta[est], rcond=None)[0]
    return theta - (coef[0][None, :] + t[:, None] * coef[1][None, :])


def with_lags(z):
    T, P = z.shape
    X = np.full((T, P * (LAGS + 1)), np.nan)
    for l in range(LAGS + 1):
        X[l:, l * P:(l + 1) * P] = z[:T - l]
    return X


def build(variant, theta, est, moy):
    lt = np.log(np.clip(theta, 1e-6, None))
    if variant == "A":
        blocks = [dev(theta, 12)]
    elif variant == "A_log":
        blocks = [dev(lt, 12)]
    elif variant == "B":
        blocks = [dev(seasonal_adjust(theta, moy, est), 12)]
    elif variant == "C":
        z = np.full(theta.shape, np.nan)
        z[12:] = theta[12:] - theta[:-12]
        blocks = [z]
    elif variant == "D":
        blocks = [dev(theta, 12), dev(theta, 60)]
    elif variant == "D_log":
        blocks = [dev(lt, 12), dev(lt, 60)]
    elif variant == "E":
        blocks = [dev(theta, 12), detrend(theta, est)]
    elif variant == "F":
        X, _ = w2.literature_inputs(theta, est, 3, LAGS)
        return X
    elif variant == "G":
        blocks = [theta]
    else:
        raise ValueError(variant)
    z = np.hstack([standardize(b, est & ~np.isnan(b).any(axis=1)) for b in blocks])
    return with_lags(z)


VARIANTS = ["A", "B", "C", "D", "E", "F", "G", "A_log", "D_log"]


def forecasts(task):
    """Pseudo-out-of-sample forecasts of one variant and one topic model at every origin."""
    variant, _, _, theta, y, origins, publish, moy = task
    T = len(y)
    f = []
    for i in origins:
        est = np.arange(T) <= i
        X = build(variant, theta, est, moy)
        okx = ~np.isnan(X).any(axis=1)
        fit = (np.arange(T) <= i - publish) & ~np.isnan(y) & okx
        c, b, _ = w2.ridge_gcv(X[fit], y[fit])
        f.append(b + X[i] @ c)
    return np.array(f)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", required=True)
    ap.add_argument("--dtm-dir", required=True)
    ap.add_argument("--window-end", type=int, default=1979)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--first-origin", default="1960-01")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--horizons", nargs="+", default=["h0", "h1"], help="h0: s..s+2, h1: s+1..s+3")
    a = ap.parse_args()
    root = common.RESULTS / "w2" / a.grid
    months = sorted(pd.read_csv(Path(a.dtm_dir) / "records.csv.gz", dtype={"month": str}, usecols=["month"]).month.unique())
    last = f"{a.window_end}-12"
    months = [m for m in months if m <= last]  # nothing after the first window's end
    T = len(months)
    pm = pd.PeriodIndex(months, freq="M")
    moy = np.asarray(pm.month)
    origins = np.flatnonzero(np.asarray(pm >= pd.Period(a.first_origin, "M")))
    growth = pd.read_csv(Path(a.dtm_dir) / "monthly_ip.csv", dtype={"month": str}).set_index("month")["growth"]
    rec = np.r_[w2.recession(months)[1:], False]

    thetas = {}
    for model, prefix in [("lda", "lda"), ("lam0", "joint_lam0")]:
        thetas[model] = [np.load(root / f"{prefix}_e{a.window_end}_s{s}" / "theta_selected.npy")[:T] for s in a.seeds]

    rows = []
    for h, (t_start, t_win) in {"h0 (s..s+2)": (0, 3), "h1 (s+1..s+3)": (1, 3)}.items():
        if h[:2] not in a.horizons:
            continue
        y = common.month_targets(months, months[-1], months[-1], t_start, t_win).target.to_numpy().copy()
        y[np.asarray(pm + (t_start + t_win - 1) > pd.Period(last, "M"))] = np.nan  # no outcome after 1979
        publish = t_start + t_win
        ok_o = origins[~np.isnan(y[origins])]
        # Benchmarks: expanding mean and direct AR (lag order by BIC at the first origin, refit at every origin)
        hm = np.array([np.nanmean(y[:i - publish + 1]) for i in ok_o])
        _, p, L = w2.ar_forecast(growth, months, y, np.arange(T) <= ok_o[0] - publish)
        A = np.column_stack([np.ones(T), L[:, :p]])
        okA = ~np.isnan(L[:, :p]).any(axis=1)
        ar = []
        for i in ok_o:
            m = (np.arange(T) <= i - publish) & ~np.isnan(y) & okA
            ar.append(A[i] @ np.linalg.lstsq(A[m], y[m], rcond=None)[0])
        ar = np.array(ar)
        yo = y[ok_o]

        def score(f, sub=None):
            m = np.ones(len(yo), bool) if sub is None else sub
            return 1 - ((yo[m] - f[m]) ** 2).sum() / ((yo[m] - hm[m]) ** 2).sum()

        rows.append({"horizon": h, "variant": "AR", "model": "-", "r2": score(ar), "r2_recession": score(ar, rec[ok_o]),
                     "r2_expansion": score(ar, ~rec[ok_o])})
        tasks = [(v, m, j, theta, y, ok_o, publish, moy) for v in VARIANTS for m, ths in thetas.items()
                 for j, theta in enumerate(ths)]
        with mp.Pool(a.workers) as pool:
            res = dict(zip([(v, m, j) for v, m, j, *_ in tasks], pool.map(forecasts, tasks)))
        for variant in VARIANTS:
            for model, ths in thetas.items():
                per_seed = [res[(variant, model, j)] for j in range(len(ths))]
                fm = np.mean(per_seed, 0)
                rows.append({"horizon": h, "variant": variant, "model": model, "r2": score(fm),
                             "r2_recession": score(fm, rec[ok_o]), "r2_expansion": score(fm, ~rec[ok_o]),
                             "r2_seeds": ", ".join(f"{score(f):.3f}" for f in per_seed)})
                print(f"{h} {variant} {model}: R2 {rows[-1]['r2']:.3f} (seeds {rows[-1]['r2_seeds']})", flush=True)
        pd.DataFrame(rows).to_csv(root / f"screen_inputs_{'_'.join(a.horizons)}.csv", index=False)

    out = pd.DataFrame(rows)
    out.to_csv(root / f"screen_inputs_{'_'.join(a.horizons)}.csv", index=False)
    pd.set_option("display.width", 200)
    for h, g in out.groupby("horizon"):
        avg = g[g.variant != "AR"].groupby("variant")[["r2", "r2_recession", "r2_expansion"]].mean()
        avg = avg.join(g[g.model == "lda"].set_index("variant")["r2"].rename("r2_lda"))
        avg = avg.join(g[g.model == "lam0"].set_index("variant")["r2"].rename("r2_lam0"))
        print(f"\n=== {h}: pseudo-out-of-sample R^2 over {a.first_origin}..{last}, "
              f"against the expanding mean (average of LDA and lam = 0; AR = {g[g.variant == 'AR'].r2.iloc[0]:.3f})")
        print(avg.sort_values("r2", ascending=False).round(3).to_string())


if __name__ == "__main__":
    main()
