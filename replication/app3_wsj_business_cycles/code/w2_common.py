"""Shared pieces of the second wave of application 3: an expanding-window evaluation with the topic models refit
every few years, the literature's handling of topic shares as forecasting inputs, benchmarks and tests.

Windows. A window ending in year E fits every topic model on the records up to E - val_years (training months),
chooses its settings on the next val_years years (validation months), and forecasts the months of years E+1 ..
E+step (the window's block). Concatenating the blocks gives one out-of-sample series per method.

Inputs (as Thorsrud 2020 and Ellingsen, Larsen & Thorsrud 2022 handle topic time series): each topic's monthly
attention is smoothed with a backward-looking moving average, a linear trend is removed and the result standardized,
the trend and the scale being estimated on the estimation months only and extended to later months; the inputs of
month t are those of months t, t-1, ..., t-lags (Ellingsen et al. use 6 lags of every predictor).
"""
import numpy as np
import pandas as pd

# NBER business cycle peaks and troughs (months); recession months run from the month after a peak to the trough.
NBER = [("1948-11", "1949-10"), ("1953-07", "1954-05"), ("1957-08", "1958-04"), ("1960-04", "1961-02"),
        ("1969-12", "1970-11"), ("1973-11", "1975-03"), ("1980-01", "1980-07"), ("1981-07", "1982-11"),
        ("1990-07", "1991-03"), ("2001-03", "2001-11"), ("2007-12", "2009-06")]


def recession(months) -> np.ndarray:
    p = pd.PeriodIndex(months, freq="M")
    out = np.zeros(len(p), dtype=bool)
    for peak, trough in NBER:
        out |= (p > pd.Period(peak, "M")) & (p <= pd.Period(trough, "M"))
    return out


def windows(first_end: int = 1979, step: int = 3, last_year: int = 2011, val_years: int = 5) -> list:
    """The windows: end year, last training month, last validation month, the block's first and last months."""
    out = []
    for end in range(first_end, last_year, step):
        out.append({"end": end, "train_end": f"{end - val_years}-12", "val_end": f"{end}-12",
                    "block_start": f"{end + 1}-01", "block_end": f"{min(end + step, last_year)}-12"})
    return out


def literature_inputs(theta: np.ndarray, est: np.ndarray, smooth: int = 3, lags: int = 6):
    """Forecasting inputs from the monthly attention theta (months x K, consecutive months).

    Per topic: backward moving average over `smooth` months; minus a linear trend (OLS on a constant and the month
    index over the estimation months `est`, a boolean mask); divided by the standard deviation of the detrended
    series over the estimation months; then the values of months t, t-1, ..., t-lags side by side. Returns
    (X: months x K*(lags+1), ok: months with the full history the smoothing and the lags need)."""
    T, K = theta.shape
    sm = pd.DataFrame(theta).rolling(smooth, min_periods=smooth).mean().to_numpy() if smooth > 1 else theta.copy()
    t = np.arange(T, dtype=float)
    fit = est & ~np.isnan(sm).any(axis=1)
    A = np.column_stack([np.ones(fit.sum()), t[fit]])
    coef = np.linalg.lstsq(A, sm[fit], rcond=None)[0]  # 2 x K
    resid = sm - (coef[0][None, :] + t[:, None] * coef[1][None, :])
    sd = resid[fit].std(axis=0)
    z = resid / np.where(sd > 0, sd, 1.0)
    X = np.full((T, K * (lags + 1)), np.nan)
    for l in range(lags + 1):
        X[l:, l * K:(l + 1) * K] = z[:T - l]
    ok = ~np.isnan(X).any(axis=1)
    return X, ok


def literature_inputs_t(att, est_idx, smooth: int = 3, lags: int = 6):
    """literature_inputs for a torch tensor att (months x K), the trend and the scale held constant in the gradient
    (computed from att detached); est_idx: indices of the estimation months. Rows without full history are zeros."""
    import torch

    T, K = att.shape
    if smooth > 1:
        win = att.unfold(0, smooth, 1).mean(-1)  # months smooth-1 .. T-1
        sm = torch.cat([torch.zeros_like(att[:smooth - 1]), win])
    else:
        sm = att
    first = smooth - 1
    est_idx = est_idx[est_idx >= first]
    with torch.no_grad():
        t = torch.arange(T, device=att.device, dtype=att.dtype)
        A = torch.stack([torch.ones_like(t[est_idx]), t[est_idx]], 1)
        coef = torch.linalg.lstsq(A, sm[est_idx].detach()).solution  # 2 x K
        trend = coef[0][None, :] + t[:, None] * coef[1][None, :]
        sd = (sm[est_idx].detach() - trend[est_idx]).std(0, unbiased=False).clamp_min(1e-12)
    z = (sm - trend) / sd
    z = torch.cat([torch.zeros_like(z[:first]), z[first:]])
    cols = [torch.cat([torch.zeros_like(z[:l]), z[:T - l]]) for l in range(lags + 1)]
    return torch.cat(cols, 1)


def history_mask(T: int, smooth: int = 3, lags: int = 6) -> np.ndarray:
    return np.arange(T) >= (smooth - 1) + lags


def ar_forecast(growth: pd.Series, months: list, target: np.ndarray, fit_mask: np.ndarray, max_p: int = 12,
                pub_lag: int = 1):
    """Direct autoregressive forecast of the target of article month s from the monthly growth published by then,
    g_{s-pub_lag}, ..., g_{s-pub_lag-p+1} (the industrial production of month s-1 comes out in month s). The lag
    order p minimizes the BIC over the months of fit_mask (the same months for every p). Returns (forecasts, p)."""
    g = growth.reindex(pd.PeriodIndex(months, freq="M").strftime("%Y-%m"))
    allg = growth.copy()
    pos = {m: i for i, m in enumerate(allg.index)}
    L = np.full((len(months), max_p), np.nan)
    for i, m in enumerate(g.index):
        j = pos[m] - pub_lag
        for l in range(max_p):
            if j - l >= 0:
                L[i, l] = allg.iloc[j - l]
    usable = fit_mask & ~np.isnan(target) & ~np.isnan(L).any(axis=1)
    best = None
    for p in range(1, max_p + 1):
        A = np.column_stack([np.ones(usable.sum()), L[usable, :p]])
        coef, *_ = np.linalg.lstsq(A, target[usable], rcond=None)
        rss = ((target[usable] - A @ coef) ** 2).sum()
        n = usable.sum()
        bic = n * np.log(rss / n) + (p + 1) * np.log(n)
        if best is None or bic < best[0]:
            best = (bic, p, coef)
    _, p, coef = best
    pred = coef[0] + L[:, :p] @ coef[1:]
    return pred, p, L


def newey_west_var(d: np.ndarray, lags: int) -> float:
    d = d - d.mean()
    n = len(d)
    v = (d @ d) / n
    for l in range(1, lags + 1):
        w = 1 - l / (lags + 1)
        v += 2 * w * (d[l:] @ d[:-l]) / n
    return v


def diebold_mariano(e1: np.ndarray, e2: np.ndarray, lags: int = 3):
    """DM test of equal squared-error loss (e1 vs e2), Newey-West variance with `lags` lags (the 3-month target
    makes errors overlap). Positive statistic: model 2 is more accurate. Returns (stat, two-sided p)."""
    from scipy.stats import norm

    d = np.asarray(e1, float) ** 2 - np.asarray(e2, float) ** 2
    stat = d.mean() / np.sqrt(newey_west_var(d, lags) / len(d))
    return float(stat), float(2 * (1 - norm.cdf(abs(stat))))


def clark_west(y: np.ndarray, f_small: np.ndarray, f_big: np.ndarray, lags: int = 3):
    """Clark-West (2007) test that the larger (nesting) model adds predictive accuracy over the smaller one.
    Returns (stat, one-sided p)."""
    from scipy.stats import norm

    y, f_small, f_big = (np.asarray(v, float) for v in (y, f_small, f_big))
    f = (y - f_small) ** 2 - ((y - f_big) ** 2 - (f_small - f_big) ** 2)
    stat = f.mean() / np.sqrt(newey_west_var(f, lags) / len(f))
    return float(stat), float(1 - norm.cdf(stat))
