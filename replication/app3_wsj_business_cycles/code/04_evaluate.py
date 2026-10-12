"""Step 04: compare runs on prediction and on topic quality.

Prediction, for every run (joint models of 03_fit_joint.py and LDA of 03_fit_lda.py), from its monthly topic
attention thetabar (theta_<which>.npy):
  - two-step regression: LASSO (and ridge) of the target on thetabar over the training months, features standardized
    on the training months, penalty chosen by the validation months' squared error (the test months never choose);
  - for joint runs, also the model's own head (prediction_<which>.csv).
Validation and test R^2 are out of sample, relative to the training mean (1 - SSE / sum of squares around the
training mean of the target).

Topic quality, from beta_<which>.npy (K x V topic-word probabilities):
  - diversity: share of distinct terms among the --diversity-n most probable terms of all topics (Dieng et al. 2020);
  - coherence: mean NPMI of the pairs among each topic's --npmi-n most probable terms, from document co-occurrence
    in the reference records (all validation and test records: text no topic model was fit on);
  - the topics' average training-period attention and the number of topics with at least 0.5% of it.

Targets and splits are those of the runs (their config.json: target window, split dates; every run must agree, and a
run's months.csv and attention must cover the same months), so the head is scored on the target it was trained for.
The two-step regressions use the head's inputs of the joint runs (--features level or change, from their configs, the
same for all; common.head_features) for every run, LDA included; months without the features' history are left
out of every score.

--lags L gives the two-step regressions the inputs of months t, t-1, ..., t-L (the own head is unchanged); months
without that history are left out of every score.

Writes results/summary_<which>_<features>[_lags<L>].csv (or --out-name) and
results/<tag>/regression_<which>_<features>[_lags<L>].csv.

Usage: python 04_evaluate.py --tags lda_k50 joint_k50_lam0 joint_k50_lam1e4 ... [--which selected --lags 6]
"""
import argparse
import json

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.linear_model import Lasso, Ridge

import common


def with_lags(z, lags: int):
    """Months x (P (lags + 1)): the inputs of months t, t-1, ..., t-lags side by side (NaN without that history)."""
    T, P = z.shape
    out = np.full((T, P * (lags + 1)), np.nan)
    for l in range(lags + 1):
        out[l:, l * P:(l + 1) * P] = z[:T - l]
    return out


def oos_r2(y, yhat, bench):
    return float(1 - ((y - yhat) ** 2).sum() / ((y - bench) ** 2).sum())


def two_step(theta, mt, model: str):
    ok = mt.target.notna() & ~np.isnan(theta).any(axis=1)
    tr = (mt.split == "train") & ok
    va = (mt.split == "val") & ok
    mu, sd = theta[tr].mean(0), theta[tr].std(0) + 1e-12
    Z = np.nan_to_num((theta - mu) / sd)
    y = mt.target.to_numpy()
    grid = np.logspace(-4, 1, 51) if model == "lasso" else np.logspace(-3, 5, 81)
    best = None
    for alpha in grid:
        est = (Lasso(alpha=alpha, max_iter=100_000) if model == "lasso" else Ridge(alpha=alpha)).fit(Z[tr], y[tr])
        err = ((y[va] - est.predict(Z[va])) ** 2).mean()
        if best is None or err < best[0]:
            best = (err, alpha, est)
    _, alpha, est = best
    yhat = np.where(ok, est.predict(Z), np.nan)
    return yhat, alpha, int((np.abs(est.coef_) > 0).sum())


def scores(y, yhat, mt):
    bench = mt.target[(mt.split == "train") & mt.target.notna()].mean()
    out = {}
    for s in ["train", "val", "test"]:
        m = ((mt.split == s) & mt.target.notna()).to_numpy() & ~np.isnan(yhat)
        out[f"{s}_r2"] = oos_r2(y[m], yhat[m], bench)
        out[f"{s}_corr"] = float(np.corrcoef(y[m], yhat[m])[0, 1])
    return out


def npmi(beta, ref, n):
    top = np.argsort(-beta, axis=1)[:, :n]
    cols = np.unique(top)
    R = ref[:, cols]
    D = ref.shape[0]
    df = np.asarray(R.sum(axis=0)).ravel() / D
    co = (R.T @ R).toarray() / D
    pos = {c: i for i, c in enumerate(cols)}
    out = []
    for k in range(beta.shape[0]):
        vals = []
        for a in range(n):
            for b in range(a + 1, n):
                i, j = pos[top[k, a]], pos[top[k, b]]
                p = co[i, j]
                vals.append(-1.0 if p == 0 else np.log(p / (df[i] * df[j])) / -np.log(p))
        out.append(np.mean(vals))
    return np.array(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="+", required=True)
    ap.add_argument("--which", default="selected")
    ap.add_argument("--diversity-n", type=int, default=25)
    ap.add_argument("--npmi-n", type=int, default=10)
    ap.add_argument("--lags", type=int, default=0, help="regressions on the inputs of months t .. t-lags")
    ap.add_argument("--out-name", default=None, help="summary file name (default summary_<which>_<features>...)")
    a = ap.parse_args()
    dtm = common.DATA / "dtm"
    rec = pd.read_csv(dtm / "records.csv.gz", dtype={"month": str})
    months = sorted(rec.month.unique())
    keys = ["train_end", "val_end", "target_start", "target_window"]
    defaults = {"train_end": "1999-12", "val_end": "2003-12", "target_start": 0, "target_window": 3}
    settings, train_ends = set(), set()
    for tag in a.tags:
        cfg = json.loads((common.RESULTS / tag / "config.json").read_text())
        train_ends.add(cfg.get("train_end", defaults["train_end"]))
        if "lam" in cfg:  # LDA has no target: the joint runs set it (or the defaults, with LDA alone)
            settings.add(tuple(cfg.get(k, defaults[k]) for k in keys))
        got = pd.read_csv(common.RESULTS / tag / "months.csv", dtype={"month": str}).month.tolist()
        assert got == months, f"{tag}: its months differ from the document-term matrix's"
    settings = settings or {tuple(defaults[k] for k in keys)}
    assert len(settings) == 1, f"runs disagree on {keys}: {settings}"
    a.train_end, a.val_end, a.target_start, a.target_window = settings.pop()
    assert train_ends == {a.train_end}, f"runs disagree on the training period: {train_ends}"
    feat = {(c.get("features", "level"), c.get("change_window", 12)) for c in
            (json.loads((common.RESULTS / t / "config.json").read_text()) for t in a.tags) if "lam" in c}
    assert len(feat) <= 1, f"joint runs disagree on the head's inputs: {feat}"
    a.features, a.change_window = feat.pop() if feat else ("level", 12)
    if a.features == "change":
        common.assert_consecutive(months)
    zero = common.head_features(np.zeros((len(months), 1)), a.features, a.change_window)[0]
    hist = ~np.isnan(with_lags(zero, a.lags)).any(axis=1)
    suffix = f"{a.features}" + (f"_lags{a.lags}" if a.lags else "")
    mt = common.month_targets(months, a.train_end, a.val_end, a.target_start, a.target_window)
    mt.loc[~hist, "target"] = np.nan
    print(f"head inputs: {a.features}" + (f" (window {a.change_window})" if a.features == "change" else "")
          + f"; regressions on months t .. t-{a.lags}")
    print(f"targets: window {a.target_window} from {a.target_start}; train to {a.train_end}, validation to "
          f"{a.val_end}", flush=True)
    X = sp.load_npz(dtm / "X.npz").tocsr()
    ref = (X[np.flatnonzero((rec.month > a.train_end).to_numpy())] > 0).astype(np.float32).tocsc()
    print(f"reference records for NPMI: {ref.shape[0]:,}", flush=True)
    y = mt.target.to_numpy()

    rows = []
    for tag in a.tags:
        res = common.RESULTS / tag
        cfg = json.loads((res / "config.json").read_text())
        theta = np.load(res / f"theta_{a.which}.npy")
        assert theta.shape[0] == len(months), f"{tag}: attention for {theta.shape[0]} months, expected {len(months)}"
        attention = theta
        theta = with_lags(common.head_features(theta, a.features, a.change_window)[0], a.lags)
        beta = np.load(res / f"beta_{a.which}.npy").astype(np.float64)
        K = beta.shape[0]
        row = {"tag": tag, "lam": cfg.get("lam", np.nan), "two_step_head": cfg.get("two_step_head", False),
               "features": a.features, "lags": a.lags, "K": K,
               "step": (res / f"{a.which}_step.txt").read_text().strip() if (res / f"{a.which}_step.txt").exists()
               else ""}
        reg = pd.DataFrame({"month": months, "split": mt.split, "target": y})
        for model in ["lasso", "ridge"]:
            yhat, alpha, nz = two_step(theta, mt, model)
            reg[model] = yhat
            row.update({f"{model}_{k}": v for k, v in scores(y, yhat, mt).items()})
            row[f"{model}_alpha"], row[f"{model}_nonzero"] = alpha, nz
        pred = res / f"predictions_{a.which}.csv"
        if pred.exists() and cfg.get("lam", 0) > 0:
            head = pd.read_csv(pred, dtype={"month": str}).set_index("month").reindex(months)["prediction"]
            reg["head"] = head.to_numpy()
            row.update({f"head_{k}": v for k, v in scores(y, head.to_numpy(), mt).items()})
        reg.to_csv(res / f"regression_{a.which}_{suffix}.csv", index=False)
        top = np.argsort(-beta, axis=1)[:, :a.diversity_n]
        row["diversity"] = len(np.unique(top)) / top.size
        coh = npmi(beta, ref, a.npmi_n)
        row["npmi"] = float(coh.mean())
        share = attention[(mt.split == "train").to_numpy()].mean(0)
        row["topics_used"] = int((share >= 0.005).sum())
        rows.append(row)
        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)
    out = pd.DataFrame(rows)
    out.to_csv(common.RESULTS / (a.out_name or f"summary_{a.which}_{suffix}.csv"), index=False)
    cols = ["tag", "lam", "two_step_head", "step", "head_val_r2", "head_test_r2", "lasso_val_r2", "lasso_test_r2", "ridge_test_r2",
            "diversity", "npmi", "topics_used"]
    print(out[[c for c in cols if c in out]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
