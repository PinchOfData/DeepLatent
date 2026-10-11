"""Paths and small helpers shared by the steps of application 3 (WSJ news -> industrial production growth).

Everything lives on scratch: code, logs and results in /scratch/Gauthier/app3_wsj/run, the cleaned corpus, the
document-term matrix and the checkpoints next to it. (Since the 2026-10-10 maintenance the GPU nodes mount the home
directory from a different server than the login and CPU nodes, so files written to /home by one are invisible to the
other; scratch is the same everywhere.) The raw WSJ file is read in place, never copied or changed.
"""
import os
from pathlib import Path

HOME = Path(os.environ.get("APP3_HOME", "/scratch/Gauthier/app3_wsj/run"))
SCRATCH = Path(os.environ.get("APP3_SCRATCH", "/scratch/Gauthier/app3_wsj"))
RAW_WSJ = Path("/home/Gauthier/deeplatent/data/WSJ_articles_full.csv")
BYBEE_VOCAB = Path("/home/Gauthier/deeplatent/data/bybee_unigrams.txt")
INDPRO = Path("/home/Gauthier/deeplatent/data/fred/INDPRO.csv")

DATA = SCRATCH / "data"  # records.csv.gz (01), dtm/ (02)
CHECKPOINTS = SCRATCH / "checkpoints"
RESULTS = HOME / "results"


def read_words(path: Path) -> list:
    return [l.strip().lower() for l in open(path) if l.strip()]


def month_targets(months: list, train_end: str, val_end: str, start: int, window: int):
    """Split and target of each article month s: the average monthly growth of industrial production (100 x log
    change) over months s+start .. s+start+window-1. Months are split by article month (training up to train_end,
    validation up to val_end, test after); a training or validation month whose target window reaches into the next
    period gets no target (NaN), so no outcome crosses a split. Returns a DataFrame (month, split, target)."""
    import numpy as np
    import pandas as pd

    ip = pd.read_csv(DATA / "dtm" / "monthly_ip.csv", dtype={"month": str}).set_index("month")["growth"]
    allm = list(ip.index)
    pos = {m: i for i, m in enumerate(allm)}
    rows = []
    for m in months:
        split = "train" if m <= train_end else "val" if m <= val_end else "test"
        win = allm[pos[m] + start: pos[m] + start + window]
        y = float(ip.loc[win].mean()) if len(win) == window and ip.loc[win].notna().all() else np.nan
        end = {"train": train_end, "val": val_end}.get(split)
        if end is not None and str(pd.Period(m, "M") + start + window - 1) > end:
            y = np.nan  # its target would use the next period's outcomes
        rows.append((m, split, y))
    return pd.DataFrame(rows, columns=["month", "split", "target"])


def head_features(theta, kind: str = "level", window: int = 12):
    """Monthly inputs of the prediction, from the monthly attention theta (months x K, consecutive months in order).

    kind = "level": the attention itself. kind = "change": the attention minus its average over the previous `window`
    months (news relative to its recent past: topic attention measures how much the news is about a subject, a level,
    while the target is a growth rate; and attention drifts out of its training range over the years). Months without
    `window` earlier months have no inputs (NaN rows, has_history False). Returns (features, has_history)."""
    import numpy as np

    T = theta.shape[0]
    if kind == "level":
        return theta, np.ones(T, dtype=bool)
    assert kind == "change", kind
    cs = np.vstack([np.zeros((1, theta.shape[1])), np.cumsum(theta, axis=0)])
    t = np.arange(window, T)
    out = np.full(theta.shape, np.nan)
    out[window:] = theta[window:] - (cs[t] - cs[t - window]) / window
    return out, np.arange(T) >= window


def assert_consecutive(months: list) -> None:
    import pandas as pd

    p = pd.PeriodIndex(months, freq="M")
    assert ((p[1:] - p[:-1]).map(lambda d: d.n) == 1).all(), "months are not consecutive"
