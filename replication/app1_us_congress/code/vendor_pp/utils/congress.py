"""Congress session utilities for mapping dates to congress numbers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


SESSIONS = [
    (103, "1993-01-05", "1995-01-04"),
    (104, "1995-01-04", "1997-01-07"),
    (105, "1997-01-07", "1999-01-06"),
    (106, "1999-01-06", "2001-01-03"),
    (107, "2001-01-03", "2003-01-07"),
    (108, "2003-01-07", "2005-01-04"),
    (109, "2005-01-04", "2007-01-04"),
    (110, "2007-01-04", "2009-01-06"),
    (111, "2009-01-06", "2011-01-05"),
    (112, "2011-01-05", "2013-01-03"),
    (113, "2013-01-03", "2015-01-06"),
    (114, "2015-01-06", "2017-01-03"),
    (115, "2017-01-03", "2019-01-03"),
    (116, "2019-01-03", "2021-01-03"),
    (117, "2021-01-03", "2023-01-03"),
    (118, "2023-01-03", "2025-01-03"),
    (119, "2025-01-03", "2027-01-03"),
]


@dataclass(frozen=True)
class Session:
    congress: int
    start: pd.Timestamp
    end: pd.Timestamp


def build_sessions(sessions: Iterable[tuple[int, str, str]] = SESSIONS) -> list[Session]:
    """Build Session objects from (congress, start_date, end_date) tuples."""
    result: list[Session] = []
    for congress, start, end in sessions:
        result.append(
            Session(
                congress=int(congress),
                start=pd.to_datetime(start),
                end=pd.to_datetime(end),
            )
        )
    result = sorted(result, key=lambda s: s.start)
    for i in range(1, len(result)):
        if result[i - 1].end != result[i].start:
            raise ValueError(
                f"Non-contiguous sessions at {result[i-1].congress}->{result[i].congress}: "
                f"{result[i-1].end} != {result[i].start}"
            )
    return result


def assign_congress(dates: pd.Series, sessions: list[Session] | None = None) -> pd.Series:
    """Assign congress numbers to dates based on session boundaries."""
    if sessions is None:
        sessions = build_sessions()

    if dates.empty:
        return pd.Series([], dtype="Int64")

    starts = np.array([s.start.to_datetime64() for s in sessions])
    ends = np.array([s.end.to_datetime64() for s in sessions])
    congresses = np.array([s.congress for s in sessions], dtype=np.int32)

    d = dates.to_numpy(dtype="datetime64[ns]", copy=False)
    idx = np.searchsorted(starts, d, side="right") - 1
    valid = (idx >= 0) & (d < ends[np.clip(idx, 0, len(ends) - 1)])
    out = np.full(shape=len(d), fill_value=np.nan, dtype=np.float64)
    out[valid] = congresses[idx[valid]]
    return pd.Series(out, index=dates.index, dtype="Int64")
