from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import matplotlib.figure


# ---------------------------- semantic colors ----------------------------

OBSERVED_COLOR = "black"  # observed / true
PLACEBO_COLOR = "grey"  # placebo / null

DEMOCRAT_COLOR = "tab:blue"
REPUBLICAN_COLOR = "tab:red"


def party_color(party: int | str) -> str:
    """Return a semantically meaningful color for party."""

    if isinstance(party, str):
        key = party.strip().lower()
        if key in {"d", "democrat", "democrats", "dem"}:
            return DEMOCRAT_COLOR
        if key in {"r", "republican", "republicans", "gop"}:
            return REPUBLICAN_COLOR
    else:
        if party == 0:
            return DEMOCRAT_COLOR
        if party == 1:
            return REPUBLICAN_COLOR

    # Fallback: still deterministic and readable.
    return "tab:purple"


# ---------------------------- semantic labels ----------------------------

SCENARIO_LABELS: dict[str, str] = {
    "fit_party_none": "Placebo: Party does not affect topic prevalence and content.",
    "fit_party_prev_only": "Party affects topic prevalence only.",
    "fit_party_cont_only": "Party affects content only.",
    "fit_party_both": "Party affects topic prevalence and content.",
}


def pretty_scenario_label(scenario: str) -> str:
    """Make scenario labels intelligible and publication-ready."""

    if scenario in SCENARIO_LABELS:
        return SCENARIO_LABELS[scenario]

    # Generic fallback: snake_case -> Sentence.
    label = scenario.replace("_", " ").strip()
    if not label:
        return scenario
    return label[0].upper() + label[1:] + ("." if not label.endswith(".") else "")


SCENARIO_COLORS: dict[str, str] = {
    # Convention: observed/true = black, placebo = grey.
    "fit_party_both": OBSERVED_COLOR,
    "fit_party_none": PLACEBO_COLOR,
    # Remaining scenarios use distinct, standard matplotlib colors.
    "fit_party_prev_only": "tab:green",
    "fit_party_cont_only": "tab:orange",
}


def scenario_color(scenario: str) -> str:
    return SCENARIO_COLORS.get(scenario, "tab:purple")


# ---------------------------- figure helpers ----------------------------


def _unique_legend_handles_labels(
    handles: Sequence,
    labels: Sequence[str],
) -> tuple[list, list[str]]:
    dedup: "OrderedDict[str, object]" = OrderedDict()
    for h, lab in zip(handles, labels):
        if lab and lab not in dedup:
            dedup[lab] = h
    return list(dedup.values()), list(dedup.keys())


def place_legend_bottom_center(
    fig: matplotlib.figure.Figure,
    axes: Iterable | None = None,
    *,
    ncol: int | None = None,
    fontsize: int | None = None,
    frameon: bool = False,
    y_offset: float = -0.02,
    bottom_margin: float = 0.22,
) -> None:
    """Place a single legend at bottom-center, outside the figure.

    This removes per-axis legends and promotes all labeled artists to a single
    figure-level legend.
    """

    axs = list(axes) if axes is not None else list(fig.axes)

    handles: list = []
    labels: list[str] = []
    for ax in axs:
        h, l = ax.get_legend_handles_labels()
        handles.extend(h)
        labels.extend(l)
        leg = ax.get_legend()
        if leg is not None:
            leg.remove()

    handles_u, labels_u = _unique_legend_handles_labels(handles, labels)
    if not labels_u:
        return

    if ncol is None:
        ncol = min(4, len(labels_u))

    # Layout first, then legend, then reserve margin for legend.
    fig.tight_layout()
    fig.legend(
        handles_u,
        labels_u,
        loc="lower center",
        bbox_to_anchor=(0.5, y_offset),
        ncol=ncol,
        frameon=frameon,
        fontsize=fontsize,
    )
    fig.subplots_adjust(bottom=bottom_margin)


def savefig_publishable(
    fig: matplotlib.figure.Figure,
    out_path: str | Path,
    *,
    dpi: int = 300,
) -> None:
    """Save with settings that keep outside legends and avoid clipping."""

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", pad_inches=0.05)


# ---------------------------- additional utilities ----------------------------


def set_all_seeds(seed: int) -> None:
    """Set random seeds for reproducibility."""
    import numpy as np
    import torch
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def smooth_series(y, window: int = 3):
    """Apply moving average smoothing to a series."""
    import numpy as np
    if window <= 1:
        return y
    y = np.asarray(y)
    kernel = np.ones(window) / window
    return np.convolve(y, kernel, mode="same")


def add_confidence_band(
    ax,
    x,
    y_mean,
    y_lo,
    y_hi,
    *,
    color: str = "tab:blue",
    alpha: float = 0.2,
    label: str | None = None,
    line_kwargs: dict | None = None,
) -> None:
    """Add a line plot with confidence band to an axis."""
    import numpy as np
    x = np.asarray(x)
    y_mean = np.asarray(y_mean)
    y_lo = np.asarray(y_lo)
    y_hi = np.asarray(y_hi)

    line_kwargs = line_kwargs or {}
    ax.plot(x, y_mean, color=color, label=label, **line_kwargs)
    ax.fill_between(x, y_lo, y_hi, color=color, alpha=alpha)


def create_diverging_colormap(
    name: str = "custom_diverging",
    n_colors: int = 256,
) -> "matplotlib.colors.Colormap":
    """Create a diverging colormap centered at white."""
    import matplotlib.colors as mcolors
    import numpy as np

    # Blue -> White -> Red
    colors = [
        (0.0, DEMOCRAT_COLOR),
        (0.5, "white"),
        (1.0, REPUBLICAN_COLOR),
    ]

    cmap = mcolors.LinearSegmentedColormap.from_list(name, [
        (0.0, mcolors.to_rgb(DEMOCRAT_COLOR)),
        (0.5, (1.0, 1.0, 1.0)),
        (1.0, mcolors.to_rgb(REPUBLICAN_COLOR)),
    ])

    return cmap


def annotate_bars(
    ax,
    bars,
    *,
    fmt: str = "{:.3f}",
    fontsize: int = 9,
    offset: tuple = (0, 3),
) -> None:
    """Add value annotations on top of bar chart bars."""
    for bar in bars:
        height = bar.get_height()
        ax.annotate(
            fmt.format(height),
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=offset,
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=fontsize,
        )


def format_axis_percent(ax, axis: str = "y") -> None:
    """Format axis ticks as percentages."""
    import matplotlib.ticker as mticker
    formatter = mticker.PercentFormatter(xmax=1.0)
    if axis in ("y", "both"):
        ax.yaxis.set_major_formatter(formatter)
    if axis in ("x", "both"):
        ax.xaxis.set_major_formatter(formatter)


def style_for_publication(
    fig=None,
    ax=None,
    *,
    grid_alpha: float = 0.3,
    spine_visible: bool = True,
) -> None:
    """Apply publication-ready styling to figure/axes."""
    if ax is not None:
        ax.grid(alpha=grid_alpha)
        if not spine_visible:
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
    if fig is not None:
        fig.tight_layout()
