"""Journal-style matplotlib defaults for manuscript figures (MS-FIG-NORM)."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator

FIGURE_DPI = 300

# Sans-serif stack: Arial/Helvetica where available, else DejaVu (bundled with mpl).
FONT_FAMILY = "sans-serif"
FONT_SANS_SERIF = ("Arial", "Helvetica", "DejaVu Sans", "Liberation Sans")

JOURNAL_RCPARAMS: dict[str, Any] = {
    "figure.dpi": FIGURE_DPI,
    "savefig.dpi": FIGURE_DPI,
    "font.family": FONT_FAMILY,
    "font.sans-serif": list(FONT_SANS_SERIF),
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "axes.linewidth": 0.8,
    "lines.linewidth": 1.25,
    "grid.linewidth": 0.6,
    "figure.autolayout": False,
}


def panel_label(index: int) -> str:
    """Return A, B, …, Z, AA, … for 0-based panel index."""
    if index < 0:
        raise ValueError(f"panel index must be >= 0, got {index}")
    n = index + 1
    label = ""
    while n:
        n, rem = divmod(n - 1, 26)
        label = chr(ord("A") + rem) + label
    return label


def savefig_kwargs(*, journal_style: bool) -> dict[str, Any]:
    """Keyword args for ``Figure.savefig`` aligned with journal export."""
    if journal_style:
        return {"dpi": FIGURE_DPI, "bbox_inches": "tight", "facecolor": "white"}
    return {"dpi": 120, "bbox_inches": "tight"}


def apply_journal_rcparams() -> None:
    """Set matplotlib rcParams for journal figures (call after ``matplotlib.use``)."""
    import matplotlib as mpl  # type: ignore

    mpl.rcParams.update(JOURNAL_RCPARAMS)


@contextmanager
def journal_style_context(*, enabled: bool = True) -> Iterator[None]:
    """Temporarily apply journal rcParams; restore previous values on exit."""
    if not enabled:
        yield
        return
    import matplotlib as mpl  # type: ignore

    prior = mpl.rcParams.copy()
    apply_journal_rcparams()
    try:
        yield
    finally:
        mpl.rcParams.update(prior)


def add_panel_label(
    ax: Any,
    label: str,
    *,
    x: float = -0.12,
    y: float = 1.06,
    fontsize: float = 11,
) -> None:
    """Bold panel letter (A/B/C) above subplot axes, journal style."""
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        fontsize=fontsize,
        fontweight="bold",
        va="bottom",
        ha="left",
        clip_on=False,
    )


def prepare_matplotlib(*, journal_style: bool) -> None:
    """Select Agg backend and optional journal rcParams."""
    import matplotlib

    matplotlib.use("Agg")
    if journal_style:
        apply_journal_rcparams()
