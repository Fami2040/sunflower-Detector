from __future__ import annotations

from typing import Any, Iterable

from harchoc.stats_ci import bin_reliability, expected_calibration_error


def reliability_and_ece(
    scores: Iterable[float],
    correct_flags: Iterable[float | bool],
    *,
    n_bins: int = 10,
) -> dict[str, Any]:
    """Reliability bins + scalar ECE for JSON reports."""
    bins = bin_reliability(scores, correct_flags, n_bins=n_bins)
    ece = expected_calibration_error(bins)
    return {
        "n_bins": n_bins,
        "ece": ece,
        "reliability": bins.to_json(),
    }
