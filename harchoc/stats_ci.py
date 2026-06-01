from __future__ import annotations

import statistics
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Literal, Sequence

StatName = Literal["mean", "median"]


@dataclass(frozen=True)
class CiResult:
    """Confidence interval for a scalar statistic."""

    stat: str
    point: float
    low: float
    high: float
    confidence: float
    method: str
    n: int

    def to_json(self) -> dict[str, Any]:
        return {
            "stat": self.stat,
            "point": self.point,
            "low": self.low,
            "high": self.high,
            "confidence": self.confidence,
            "method": self.method,
            "n": self.n,
        }


def _as_floats(values: Iterable[float]) -> list[float]:
    out: list[float] = []
    for v in values:
        try:
            out.append(float(v))
        except (TypeError, ValueError):
            continue
    return out


def _stat_fn(name: StatName) -> Callable[[Sequence[float]], float]:
    if name == "median":
        return statistics.median
    return statistics.fmean


def percentile_ci(
    values: Iterable[float],
    *,
    stat: StatName = "mean",
    confidence: float = 0.95,
) -> CiResult | None:
    """
    Normal-approximation CI using sample std (fallback for tiny n or no scipy).
    Returns None when n < 2.
    """
    xs = _as_floats(values)
    n = len(xs)
    if n < 2:
        return None
    fn = _stat_fn(stat)
    point = float(fn(xs))
    if n < 3 or stat == "median":
        # Use min/max spread for median or very small n.
        low, high = float(min(xs)), float(max(xs))
        return CiResult(
            stat=stat,
            point=point,
            low=low,
            high=high,
            confidence=confidence,
            method="min_max",
            n=n,
        )
    sd = float(statistics.stdev(xs))
    # z for 95% ≈ 1.96; generalize via scipy if needed — keep simple for CI-light paths.
    z = 1.96 if confidence >= 0.95 else 1.645
    half = z * sd / (n**0.5)
    return CiResult(
        stat=stat,
        point=point,
        low=point - half,
        high=point + half,
        confidence=confidence,
        method="normal_approx",
        n=n,
    )


def bootstrap_ci(
    values: Iterable[float],
    *,
    stat: StatName = "mean",
    n_resamples: int = 999,
    confidence: float = 0.95,
    random_state: int | None = 0,
) -> CiResult:
    """
    Bootstrap percentile CI using scipy.stats.bootstrap (requires scipy).
    """
    xs = _as_floats(values)
    if not xs:
        raise ValueError("bootstrap_ci requires at least one value")
    fn = _stat_fn(stat)
    point = float(fn(xs))
    if len(xs) < 2:
        return CiResult(
            stat=stat,
            point=point,
            low=point,
            high=point,
            confidence=confidence,
            method="degenerate",
            n=len(xs),
        )

    import numpy as np
    from scipy import stats as sp_stats  # type: ignore

    arr = np.asarray(xs, dtype=float)

    if stat == "median":
        statistic = np.median  # type: ignore[assignment]
    else:
        statistic = np.mean  # type: ignore[assignment]

    res = sp_stats.bootstrap(
        (arr,),
        statistic=statistic,
        n_resamples=int(n_resamples),
        confidence_level=float(confidence),
        method="percentile",
        random_state=random_state,
    )
    low, high = float(res.confidence_interval.low), float(res.confidence_interval.high)
    return CiResult(
        stat=stat,
        point=point,
        low=low,
        high=high,
        confidence=confidence,
        method="bootstrap_percentile",
        n=len(xs),
    )


def ci_for_values(
    values: Iterable[float],
    *,
    stat: StatName = "mean",
    confidence: float = 0.95,
    n_resamples: int = 999,
    prefer_bootstrap: bool = True,
) -> CiResult | None:
    """Pick bootstrap when scipy is available and n>=2; else percentile fallback."""
    xs = _as_floats(values)
    if not xs:
        return None
    if len(xs) < 2:
        p = float(xs[0])
        return CiResult(
            stat=stat,
            point=p,
            low=p,
            high=p,
            confidence=confidence,
            method="single_sample",
            n=1,
        )
    if prefer_bootstrap and len(xs) >= 5:
        try:
            return bootstrap_ci(
                xs, stat=stat, n_resamples=n_resamples, confidence=confidence
            )
        except (ImportError, ValueError):
            pass
    return percentile_ci(xs, stat=stat, confidence=confidence)


@dataclass(frozen=True)
class ReliabilityBins:
    """Binned scores for reliability / ECE computation."""

    bin_edges: tuple[float, ...]
    bin_counts: tuple[int, ...]
    bin_accuracies: tuple[float | None, ...]
    bin_confidences: tuple[float | None, ...]

    def to_json(self) -> dict[str, Any]:
        return {
            "bin_edges": list(self.bin_edges),
            "bin_counts": list(self.bin_counts),
            "bin_accuracies": [None if a is None else float(a) for a in self.bin_accuracies],
            "bin_confidences": [None if c is None else float(c) for c in self.bin_confidences],
        }


def bin_reliability(
    scores: Iterable[float],
    correct_flags: Iterable[float | bool],
    *,
    n_bins: int = 10,
) -> ReliabilityBins:
    """
    Histogram scores into equal-width bins [0,1] (clamped) and compute per-bin
    mean confidence and accuracy (fraction correct).
    """
    pairs = [(float(s), 1.0 if bool(c) else 0.0) for s, c in zip(scores, correct_flags, strict=False)]
    if not pairs:
        edges = tuple(0.0 if i == 0 else i / n_bins for i in range(n_bins + 1))
        z = (0,) * n_bins
        return ReliabilityBins(
            bin_edges=edges,
            bin_counts=z,
            bin_accuracies=tuple(None for _ in range(n_bins)),
            bin_confidences=tuple(None for _ in range(n_bins)),
        )

    nb = max(1, int(n_bins))
    edges = tuple(i / nb for i in range(nb + 1))

    counts = [0] * nb
    sum_conf = [0.0] * nb
    sum_acc = [0.0] * nb

    for score, acc in pairs:
        s = min(1.0, max(0.0, score))
        # bin index: [edge_i, edge_{i+1})
        idx = min(nb - 1, int(s * nb) if s < 1.0 else nb - 1)
        counts[idx] += 1
        sum_conf[idx] += s
        sum_acc[idx] += acc

    accs: list[float | None] = []
    confs: list[float | None] = []
    for i in range(nb):
        if counts[i] <= 0:
            accs.append(None)
            confs.append(None)
        else:
            accs.append(sum_acc[i] / counts[i])
            confs.append(sum_conf[i] / counts[i])

    return ReliabilityBins(
        bin_edges=edges,
        bin_counts=tuple(counts),
        bin_accuracies=tuple(accs),
        bin_confidences=tuple(confs),
    )


def expected_calibration_error(bins: ReliabilityBins) -> float | None:
    """Weighted |accuracy - confidence| across non-empty bins."""
    total = sum(bins.bin_counts)
    if total <= 0:
        return None
    ece = 0.0
    for n, acc, conf in zip(bins.bin_counts, bins.bin_accuracies, bins.bin_confidences, strict=True):
        if n <= 0 or acc is None or conf is None:
            continue
        ece += (n / total) * abs(acc - conf)
    return float(ece)
