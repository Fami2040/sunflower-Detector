"""Area strata and conf×taxonomy grid helpers for error_analysis."""

from __future__ import annotations

from typing import Any

from harchoc.detection_match import _as_xyxy

# COCO object-size thresholds on sqrt(bbox area) ≈ side length in pixels.
DEFAULT_AREA_SQRT_THRESHOLDS: tuple[float, float] = (32.0, 96.0)
AREA_STRATA_LABELS: tuple[str, ...] = ("small", "medium", "large")

DEFAULT_CONF_BIN_EDGES: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0)

# Granular instance types used for strata / conf grid (FP split matches fp_breakdown).
INSTANCE_ERROR_TYPES: tuple[str, ...] = (
    "tp",
    "fn",
    "fp_background",
    "fp_localization",
    "fp_classification",
    "dupe",
)


def bbox_sqrt_area(box: Any) -> float:
    """Geometric side length sqrt(w*h) for an xyxy or xywh bbox."""
    x1, y1, x2, y2 = _as_xyxy(box)
    w = max(0.0, x2 - x1)
    h = max(0.0, y2 - y1)
    return (w * h) ** 0.5


def area_stratum_label(
    sqrt_area: float,
    *,
    thresholds: tuple[float, float] = DEFAULT_AREA_SQRT_THRESHOLDS,
) -> str:
    t_small, t_large = thresholds
    if sqrt_area < t_small:
        return "small"
    if sqrt_area < t_large:
        return "medium"
    return "large"


def conf_bin_label(
    score: float | None,
    *,
    edges: tuple[float, ...] = DEFAULT_CONF_BIN_EDGES,
) -> str | None:
    """Return half-open bin label [lo, hi) for score, or None if score is missing."""
    if score is None:
        return None
    sc = float(score)
    if sc < edges[0] or sc > edges[-1]:
        # Clamp into outer bins for numerical edge cases.
        if sc < edges[0]:
            sc = edges[0]
        elif sc >= edges[-1]:
            sc = max(edges[-2], edges[-1] - 1e-9)
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        if i == len(edges) - 2:
            if lo <= sc <= hi:
                return f"[{lo:.2f},{hi:.2f}]"
        elif lo <= sc < hi:
            return f"[{lo:.2f},{hi:.2f})"
    return None


def _empty_strata_counts() -> dict[str, dict[str, int]]:
    return {et: {label: 0 for label in AREA_STRATA_LABELS} for et in INSTANCE_ERROR_TYPES}


def _empty_conf_grid(
    *,
    edges: tuple[float, ...] = DEFAULT_CONF_BIN_EDGES,
) -> dict[str, dict[str, int]]:
    labels = [_conf_bin_label_for_index(i, edges=edges) for i in range(len(edges) - 1)]
    return {lbl: {et: 0 for et in INSTANCE_ERROR_TYPES if et != "fn"} for lbl in labels}


def _conf_bin_label_for_index(i: int, *, edges: tuple[float, ...]) -> str:
    lo, hi = edges[i], edges[i + 1]
    if i == len(edges) - 2:
        return f"[{lo:.2f},{hi:.2f}]"
    return f"[{lo:.2f},{hi:.2f})"


def build_bbox_area_strata(
    events: list[dict[str, Any]],
    *,
    thresholds: tuple[float, float] = DEFAULT_AREA_SQRT_THRESHOLDS,
) -> dict[str, Any]:
    """Count instances per error type × area stratum (small/medium/large)."""
    by_error_type = _empty_strata_counts()
    for ev in events:
        et = str(ev.get("error_type") or "")
        if et not in by_error_type:
            continue
        bbox = ev.get("bbox")
        if bbox is None:
            continue
        label = area_stratum_label(bbox_sqrt_area(bbox), thresholds=thresholds)
        by_error_type[et][label] += 1
    return {
        "method": "sqrt_area",
        "thresholds": list(thresholds),
        "labels": list(AREA_STRATA_LABELS),
        "by_error_type": by_error_type,
    }


def build_conf_taxonomy_grid(
    events: list[dict[str, Any]],
    *,
    edges: tuple[float, ...] = DEFAULT_CONF_BIN_EDGES,
) -> dict[str, Any]:
    """2D counts: confidence bin × granular error type (FN omitted — no score)."""
    counts = _empty_conf_grid(edges=edges)
    bin_labels = list(counts.keys())
    for ev in events:
        et = str(ev.get("error_type") or "")
        if et == "fn":
            continue
        score = ev.get("score")
        if score is None:
            continue
        lbl = conf_bin_label(float(score), edges=edges)
        if lbl is None or lbl not in counts:
            continue
        if et not in counts[lbl]:
            continue
        counts[lbl][et] += 1
    return {
        "conf_bin_edges": list(edges),
        "conf_bin_labels": bin_labels,
        "error_types": [et for et in INSTANCE_ERROR_TYPES if et != "fn"],
        "counts": counts,
    }
