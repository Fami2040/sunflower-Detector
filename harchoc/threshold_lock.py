"""Locked confidence threshold helpers (val sweep → test reporting)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_locked_match_iou(path: str | Path) -> float | None:
    """Read match IoU from val sweep metadata (selected.nms_iou or match.iou)."""
    p = Path(path).expanduser()
    obj = json.loads(p.read_text("utf-8"))
    if not isinstance(obj, dict):
        return None
    selected = obj.get("selected")
    if isinstance(selected, dict):
        nms = selected.get("nms_iou")
        if isinstance(nms, dict) and nms.get("selected") is not None:
            return float(nms["selected"])
    match = obj.get("match")
    if isinstance(match, dict) and match.get("iou") is not None:
        return float(match["iou"])
    return None


def load_locked_conf(path: str | Path) -> float:
    """
    Read conf_thr from a threshold_sweep_run.v1 JSON.

    Prefers locked.row.conf_thr when the sweep carries a locked operating point
    (``locked.mode`` in ``fixed_conf`` / ``locked``, or ``locked.row.conf_thr``
    is set). Falls back to selected.row for val-only sweeps without a locked block.
    """
    p = Path(path).expanduser()
    obj = json.loads(p.read_text("utf-8"))
    if not isinstance(obj, dict):
        raise ValueError(f"Expected JSON object in {p}")
    locked = obj.get("locked")
    if isinstance(locked, dict):
        row = locked.get("row")
        mode = locked.get("mode")
        prefer_locked = mode in ("fixed_conf", "locked")
        if isinstance(row, dict) and row.get("conf_thr") is not None:
            prefer_locked = True
        if prefer_locked and isinstance(row, dict) and row.get("conf_thr") is not None:
            return float(row["conf_thr"])
    selected = obj.get("selected")
    if isinstance(selected, dict):
        row = selected.get("row")
        if isinstance(row, dict) and row.get("conf_thr") is not None:
            return float(row["conf_thr"])
    raise ValueError(f"No conf_thr in selected.row or locked.row: {p}")


def metrics_row_at_conf(
    *,
    gt: Any,
    preds: Any,
    conf_thr: float,
    iou_thr: float,
    category_aware: bool,
    n_images: int,
) -> dict[str, Any]:
    """Compute TP/FP/FN metrics at a single confidence (same shape as sweep rows)."""
    from harchoc.detection_match import match_counts_for_threshold

    counts = match_counts_for_threshold(
        gt=gt,
        preds=preds,
        conf_thr=float(conf_thr),
        iou_thr=float(iou_thr),
        category_aware=category_aware,
    )
    tp, fp, fn = int(counts["tp"]), int(counts["fp"]), int(counts["fn"])
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0
    fp_per_image = (fp / n_images) if n_images else 0.0
    return {
        "conf_thr": float(conf_thr),
        **counts,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "fp_per_image": fp_per_image,
    }


def build_locked_block(
    *,
    row: dict[str, Any],
    source: str | None = None,
    counting_metrics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    out: dict[str, Any] = {"mode": "fixed_conf", "row": row}
    if source:
        out["source"] = source
    if counting_metrics is not None:
        out["counting_metrics"] = counting_metrics
    return out
