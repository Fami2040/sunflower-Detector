"""Eval-only head ROI mask on predictions before counting @ locked conf (P2-HEAD-ROI-EVAL)."""

from __future__ import annotations

import copy
from typing import Any

from harchoc.detection_match import (
    _as_xyxy,
    _extract_boxes,
    _filter_preds_by_conf,
    _index_records_by_image_id,
    match_counts_for_threshold,
    per_image_detection_counts,
)
from harchoc.error_analysis_core import _counting_metrics, analyze_errors
from harchoc.schemas import with_schema_version
from harchoc.threshold_lock import load_locked_conf, load_locked_match_iou

HEAD_ROI_EVAL_SCHEMA = "head_roi_eval.v1"
DEFAULT_MARGIN_FRAC = 0.02


def gt_union_head_roi_xyxy(
    gt_boxes: list[dict[str, Any]],
    *,
    width: int,
    height: int,
    margin_frac: float = DEFAULT_MARGIN_FRAC,
) -> tuple[float, float, float, float] | None:
    """Axis-aligned ROI enclosing all GT boxes, expanded by margin_frac of image span."""
    if not gt_boxes:
        return None
    x1 = min(b["bbox"][0] for b in gt_boxes)
    y1 = min(b["bbox"][1] for b in gt_boxes)
    x2 = max(b["bbox"][2] for b in gt_boxes)
    y2 = max(b["bbox"][3] for b in gt_boxes)
    mx = float(margin_frac) * float(width)
    my = float(margin_frac) * float(height)
    return (
        max(0.0, x1 - mx),
        max(0.0, y1 - my),
        min(float(width), x2 + mx),
        min(float(height), y2 + my),
    )


def bbox_center_in_roi(
    bbox: tuple[float, float, float, float],
    roi: tuple[float, float, float, float],
) -> bool:
    cx = 0.5 * (bbox[0] + bbox[2])
    cy = 0.5 * (bbox[1] + bbox[3])
    rx1, ry1, rx2, ry2 = roi
    return rx1 <= cx <= rx2 and ry1 <= cy <= ry2


def _image_wh(rec: dict[str, Any], gt_boxes: list[dict[str, Any]]) -> tuple[int, int]:
    w = rec.get("width")
    h = rec.get("height")
    if w is not None and h is not None:
        return int(w), int(h)
    if not gt_boxes:
        return 0, 0
    x2 = max(b["bbox"][2] for b in gt_boxes)
    y2 = max(b["bbox"][3] for b in gt_boxes)
    return max(1, int(round(x2))), max(1, int(round(y2)))


def apply_head_roi_mask_to_preds(
    preds: Any,
    gt: Any,
    *,
    margin_frac: float = DEFAULT_MARGIN_FRAC,
) -> tuple[Any, dict[str, Any]]:
    """
    Drop detections whose box center lies outside the per-image GT-union head ROI.

    Images with no GT boxes are left unchanged (no ROI defined).
    """
    gt_by_img = _index_records_by_image_id(gt)
    preds_by_img = _index_records_by_image_id(preds)
    masked = copy.deepcopy(preds)
    masked_by_img = _index_records_by_image_id(masked)

    n_removed = 0
    n_images_masked = 0
    n_images_no_gt = 0

    for img_id, pr_rec in masked_by_img.items():
        gt_rec = gt_by_img.get(img_id, {"annotations": []})
        gt_boxes = _extract_boxes(gt_rec, key="annotations")
        if not gt_boxes:
            n_images_no_gt += 1
            continue
        w, h = _image_wh(gt_rec, gt_boxes)
        roi = gt_union_head_roi_xyxy(gt_boxes, width=w, height=h, margin_frac=margin_frac)
        if roi is None:
            continue
        dets = pr_rec.get("detections") or []
        if not isinstance(dets, list):
            continue
        kept: list[dict[str, Any]] = []
        for det in dets:
            if not isinstance(det, dict) or det.get("bbox") is None:
                kept.append(det)
                continue
            box = _as_xyxy(det["bbox"])
            if bbox_center_in_roi(box, roi):
                kept.append(det)
            else:
                n_removed += 1
        pr_rec["detections"] = kept
        n_images_masked += 1

    stats = {
        "margin_frac": float(margin_frac),
        "method": "gt_union_bbox_center_filter",
        "n_preds_removed": n_removed,
        "n_images_with_roi_mask": n_images_masked,
        "n_images_no_gt_roi_skipped": n_images_no_gt,
    }
    return masked, stats


def run_head_roi_eval(
    *,
    gt: Any,
    preds: Any,
    locked_conf_from: str,
    weights: str | None = None,
    split_file: str | None = None,
    margin_frac: float = DEFAULT_MARGIN_FRAC,
    iou_thr: float | None = None,
    iou_bg_thr: float = 0.1,
) -> dict[str, Any]:
    """Compare baseline vs head-ROI-masked metrics at val-locked conf."""
    conf_thr = load_locked_conf(locked_conf_from)
    match_iou = float(iou_thr) if iou_thr is not None else load_locked_match_iou(locked_conf_from)
    if match_iou is None:
        match_iou = 0.5

    masked_preds, mask_stats = apply_head_roi_mask_to_preds(
        preds, gt, margin_frac=margin_frac
    )

    baseline_counts = match_counts_for_threshold(
        gt=gt, preds=preds, conf_thr=conf_thr, iou_thr=match_iou
    )
    masked_counts = match_counts_for_threshold(
        gt=gt, preds=masked_preds, conf_thr=conf_thr, iou_thr=match_iou
    )

    baseline_per_img = per_image_detection_counts(gt=gt, preds=preds, conf_thr=conf_thr)
    masked_per_img = per_image_detection_counts(
        gt=gt, preds=masked_preds, conf_thr=conf_thr
    )
    baseline_cm = _counting_metrics(baseline_per_img)
    masked_cm = _counting_metrics(masked_per_img)

    baseline_err = analyze_errors(
        gt=gt,
        preds=preds,
        conf_thr=conf_thr,
        iou_thr=match_iou,
        iou_bg_thr=iou_bg_thr,
    )
    masked_err = analyze_errors(
        gt=gt,
        preds=masked_preds,
        conf_thr=conf_thr,
        iou_thr=match_iou,
        iou_bg_thr=iou_bg_thr,
    )

    payload: dict[str, Any] = {
        "status": "complete",
        "locked_conf_from": str(locked_conf_from),
        "locked_conf": float(conf_thr),
        "match_iou": float(match_iou),
        "weights": weights,
        "split_file": split_file,
        "roi": mask_stats,
        "baseline": {
            "match_counts": baseline_counts,
            "counting_metrics": baseline_cm,
            "fp_breakdown": baseline_err.get("fp_breakdown"),
        },
        "masked": {
            "match_counts": masked_counts,
            "counting_metrics": masked_cm,
            "fp_breakdown": masked_err.get("fp_breakdown"),
        },
        "delta": {
            "count_mae": float(masked_cm["mae"]) - float(baseline_cm["mae"]),
            "fp": int(masked_counts["fp"]) - int(baseline_counts["fp"]),
            "tp": int(masked_counts["tp"]) - int(baseline_counts["tp"]),
            "fn": int(masked_counts["fn"]) - int(baseline_counts["fn"]),
            "background_fp": int((masked_err.get("fp_breakdown") or {}).get("background", 0))
            - int((baseline_err.get("fp_breakdown") or {}).get("background", 0)),
        },
    }
    return with_schema_version(payload, schema_version=HEAD_ROI_EVAL_SCHEMA)


def dry_run_payload(
    *,
    locked_conf_from: str,
    weights: str,
    split_file: str,
    out_path: str,
    device: str,
    gt_json: str | None = None,
    preds_json: str | None = None,
) -> dict[str, Any]:
    return with_schema_version(
        {
            "status": "dry_run",
            "locked_conf_from": locked_conf_from,
            "weights": weights,
            "split_file": split_file,
            "device": device,
            "out": out_path,
            "gt_json": gt_json,
            "preds_json": preds_json,
            "note": "Plan only; run without --dry-run to export preds and compute masked metrics.",
        },
        schema_version=HEAD_ROI_EVAL_SCHEMA,
    )
