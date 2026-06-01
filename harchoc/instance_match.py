"""Greedy instance-level prediction classification shared by error analysis and confusion matrix."""

from __future__ import annotations

from typing import Any, Literal

from harchoc.detection_match import _iou_xyxy, _max_iou_to_gts

MatchOutcome = Literal["tp", "dupe", "cls_confusion", "fp_localization", "fp_background"]


def find_same_class_tp_match(
    p: dict[str, Any],
    gt_boxes: list[dict[str, Any]],
    gt_used: list[bool],
    *,
    iou_thr: float,
) -> int:
    """Return index of best unused same-class GT with IoU>=iou_thr, or -1."""
    pc = int(p["category_id"])
    best_i, best_iou = -1, 0.0
    for i, g in enumerate(gt_boxes):
        if gt_used[i] or int(g["category_id"]) != pc:
            continue
        iou = _iou_xyxy(p["bbox"], g["bbox"])
        if iou >= iou_thr and iou > best_iou:
            best_i, best_iou = i, iou
    return best_i


def classify_unmatched_prediction(
    p: dict[str, Any],
    gt_boxes: list[dict[str, Any]],
    gt_used: list[bool],
    *,
    iou_thr: float,
    iou_bg_thr: float = 0.1,
) -> tuple[MatchOutcome, int | None]:
    """
    Classify a prediction that did not match as same-class TP.

    Returns (outcome, gt_index) where gt_index is set for cls_confusion.
    """
    pc = int(p["category_id"])

    for i, g in enumerate(gt_boxes):
        if not gt_used[i] or int(g["category_id"]) != pc:
            continue
        if _iou_xyxy(p["bbox"], g["bbox"]) >= iou_thr:
            return "dupe", None

    for i, g in enumerate(gt_boxes):
        if int(p["category_id"]) == int(g["category_id"]):
            continue
        if _iou_xyxy(p["bbox"], g["bbox"]) >= iou_thr:
            return "cls_confusion", i

    if not gt_boxes:
        return "fp_background", None

    overlapped_same_cls_low_iou = False
    for i, g in enumerate(gt_boxes):
        if int(p["category_id"]) != int(g["category_id"]):
            continue
        iou = _iou_xyxy(p["bbox"], g["bbox"])
        if iou_bg_thr <= iou < iou_thr:
            overlapped_same_cls_low_iou = True
            break

    if overlapped_same_cls_low_iou:
        return "fp_localization", None

    if _max_iou_to_gts(p["bbox"], gt_boxes) < iou_bg_thr:
        return "fp_background", None

    return "fp_localization", None
