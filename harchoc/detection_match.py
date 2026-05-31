"""Shared detection matching utilities (IoU, COCO-like box extract, greedy match)."""

from __future__ import annotations

from typing import Any, Iterable


def _as_xyxy(box: Any) -> tuple[float, float, float, float]:
    """
    Accept either [x1,y1,x2,y2] or COCO-ish [x,y,w,h].
    We treat 4-tuples where x2>x1 and y2>y1 as xyxy; otherwise assume xywh.
    """
    if not isinstance(box, (list, tuple)) or len(box) != 4:
        raise ValueError(f"bbox must be a 4-list, got: {box!r}")
    x1, y1, a, b = [float(x) for x in box]
    if a > x1 and b > y1:
        return x1, y1, a, b
    return x1, y1, x1 + a, y1 + b


def _iou_xyxy(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    aa = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    bb = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = aa + bb - inter
    return inter / union if union > 0 else 0.0


def _iter_image_records(obj: Any) -> Iterable[dict[str, Any]]:
    """
    Normalize a few convenient JSON shapes into per-image records:
    - {"images":[{...,"annotations":[...]}]} (GT)
    - {"images":[{...,"detections":[...]}]} (preds)
    - {"by_image": {"img1":[...], ...}} with optional image metadata
    """
    if isinstance(obj, dict) and isinstance(obj.get("images"), list):
        for rec in obj["images"]:
            if isinstance(rec, dict):
                yield rec
        return
    if isinstance(obj, dict) and isinstance(obj.get("by_image"), dict):
        for k, v in obj["by_image"].items():
            yield {"image_id": k, "detections": v}
        return
    raise ValueError("Unsupported JSON format; expected top-level {'images':[...]} or {'by_image':{...}}")


def _image_id_from_record(rec: dict[str, Any]) -> str:
    return str(rec.get("image_id") or rec.get("id") or rec.get("file_name") or "")


def _index_records_by_image_id(obj: Any) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for rec in _iter_image_records(obj):
        out[_image_id_from_record(rec)] = rec
    return out


def _extract_boxes(rec: dict[str, Any], *, key: str) -> list[dict[str, Any]]:
    raw = rec.get(key) or []
    if not isinstance(raw, list):
        raise ValueError(f"Expected {key} list, got {type(raw)}")
    out: list[dict[str, Any]] = []
    for ann in raw:
        if not isinstance(ann, dict):
            continue
        bbox = _as_xyxy(ann.get("bbox"))
        out.append(
            {
                "bbox": bbox,
                "category_id": int(ann.get("category_id", ann.get("class_id", 0))),
                "score": float(ann.get("score")) if "score" in ann and ann.get("score") is not None else None,
            }
        )
    return out


def _filter_preds_by_conf(pr_boxes: list[dict[str, Any]], conf_thr: float) -> list[dict[str, Any]]:
    pr_filt = [p for p in pr_boxes if (p["score"] is None or float(p["score"]) >= conf_thr)]
    pr_filt.sort(key=lambda x: float(x["score"] or 0.0), reverse=True)
    return pr_filt


def _max_iou_to_gts(
    pred_box: tuple[float, float, float, float],
    gt_boxes: list[dict[str, Any]],
    *,
    same_class: int | None = None,
) -> float:
    best = 0.0
    for g in gt_boxes:
        if same_class is not None and int(g["category_id"]) != same_class:
            continue
        best = max(best, _iou_xyxy(pred_box, g["bbox"]))
    return best


def _pred_pred_max_iou(
    idx: int, pr_boxes: list[dict[str, Any]], *, same_class: int
) -> float:
    p = pr_boxes[idx]
    best = 0.0
    for j, q in enumerate(pr_boxes):
        if j == idx or int(q["category_id"]) != same_class:
            continue
        best = max(best, _iou_xyxy(p["bbox"], q["bbox"]))
    return best


def match_counts_for_threshold(
    *,
    gt: Any,
    preds: Any,
    conf_thr: float,
    iou_thr: float = 0.5,
    category_aware: bool = True,
) -> dict[str, int]:
    """
    Greedy per-image matching. Counts are totals across all images.
    - TP: prediction matched to a GT (IoU>=iou_thr, and same class if category_aware).
    - FP: prediction not matched.
    - FN: GT not matched.
    """
    gt_by_img = _index_records_by_image_id(gt)
    preds_by_img = _index_records_by_image_id(preds)

    all_ids = set(gt_by_img.keys()) | set(preds_by_img.keys())
    tp = fp = fn = 0

    for img_id in sorted(all_ids):
        gt_rec = gt_by_img.get(img_id, {"annotations": []})
        pr_rec = preds_by_img.get(img_id, {"detections": []})
        gt_boxes = _extract_boxes(gt_rec, key="annotations")
        pr_filt = _filter_preds_by_conf(_extract_boxes(pr_rec, key="detections"), conf_thr)

        gt_used = [False] * len(gt_boxes)
        for p in pr_filt:
            best_i, best_iou = -1, 0.0
            for i, g in enumerate(gt_boxes):
                if gt_used[i]:
                    continue
                if category_aware and int(p["category_id"]) != int(g["category_id"]):
                    continue
                iou = _iou_xyxy(p["bbox"], g["bbox"])
                if iou >= iou_thr and iou > best_iou:
                    best_i, best_iou = i, iou
            if best_i >= 0:
                gt_used[best_i] = True
                tp += 1
            else:
                fp += 1
        fn += sum(1 for used in gt_used if not used)

    return {"tp": tp, "fp": fp, "fn": fn}


def image_ids_union(gt: Any, preds: Any) -> list[str]:
    gt_ids = [_image_id_from_record(rec) for rec in _iter_image_records(gt)]
    pr_ids = [_image_id_from_record(rec) for rec in _iter_image_records(preds)]
    return sorted(set(gt_ids) | set(pr_ids))


def per_image_detection_counts(
    *,
    gt: Any,
    preds: Any,
    conf_thr: float,
    category_aware: bool = True,
) -> dict[str, dict[str, Any]]:
    """
    Per-image GT vs filtered-prediction counts for counting MAE (n_pred = boxes above conf).
    """
    gt_by_img = _index_records_by_image_id(gt)
    preds_by_img = _index_records_by_image_id(preds)
    all_ids = set(gt_by_img.keys()) | set(preds_by_img.keys())
    out: dict[str, dict[str, Any]] = {}
    for img_id in sorted(all_ids):
        gt_rec = gt_by_img.get(img_id, {"annotations": []})
        pr_rec = preds_by_img.get(img_id, {"detections": []})
        gt_boxes = _extract_boxes(gt_rec, key="annotations")
        pr_filt = _filter_preds_by_conf(_extract_boxes(pr_rec, key="detections"), conf_thr)
        out[img_id] = {
            "image_id": img_id,
            "n_gt": len(gt_boxes),
            "n_pred": len(pr_filt),
        }
    return out
