"""TIDE-style bucket summaries and ambiguous×FP cross-tabs for error_analysis."""

from __future__ import annotations

import json
import math
import tempfile
from pathlib import Path
from typing import Any

TIDE_BUCKET_KEYS = ("Miss", "Loc", "Bkg", "Cls", "Dupe")
FP_BUCKET_KEYS = ("tp", "background", "localization", "classification", "dupe")
TIDECV_COMPARE_V1 = "tidecv_compare.v1"

_TIDECV_DELTA_AP_ALIASES: dict[str, str] = {
    "miss": "Miss",
    "localization": "Loc",
    "loc": "Loc",
    "background": "Bkg",
    "bkg": "Bkg",
    "classification": "Cls",
    "cls": "Cls",
    "dupe": "Dupe",
    "duplicate": "Dupe",
}

SUNFLOWER_COCO_CATEGORIES: list[dict[str, Any]] = [
    {"id": 0, "name": "developed"},
    {"id": 1, "name": "aborted"},
]


def _xyxy_to_coco_xywh(bbox: tuple[float, float, float, float]) -> list[float]:
    x1, y1, x2, y2 = bbox
    return [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]


def _bbox_xywh_to_segmentation(bbox_xywh: list[float]) -> list[list[float]]:
    x, y, w, h = bbox_xywh
    return [[x, y, x + w, y, x + w, y + h, x, y + h]]


def _image_size_from_xyxy_boxes(boxes: list[tuple[float, float, float, float]]) -> tuple[int, int]:
    if not boxes:
        return 640, 640
    max_x = max(b[2] for b in boxes)
    max_y = max(b[3] for b in boxes)
    return max(1, int(math.ceil(max_x))), max(1, int(math.ceil(max_y)))


def export_coco_gt_for_tide(
    gt: Any,
    *,
    categories: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """
    Convert harchoc ``eval_export`` GT JSON to COCO instances format for ``tidecv``.

    Accepts ``{"images": [{image_id, file_name, annotations:[{bbox, category_id}]}]}``.
    Bboxes may be xyxy (harchoc default) or COCO xywh; tidecv expects xywh + segmentation.
    """
    from harchoc.detection_match import _extract_boxes, _index_records_by_image_id

    cats = list(categories or SUNFLOWER_COCO_CATEGORIES)
    gt_by_img = _index_records_by_image_id(gt)
    images: list[dict[str, Any]] = []
    annotations: list[dict[str, Any]] = []
    ann_id = 1

    for img_id in sorted(gt_by_img):
        rec = gt_by_img[img_id]
        gt_boxes = _extract_boxes(rec, key="annotations")
        width, height = _image_size_from_xyxy_boxes([b["bbox"] for b in gt_boxes])
        images.append(
            {
                "id": img_id,
                "file_name": str(rec.get("file_name") or img_id),
                "width": width,
                "height": height,
            }
        )
        for ann in gt_boxes:
            bbox_xywh = _xyxy_to_coco_xywh(ann["bbox"])
            annotations.append(
                {
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": int(ann["category_id"]),
                    "bbox": bbox_xywh,
                    "area": bbox_xywh[2] * bbox_xywh[3],
                    "iscrowd": 0,
                    "segmentation": _bbox_xywh_to_segmentation(bbox_xywh),
                }
            )
            ann_id += 1

    return {"images": images, "annotations": annotations, "categories": cats}


def export_coco_predictions_for_tide(*, gt: Any, preds: Any) -> list[dict[str, Any]]:
    """
    Convert harchoc ``eval_export`` preds JSON to COCO results list for ``tidecv``.

    Returns a flat list of ``{image_id, category_id, score, bbox}`` with COCO xywh bboxes.
    Predictions on images present only in preds (no GT row) are included.
    """
    from harchoc.detection_match import _extract_boxes, _index_records_by_image_id

    gt_by_img = _index_records_by_image_id(gt)
    preds_by_img = _index_records_by_image_id(preds)
    all_ids = set(gt_by_img) | set(preds_by_img)

    dets: list[dict[str, Any]] = []
    for img_id in sorted(all_ids):
        rec = preds_by_img.get(img_id, {"detections": []})
        for p in _extract_boxes(rec, key="detections"):
            if p["score"] is None:
                continue
            dets.append(
                {
                    "image_id": img_id,
                    "category_id": int(p["category_id"]),
                    "score": float(p["score"]),
                    "bbox": _xyxy_to_coco_xywh(p["bbox"]),
                }
            )
    return dets


def build_ambiguous_fp_crosstab(
    pred_outcomes: list[dict[str, Any]],
    *,
    conf_band: list[float],
) -> dict[str, Any]:
    """
    Cross-tabulate ambiguous detections against error buckets.

    Each entry in ``pred_outcomes`` must include ``ambiguous`` (bool), optional
    ``flags`` (list[str]), and ``bucket`` (one of FP_BUCKET_KEYS).
    """
    by_bucket: dict[str, dict[str, int]] = {
        b: {"ambiguous": 0, "not_ambiguous": 0} for b in FP_BUCKET_KEYS
    }
    by_flag: dict[str, dict[str, int]] = {}

    for row in pred_outcomes:
        bucket = str(row.get("bucket") or "background")
        if bucket not in by_bucket:
            bucket = "background"
        amb_key = "ambiguous" if bool(row.get("ambiguous")) else "not_ambiguous"
        by_bucket[bucket][amb_key] += 1
        if bool(row.get("ambiguous")):
            for flag in row.get("flags") or []:
                f = str(flag)
                by_flag.setdefault(f, {b: 0 for b in FP_BUCKET_KEYS})
                by_flag[f][bucket] = int(by_flag[f].get(bucket, 0)) + 1

    amb_total = sum(v["ambiguous"] for v in by_bucket.values())
    fp_buckets = ("background", "localization", "classification", "dupe")
    amb_among_fp = sum(by_bucket[b]["ambiguous"] for b in fp_buckets)

    return {
        "conf_band": list(conf_band),
        "by_bucket": by_bucket,
        "by_flag": by_flag,
        "totals": {
            "n_predictions": len(pred_outcomes),
            "n_ambiguous": amb_total,
            "n_ambiguous_among_fp_buckets": amb_among_fp,
        },
    }


def build_tide_bucket_summary(
    *,
    counts: dict[str, Any],
    fp_breakdown: dict[str, Any],
    match: dict[str, Any] | None = None,
    map50: float | None = None,
) -> dict[str, Any]:
    """
    TIDE-aligned bucket counts with count-share proxy for delta-AP impact.

    Official per-bucket delta-AP uses ``tidecv`` when available; this summary uses
    error-mass shares for reviewer-facing bars (see ``tidecv_compare.v1``).
    """
    buckets = {
        "Miss": int(counts.get("fn") or 0),
        "Loc": int(fp_breakdown.get("localization") or 0),
        "Bkg": int(fp_breakdown.get("background") or 0),
        "Cls": int(fp_breakdown.get("classification") or 0),
        "Dupe": int(fp_breakdown.get("dupe") or 0),
    }
    tp = int(counts.get("tp") or 0)
    n_errors = sum(buckets.values())
    delta_ap_share = {
        k: (float(v) / n_errors if n_errors else 0.0) for k, v in buckets.items()
    }
    delta_ap_estimate: dict[str, float] | None = None
    if map50 is not None and n_errors:
        gap = max(0.0, 1.0 - float(map50))
        delta_ap_estimate = {k: float(v) / n_errors * gap for k, v in buckets.items()}

    loc_cls_ratio: float | None = None
    loc_mass = int(buckets["Loc"]) + int(buckets["Bkg"])
    cls_mass = int(buckets["Cls"])
    if cls_mass:
        loc_cls_ratio = loc_mass / cls_mass

    dominant = max(buckets, key=lambda k: buckets[k]) if n_errors else None
    out: dict[str, Any] = {
        "method": "tide_bucket_count_proxy.v1",
        "note": (
            "Count-share proxy for TIDE delta-AP; official delta-AP via tidecv in tidecv_compare.v1."
        ),
        "match": dict(match or {}),
        "buckets": buckets,
        "tp": tp,
        "n_errors": n_errors,
        "delta_ap_share": delta_ap_share,
        "dominant_bucket": dominant,
        "localization_dominates_classification": (
            loc_mass > cls_mass if (loc_mass or cls_mass) else None
        ),
        "loc_plus_bkg_over_cls_ratio": loc_cls_ratio,
    }
    if delta_ap_estimate is not None:
        out["delta_ap_estimate"] = delta_ap_estimate
        if map50 is not None:
            out["map50_reference"] = float(map50)
    return out


def default_tidecv_compare_path(report_path: str | Path) -> str:
    """Sidecar path next to ``error_analysis`` report (``*_tidecv_compare.json``)."""
    p = Path(report_path)
    stem = p.stem
    if stem.endswith("_report"):
        stem = stem[: -len("_report")]
    return str(p.with_name(f"{stem}_tidecv_compare.json"))


def _normalize_tidecv_delta_ap(delta_ap: dict[str, Any]) -> dict[str, float]:
    out: dict[str, float] = {k: 0.0 for k in TIDE_BUCKET_KEYS}
    for raw_key, val in delta_ap.items():
        key = str(raw_key).strip()
        norm = _TIDECV_DELTA_AP_ALIASES.get(key.lower(), key)
        if norm in out:
            out[norm] = float(val)
    return out


def build_tidecv_compare(
    *,
    tide_bucket_summary: dict[str, Any],
    tidecv_result: dict[str, Any],
) -> dict[str, Any]:
    """
    Proxy bucket shares vs optional official ``tidecv`` ``delta_ap`` / ``ap50``.

    Emitted as ``tidecv_compare.v1`` by ``error_analysis.py --tidecv``.
    """
    status = str(tidecv_result.get("status") or "skipped")
    adapter = tidecv_result.get("adapter")
    adapter_ok = bool(tidecv_result.get("adapter_ok", isinstance(adapter, dict)))
    skipped_reason: str | None = None
    if status in ("skipped", "error"):
        skipped_reason = str(
            tidecv_result.get("reason") or tidecv_result.get("error") or status
        )

    proxy_shares = dict(tide_bucket_summary.get("delta_ap_share") or {})
    proxy_buckets = dict(tide_bucket_summary.get("buckets") or {})
    official_raw = tidecv_result.get("delta_ap") if status == "ok" else None
    official_norm = _normalize_tidecv_delta_ap(official_raw) if isinstance(official_raw, dict) else None

    comparison: dict[str, dict[str, float | None]] | None = None
    if official_norm is not None:
        comparison = {
            k: {
                "proxy_share": float(proxy_shares.get(k) or 0.0),
                "tidecv_delta_ap": float(official_norm.get(k) or 0.0),
            }
            for k in TIDE_BUCKET_KEYS
        }

    out: dict[str, Any] = {
        "status": status,
        "adapter_ok": adapter_ok,
        "skipped_reason": skipped_reason,
        "adapter": dict(adapter) if isinstance(adapter, dict) else None,
        "proxy": {
            "method": tide_bucket_summary.get("method"),
            "buckets": proxy_buckets,
            "delta_ap_share": proxy_shares,
            "n_errors": tide_bucket_summary.get("n_errors"),
            "dominant_bucket": tide_bucket_summary.get("dominant_bucket"),
        },
        "tidecv": {
            "status": status,
            "ap50": tidecv_result.get("ap50") if status == "ok" else None,
            "delta_ap": official_raw if status == "ok" else None,
            "delta_ap_normalized": official_norm,
        },
    }
    if comparison is not None:
        out["comparison"] = comparison
    if tidecv_result.get("install_hint"):
        out["install_hint"] = tidecv_result["install_hint"]
    return out


def _tidecv_adapter_meta(*, gt: Any, preds: Any) -> dict[str, Any]:
    coco_gt = export_coco_gt_for_tide(gt)
    coco_preds = export_coco_predictions_for_tide(gt=gt, preds=preds)
    return {
        "n_images": len(coco_gt.get("images") or []),
        "n_gt_annotations": len(coco_gt.get("annotations") or []),
        "n_predictions": len(coco_preds),
        "coco_gt": coco_gt,
        "coco_preds": coco_preds,
    }


def try_run_tidecv(*, gt: Any, preds: Any) -> dict[str, Any] | None:
    """
    Optional official TIDE run when ``tidecv`` is installed.

    Always converts harchoc GT/preds JSON to COCO format first. When ``tidecv`` is
    missing, returns a structured skip with adapter counts (no silent pass).
    Proxy-vs-official comparison is assembled in ``build_tidecv_compare`` →
    ``tidecv_compare.v1``.
    """
    try:
        adapter = _tidecv_adapter_meta(gt=gt, preds=preds)
    except Exception as exc:
        return {
            "status": "skipped",
            "adapter_ok": False,
            "reason": "COCO adapter failed",
            "error": f"{type(exc).__name__}: {exc}",
        }

    adapter_summary = {
        "n_images": adapter["n_images"],
        "n_gt_annotations": adapter["n_gt_annotations"],
        "n_predictions": adapter["n_predictions"],
    }
    if adapter["n_predictions"] == 0:
        return {
            "status": "skipped",
            "adapter_ok": True,
            "reason": "no scored predictions after COCO export",
            "adapter": adapter_summary,
        }

    try:
        from tidecv import TIDE  # type: ignore
        import tidecv.datasets as tide_datasets  # type: ignore
    except ImportError:
        return {
            "status": "skipped",
            "adapter_ok": True,
            "reason": "tidecv not installed",
            "install_hint": "pip install tidecv",
            "adapter": adapter_summary,
        }

    try:
        with tempfile.TemporaryDirectory(prefix="harchoc_tidecv_") as tmp:
            gt_path = Path(tmp) / "gt_coco.json"
            pred_path = Path(tmp) / "preds_coco.json"
            gt_path.write_text(json.dumps(adapter["coco_gt"]), encoding="utf-8")
            pred_path.write_text(json.dumps(adapter["coco_preds"]), encoding="utf-8")

            gt_data = tide_datasets.COCO(str(gt_path), name="harchoc_gt")
            pred_data = tide_datasets.COCOResult(str(pred_path), name="harchoc_preds")
            tide = TIDE(pos_threshold=0.5, background_threshold=0.1, mode=TIDE.BOX)
            run = tide.evaluate(gt_data, pred_data, mode=TIDE.BOX, name="harchoc_preds")
            main_errors = tide.get_main_errors().get("harchoc_preds") or {}

        return {
            "status": "ok",
            "adapter_ok": True,
            "adapter": adapter_summary,
            "ap50": float(run.ap),
            "delta_ap": {str(k): float(v) for k, v in main_errors.items()},
            "note": "Official tidecv @ IoU 0.5; proxy comparison in tidecv_compare.v1.",
        }
    except Exception as exc:
        return {
            "status": "error",
            "adapter_ok": True,
            "reason": f"tidecv evaluate failed: {type(exc).__name__}: {exc}",
            "adapter": adapter_summary,
        }
