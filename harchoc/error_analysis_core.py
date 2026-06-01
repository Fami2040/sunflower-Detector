"""Core error taxonomy / counting metrics (shared by script and HSP eval chain)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from harchoc.detection_match import (
    _as_xyxy,
    _extract_boxes,
    _filter_preds_by_conf,
    _index_records_by_image_id,
    _iou_xyxy,
    _max_iou_to_gts,
    _pred_pred_max_iou,
)
from harchoc.error_taxonomy import build_bbox_area_strata, build_conf_taxonomy_grid
from harchoc.instance_match import classify_unmatched_prediction, find_same_class_tp_match
from harchoc.tide_summary import build_ambiguous_fp_crosstab, build_tide_bucket_summary

def _try_import_pil() -> tuple[Any | None, str | None]:
    try:
        from PIL import Image  # type: ignore

        return Image, None
    except Exception as e:  # pragma: no cover - environment dependent
        return None, f"PIL not available: {type(e).__name__}: {e}"


def _clamp_xyxy_to_image(
    box: tuple[float, float, float, float], *, width: int, height: int
) -> tuple[int, int, int, int] | None:
    x1, y1, x2, y2 = box
    ix1 = max(0, min(int(round(x1)), width))
    iy1 = max(0, min(int(round(y1)), height))
    ix2 = max(0, min(int(round(x2)), width))
    iy2 = max(0, min(int(round(y2)), height))
    if ix2 <= ix1 or iy2 <= iy1:
        return None
    return ix1, iy1, ix2, iy2


def _counting_metrics(per_image: dict[str, dict[str, Any]]) -> dict[str, Any]:
    from harchoc.stats_ci import ci_for_values

    errs: list[float] = []
    abs_errs: list[float] = []
    gt_counts: list[int] = []
    for rec in per_image.values():
        n_gt = int(rec.get("n_gt") or 0)
        n_pred = int(rec.get("n_pred") or 0)
        gt_counts.append(n_gt)
        e = float(n_pred - n_gt)
        errs.append(e)
        abs_errs.append(abs(e))
    if not errs:
        return {"mae": 0.0, "rmse": 0.0, "rrmse": None, "n_images": 0}
    mae = sum(abs_errs) / len(abs_errs)
    rmse = (sum(e * e for e in errs) / len(errs)) ** 0.5
    mean_gt = sum(gt_counts) / len(gt_counts) if gt_counts else 0.0
    rrmse = (rmse / mean_gt) if mean_gt > 0 else None
    out: dict[str, Any] = {"mae": mae, "rmse": rmse, "rrmse": rrmse, "n_images": len(errs)}
    mae_ci = ci_for_values(abs_errs, stat="mean")
    if mae_ci is not None:
        out["mae_ci"] = mae_ci.to_json()
    err_ci = ci_for_values(errs, stat="mean")
    if err_ci is not None:
        out["signed_error_mean_ci"] = err_ci.to_json()
    return out


def _counting_metrics_excluding_conf_band(
    per_image: dict[str, dict[str, Any]],
    *,
    gt_by_img: dict[str, Any],
    preds_by_img: dict[str, Any],
    conf_thr: float,
    conf_hi: float,
) -> dict[str, Any]:
    """Count MAE after dropping preds with score in the ambiguous low-conf band (<= conf_hi)."""
    adjusted: dict[str, dict[str, Any]] = {}
    for img_id in per_image:
        gt_rec = gt_by_img.get(img_id, {"annotations": []})
        pr_rec = preds_by_img.get(img_id, {"detections": []})
        gt_boxes = _extract_boxes(gt_rec, key="annotations")
        pr_filt = _filter_preds_by_conf(_extract_boxes(pr_rec, key="detections"), conf_thr)
        n_pred_excl = sum(
            1 for p in pr_filt if float(p.get("score") or 0.0) > float(conf_hi)
        )
        adjusted[img_id] = {
            "n_gt": len(gt_boxes),
            "n_pred": n_pred_excl,
            "count_error": n_pred_excl - len(gt_boxes),
        }
    out = _counting_metrics(adjusted)
    out["excludes_conf_band_upper"] = float(conf_hi)
    out["note"] = "Count MAE with detections in ambiguous low-conf band removed (score <= conf_hi)."
    return out


def analyze_errors(
    *,
    gt: Any,
    preds: Any,
    conf_thr: float = 0.25,
    iou_thr: float = 0.5,
    iou_bg_thr: float = 0.1,
    ambiguity_conf_low: float | None = None,
    ambiguity_conf_high: float | None = None,
    pred_overlap_thr: float = 0.5,
) -> dict[str, Any]:
    """
    Light-mode error taxonomy from GT labels vs predictions.
    Taxonomy counts:
    - tp: matched (same class, IoU>=iou_thr)
    - fp: prediction not matched to any GT (localization/background buckets)
    - dupe: overlaps an already-matched GT (same class, IoU>=iou_thr)
    - fn: GT not matched to any prediction (same class)
    - cls_confusion: prediction overlaps a GT (IoU>=iou_thr) but wrong class

    Background vs localization uses dual IoU: t_b=iou_bg_thr (TIDE-style), t_f=iou_thr.
    """
    amb_lo = float(conf_thr) if ambiguity_conf_low is None else float(ambiguity_conf_low)
    amb_hi = float(conf_thr) + 0.15 if ambiguity_conf_high is None else float(ambiguity_conf_high)
    gt_by_img = _index_records_by_image_id(gt)
    preds_by_img = _index_records_by_image_id(preds)

    all_ids = set(gt_by_img.keys()) | set(preds_by_img.keys())
    counts = {"tp": 0, "fp": 0, "fn": 0, "cls_confusion": 0, "dupe": 0}
    fp_breakdown_counts = {"background": 0, "localization": 0, "classification": 0, "dupe": 0}
    per_image: dict[str, dict[str, Any]] = {}
    ambiguous_total = 0
    instance_events: list[dict[str, Any]] = []
    pred_outcomes: list[dict[str, Any]] = []

    for img_id in sorted(all_ids):
        gt_rec = gt_by_img.get(img_id, {"annotations": []})
        pr_rec = preds_by_img.get(img_id, {"detections": []})
        file_name = str(gt_rec.get("file_name") or pr_rec.get("file_name") or img_id)

        gt_boxes = _extract_boxes(gt_rec, key="annotations")
        pr_filt = _filter_preds_by_conf(_extract_boxes(pr_rec, key="detections"), conf_thr)

        gt_used = [False] * len(gt_boxes)
        img_tp = img_fp = img_fn = img_conf = img_dupe = 0
        img_fp_breakdown = {"background": 0, "localization": 0, "classification": 0, "dupe": 0}
        fp_examples: list[dict[str, Any]] = []
        ambiguous_detections: list[dict[str, Any]] = []

        for pi, p in enumerate(pr_filt):
            score = p.get("score")
            amb_flags: list[str] = []
            if score is not None:
                sc = float(score)
                if amb_lo <= sc <= amb_hi:
                    amb_flags.append("low_conf_band")
            if _pred_pred_max_iou(pi, pr_filt, same_class=int(p["category_id"])) >= pred_overlap_thr:
                amb_flags.append("pred_pred_overlap")
            if amb_flags:
                ambiguous_total += 1
                ambiguous_detections.append(
                    {
                        "image_id": img_id,
                        "file_name": file_name,
                        "bbox": p["bbox"],
                        "category_id": int(p["category_id"]),
                        "score": p["score"],
                        "flags": amb_flags,
                    }
                )

            best_i = find_same_class_tp_match(p, gt_boxes, gt_used, iou_thr=iou_thr)
            if best_i >= 0:
                gt_used[best_i] = True
                img_tp += 1
                pred_outcomes.append(
                    {"ambiguous": bool(amb_flags), "flags": list(amb_flags), "bucket": "tp"}
                )
                instance_events.append(
                    {
                        "error_type": "tp",
                        "bbox": p["bbox"],
                        "score": p.get("score"),
                        "image_id": img_id,
                    }
                )
                continue

            outcome, _confused_gt = classify_unmatched_prediction(
                p, gt_boxes, gt_used, iou_thr=iou_thr, iou_bg_thr=iou_bg_thr
            )
            if outcome == "dupe":
                img_dupe += 1
                img_fp_breakdown["dupe"] += 1
                pred_outcomes.append(
                    {"ambiguous": bool(amb_flags), "flags": list(amb_flags), "bucket": "dupe"}
                )
                instance_events.append(
                    {
                        "error_type": "dupe",
                        "bbox": p["bbox"],
                        "score": p.get("score"),
                        "image_id": img_id,
                    }
                )
                fp_examples.append(
                    {
                        "image_id": img_id,
                        "file_name": file_name,
                        "bbox": p["bbox"],
                        "category_id": int(p["category_id"]),
                        "score": p["score"],
                        "error_type": "dupe",
                    }
                )
                continue

            if outcome == "cls_confusion":
                img_conf += 1
                img_fp_breakdown["classification"] += 1
                pred_outcomes.append(
                    {
                        "ambiguous": bool(amb_flags),
                        "flags": list(amb_flags),
                        "bucket": "classification",
                    }
                )
                instance_events.append(
                    {
                        "error_type": "fp_classification",
                        "bbox": p["bbox"],
                        "score": p.get("score"),
                        "image_id": img_id,
                    }
                )
                fp_examples.append(
                    {
                        "image_id": img_id,
                        "file_name": file_name,
                        "bbox": p["bbox"],
                        "category_id": int(p["category_id"]),
                        "score": p["score"],
                        "error_type": "classification",
                    }
                )
                continue

            img_fp += 1
            if outcome == "fp_background":
                img_fp_breakdown["background"] += 1
                err_type = "background"
                pred_outcomes.append(
                    {
                        "ambiguous": bool(amb_flags),
                        "flags": list(amb_flags),
                        "bucket": "background",
                    }
                )
                instance_events.append(
                    {
                        "error_type": "fp_background",
                        "bbox": p["bbox"],
                        "score": p.get("score"),
                        "image_id": img_id,
                    }
                )
            else:
                img_fp_breakdown["localization"] += 1
                err_type = "localization"
                pred_outcomes.append(
                    {
                        "ambiguous": bool(amb_flags),
                        "flags": list(amb_flags),
                        "bucket": "localization",
                    }
                )
                instance_events.append(
                    {
                        "error_type": "fp_localization",
                        "bbox": p["bbox"],
                        "score": p.get("score"),
                        "image_id": img_id,
                    }
                )
            fp_examples.append(
                {
                    "image_id": img_id,
                    "file_name": file_name,
                    "bbox": p["bbox"],
                    "category_id": int(p["category_id"]),
                    "score": p["score"],
                    "error_type": err_type,
                }
            )

        img_fn = sum(1 for used in gt_used if not used)
        for i, g in enumerate(gt_boxes):
            if not gt_used[i]:
                instance_events.append(
                    {
                        "error_type": "fn",
                        "bbox": g["bbox"],
                        "score": None,
                        "image_id": img_id,
                    }
                )

        counts["tp"] += img_tp
        counts["fp"] += img_fp
        counts["fn"] += img_fn
        counts["cls_confusion"] += img_conf
        counts["dupe"] += img_dupe
        for k in fp_breakdown_counts:
            fp_breakdown_counts[k] += int(img_fp_breakdown[k])

        per_image[img_id] = {
            "image_id": img_id,
            "file_name": file_name,
            "tp": img_tp,
            "fp": img_fp,
            "fn": img_fn,
            "cls_confusion": img_conf,
            "dupe": img_dupe,
            "fp_breakdown": img_fp_breakdown,
            "fp_examples": fp_examples,
            "n_gt": len(gt_boxes),
            "n_pred": len(pr_filt),
            "count_error": len(pr_filt) - len(gt_boxes),
            "ambiguous_detections": ambiguous_detections,
        }

    top_fp = sorted(per_image.values(), key=lambda r: (int(r["fp"]), int(r["cls_confusion"])), reverse=True)

    fp_examples_all: list[dict[str, Any]] = []
    for r in per_image.values():
        fp_examples_all.extend(list(r.get("fp_examples") or []))
    fp_examples_all.sort(key=lambda x: float(x.get("score") or 0.0), reverse=True)

    # Reviewer-facing split: "localization" (background + low-IoU) vs "classification" (cross-class confusion).
    error_taxonomy = {
        "localization": {
            "fp": int(fp_breakdown_counts["background"]) + int(fp_breakdown_counts["localization"]),
            "fp_background": int(fp_breakdown_counts["background"]),
            "fp_low_iou": int(fp_breakdown_counts["localization"]),
            "fn": int(counts["fn"]),
        },
        "classification": {
            "fp": int(fp_breakdown_counts["classification"]),
            "fp_cross_class_confusion": int(fp_breakdown_counts["classification"]),
        },
        "duplicate": {
            "fp_dupe": int(fp_breakdown_counts["dupe"]),
        },
    }

    counting = _counting_metrics(per_image)
    counting_excl_ambiguous_band = _counting_metrics_excluding_conf_band(
        per_image,
        gt_by_img=gt_by_img,
        preds_by_img=preds_by_img,
        conf_thr=float(conf_thr),
        conf_hi=amb_hi,
    )
    bbox_area_strata = build_bbox_area_strata(instance_events)
    conf_taxonomy_grid = build_conf_taxonomy_grid(instance_events)
    ambiguous_fp_crosstab = build_ambiguous_fp_crosstab(
        pred_outcomes, conf_band=[amb_lo, amb_hi]
    )
    tide_bucket_summary = build_tide_bucket_summary(
        counts=counts,
        fp_breakdown=fp_breakdown_counts,
        match={"conf": conf_thr, "iou": iou_thr, "iou_bg": iou_bg_thr},
    )

    return {
        "counts": counts,
        "fp_breakdown": fp_breakdown_counts,
        "error_taxonomy": error_taxonomy,
        "bbox_area_strata": bbox_area_strata,
        "conf_taxonomy_grid": conf_taxonomy_grid,
        "counting_metrics": counting,
        "ambiguous_summary": {
            "n_ambiguous_detections": ambiguous_total,
            "conf_band": [amb_lo, amb_hi],
            "pred_overlap_thr": pred_overlap_thr,
        },
        "ambiguous_fp_crosstab": ambiguous_fp_crosstab,
        "tide_bucket_summary": tide_bucket_summary,
        "counting_metrics_excl_ambiguous_band": counting_excl_ambiguous_band,
        "match": {"conf": conf_thr, "iou": iou_thr, "iou_bg": iou_bg_thr},
        "top_fp_images": top_fp,
        "fp_examples": fp_examples_all,
    }


def export_topk_fp_crops(
    *,
    fp_examples: list[dict[str, Any]],
    out_dir: str | Path,
    topk: int,
    dataset_root: str | Path | None = None,
) -> dict[str, Any]:
    """
    Export crops for top-K false positives (by score).
    Requires image paths in `file_name`. Uses PIL only if available.
    """
    Image, pil_reason = _try_import_pil()
    if Image is None:
        return {"status": "skipped", "reason": pil_reason, "exported": 0, "out_dir": str(Path(out_dir))}

    out_dir_p = Path(out_dir)
    out_dir_p.mkdir(parents=True, exist_ok=True)

    exported: list[dict[str, Any]] = []
    attempted = 0
    for i, ex in enumerate(fp_examples[: max(0, int(topk))]):
        attempted += 1
        file_name = str(ex.get("file_name") or "")
        if not file_name:
            exported.append({**ex, "status": "skipped", "reason": "missing file_name"})
            continue

        img_path = Path(file_name)
        if not img_path.is_absolute() and dataset_root:
            img_path = Path(dataset_root) / img_path
        if not img_path.exists():
            exported.append({**ex, "status": "skipped", "reason": f"image not found: {img_path}"})
            continue

        bbox = ex.get("bbox")
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            exported.append({**ex, "status": "skipped", "reason": f"bad bbox: {bbox!r}"})
            continue

        try:
            with Image.open(img_path) as im:
                w, h = int(im.size[0]), int(im.size[1])
                xyxy = _clamp_xyxy_to_image(_as_xyxy(bbox), width=w, height=h)
                if xyxy is None:
                    exported.append({**ex, "status": "skipped", "reason": "empty crop after clamp"})
                    continue
                crop = im.crop(xyxy)
                out_name = f"fp_{i:05d}_{ex.get('error_type','fp')}_c{int(ex.get('category_id',0))}_s{float(ex.get('score') or 0.0):.3f}.png"
                out_path = out_dir_p / out_name
                crop.save(out_path)
                exported.append(
                    {
                        **ex,
                        "status": "ok",
                        "image_path": str(img_path),
                        "crop_path": str(out_path),
                        "crop_xyxy": list(xyxy),
                        "image_size": [w, h],
                    }
                )
        except Exception as e:  # pragma: no cover - I/O/decoder dependent
            exported.append({**ex, "status": "skipped", "reason": f"{type(e).__name__}: {e}"})

    return {
        "status": "ok",
        "out_dir": str(out_dir_p),
        "attempted": attempted,
        "exported": sum(1 for r in exported if r.get("status") == "ok"),
        "results": exported,
    }
