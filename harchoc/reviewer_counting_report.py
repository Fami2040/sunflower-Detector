"""Reviewer counting metrics: pooled MAE, per-class totals, relative-error distribution."""

from __future__ import annotations

import statistics
from pathlib import Path
from typing import Any

from harchoc.detection_match import (
    _extract_boxes,
    _filter_preds_by_conf,
    _index_records_by_image_id,
    per_image_detection_counts,
)
from harchoc.json_io import load_json
from harchoc.schemas import with_schema_version
from harchoc.sunflower_dataset import CLASS_NAMES_DICT
from harchoc.threshold_lock import load_locked_conf, load_locked_match_iou
from harchoc.threshold_protocol import counting_metrics_at_conf


def per_image_relative_error_summary(
    per_image: dict[str, dict[str, Any]],
) -> dict[str, float | int]:
    """Mean/median relative error (%) and fraction of images below thresholds."""
    rel: list[float] = []
    for rec in per_image.values():
        n_gt = int(rec.get("n_gt") or 0)
        if n_gt <= 0:
            continue
        n_pred = int(rec.get("n_pred") or 0)
        rel.append(100.0 * abs(n_pred - n_gt) / float(n_gt))
    if not rel:
        return {
            "mean": 0.0,
            "median": 0.0,
            "pct_below_2": 0.0,
            "pct_below_5": 0.0,
            "pct_below_10": 0.0,
            "n_with_gt": 0,
        }
    n = len(rel)
    return {
        "mean": round(statistics.mean(rel), 2),
        "median": round(statistics.median(rel), 2),
        "pct_below_2": round(100.0 * sum(x < 2.0 for x in rel) / n, 1),
        "pct_below_5": round(100.0 * sum(x < 5.0 for x in rel) / n, 1),
        "pct_below_10": round(100.0 * sum(x < 10.0 for x in rel) / n, 1),
        "n_with_gt": n,
    }


def per_class_detection_totals(
    *,
    gt: Any,
    preds: Any,
    conf_thr: float,
) -> dict[str, dict[str, int]]:
    """Pooled GT vs pred box counts by class name (boxes above conf for preds)."""
    gt_by_img = _index_records_by_image_id(gt)
    preds_by_img = _index_records_by_image_id(preds)
    totals: dict[str, dict[str, int]] = {
        name: {"n_gt": 0, "n_pred": 0} for name in CLASS_NAMES_DICT.values()
    }
    id_to_name = {int(k): v for k, v in CLASS_NAMES_DICT.items()}
    all_ids = set(gt_by_img.keys()) | set(preds_by_img.keys())
    for img_id in all_ids:
        gt_rec = gt_by_img.get(img_id, {"annotations": []})
        pr_rec = preds_by_img.get(img_id, {"detections": []})
        for ann in _extract_boxes(gt_rec, key="annotations"):
            name = id_to_name.get(int(ann["category_id"]), str(ann["category_id"]))
            if name not in totals:
                totals[name] = {"n_gt": 0, "n_pred": 0}
            totals[name]["n_gt"] += 1
        pr_filt = _filter_preds_by_conf(_extract_boxes(pr_rec, key="detections"), conf_thr)
        for ann in pr_filt:
            name = id_to_name.get(int(ann["category_id"]), str(ann["category_id"]))
            if name not in totals:
                totals[name] = {"n_gt": 0, "n_pred": 0}
            totals[name]["n_pred"] += 1
    return totals


def mean_gt_boxes_per_image(per_image: dict[str, dict[str, Any]]) -> float:
    if not per_image:
        return 0.0
    return sum(int(r.get("n_gt") or 0) for r in per_image.values()) / float(len(per_image))


def counting_block_for_pair(
    *,
    gt: Any,
    preds: Any,
    conf_thr: float,
    category_aware: bool = True,
) -> dict[str, Any]:
    """Pooled counting + per-class totals + relative-error distribution."""
    per_image = per_image_detection_counts(
        gt=gt,
        preds=preds,
        conf_thr=float(conf_thr),
        category_aware=category_aware,
    )
    pooled = counting_metrics_at_conf(
        gt=gt,
        preds=preds,
        conf_thr=float(conf_thr),
        category_aware=category_aware,
    )
    return {
        "mean_gt_boxes_per_image": round(mean_gt_boxes_per_image(per_image), 2),
        "pooled": pooled,
        "per_class_totals": per_class_detection_totals(
            gt=gt, preds=preds, conf_thr=float(conf_thr)
        ),
        "per_image_relative_error_pct": per_image_relative_error_summary(per_image),
        "n_images": len(per_image),
    }


def extract_protocol_from_eval(eval_doc: dict[str, Any] | None) -> dict[str, Any]:
    if not eval_doc:
        return {}
    pred_export = eval_doc.get("pred_export")
    if not isinstance(pred_export, dict):
        return {}
    out: dict[str, Any] = {}
    for key in ("export_conf", "export_max_det", "export_iou", "export_device"):
        if pred_export.get(key) is not None:
            out[key] = pred_export[key]
    return out


def _dual_metric_snapshot(path: str | Path) -> dict[str, Any] | None:
    p = Path(path)
    if not p.is_file():
        return None
    doc = load_json(p)
    if not isinstance(doc, dict):
        return None
    rows = doc.get("rows")
    return {
        "path": str(p),
        "operating_point": doc.get("operating_point"),
        "counting_sources": doc.get("counting_sources"),
        "rows": rows if isinstance(rows, list) else None,
    }


def build_reviewer_counting_report(
    *,
    locked_conf: float,
    locked_conf_from: str,
    test_gt_path: str,
    test_preds_path: str,
    val_gt_path: str | None = None,
    val_preds_path: str | None = None,
    eval_test_path: str | None = None,
    dual_metric_path: str | None = None,
    split_file: str | None = None,
    weights: str | None = None,
    comparisons: list[dict[str, Any]] | None = None,
    manuscript_docx: dict[str, Any] | None = None,
    match_iou: float | None = None,
    category_aware: bool = True,
) -> dict[str, Any]:
    """Assemble reviewer_counting_report.v1 from on-disk GT/preds exports."""
    gt_test = load_json(test_gt_path)
    preds_test = load_json(test_preds_path)
    test_block = counting_block_for_pair(
        gt=gt_test,
        preds=preds_test,
        conf_thr=locked_conf,
        category_aware=category_aware,
    )

    eval_test = None
    if eval_test_path and Path(eval_test_path).is_file():
        eval_test = load_json(eval_test_path)
        if not isinstance(eval_test, dict):
            eval_test = None

    protocol = extract_protocol_from_eval(eval_test)
    if match_iou is not None:
        protocol["match_iou"] = float(match_iou)
    protocol["locked_conf"] = float(locked_conf)
    protocol["category_aware"] = bool(category_aware)

    val_block: dict[str, Any] | None = None
    if val_gt_path and val_preds_path and Path(val_gt_path).is_file() and Path(val_preds_path).is_file():
        val_block = counting_block_for_pair(
            gt=load_json(val_gt_path),
            preds=load_json(val_preds_path),
            conf_thr=locked_conf,
            category_aware=category_aware,
        )

    comparison_rows: list[dict[str, Any]] = []
    for spec in comparisons or []:
        label = str(spec.get("label") or "comparison")
        gt_p = str(spec.get("gt_json") or test_gt_path)
        pr_p = str(spec.get("preds_json") or "")
        if not pr_p or not Path(pr_p).is_file():
            comparison_rows.append(
                {
                    "label": label,
                    "status": "missing_preds",
                    "preds_json": pr_p,
                }
            )
            continue
        block = counting_block_for_pair(
            gt=load_json(gt_p),
            preds=load_json(pr_p),
            conf_thr=locked_conf,
            category_aware=category_aware,
        )
        comparison_rows.append(
            {
                "label": label,
                "status": "ok",
                "gt_json": gt_p,
                "preds_json": pr_p,
                "test_mae": block["pooled"].get("mae"),
                "test_mae_ci": block["pooled"].get("mae_ci"),
                "test_rrmse": block["pooled"].get("rrmse"),
                "per_class_totals": block["per_class_totals"],
                "per_image_relative_error_pct": block["per_image_relative_error_pct"],
            }
        )

    payload: dict[str, Any] = {
        "status": "ok",
        "n_images": test_block["n_images"],
        "split_file": split_file,
        "weights": weights,
        "locked_conf": float(locked_conf),
        "locked_conf_from": locked_conf_from,
        "protocol": protocol,
        "inputs": {
            "test_gt_json": test_gt_path,
            "test_preds_json": test_preds_path,
            "val_gt_json": val_gt_path,
            "val_preds_json": val_preds_path,
            "eval_test_json": eval_test_path,
            "dual_metric_json": dual_metric_path,
        },
        "test": test_block,
        "pooled": test_block["pooled"],
        "per_image_relative_error_pct": test_block["per_image_relative_error_pct"],
        "per_class_totals": test_block["per_class_totals"],
        "mean_gt_boxes_per_image": test_block["mean_gt_boxes_per_image"],
    }
    if val_block is not None:
        payload["val"] = val_block
    dm = _dual_metric_snapshot(dual_metric_path) if dual_metric_path else None
    if dm is not None:
        payload["dual_metric"] = dm
    if comparison_rows:
        payload["comparisons"] = comparison_rows
    if manuscript_docx:
        payload["manuscript_docx"] = dict(manuscript_docx)
    return with_schema_version(payload, schema_version="reviewer_counting_report.v1")


def build_dry_run_report(
    *,
    out: str,
    fields: dict[str, Any],
) -> dict[str, Any]:
    """Placeholder when GT/preds are absent; documents intended inputs."""
    inputs = {
        "locked_conf_from": fields.get("locked_conf_from") or "reports/hsp/threshold_val.json",
        "test_gt_json": fields.get("gt_test") or "reports/hsp/gt_test.json",
        "test_preds_json": fields.get("preds_test") or "reports/hsp/preds_test.json",
        "val_gt_json": fields.get("gt_val") or "reports/hsp/gt_val.json",
        "val_preds_json": fields.get("preds_val") or "reports/hsp/preds_val.json",
        "eval_test_json": fields.get("eval_test") or "reports/hsp/eval_test.json",
        "dual_metric_json": fields.get("dual_metric") or "reports/hsp/dual_metric.json",
    }
    payload: dict[str, Any] = {
        "status": "dry-run",
        "script": "reviewer-counting",
        "out": str(Path(out)),
        "inputs": inputs,
        "weights": fields.get("weights") or "models/best2.pt",
        "regenerate_from_weights": [
            "mamba run -n harchoc python scripts/experiment.py repro",
            "mamba run -n harchoc python scripts/experiment.py reviewer-counting",
        ],
        "n_images": None,
        "locked_conf": None,
        "pooled": {"mae": None, "mae_ci": None, "rrmse": None},
        "per_image_relative_error_pct": {
            "mean": None,
            "median": None,
            "pct_below_2": None,
        },
    }
    return with_schema_version(payload, schema_version="reviewer_counting_report.v1")


def merge_reviewer_counting_fields(fields: dict[str, Any]) -> dict[str, Any]:
    """Normalize CLI/config field names."""
    return {
        "dry_run": bool(fields.get("dry_run")),
        "out": fields.get("out") or "reports/reviewer2_counting_metrics_computed.json",
        "locked_conf_from": fields.get("locked_conf_from") or "reports/hsp/threshold_val.json",
        "gt_test": fields.get("gt_test") or fields.get("gt_json"),
        "preds_test": fields.get("preds_test") or fields.get("preds_json"),
        "gt_val": fields.get("gt_val"),
        "preds_val": fields.get("preds_val"),
        "eval_test": fields.get("eval_test"),
        "dual_metric": fields.get("dual_metric"),
        "split_file": fields.get("split_file") or "data/splits/test.txt",
        "weights": fields.get("weights"),
        "comparisons": fields.get("comparisons"),
        "manuscript_docx": fields.get("manuscript_docx"),
        "category_aware": fields.get("category_aware", True),
    }


def run_reviewer_counting(fields: dict[str, Any]) -> dict[str, Any]:
    """Build report dict (caller writes JSON)."""
    norm = merge_reviewer_counting_fields(fields)
    out_path = str(norm["out"])
    locked_from = str(norm["locked_conf_from"])

    test_gt = str(norm["gt_test"] or "reports/hsp/gt_test.json")
    test_preds = str(norm["preds_test"] or "reports/hsp/preds_test.json")
    missing = [p for p in (test_gt, test_preds) if not Path(p).is_file()]

    if norm["dry_run"] or missing:
        return build_dry_run_report(out=out_path, fields=norm)

    locked_conf = load_locked_conf(locked_from)
    match_iou = load_locked_match_iou(locked_from)

    manuscript = norm.get("manuscript_docx")
    if manuscript is None:
        manuscript = {"mean_relative_pct": 13.2, "pct_below_2": 80}

    return build_reviewer_counting_report(
        locked_conf=locked_conf,
        locked_conf_from=locked_from,
        test_gt_path=test_gt,
        test_preds_path=test_preds,
        val_gt_path=str(norm["gt_val"]) if norm.get("gt_val") else "reports/hsp/gt_val.json",
        val_preds_path=str(norm["preds_val"]) if norm.get("preds_val") else "reports/hsp/preds_val.json",
        eval_test_path=str(norm["eval_test"]) if norm.get("eval_test") else "reports/hsp/eval_test.json",
        dual_metric_path=str(norm["dual_metric"]) if norm.get("dual_metric") else "reports/hsp/dual_metric.json",
        split_file=str(norm.get("split_file") or ""),
        weights=str(norm["weights"]) if norm.get("weights") else "models/best2.pt",
        comparisons=norm.get("comparisons"),
        manuscript_docx=manuscript if isinstance(manuscript, dict) else None,
        match_iou=match_iou,
        category_aware=bool(norm.get("category_aware", True)),
    )
