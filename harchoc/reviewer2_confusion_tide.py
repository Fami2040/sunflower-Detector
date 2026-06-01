"""Reviewer 2 §11 confusion + TIDE bucket audit (CPU recompute from HSP exports)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from harchoc.detection_confusion import confusion_matrix_from_exports
from harchoc.error_analysis_core import analyze_errors
from harchoc.hsp_export_protocol import EXPORT_IOU
from harchoc.schemas import with_schema_version
from harchoc.tide_summary import build_tide_bucket_summary

REVIEWER2_CONFUSION_TIDE_V1 = "reviewer2_confusion_tide.v1"
SECTION11_IOU = 0.5
SECTION11_IOU_BG = 0.1
DEFAULT_OUT = "reports/reviewer2_confusion_tide.json"

DEFAULT_PATHS = {
    "gt_json": "reports/hsp/gt_test.json",
    "preds_json": "reports/hsp/preds_test.json",
    "locked_conf_from": "reports/hsp/threshold_val.json",
    "error_report": "reports/hsp/error_test_report.json",
    "tide_summary": "reports/hsp/tide_bucket_summary.json",
    "confusion_iou_0_3": "reports/hsp/best2_test_confusion.json",
}


def _pct(n: float, d: float) -> float | None:
    if not d:
        return None
    return 100.0 * n / d


def _approx_equal(a: float, b: float, *, rtol: float = 1e-9, atol: float = 1e-6) -> bool:
    return abs(float(a) - float(b)) <= atol + rtol * max(abs(float(a)), abs(float(b)), 1.0)


def _audit_row(
    *,
    claim: str,
    api: str,
    expected: Any,
    actual: Any,
    match: bool,
) -> dict[str, Any]:
    return {
        "claim": claim,
        "api": api,
        "expected": expected,
        "actual": actual,
        "match": bool(match),
    }


def build_reviewer2_confusion_tide_payload(
    *,
    gt: dict[str, Any],
    preds: dict[str, Any],
    conf_thr: float,
    iou_bg_thr: float = SECTION11_IOU_BG,
    stored_error: dict[str, Any] | None = None,
    stored_tide: dict[str, Any] | None = None,
    stored_confusion_iou_0_3: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
  Recompute §11 matcher stats (IoU 0.5), TIDE proxy buckets, and optional IoU 0.3 matrix.

  Uses ``analyze_errors`` + ``confusion_matrix_from_exports`` (same matcher as ``error_analysis.py``).
    """
    summary = analyze_errors(
        gt=gt,
        preds=preds,
        conf_thr=float(conf_thr),
        iou_thr=SECTION11_IOU,
        iou_bg_thr=float(iou_bg_thr),
    )
    counts = dict(summary["counts"])
    fp_breakdown = dict(summary["fp_breakdown"])
    match = dict(summary.get("match") or {"conf": conf_thr, "iou": SECTION11_IOU, "iou_bg": iou_bg_thr})

    acc_05 = confusion_matrix_from_exports(
        gt,
        preds,
        conf_thr=float(conf_thr),
        iou_thr=SECTION11_IOU,
        iou_bg_thr=float(iou_bg_thr),
    )
    acc_03 = confusion_matrix_from_exports(
        gt,
        preds,
        conf_thr=float(conf_thr),
        iou_thr=float(EXPORT_IOU),
        iou_bg_thr=float(iou_bg_thr),
    )
    tide_recomputed = build_tide_bucket_summary(
        counts=counts,
        fp_breakdown=fp_breakdown,
        match=match,
    )

    fp = int(counts.get("fp") or 0)
    tp = int(counts.get("tp") or 0)
    cls = int(counts.get("cls_confusion") or 0)
    loc = int(fp_breakdown.get("localization") or 0)
    bkg = int(fp_breakdown.get("background") or 0)

    derived = {
        "n_images": int(summary.get("counting_metrics", {}).get("n_images") or acc_05.n_images),
        "fp_loc_pct": _pct(loc, fp),
        "fp_bkg_pct": _pct(bkg, fp),
        "cls_over_tp_pct": _pct(cls, tp),
        "matrix_off_diagonal_cls": int(acc_05.matrix[0][1] + acc_05.matrix[1][0]),
        "delta_ap_share_loc_pct": 100.0 * float(tide_recomputed["delta_ap_share"]["Loc"]),
        "delta_ap_share_bkg_pct": 100.0 * float(tide_recomputed["delta_ap_share"]["Bkg"]),
        "delta_ap_share_cls_pct": 100.0 * float(tide_recomputed["delta_ap_share"]["Cls"]),
        "delta_ap_share_miss_pct": 100.0 * float(tide_recomputed["delta_ap_share"]["Miss"]),
    }

    parity: dict[str, bool | None] = {
        "confusion_iou_0_5_matches_error_counts": all(
            acc_05.stats[k] == int(counts[k])
            for k in ("tp", "fp", "fn", "cls_confusion", "dupe")
        ),
        "off_diagonal_equals_cls_confusion": acc_05.stats["cls_confusion"]
        == derived["matrix_off_diagonal_cls"],
        "recomputed_tide_equals_stored": None,
        "recomputed_counts_equal_stored_error": None,
        "stored_confusion_iou_0_3_matches_recompute": None,
        "stored_confusion_iou_0_3_differs_from_section11_fp": None,
    }

    if stored_tide is not None:
        parity["recomputed_tide_equals_stored"] = (
            tide_recomputed.get("buckets") == stored_tide.get("buckets")
            and tide_recomputed.get("loc_plus_bkg_over_cls_ratio")
            == stored_tide.get("loc_plus_bkg_over_cls_ratio")
        )
    if stored_error is not None:
        stored_counts = stored_error.get("counts") or {}
        parity["recomputed_counts_equal_stored_error"] = all(
            int(counts[k]) == int(stored_counts[k])
            for k in ("tp", "fp", "fn", "cls_confusion", "dupe")
        )
    if stored_confusion_iou_0_3 is not None:
        stored_stats = stored_confusion_iou_0_3.get("stats") or {}
        parity["stored_confusion_iou_0_3_matches_recompute"] = dict(acc_03.stats) == dict(stored_stats)
        parity["stored_confusion_iou_0_3_differs_from_section11_fp"] = int(stored_stats.get("fp") or 0) != fp

    audit: list[dict[str, Any]] = []
    if stored_error is not None:
        sc = stored_error.get("counts") or {}
        sfb = stored_error.get("fp_breakdown") or {}
        audit.extend(
            [
                _audit_row(
                    claim="Test n_images = 109",
                    api="error_test_report.json → counting_metrics.n_images",
                    expected=109,
                    actual=derived["n_images"],
                    match=derived["n_images"] == 109,
                ),
                _audit_row(
                    claim="FP count @ §11 IoU 0.5",
                    api="error_test_report.json → counts.fp",
                    expected=int(sc.get("fp") or 0),
                    actual=fp,
                    match=int(sc.get("fp") or 0) == fp,
                ),
                _audit_row(
                    claim="62% of FPs are localization",
                    api="fp_breakdown.localization / counts.fp",
                    expected=round(100.0 * int(sfb.get("localization") or 0) / max(int(sc.get("fp") or 0), 1), 2),
                    actual=round(derived["fp_loc_pct"] or 0.0, 2),
                    match=_approx_equal(derived["fp_loc_pct"] or 0.0, 62.19, atol=0.1),
                ),
                _audit_row(
                    claim="38% of FPs are background",
                    api="fp_breakdown.background / counts.fp",
                    expected=round(100.0 * int(sfb.get("background") or 0) / max(int(sc.get("fp") or 0), 1), 2),
                    actual=round(derived["fp_bkg_pct"] or 0.0, 2),
                    match=_approx_equal(derived["fp_bkg_pct"] or 0.0, 37.81, atol=0.1),
                ),
                _audit_row(
                    claim="cls confusion ~6.6% of TP",
                    api="counts.cls_confusion / counts.tp",
                    expected=round(100.0 * int(sc.get("cls_confusion") or 0) / max(int(sc.get("tp") or 0), 1), 2),
                    actual=round(derived["cls_over_tp_pct"] or 0.0, 2),
                    match=_approx_equal(derived["cls_over_tp_pct"] or 0.0, 6.64, atol=0.05),
                ),
            ]
        )
    if stored_tide is not None:
        audit.append(
            _audit_row(
                claim="localization_dominates_classification",
                api="tide_bucket_summary.json",
                expected=True,
                actual=tide_recomputed.get("localization_dominates_classification"),
                match=bool(tide_recomputed.get("localization_dominates_classification")),
            )
        )
        audit.append(
            _audit_row(
                claim="loc_plus_bkg_over_cls_ratio ~15.07",
                api="tide_bucket_summary.json → loc_plus_bkg_over_cls_ratio",
                expected=stored_tide.get("loc_plus_bkg_over_cls_ratio"),
                actual=tide_recomputed.get("loc_plus_bkg_over_cls_ratio"),
                match=_approx_equal(
                    float(tide_recomputed.get("loc_plus_bkg_over_cls_ratio") or 0.0),
                    float(stored_tide.get("loc_plus_bkg_over_cls_ratio") or 0.0),
                    atol=1e-3,
                ),
            )
        )
    if stored_confusion_iou_0_3 is not None:
        sfp = int((stored_confusion_iou_0_3.get("stats") or {}).get("fp") or 0)
        audit.append(
            _audit_row(
                claim="best2_test_confusion.json fp ≠ §11 fp (IoU 0.3 vs 0.5)",
                api="best2_test_confusion.json stats.fp vs error counts.fp",
                expected=f"{sfp} ≠ {fp}",
                actual=f"{sfp} ≠ {fp}",
                match=sfp != fp,
            )
        )

    return {
        "protocol": {
            "section11": {
                "conf": float(conf_thr),
                "iou": SECTION11_IOU,
                "iou_bg": float(iou_bg_thr),
                "note": "MS-FP-LOC-NARR / gap §11 — error_analysis + TIDE proxy",
            },
            "hsp_confusion_export": {
                "iou": float(EXPORT_IOU),
                "note": "eval.py --confusion-matrix-* default match IoU (counting-first)",
            },
        },
        "recomputed": {
            "counts": counts,
            "fp_breakdown": fp_breakdown,
            "match": match,
            "derived_metrics": derived,
            "confusion_iou_0_5": acc_05.to_payload(conf_thr=float(conf_thr), iou_thr=SECTION11_IOU),
            "confusion_iou_0_3": acc_03.to_payload(
                conf_thr=float(conf_thr), iou_thr=float(EXPORT_IOU)
            ),
            "tide_bucket_summary": tide_recomputed,
        },
        "parity": parity,
        "audit": audit,
    }


def reproduce_commands(*, config_path: str = "configs/experiments/error_analysis_test.json") -> dict[str, list[str]]:
    """CLI strings for appendix (CPU audit vs full HSP regen)."""
    return {
        "cpu_audit": [
            "PYTHONPATH=. HARCHOC_ALLOW_BASE_PYTHON=1 python scripts/experiment.py reviewer2-confusion",
            "PYTHONPATH=. HARCHOC_ALLOW_BASE_PYTHON=1 python scripts/experiment.py reviewer2-confusion --dry-run",
            "PYTHONPATH=. HARCHOC_ALLOW_BASE_PYTHON=1 python scripts/eval.py "
            "--confusion-matrix-only --confusion-from-exports "
            "--export-gt-json reports/hsp/gt_test.json --export-preds-json reports/hsp/preds_test.json "
            "--confusion-matrix-out reports/hsp/best2_test_confusion.json "
            "--locked-conf-from reports/hsp/threshold_val.json --weights models/best2.pt "
            "--out reports/hsp/best2_confusion_from_exports_run.json",
        ],
        "gpu_regenerate_exports": [
            f"mamba run -n harchoc python scripts/error_analysis.py --config {config_path}",
            "mamba run -n harchoc python scripts/eval.py --weights models/best2.pt "
            "--export-only --split-file data/splits/test.txt "
            "--out reports/hsp/eval_test.json",
            "mamba run -n harchoc python scripts/eval.py --confusion-matrix-only "
            "--weights models/best2.pt --confusion-matrix-splits test "
            "--confusion-matrix-out reports/hsp/best2 --locked-conf-from reports/hsp/threshold_val.json",
        ],
    }


def run_reviewer2_confusion_tide(
    *,
    repo_root: Path,
    out: str | Path = DEFAULT_OUT,
    gt_json: str | None = None,
    preds_json: str | None = None,
    locked_conf_from: str | None = None,
    error_report: str | None = None,
    tide_summary: str | None = None,
    confusion_iou_0_3: str | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    from harchoc.json_io import load_json
    from harchoc.threshold_lock import load_locked_conf

    paths = {**DEFAULT_PATHS}
    if gt_json:
        paths["gt_json"] = gt_json
    if preds_json:
        paths["preds_json"] = preds_json
    if locked_conf_from:
        paths["locked_conf_from"] = locked_conf_from
    if error_report:
        paths["error_report"] = error_report
    if tide_summary:
        paths["tide_summary"] = tide_summary
    if confusion_iou_0_3:
        paths["confusion_iou_0_3"] = confusion_iou_0_3

    repro = reproduce_commands()
    if dry_run:
        return with_schema_version(
            {
                "status": "dry-run",
                "script": "reviewer2-confusion",
                "out": str(Path(out)),
                "inputs": paths,
                "reproduce": repro,
            },
            schema_version=REVIEWER2_CONFUSION_TIDE_V1,
        )

    def _load_optional(rel: str) -> dict[str, Any] | None:
        p = (repo_root / rel).resolve()
        return load_json(p) if p.is_file() else None

    locked_path = (repo_root / paths["locked_conf_from"]).resolve()
    conf_thr = load_locked_conf(str(locked_path)) if locked_path.is_file() else 0.15

    gt = load_json((repo_root / paths["gt_json"]).resolve())
    preds = load_json((repo_root / paths["preds_json"]).resolve())
    stored_error = _load_optional(paths["error_report"])
    stored_tide = _load_optional(paths["tide_summary"])
    stored_cm03 = _load_optional(paths["confusion_iou_0_3"])

    body = build_reviewer2_confusion_tide_payload(
        gt=gt,
        preds=preds,
        conf_thr=conf_thr,
        stored_error=stored_error,
        stored_tide=stored_tide,
        stored_confusion_iou_0_3=stored_cm03,
    )
    payload = with_schema_version(
        {
            "status": "ok",
            "inputs": paths,
            "locked_conf_from": str(locked_path.relative_to(repo_root))
            if locked_path.is_file()
            else paths["locked_conf_from"],
            "reproduce": repro,
            **body,
        },
        schema_version=REVIEWER2_CONFUSION_TIDE_V1,
    )
    return payload
