from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import sys

_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))
from harchoc.script_entry import bootstrap_repo_imports

bootstrap_repo_imports()

from harchoc.config_coerce import as_dict, coerce_float, coerce_int, optional_str
from harchoc.datasets import describe_dataset, resolve_dataset
from harchoc.error_analysis_core import analyze_errors, export_topk_fp_crops
from harchoc.run_metadata import collect_run_metadata
from harchoc.error_analysis_schema import (
    ERROR_ANALYSIS_REPORT_V1,
    ERROR_ANALYSIS_SUMMARY_V1,
    validate_error_analysis_payload,
)
from harchoc.schemas import with_schema_version
from harchoc.hsp_weights import HSP_DETECTION_WEIGHTS
from harchoc.threshold_lock import load_locked_conf
from harchoc.experiment_cli import add_locked_conf_args
from scripts._common_cli import (
    add_dataset_args,
    add_dry_run_arg,
    add_light_mode_arg,
    cli_print,
    eprint,
    read_json,
    require_existing_dir,
    resolve_light_gt_preds,
    write_json,
)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Export false positive analysis artifacts (scaffold).")
    p.add_argument(
        "--config",
        action="append",
        default=[],
        help="Path to JSON experiment config. Can be repeated; later entries override earlier.",
    )
    add_dataset_args(p)
    add_dry_run_arg(p)
    add_light_mode_arg(p)
    p.add_argument("--weights", default=HSP_DETECTION_WEIGHTS, help="Path to model weights.")
    p.add_argument("--out", default="reports/error_analysis/summary.json", help="Where to write JSON summary.")
    p.add_argument("--report", default="reports/error_analysis/report.json", help="Where to write taxonomy JSON report.")
    p.add_argument("--topk", type=int, default=50, help="How many examples to export.")
    p.add_argument("--export-fp-crops", action="store_true", help="Export top-K FP crops if images available.")
    p.add_argument("--fp-crops-dir", default="reports/error_analysis/fp_crops", help="Directory for FP crops.")
    p.add_argument("--fp-crops-topk", type=int, default=100, help="Top-K FP crops to export (by score).")
    p.add_argument("--gt-json", default="", help="GT labels JSON.")
    p.add_argument("--preds-json", default="", help="Predictions/detections JSON.")
    p.add_argument("--conf", type=float, default=0.25, help="Confidence threshold (preds score filter).")
    p.add_argument("--iou", type=float, default=0.5, help="IoU threshold for matching (t_f).")
    p.add_argument(
        "--iou-bg",
        type=float,
        default=0.1,
        help="Background IoU threshold (t_b); max IoU below this => background FP.",
    )
    add_locked_conf_args(p)
    p.add_argument(
        "--tide-out",
        default="",
        help="Optional path for TIDE bucket summary JSON (e.g. reports/hsp/tide_bucket_summary.json).",
    )
    p.add_argument(
        "--tidecv",
        action="store_true",
        help="Attempt official tidecv delta-AP (optional dep); writes tidecv_compare.v1 sidecar.",
    )
    p.add_argument(
        "--tidecv-compare-out",
        default="",
        help="Path for tidecv_compare.v1 JSON (default: sibling of --report, *_tidecv_compare.json).",
    )
    p.add_argument(
        "--confusion-matrix-out",
        default="",
        help="Optional 3×3 class confusion matrix JSON (from --gt-json / --preds-json; no re-inference).",
    )
    args = p.parse_args(argv)

    from harchoc.experiment_cli import (
        apply_dataset_args,
        merge_config_objects,
        pick_cli_or_section,
        section_and_dataset_from_config,
    )

    config_obj = merge_config_objects(list(args.config))
    analysis_cfg_obj, dataset_cfg_obj = section_and_dataset_from_config(config_obj, "error_analysis")
    apply_dataset_args(args, dataset_cfg_obj)

    def _pick(name: str, *, default: object) -> object:
        return pick_cli_or_section(args, name, section_cfg=analysis_cfg_obj, default=default)

    args.weights = str(_pick("weights", default=HSP_DETECTION_WEIGHTS))
    args.out = str(_pick("out", default="reports/hsp/error_analysis_summary.json"))
    args.report = str(_pick("report", default="reports/hsp/error_test_report.json"))
    args.topk = coerce_int(_pick("topk", default=50)) or 50
    args.export_fp_crops = bool(_pick("export_fp_crops", default=False))
    args.fp_crops_dir = str(_pick("fp_crops_dir", default="reports/hsp/error_fp_crops"))
    args.fp_crops_topk = coerce_int(_pick("fp_crops_topk", default=100)) or 100
    args.gt_json = str(_pick("gt_json", default=""))
    args.preds_json = str(_pick("preds_json", default=""))
    args.conf = coerce_float(_pick("conf", default=0.25)) or 0.25
    args.iou = coerce_float(_pick("iou", default=0.5)) or 0.5
    args.iou_bg = coerce_float(_pick("iou_bg", default=0.1)) or 0.1
    args.locked_conf_from = str(_pick("locked_conf_from", default=""))
    args.tide_out = str(_pick("tide_out", default=""))
    args.tidecv = bool(_pick("tidecv", default=False))
    args.tidecv_compare_out = str(_pick("tidecv_compare_out", default=""))
    args.confusion_matrix_out = str(_pick("confusion_matrix_out", default=""))
    args.light = bool(_pick("light", default=False))
    args.dry_run = bool(_pick("dry_run", default=False))

    repo_root = Path(__file__).resolve().parents[1]

    locked_from = (args.locked_conf_from or "").strip()
    if locked_from:
        args.conf = load_locked_conf(locked_from)

    if args.dry_run:
        out_path = write_json(
            args.out,
            with_schema_version(
                {
                "status": "dry-run",
                "script": "error_analysis",
                "light": bool(args.light),
                "meta": collect_run_metadata(
                    repo_root=repo_root,
                    dataset_manifest=Path(args.manifest),
                    extra_files={
                        "weights": str(Path(args.weights)),
                        "gt_json": str(Path(args.gt_json)) if str(args.gt_json).strip() else "",
                        "preds_json": str(Path(args.preds_json)) if str(args.preds_json).strip() else "",
                    },
                ),
                "weights": str(Path(args.weights)),
                "topk": args.topk,
                "out": str(Path(args.out)),
                "report": str(Path(args.report)),
                },
                schema_version=ERROR_ANALYSIS_SUMMARY_V1,
            ),
        )
        cli_print(f"Wrote {out_path}")
        return 0

    gt_json = (args.gt_json or "").strip()
    preds_json = (args.preds_json or "").strip()
    if args.light:
        gt_path, preds_path = resolve_light_gt_preds(repo_root=repo_root, gt_json=gt_json, preds_json=preds_json)
        gt_json, preds_json = str(gt_path), str(preds_path)
    elif not gt_json or not preds_json:
        eprint("error_analysis: requires --gt-json and --preds-json, or --light (data/examples/).")
        eprint("See data/examples/README.md for exporting real eval preds.")
        raise SystemExit(2)

    spec = resolve_dataset(
        manifest_path=args.manifest,
        default_dataset_name=args.default_dataset_name,
        dataset_name=optional_str(args.dataset_name),
        dataset_root=optional_str(args.dataset_root),
        yolo_data_yaml=optional_str(args.yolo_data_yaml),
    )
    if args.light:
        dataset_desc: dict[str, Any] = (
            as_dict(describe_dataset(spec))
            if spec.root.is_dir()
            else {"note": "light mode without DATASET_ROOT"}
        )
    else:
        require_existing_dir(spec.root, what="Dataset root", hint="Export DATASET_ROOT=/path/to/extracted/dataset")
        dataset_desc = as_dict(describe_dataset(spec))

    gt_obj = read_json(gt_json)
    preds_obj = read_json(preds_json)
    summary = analyze_errors(
        gt=gt_obj,
        preds=preds_obj,
        conf_thr=float(args.conf),
        iou_thr=float(args.iou),
        iou_bg_thr=float(args.iou_bg),
    )

    payload = with_schema_version(
        {
        "status": "ok",
        "light": bool(args.light),
        "meta": collect_run_metadata(
            repo_root=repo_root,
            dataset_manifest=spec.manifest_path,
            extra_files={
                "weights": str(Path(args.weights)),
                "gt_json": str(Path(gt_json)),
                "preds_json": str(Path(preds_json)),
            },
        ),
        "dataset": {"description": dataset_desc},
        "weights": str(Path(args.weights)),
        "inputs": {"gt_json": str(Path(gt_json)), "preds_json": str(Path(preds_json))},
        "match": summary.get("match") or {"conf": float(args.conf), "iou": float(args.iou)},
        "counts": summary["counts"],
        "fp_breakdown": summary["fp_breakdown"],
        "error_taxonomy": summary["error_taxonomy"],
        "bbox_area_strata": summary["bbox_area_strata"],
        "conf_taxonomy_grid": summary["conf_taxonomy_grid"],
        "counting_metrics": summary["counting_metrics"],
        "counting_metrics_excl_ambiguous_band": summary["counting_metrics_excl_ambiguous_band"],
        "ambiguous_summary": summary["ambiguous_summary"],
        "ambiguous_fp_crosstab": summary["ambiguous_fp_crosstab"],
        "tide_bucket_summary": summary["tide_bucket_summary"],
        "top_fp_images": summary["top_fp_images"][: int(args.topk)],
        },
        schema_version=ERROR_ANALYSIS_SUMMARY_V1,
    )
    validate_error_analysis_payload(payload, schema_version=ERROR_ANALYSIS_SUMMARY_V1)
    out_path = write_json(args.out, payload)
    cli_print(f"Wrote {out_path}")

    report = with_schema_version(
        {
        "status": "ok",
        "light": bool(args.light),
        "meta": collect_run_metadata(
            repo_root=repo_root,
            dataset_manifest=spec.manifest_path,
            extra_files={
                "gt_json": str(Path(gt_json)),
                "preds_json": str(Path(preds_json)),
            },
        ),
        "dataset": {"description": dataset_desc},
        "inputs": {"gt_json": str(Path(gt_json)), "preds_json": str(Path(preds_json))},
        "match": summary.get("match") or {"conf": float(args.conf), "iou": float(args.iou)},
        "counts": summary["counts"],
        "fp_breakdown": summary["fp_breakdown"],
        "error_taxonomy": summary["error_taxonomy"],
        "bbox_area_strata": summary["bbox_area_strata"],
        "conf_taxonomy_grid": summary["conf_taxonomy_grid"],
        "counting_metrics": summary["counting_metrics"],
        "counting_metrics_excl_ambiguous_band": summary["counting_metrics_excl_ambiguous_band"],
        "ambiguous_summary": summary["ambiguous_summary"],
        "ambiguous_fp_crosstab": summary["ambiguous_fp_crosstab"],
        "tide_bucket_summary": summary["tide_bucket_summary"],
        "top_fp_examples": summary["fp_examples"][: int(args.topk)],
        },
        schema_version=ERROR_ANALYSIS_REPORT_V1,
    )
    validate_error_analysis_payload(report, schema_version=ERROR_ANALYSIS_REPORT_V1)

    if bool(args.export_fp_crops):
        report["fp_crops"] = export_topk_fp_crops(
            fp_examples=summary["fp_examples"],
            out_dir=args.fp_crops_dir,
            topk=int(args.fp_crops_topk),
            dataset_root=spec.root,
        )
    else:
        report["fp_crops"] = {"status": "skipped", "reason": "not requested (pass --export-fp-crops)"}

    if bool(args.tidecv):
        from harchoc.tide_summary import (
            build_tidecv_compare,
            default_tidecv_compare_path,
            try_run_tidecv,
        )

        tidecv_result = try_run_tidecv(gt=gt_obj, preds=preds_obj)
        compare_body = build_tidecv_compare(
            tide_bucket_summary=summary["tide_bucket_summary"],
            tidecv_result=tidecv_result or {},
        )
        compare_path = (args.tidecv_compare_out or "").strip() or default_tidecv_compare_path(args.report)
        compare_payload = with_schema_version(compare_body, schema_version="tidecv_compare.v1")
        compare_written = write_json(compare_path, compare_payload)
        cli_print(f"Wrote {compare_written}")

        report["tidecv"] = tidecv_result
        report["tidecv_compare"] = {
            "path": str(Path(compare_written)),
            "status": compare_body["status"],
            "adapter_ok": compare_body["adapter_ok"],
            "skipped_reason": compare_body.get("skipped_reason"),
        }

    report_path = write_json(args.report, report)
    cli_print(f"Wrote {report_path}")

    tide_out = (args.tide_out or "").strip()
    if tide_out:
        tide_payload = with_schema_version(
            {
                "status": "ok",
                "inputs": {"gt_json": str(Path(gt_json)), "preds_json": str(Path(preds_json))},
                **dict(summary["tide_bucket_summary"]),
            },
            schema_version="tide_bucket_summary.v1",
        )
        tide_path = write_json(tide_out, tide_payload)
        cli_print(f"Wrote {tide_path}")

    confusion_out = (args.confusion_matrix_out or "").strip()
    if confusion_out:
        from harchoc.detection_confusion import (
            confusion_matrix_from_exports,
            format_confusion_matrix_text,
            resolve_match_settings,
        )

        cm_conf = float(args.conf)
        cm_iou = float(args.iou)
        if locked_from:
            cm_conf, cm_iou = resolve_match_settings(
                conf=cm_conf,
                iou=cm_iou,
                locked_conf_from=locked_from,
            )
        acc = confusion_matrix_from_exports(
            gt_obj,
            preds_obj,
            conf_thr=cm_conf,
            iou_thr=cm_iou,
        )
        cm_payload = acc.to_payload(
            conf_thr=cm_conf,
            iou_thr=cm_iou,
            weights=str(Path(args.weights)),
        )
        cm_path = write_json(confusion_out, cm_payload)
        cli_print(format_confusion_matrix_text(acc.matrix, acc.stats))
        cli_print(f"Wrote {cm_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

