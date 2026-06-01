"""Val threshold sweep; Methods prose: reports/hsp/p0_summary.md § Operating point."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import sys; from pathlib import Path; _r = Path(__file__).resolve().parent.parent; (str(_r) not in sys.path) and sys.path.insert(0, str(_r)); from harchoc.script_entry import bootstrap_repo_imports; bootstrap_repo_imports()

from harchoc.datasets import describe_dataset, resolve_dataset
from harchoc.strict_ml import capture_failure, fail_or_warn
from harchoc.detection_match import image_ids_union, match_counts_for_threshold
from harchoc.experiment_config import load_config_json, merge_experiment_config, script_section_from_config
from harchoc.hsp_weights import HSP_DETECTION_WEIGHTS
from harchoc.fp_budget_sweep import (
    build_fp_budget_sweep_payload,
    write_fp_budget_sweep,
)
from harchoc.threshold_lock import (
    build_locked_block,
    load_locked_conf,
    load_locked_match_iou,
    metrics_row_at_conf,
)
from harchoc.threshold_protocol import (
    SelectMode,
    build_iou_grid,
    counting_metrics_at_conf,
    enforce_tuning_guardrails,
    infer_split_role,
    resolve_dataset_root_for_splits,
    select_best_iou_from_grid,
    select_min_count_mae,
    select_operating_point as _select_operating_point,
    tuning_active,
)
from harchoc.run_metadata import collect_run_metadata
from harchoc.schemas import with_schema_version
from harchoc.calibration_metrics import reliability_and_ece
from harchoc.platt import apply_calibration_to_preds
from scripts._common_cli import (
    add_dataset_args,
    add_dry_run_arg,
    add_light_mode_arg,
    cli_print,
    eprint,
    read_json,
    require_conda_env,
    require_existing_dir,
    resolve_light_gt_preds,
    write_json,
)


def _metrics(tp: int, fp: int, fn: int) -> dict[str, float]:
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0
    return {"precision": prec, "recall": rec, "f1": f1}


def write_compact_csv(path: str | Path, rows: list[dict[str, Any]]) -> Path:
    """
    Write a compact CSV of sweep rows (no pandas).
    The schema is stable so downstream tools can parse it reliably.
    """
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)

    fields = [
        "conf_thr",
        "tp",
        "fp",
        "fn",
        "precision",
        "recall",
        "f1",
        "fp_per_image",
    ]

    with out.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in fields})
    return out


SelectMode = SelectMode  # re-export for tests


def _best_f1(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    from harchoc.threshold_protocol import _best_f1 as _best_f1_impl

    return _best_f1_impl(rows)


def select_operating_point(
    rows: list[dict[str, Any]],
    *,
    mode: SelectMode = "best_f1",
    min_recall: float | None = None,
    min_precision: float | None = None,
    max_fp_per_image: float | None = None,
) -> dict[str, Any] | None:
    """Re-export from harchoc.threshold_protocol (tests import from here)."""
    return _select_operating_point(
        rows,
        mode=mode,
        min_recall=min_recall,
        min_precision=min_precision,
        max_fp_per_image=max_fp_per_image,
    )


def _select_row(
    rows: list[dict[str, Any]],
    *,
    mode: str,
    gt_obj: Any,
    preds_obj: Any,
    iou_thr: float,
    category_aware: bool,
    min_recall: float | None = None,
    min_precision: float | None = None,
    max_fp_per_image: float | None = None,
) -> dict[str, Any] | None:
    if mode == "min_count_mae":
        return select_min_count_mae(
            rows,
            gt=gt_obj,
            preds=preds_obj,
            iou_thr=float(iou_thr),
            category_aware=category_aware,
        )
    return select_operating_point(
        rows,
        mode=mode,  # type: ignore[arg-type]
        min_recall=min_recall,
        min_precision=min_precision,
        max_fp_per_image=max_fp_per_image,
    )


def _linspace(tmin: float, tmax: float, steps: int) -> list[float]:
    if steps <= 1:
        return [float(tmin)]
    if tmax < tmin:
        tmin, tmax = tmax, tmin
    return [tmin + (tmax - tmin) * (i / (steps - 1)) for i in range(steps)]


def _sweep_rows(
    *,
    gt_obj: Any,
    preds_obj: Any,
    thresholds: list[float],
    iou_thr: float,
    category_aware: bool,
    n_images: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for thr in thresholds:
        counts = match_counts_for_threshold(
            gt=gt_obj,
            preds=preds_obj,
            conf_thr=thr,
            iou_thr=float(iou_thr),
            category_aware=category_aware,
        )
        m = _metrics(counts["tp"], counts["fp"], counts["fn"])
        fp_per_image = (counts["fp"] / n_images) if n_images else 0.0
        rows.append({"conf_thr": thr, **counts, **m, "fp_per_image": fp_per_image})
    return rows


def main(argv: list[str] | None = None) -> int:
    require_conda_env()
    p = argparse.ArgumentParser(
        description=(
            "Sweep confidence thresholds on val exports, lock conf_thr, then report on test with "
            "--locked-conf-from (no re-selection on test). Tuning (--select, --iou-grid, --calibrate) "
            "must not use the test split."
        ),
    )
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
    p.add_argument("--out", default="reports/thresholds/sweep.json", help="Where to write sweep JSON.")
    p.add_argument("--csv-out", default="", help="Optional: write compact CSV alongside JSON.")
    p.add_argument("--min", dest="tmin", type=float, default=0.05, help="Min threshold.")
    p.add_argument("--max", dest="tmax", type=float, default=0.95, help="Max threshold.")
    p.add_argument("--steps", type=int, default=19, help="Number of thresholds.")
    p.add_argument("--gt-json", default="", help="GT labels JSON (synthetic-friendly).")
    p.add_argument("--preds-json", default="", help="Predictions/detections JSON.")
    p.add_argument("--iou", type=float, default=0.5, help="IoU threshold for greedy GT matching (single value).")
    p.add_argument(
        "--iou-grid",
        type=float,
        action="append",
        default=None,
        dest="iou_grid",
        metavar="IOU",
        help="Repeatable NMS/match IoU values to search on val before conf lock (omit for legacy single --iou).",
    )
    p.add_argument("--iou-min", type=float, default=None, help="IoU grid lower bound (with --iou-max and --iou-steps).")
    p.add_argument("--iou-max", type=float, default=None, help="IoU grid upper bound.")
    p.add_argument("--iou-steps", type=int, default=None, help="Number of IoU grid steps (default 5 when min/max set).")
    p.add_argument("--class-agnostic", action="store_true", help="Ignore category_id when matching.")
    p.add_argument(
        "--split-file",
        default="",
        help="Optional split list (e.g. data/splits/val.txt) for val/test role detection and guardrails.",
    )
    p.add_argument(
        "--allow-test-tuning",
        action="store_true",
        help="Override guardrail: allow --select / IoU grid on test (not for manuscript metrics).",
    )
    p.add_argument(
        "--select",
        choices=["best_f1", "constraints", "min_count_mae"],
        default="best_f1",
        help=(
            "Operating point on val only: best_f1, min_count_mae, or constraints. "
            "Forbidden on test unless --locked-conf-from is set (test reports the locked row only)."
        ),
    )
    p.add_argument("--min-recall", type=float, default=None, help="Constraint: require recall >= this (only for --select constraints).")
    p.add_argument(
        "--min-precision", type=float, default=None, help="Constraint: require precision >= this (only for --select constraints)."
    )
    p.add_argument(
        "--max-fp-per-image", type=float, default=None, help="Constraint: require fp_per_image <= this (only for --select constraints)."
    )
    p.add_argument(
        "--fixed-conf",
        type=float,
        default=None,
        help="Evaluate metrics at this confidence without re-selecting (val-locked → test report).",
    )
    p.add_argument(
        "--locked-conf-from",
        default="",
        help=(
            "Read conf_thr (and match IoU when present) from a val sweep JSON; evaluate test preds "
            "at that fixed operating point without re-running --select."
        ),
    )
    p.add_argument("--run-yolo", action="store_true", help="(Optional) Run ultralytics YOLO on a few images if --preds-json omitted.")
    p.add_argument("--images", nargs="*", default=[], help="Image paths for --run-yolo mode (small list).")
    p.add_argument(
        "--calibrate",
        choices=["none", "isotonic", "platt"],
        default="none",
        help="Post-hoc score calibration on val preds before sweep (monotone; mAP ranking preserved approximately).",
    )
    p.add_argument(
        "--calibration-metrics",
        action="store_true",
        help="Add reliability bins + ECE block to output (uses detection score labels at --iou).",
    )
    p.add_argument(
        "--fp-budget-sweep-out",
        default="",
        help="Optional: write fp_budget_sweep.v1 ablation JSON (P1-FP-BUDGET constraint grid).",
    )
    p.add_argument(
        "--fp-budget-grid",
        type=float,
        action="append",
        default=None,
        dest="fp_budget_grid",
        metavar="FP_PER_IMAGE",
        help="Repeatable max_fp_per_image caps for constraint ablation (default grid in harchoc.fp_budget_sweep).",
    )
    p.add_argument(
        "--sweep-from",
        default="",
        help="Reuse rows/match IoU from an existing threshold_sweep_run.v1 (skip conf re-sweep).",
    )
    args = p.parse_args(argv)

    # Merge all --config entries (left-to-right). Expected section: threshold_sweep.
    config_obj: dict[str, Any] = {}
    for raw in args.config:
        cfg = load_config_json(raw)
        config_obj = merge_experiment_config(config=config_obj, cli=cfg)
    dataset_cfg = config_obj.get("dataset")
    dataset_cfg_obj = dataset_cfg if isinstance(dataset_cfg, dict) else {}
    sweep_cfg_obj = script_section_from_config(config_obj, "threshold_sweep")

    def _pick(name: str, *, default: object) -> object:
        cli_v = getattr(args, name)
        if cli_v != default:
            return cli_v
        if name in sweep_cfg_obj:
            return sweep_cfg_obj[name]
        return default

    def _pick_dataset(name: str, *, default: object) -> object:
        cli_v = getattr(args, name)
        if cli_v != default:
            return cli_v
        if name in dataset_cfg_obj:
            return dataset_cfg_obj[name]
        return default

    args.manifest = str(_pick_dataset("manifest", default="data/manifest.json"))
    args.default_dataset_name = str(_pick_dataset("default_dataset_name", default="sunflower-cvat-2500"))
    args.dataset_name = _pick_dataset("dataset_name", default=None)  # type: ignore[assignment]
    args.dataset_root = _pick_dataset("dataset_root", default=None)  # type: ignore[assignment]
    args.yolo_data_yaml = _pick_dataset("yolo_data_yaml", default=None)  # type: ignore[assignment]

    args.weights = str(_pick("weights", default=HSP_DETECTION_WEIGHTS))
    args.out = str(_pick("out", default="reports/thresholds/sweep.json"))
    args.csv_out = str(_pick("csv_out", default=""))  # type: ignore[attr-defined]
    args.tmin = float(_pick("tmin", default=0.05))
    args.tmax = float(_pick("tmax", default=0.95))
    args.steps = int(_pick("steps", default=19))
    args.gt_json = str(_pick("gt_json", default=""))
    args.preds_json = str(_pick("preds_json", default=""))
    args.iou = float(_pick("iou", default=0.5))
    args.iou_grid = list(_pick("iou_grid", default=None) or [])  # type: ignore[attr-defined]
    args.iou_min = _pick("iou_min", default=None)  # type: ignore[attr-defined]
    args.iou_max = _pick("iou_max", default=None)  # type: ignore[attr-defined]
    args.iou_steps = _pick("iou_steps", default=None)  # type: ignore[attr-defined]
    args.split_file = str(_pick("split_file", default=""))  # type: ignore[attr-defined]
    args.allow_test_tuning = bool(_pick("allow_test_tuning", default=False))  # type: ignore[attr-defined]
    args.class_agnostic = bool(_pick("class_agnostic", default=False))
    args.select = str(_pick("select", default="best_f1"))  # type: ignore[assignment]
    args.min_recall = _pick("min_recall", default=None)  # type: ignore[assignment]
    args.min_precision = _pick("min_precision", default=None)  # type: ignore[assignment]
    args.max_fp_per_image = _pick("max_fp_per_image", default=None)  # type: ignore[assignment]
    args.fixed_conf = _pick("fixed_conf", default=None)  # type: ignore[assignment]
    args.locked_conf_from = str(_pick("locked_conf_from", default=""))  # type: ignore[attr-defined]
    args.run_yolo = bool(_pick("run_yolo", default=False))
    args.images = list(_pick("images", default=[]))  # type: ignore[assignment]
    args.dry_run = bool(_pick("dry_run", default=False))
    args.light = bool(_pick("light", default=False))  # type: ignore[attr-defined]
    args.calibrate = str(_pick("calibrate", default="none"))  # type: ignore[attr-defined]
    args.calibration_metrics = bool(_pick("calibration_metrics", default=False))  # type: ignore[attr-defined]
    args.fp_budget_sweep_out = str(_pick("fp_budget_sweep_out", default=""))  # type: ignore[attr-defined]
    args.fp_budget_grid = list(_pick("fp_budget_grid", default=None) or [])  # type: ignore[attr-defined]
    args.sweep_from = str(_pick("sweep_from", default=""))  # type: ignore[attr-defined]

    repo_root = Path(__file__).resolve().parents[1]

    if args.dry_run:
        out_path = write_json(
            args.out,
            with_schema_version(
                {
                "status": "dry-run",
                "script": "threshold_sweep",
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
                "thresholds": {"min": args.tmin, "max": args.tmax, "steps": args.steps},
                "out": str(Path(args.out)),
                },
                schema_version="threshold_sweep_run.v1",
            ),
        )
        cli_print(f"Wrote {out_path}")
        return 0

    gt_json = (args.gt_json or "").strip()
    preds_json = (args.preds_json or "").strip()
    if args.light:
        gt_path, preds_path = resolve_light_gt_preds(repo_root=repo_root, gt_json=gt_json, preds_json=preds_json)
        gt_json, preds_json = str(gt_path), str(preds_path)

    spec = resolve_dataset(
        manifest_path=args.manifest,
        default_dataset_name=args.default_dataset_name,
        dataset_name=args.dataset_name,
        dataset_root=args.dataset_root,
        yolo_data_yaml=args.yolo_data_yaml,
    )
    if args.light:
        dataset_desc: dict[str, Any] = (
            describe_dataset(spec) if spec.root.is_dir() else {"note": "light mode without DATASET_ROOT"}
        )
    else:
        require_existing_dir(spec.root, what="Dataset root", hint="Export DATASET_ROOT=/path/to/extracted/dataset")
        dataset_desc = describe_dataset(spec)

    gt_obj: Any | None = read_json(gt_json) if gt_json else None
    preds_obj: Any | None = read_json(preds_json) if preds_json else None

    if preds_obj is None and args.run_yolo:
        weights = Path(args.weights)
        if not weights.exists():
            raise SystemExit(f"--run-yolo requires existing --weights, not found: {weights}")
        if not args.images:
            raise SystemExit("--run-yolo requires --images (small list of paths)")
        try:
            from ultralytics import YOLO  # type: ignore
        except Exception as ex:
            raise SystemExit(f"ultralytics not available for --run-yolo: {ex}") from ex
        model = YOLO(str(weights))
        images = [str(Path(p).expanduser().resolve()) for p in args.images]
        results = model.predict(images, verbose=False)
        preds_images: list[dict[str, Any]] = []
        for img_path, r in zip(images, results, strict=False):
            dets: list[dict[str, Any]] = []
            with capture_failure(f"parse YOLO boxes for {img_path}") as cap:
                boxes = r.boxes
                for i in range(len(boxes)):
                    b = boxes.xyxy[i].tolist()
                    dets.append(
                        {
                            "bbox": [float(b[0]), float(b[1]), float(b[2]), float(b[3])],
                            "category_id": int(boxes.cls[i].item()),
                            "score": float(boxes.conf[i].item()),
                        }
                    )
            if cap.failed:
                fail_or_warn(f"{cap.context}: {cap.exc_type}: {cap.exc_msg}")
            preds_images.append({"image_id": img_path, "file_name": img_path, "detections": dets})
        preds_obj = {"images": preds_images}

    if gt_obj is None or preds_obj is None:
        eprint("threshold_sweep: requires --gt-json and --preds-json, or --light (data/examples/).")
        eprint("If you want optional inference, pass --run-yolo --weights ... --images ...")
        eprint("See data/examples/README.md for exporting real eval preds.")
        raise SystemExit(2)

    ds_root = resolve_dataset_root_for_splits(repo_root=repo_root, dataset_root=spec.root if spec.root.is_dir() else None)
    split_role, split_hints = infer_split_role(
        gt_json=gt_json,
        preds_json=preds_json,
        split_file=(args.split_file or "").strip() or None,
        gt=gt_obj,
        repo_root=repo_root,
        dataset_root=ds_root,
    )

    iou_grid = build_iou_grid(
        iou=float(args.iou),
        iou_grid=args.iou_grid or None,
        iou_min=args.iou_min,
        iou_max=args.iou_max,
        iou_steps=args.iou_steps,
    )
    locked_from = (args.locked_conf_from or "").strip()
    enforce_tuning_guardrails(
        split_role,
        locked_conf_from=locked_from or None,
        iou_grid=iou_grid,
        calibrate=str(args.calibrate or "none"),
        allow_test_tuning=bool(args.allow_test_tuning),
    )

    match_iou = float(args.iou)
    if locked_from:
        locked_iou = load_locked_match_iou(locked_from)
        if locked_iou is not None:
            match_iou = float(locked_iou)
        iou_grid = None

    calibrate_mode = str(args.calibrate or "none")
    calibration_block: dict[str, Any] | None = None
    if calibrate_mode != "none" and not locked_from:
        preds_obj, calibration_block = apply_calibration_to_preds(
            preds_obj,
            calibrate=calibrate_mode,
            gt=gt_obj,
            iou_thr=match_iou,
            category_aware=not bool(args.class_agnostic),
        )

    image_ids = image_ids_union(gt_obj, preds_obj)
    n_images = len(image_ids)
    thresholds = [max(0.0, min(1.0, float(x))) for x in _linspace(args.tmin, args.tmax, args.steps)]
    category_aware = not bool(args.class_agnostic)

    nms_iou_block: dict[str, Any] | None = None
    if iou_grid and len(iou_grid) > 1 and tuning_active(
        locked_conf_from=locked_from or None,
        iou_grid=iou_grid,
        calibrate=calibrate_mode,
    ):
        grid_results: list[dict[str, Any]] = []
        for iou_v in iou_grid:
            grid_rows = _sweep_rows(
                gt_obj=gt_obj,
                preds_obj=preds_obj,
                thresholds=thresholds,
                iou_thr=float(iou_v),
                category_aware=category_aware,
                n_images=n_images,
            )
            grid_sel = _select_row(
                grid_rows,
                mode=str(args.select),
                gt_obj=gt_obj,
                preds_obj=preds_obj,
                iou_thr=float(iou_v),
                category_aware=category_aware,
                min_recall=args.min_recall,
                min_precision=args.min_precision,
                max_fp_per_image=args.max_fp_per_image,
            )
            grid_results.append({"iou": float(iou_v), "selected_row": grid_sel, "best_f1": _best_f1(grid_rows)})
        match_iou, _ = select_best_iou_from_grid(grid_results)
        nms_iou_block = {"grid": iou_grid, "results": grid_results, "selected": match_iou}

    rows = _sweep_rows(
        gt_obj=gt_obj,
        preds_obj=preds_obj,
        thresholds=thresholds,
        iou_thr=match_iou,
        category_aware=category_aware,
        n_images=n_images,
    )

    selected = _select_row(
        rows,
        mode=str(args.select),
        gt_obj=gt_obj,
        preds_obj=preds_obj,
        iou_thr=match_iou,
        category_aware=category_aware,
        min_recall=args.min_recall,
        min_precision=args.min_precision,
        max_fp_per_image=args.max_fp_per_image,
    )
    selected_ok = selected is not None
    if selected is None:
        selected = _best_f1(rows)

    fixed_conf: float | None = None
    locked_source: str | None = None
    if locked_from:
        locked_source = locked_from
        fixed_conf = load_locked_conf(locked_from)
    elif args.fixed_conf is not None:
        fixed_conf = float(args.fixed_conf)

    locked_block: dict[str, Any] | None = None
    if fixed_conf is not None:
        locked_row = metrics_row_at_conf(
            gt=gt_obj,
            preds=preds_obj,
            conf_thr=fixed_conf,
            iou_thr=match_iou,
            category_aware=category_aware,
            n_images=n_images,
        )
        counting: dict[str, Any] | None = None
        if locked_from:
            counting = counting_metrics_at_conf(
                gt=gt_obj,
                preds=preds_obj,
                conf_thr=fixed_conf,
                iou_thr=match_iou,
                category_aware=category_aware,
            )
        locked_block = build_locked_block(row=locked_row, source=locked_source, counting_metrics=counting)

    payload = with_schema_version(
        {
        "status": "ok",
        "light": bool(args.light),
        "meta": collect_run_metadata(
            repo_root=repo_root,
            dataset_manifest=spec.manifest_path,
            extra_files={
                "weights": str(Path(args.weights)),
                "gt_json": str(Path(gt_json)) if gt_json else "",
                "preds_json": str(Path(preds_json)) if preds_json else "",
            },
        ),
        "dataset": {"description": dataset_desc},
        "weights": str(Path(args.weights)),
        "inputs": {"gt_json": str(Path(gt_json)) if gt_json else None, "preds_json": str(Path(preds_json)) if preds_json else None},
        "eval_target": {"split_role": split_role, "hints": split_hints},
        "match": {"iou": match_iou, "category_aware": category_aware},
        "thresholds": {"min": args.tmin, "max": args.tmax, "steps": args.steps, "values": thresholds},
        "images": {"n": n_images},
        "rows": rows,
        "selected": {
            "mode": str(args.select),
            "constraints": {
                "min_recall": args.min_recall,
                "min_precision": args.min_precision,
                "max_fp_per_image": args.max_fp_per_image,
            },
            "constraints_satisfied": bool(selected_ok) if str(args.select) == "constraints" else None,
            "row": selected,
            **({"nms_iou": nms_iou_block} if nms_iou_block is not None else {}),
        },
        "best_f1": _best_f1(rows),
        **({"locked": locked_block} if locked_block is not None else {}),
        },
        schema_version="threshold_sweep_run.v1",
    )
    if bool(args.calibration_metrics):
        from harchoc.platt import collect_detection_calibration_pairs

        scores, labels = collect_detection_calibration_pairs(
            gt=gt_obj,
            preds=preds_obj,
            iou_thr=match_iou,
            category_aware=category_aware,
        )
        payload["calibration_metrics"] = reliability_and_ece(scores, labels)
    if calibration_block is not None:
        payload["calibration"] = calibration_block
    out_path = write_json(args.out, payload)
    cli_print(f"Wrote {out_path}")
    if str(args.csv_out).strip():
        csv_path = write_compact_csv(args.csv_out, rows)
        cli_print(f"Wrote {csv_path}")
    fp_out = (args.fp_budget_sweep_out or "").strip()
    if fp_out:
        ablation = build_fp_budget_sweep_payload(
            gt=gt_obj,
            preds=preds_obj,
            rows=rows,
            iou_thr=match_iou,
            category_aware=category_aware,
            n_images=n_images,
            fp_budget_grid=(args.fp_budget_grid or None) if args.fp_budget_grid else None,
            sweep_val_path=str(out_path),
            gt_json=gt_json,
            preds_json=preds_json,
            split_role=split_role,
            split_hints=split_hints,
        )
        fp_path = write_fp_budget_sweep(fp_out, ablation)
        cli_print(f"Wrote {fp_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

