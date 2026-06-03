from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from typing import Any

import sys

_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))
from harchoc.script_entry import bootstrap_repo_imports

bootstrap_repo_imports()

from harchoc.datasets import describe_dataset, resolve_dataset
from harchoc.hsp_weights import HSP_DETECTION_WEIGHTS, resolve_detection_weights
from harchoc.splits_io import materialize_abs_split_list
from harchoc.ultralytics_eval import run_val
from harchoc.yaml_minimal import parse_names_and_nc
from harchoc.run_metadata import collect_run_metadata
from harchoc.schemas import with_schema_version
from scripts._common_cli import (
    add_dataset_args,
    add_dry_run_arg,
    cli_print,
    require_conda_env,
    require_existing_dir,
    write_json,
)


def _resolve_weights(raw: str | None) -> Path:
    return resolve_detection_weights(raw)


def _resolve_split_source(*, split_file_arg: str | None, dataset_root: Path) -> tuple[dict[str, object], Path]:
    if split_file_arg and split_file_arg.strip():
        p = Path(split_file_arg).expanduser()
        return ({"kind": "split_file", "path": str(p)}, p)

    default_split = Path("data/splits/test.txt")
    if default_split.is_file():
        return ({"kind": "split_file", "path": str(default_split), "role": "test"}, default_split)

    raise SystemExit(
        "Refusing to evaluate on a non-test split.\n"
        "Expected a tracked test split list at data/splits/test.txt or an explicit --split-file.\n"
        "Fix: run scripts/make_splits.py (or provide --split-file data/splits/test.txt)."
    )


def _val_entry_for_yaml(*, dataset_root: Path, val_source: Path, out_dir: Path) -> str:
    """
    Ultralytics `val` may be a directory or a .txt list. For split lists with paths
    relative to the dataset root, materialize a txt of absolute image paths.
    """
    if val_source.is_file() and val_source.suffix.lower() == ".txt":
        out_txt = out_dir / "eval_val_abs_paths.txt"
        materialize_abs_split_list(
            split_source=val_source, dataset_root=dataset_root, out_path=out_txt
        )
        return str(out_txt.resolve())

    try:
        return str(val_source.resolve().relative_to(dataset_root.resolve()))
    except Exception:
        return str(val_source.resolve())


def _write_eval_data_yaml(
    *,
    out_dir: Path,
    dataset_root: Path,
    val_source: Path,
    names: dict[int, str] | None,
    nc: int | None,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / "eval_data.yaml"

    val_str = _val_entry_for_yaml(dataset_root=dataset_root, val_source=val_source, out_dir=out_dir)

    # Ultralytics requires both `train` and `val` keys even for pure evaluation.
    train_default = dataset_root / "images" / "train"
    try:
        train_str = (
            str(train_default.resolve().relative_to(dataset_root.resolve()))
            if train_default.is_dir()
            else val_str
        )
    except Exception:
        train_str = val_str

    lines: list[str] = []
    lines.append(f"path: {dataset_root.resolve()}")
    lines.append(f"train: {train_str}")
    lines.append(f"val: {val_str}")
    if nc is not None:
        lines.append(f"nc: {nc}")
    if names:
        lines.append("names:")
        for k in sorted(names.keys()):
            v = names[k]
            lines.append(f"  {k}: {v}")

    p.write_text("\n".join(lines) + "\n", "utf-8")
    return p


def _extract_ultralytics_metrics(
    res: object,
    *,
    strict_warnings: Any | None = None,
) -> tuple[float | None, float | None, list[dict[str, object]] | None]:
    """
    Best-effort extraction across ultralytics versions.
    Returns: (map50, map50_95, per_class_list).

    On parse failure, records structured warnings via *strict_warnings* (raises
    only when a warn call uses ``raise_if_strict=True`` under ``HARCHOC_STRICT_ML``).
    """
    box = getattr(res, "box", None)
    if box is None:
        if strict_warnings is not None:
            strict_warnings.warn(
                "ultralytics_metrics_missing_box",
                "validation result has no .box metrics",
                raise_if_strict=False,
            )
        return None, None, None

    map50 = getattr(box, "map50", None)
    map50_95 = getattr(box, "map", None)
    try:
        map50_f = float(map50) if map50 is not None else None
    except Exception as ex:
        map50_f = None
        if strict_warnings is not None:
            strict_warnings.warn("ultralytics_metrics_map50", str(ex), raise_if_strict=False)
    try:
        map5095_f = float(map50_95) if map50_95 is not None else None
    except Exception as ex:
        map5095_f = None
        if strict_warnings is not None:
            strict_warnings.warn("ultralytics_metrics_map5095", str(ex), raise_if_strict=False)

    per_class = None
    maps = getattr(box, "maps", None)
    names = getattr(res, "names", None)
    if maps is not None:
        try:
            per_class = []
            for i, v in enumerate(list(maps)):
                cls_name = None
                if isinstance(names, dict):
                    cls_name = names.get(i)
                per_class.append({"class_id": i, "class_name": cls_name, "mAP50_95": float(v)})
        except Exception as ex:
            per_class = None
            if strict_warnings is not None:
                strict_warnings.warn("ultralytics_metrics_per_class", str(ex), raise_if_strict=False)

    return map50_f, map5095_f, per_class


def _image_split_dir_from_split_file(*, dataset_root: Path, split_file: Path) -> Path | None:
    """Infer images/<split> directory from the first entry in a split list file."""
    for ln in split_file.read_text("utf-8", errors="ignore").splitlines():
        rel = ln.strip()
        if not rel or rel.startswith("#"):
            continue
        p = Path(rel)
        if "images" in p.parts:
            idx = p.parts.index("images")
            if idx + 1 < len(p.parts):
                split = p.parts[idx + 1]
                return (dataset_root / "images" / split).resolve()
        # Fallback: treat path as under images/val
        return (dataset_root / "images" / "val").resolve()
    return None


def _infer_nc_from_labels(*, dataset_root: Path, split_dir: Path) -> int | None:
    """
    Infer `nc` (number of classes) by scanning YOLO label files for the split.
    This is a fallback for datasets that ship without a `data.yaml`.
    """
    labels_dir = dataset_root / "labels" / split_dir.name
    if not labels_dir.is_dir():
        return None

    max_cls = -1
    for p in labels_dir.glob("*.txt"):
        txt = p.read_text("utf-8", errors="ignore").strip()
        if not txt:
            continue
        for line in txt.splitlines():
            line = line.strip()
            if not line:
                continue
            head = line.split(maxsplit=1)[0]
            try:
                c = int(head)
            except Exception:
                continue
            if c > max_cls:
                max_cls = c

    return (max_cls + 1) if max_cls >= 0 else None


def main(argv: list[str] | None = None) -> int:
    require_conda_env()
    p = argparse.ArgumentParser(description="Evaluate a trained model on a dataset split.")
    add_dataset_args(p)
    add_dry_run_arg(p)
    p.add_argument(
        "--weights",
        default=None,
        help=f"Path to model weights. Default: DETECTION_MODEL env var or {HSP_DETECTION_WEIGHTS}",
    )
    p.add_argument(
        "--split-file",
        default=None,
        help="Path to a text file listing images (one per line). If omitted, uses data/splits/test.txt. (This script is intentionally test-only.)",
    )
    p.add_argument("--out", default="reports/eval.json", help="Where to write evaluation JSON.")
    p.add_argument(
        "--max-det",
        type=int,
        default=None,
        help="Optional max detections per image for val() (lower is faster; train default is 3000).",
    )
    p.add_argument(
        "--imgsz",
        type=int,
        default=None,
        help="Image size for val() (match train imgsz, e.g. 1280 for bench matrix parity).",
    )
    p.add_argument(
        "--export-gt-json",
        default=None,
        help="Write GT boxes JSON for the eval split (threshold_sweep / error_analysis schema).",
    )
    p.add_argument(
        "--export-preds-json",
        default=None,
        help="Write model predictions JSON for the eval split (low conf by default).",
    )
    p.add_argument(
        "--export-conf",
        type=float,
        default=0.001,
        help="Confidence threshold for --export-preds-json (keep low for PR tail).",
    )
    p.add_argument(
        "--export-iou",
        type=float,
        default=0.3,
        help="NMS IoU for --export-preds-json (document alongside exports).",
    )
    p.add_argument(
        "--export-max-det",
        type=int,
        default=None,
        help="max_det for prediction export (defaults to --max-det or 3000).",
    )
    p.add_argument(
        "--export-only",
        action="store_true",
        help="Skip Ultralytics val() mAP; only write --export-gt-json / --export-preds-json.",
    )
    p.add_argument(
        "--export-device",
        default=None,
        help="Device for prediction export (default: HARCHOC_EXPORT_DEVICE or cuda). Use cpu if GPU OOM @ 1280.",
    )
    p.add_argument(
        "--locked-conf-from",
        default="",
        help="Val threshold JSON: counting MAE at locked conf_thr (export stays at --export-conf unless --locked-export-conf).",
    )
    p.add_argument(
        "--locked-export-conf",
        action="store_true",
        help="Also run prediction export at locked conf (default: export at --export-conf, count at locked).",
    )
    p.add_argument(
        "--device",
        default=None,
        help="Device for val() mAP (default: --export-device, else HARCHOC_EXPORT_DEVICE, else cuda). Use cpu if GPU busy/OOM.",
    )
    p.add_argument(
        "--confusion-matrix-out",
        default=None,
        help="Write confusion matrix JSON path, or output prefix when --confusion-matrix-splits is set.",
    )
    p.add_argument(
        "--confusion-matrix-splits",
        default=None,
        help="Comma-separated split roles (e.g. test,train) or JSON map role→split file. "
        "One model load; writes {prefix}_{role}_confusion.json.",
    )
    p.add_argument(
        "--confusion-matrix-only",
        action="store_true",
        help="Skip val() mAP; only run confusion matrix (requires --confusion-matrix-out or --confusion-matrix-splits).",
    )
    p.add_argument(
        "--confusion-from-exports",
        action="store_true",
        help="Build confusion from existing --export-gt-json / --export-preds-json (CPU, no predict). "
        "Incompatible with --confusion-matrix-splits.",
    )
    args = p.parse_args(argv)

    weights = _resolve_weights(args.weights)
    confusion_out = (args.confusion_matrix_out or "").strip()
    confusion_splits_raw = (args.confusion_matrix_splits or "").strip()
    if args.confusion_matrix_only and not confusion_out and not confusion_splits_raw:
        raise SystemExit(
            "--confusion-matrix-only requires --confusion-matrix-out and/or --confusion-matrix-splits."
        )
    if confusion_splits_raw and not confusion_out:
        raise SystemExit("--confusion-matrix-splits requires --confusion-matrix-out as output prefix.")
    if args.confusion_from_exports and confusion_splits_raw:
        raise SystemExit(
            "--confusion-from-exports does not support --confusion-matrix-splits; "
            "run once per split with matching gt/preds exports."
        )
    if args.confusion_from_exports and not (args.export_gt_json and args.export_preds_json):
        raise SystemExit(
            "--confusion-from-exports requires --export-gt-json and --export-preds-json."
        )

    # Resolve dataset + split source even in dry-run, but do not require anything to exist.
    spec = resolve_dataset(
        manifest_path=args.manifest,
        default_dataset_name=args.default_dataset_name,
        dataset_name=args.dataset_name,
        dataset_root=args.dataset_root,
        yolo_data_yaml=args.yolo_data_yaml,
    )
    split_source, split_path = _resolve_split_source(split_file_arg=args.split_file, dataset_root=spec.root)
    repo_root = Path(__file__).resolve().parents[1]
    meta = collect_run_metadata(
        repo_root=repo_root,
        dataset_manifest=spec.manifest_path,
        extra_files={
            "weights": str(weights),
            "split_source": str(split_path),
        },
    )

    if args.dry_run:
        out_path = write_json(
            args.out,
            with_schema_version(
                {
                "status": "dry-run",
                "script": "eval",
                "meta": meta,
                "dataset": {"description": describe_dataset(spec)},
                "weights": str(weights),
                "split_source": split_source,
                "eval_target": {"split_role": "test"},
                "max_det": args.max_det,
                "imgsz": args.imgsz,
                "export_gt_json": args.export_gt_json,
                "export_preds_json": args.export_preds_json,
                "export_conf": args.export_conf,
                "export_iou": args.export_iou,
                "export_only": bool(args.export_only),
                "confusion_matrix_out": confusion_out or None,
                "confusion_matrix_splits": confusion_splits_raw or None,
                "confusion_matrix_only": bool(args.confusion_matrix_only),
                "confusion_from_exports": bool(args.confusion_from_exports),
                "strict_warnings": [],
                "out": str(Path(args.out)),
                },
                schema_version="eval_run.v1",
            ),
        )
        cli_print(f"Wrote {out_path}")
        return 0

    export_gt_early = (args.export_gt_json or "").strip()
    export_preds_early = (args.export_preds_json or "").strip()
    exports_only_cm = bool(
        args.confusion_matrix_only
        and args.confusion_from_exports
        and export_gt_early
        and export_preds_early
    )
    if exports_only_cm:
        for label, rel in (("GT export", export_gt_early), ("preds export", export_preds_early)):
            p = Path(rel).expanduser()
            if not p.is_file():
                raise SystemExit(f"{label} does not exist: {p}")
    else:
        require_existing_dir(spec.root, what="Dataset root", hint="Export DATASET_ROOT=/path/to/extracted/dataset")
        if not weights.expanduser().is_file():
            raise SystemExit(f"Model weights do not exist: {weights}")

    export_gt = export_gt_early
    export_preds = export_preds_early

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    sp = split_path.expanduser()
    data_yaml_path: Path | None = None
    if exports_only_cm:
        split_role = "test"
        export_lower = f"{export_gt} {export_preds}".lower()
        if "val" in export_lower:
            split_role = "val"
        elif "train" in export_lower:
            split_role = "train"
    else:
        if split_source.get("kind") == "split_file":
            if not sp.is_file():
                raise SystemExit(f"Split file does not exist: {sp}")
        else:
            require_existing_dir(sp, what="Split directory")

        names: dict[int, str] | None = None
        nc: int | None = None
        yolo_data_yaml = spec.yolo_data_yaml or (
            spec.root / "data.yaml" if (spec.root / "data.yaml").is_file() else None
        )
        if yolo_data_yaml is not None and yolo_data_yaml.is_file():
            names, nc = parse_names_and_nc(yolo_data_yaml)
        elif split_source.get("kind") == "dir":
            nc = _infer_nc_from_labels(dataset_root=spec.root, split_dir=sp)
        elif split_source.get("kind") == "split_file":
            img_split = _image_split_dir_from_split_file(dataset_root=spec.root, split_file=sp)
            if img_split is not None:
                nc = _infer_nc_from_labels(dataset_root=spec.root, split_dir=img_split)

        if nc is None:
            nc = 2
        if names is None and nc == 2:
            from harchoc.sunflower_dataset import CLASS_NAMES_DICT

            names = dict(CLASS_NAMES_DICT)

        data_yaml_path = _write_eval_data_yaml(
            out_dir=out_path.parent,
            dataset_root=spec.root,
            val_source=sp,
            names=names,
            nc=nc,
        )

        split_role = "test"
        split_name = sp.name.lower()
        if "val" in split_name:
            split_role = "val"
        elif "train" in split_name:
            split_role = "train"

    if export_gt or export_preds:
        if not export_gt or not export_preds:
            raise SystemExit("--export-gt-json and --export-preds-json must be set together.")

    from harchoc.strict_ml import StrictWarnings

    strict_warnings = StrictWarnings()
    map50: float | None = None
    map50_95: float | None = None
    per_class: list[dict[str, object]] | None = None
    runtime_s = 0.0
    eval_device: str | None = None
    cuda_visible_prior = os.environ.get("CUDA_VISIBLE_DEVICES")

    if not args.export_only and not args.confusion_matrix_only:
        eval_device = args.device
        if eval_device is None:
            eval_device = args.export_device
        if eval_device is None:
            eval_device = (os.getenv("HARCHOC_EXPORT_DEVICE") or "").strip() or None
        if data_yaml_path is None:
            raise SystemExit("internal: data_yaml missing for validation eval")
        t0 = time.perf_counter()
        res = run_val(
            weights,
            data_yaml_path,
            imgsz=args.imgsz,
            max_det=args.max_det,
            split="val",
            device=eval_device,
        )
        runtime_s = time.perf_counter() - t0
        map50, map50_95, per_class = _extract_ultralytics_metrics(res, strict_warnings=strict_warnings)

    payload = with_schema_version(
        {
        "status": "ok",
        "meta": meta,
        "dataset": {"description": describe_dataset(spec)},
        "weights": str(weights),
        "split_source": split_source,
        "eval_target": {"split_role": split_role},
        "max_det": args.max_det,
        "imgsz": args.imgsz,
        "generated_data_yaml": str(data_yaml_path) if data_yaml_path is not None else None,
        "confusion_from_exports": bool(args.confusion_from_exports),
        "export_only": bool(args.export_only),
        "device": eval_device,
        "mAP50": map50,
        "mAP50_95": map50_95,
        "per_class": per_class,
        "runtime_s": runtime_s,
        "strict_warnings": strict_warnings.as_list(),
        },
        schema_version="eval_run.v1",
    )

    if export_gt and export_preds and not exports_only_cm:
        from harchoc.eval_export import export_gt_preds_json

        locked_from = (args.locked_conf_from or "").strip()
        export_conf = float(args.export_conf)
        counting_conf = export_conf
        match_iou = float(args.export_iou)
        category_aware = True
        if locked_from:
            from harchoc.threshold_lock import load_locked_conf
            from harchoc.domain_count_mae import match_settings_from_threshold_json

            counting_conf = load_locked_conf(locked_from)
            match_iou, category_aware = match_settings_from_threshold_json(locked_from)
            if bool(args.locked_export_conf):
                export_conf = counting_conf

        export_max_det = args.export_max_det
        if export_max_det is None:
            export_max_det = int(args.max_det) if args.max_det is not None else 3000
        export_device = args.export_device
        if export_device is None:
            export_device = (os.getenv("HARCHOC_EXPORT_DEVICE") or "").strip() or None
        export_info = export_gt_preds_json(
            split_file=sp,
            dataset_root=spec.root,
            weights=weights,
            gt_out=Path(export_gt),
            preds_out=Path(export_preds),
            conf=float(export_conf),
            iou=float(args.export_iou),
            max_det=int(export_max_det),
            device=export_device,
            strict_warnings=strict_warnings,
        )
        payload["pred_export"] = export_info
        payload["strict_warnings"] = strict_warnings.as_list()
        if locked_from:
            from harchoc.json_io import load_json
            from harchoc.threshold_protocol import counting_metrics_at_conf

            gt_obj = load_json(export_gt)
            preds_obj = load_json(export_preds)
            counting = counting_metrics_at_conf(
                gt=gt_obj,
                preds=preds_obj,
                conf_thr=float(counting_conf),
                iou_thr=float(match_iou),
                category_aware=category_aware,
            )
            payload["locked_conf"] = float(counting_conf)
            payload["locked_conf_from"] = locked_from
            payload["counting_metrics"] = counting
            payload["count_mae"] = counting.get("mae")

    if confusion_out or confusion_splits_raw:
        from harchoc.detection_confusion import (
            confusion_matrix_from_exports,
            confusion_matrix_multi_split,
            confusion_matrix_out_path,
            confusion_matrix_streaming,
            format_confusion_matrix_text,
            parse_confusion_matrix_splits,
            resolve_match_settings,
        )

        cm_conf, cm_iou = resolve_match_settings(
            conf=float(args.export_conf),
            iou=float(args.export_iou),
            locked_conf_from=(args.locked_conf_from or "").strip() or None,
        )
        export_max_det = args.export_max_det
        if export_max_det is None:
            export_max_det = int(args.max_det) if args.max_det is not None else 3000
        export_device = args.export_device
        if export_device is None:
            export_device = (os.getenv("HARCHOC_EXPORT_DEVICE") or "").strip() or "cuda"
        if args.confusion_from_exports:
            export_device = "cpu"

        confusion_payloads: dict[str, Any] = {}
        if confusion_splits_raw:
            splits = parse_confusion_matrix_splits(confusion_splits_raw, repo_root=repo_root)
            for role, split_path in splits.items():
                if not split_path.is_file():
                    raise SystemExit(f"Split file for {role!r} does not exist: {split_path}")
            multi = confusion_matrix_multi_split(
                weights=weights,
                splits=splits,
                dataset_root=spec.root,
                conf_thr=cm_conf,
                iou_thr=cm_iou,
                export_conf=float(args.export_conf),
                export_iou=float(args.export_iou),
                max_det=int(export_max_det),
                device=export_device,
                imgsz=args.imgsz,
                strict_warnings=strict_warnings,
            )
            out_prefix = Path(confusion_out)
            for role, (acc, cm_runtime_s) in multi.items():
                cm_payload = acc.to_payload(
                    conf_thr=cm_conf,
                    iou_thr=cm_iou,
                    split_role=role,
                    weights=str(weights),
                    export_conf=float(args.export_conf),
                    export_iou=float(args.export_iou),
                    export_device=export_device,
                    runtime_s=cm_runtime_s,
                )
                cm_path = write_json(confusion_matrix_out_path(out_prefix, role), cm_payload)
                cli_print(format_confusion_matrix_text(acc.matrix, acc.stats, title=f"{role} split"))
                cli_print(f"Wrote {cm_path}")
                confusion_payloads[role] = {"path": str(cm_path), **cm_payload}
            payload["confusion_matrices"] = confusion_payloads
        else:
            if args.confusion_from_exports:
                from harchoc.json_io import load_json

                t0 = time.perf_counter()
                gt_obj = load_json(Path(export_gt).expanduser())
                preds_obj = load_json(Path(export_preds).expanduser())
                acc = confusion_matrix_from_exports(
                    gt_obj,
                    preds_obj,
                    conf_thr=cm_conf,
                    iou_thr=cm_iou,
                )
                cm_runtime_s = time.perf_counter() - t0
            else:
                acc, cm_runtime_s = confusion_matrix_streaming(
                    weights=weights,
                    split_file=sp,
                    dataset_root=spec.root,
                    conf_thr=cm_conf,
                    iou_thr=cm_iou,
                    export_conf=float(args.export_conf),
                    export_iou=float(args.export_iou),
                    max_det=int(export_max_det),
                    device=export_device,
                    imgsz=args.imgsz,
                    strict_warnings=strict_warnings,
                )
            cm_payload = acc.to_payload(
                conf_thr=cm_conf,
                iou_thr=cm_iou,
                split_role=split_role,
                weights=str(weights),
                export_conf=float(args.export_conf),
                export_iou=float(args.export_iou),
                export_device=export_device,
                runtime_s=cm_runtime_s,
            )
            cm_path = write_json(confusion_out, cm_payload)
            cli_print(format_confusion_matrix_text(acc.matrix, acc.stats, title=f"{split_role} split"))
            cli_print(f"Wrote {cm_path}")
            payload["confusion_matrix"] = {"path": str(Path(cm_path)), **cm_payload}

    out_path = write_json(args.out, payload)
    cli_print(f"Wrote {out_path}")
    if export_gt:
        cli_print(f"Wrote GT export {export_gt}")
        cli_print(f"Wrote preds export {export_preds}")
    from harchoc.post_train_eval import restore_cuda_visible_devices_after_ultralytics_cpu

    restore_cuda_visible_devices_after_ultralytics_cpu(cuda_visible_prior)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

