from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))
from harchoc.script_entry import bootstrap_repo_imports

bootstrap_repo_imports()

from harchoc.data_yaml import ensure_data_yaml
from harchoc.train_config import (
    effective_train_aug_merged,
    resolve_train_config_extends,
    warn_train_schedule_close_mosaic,
)
from harchoc.train_kwargs import (
    effective_train_cfg_with_freeze,
    forwarded_keys_from_train_cfg,
    forwarded_train_keys,
    load_ultralytics_train_model,
    resolve_freeze_policy,
    ultralytics_train_kwargs,
)
from harchoc.rtdetr_limits import validate_rtdetr_query_cap
from harchoc.training_budget import enforce_budget
from harchoc.splits_io import materialize_abs_split_list
from harchoc.yaml_minimal import parse_minimal_yaml_flat
from harchoc.datasets import describe_dataset, resolve_dataset
from harchoc.run_metadata import collect_run_metadata
from harchoc.strict_ml import append_capture_warning, capture_failure, ml_warnings_sink
from harchoc.resource_snapshot import snapshot_from_train_cfg
from harchoc.schemas import with_schema_version
from scripts._common_cli import add_dataset_args, add_dry_run_arg, require_conda_env, require_existing_dir, write_json

# Baseline recipe from yolo-sunflower-seed-detector/training.py (overridable via --config JSON).
# Dataset: dense YOLO boxes, class 0=developed, class 1=aborted (see data/README.md).
# RT-DETR matrix: set ``amp`` and ``grad_clip`` in train JSON (e.g. train_bench_rtdetr-l.json);
# forwarded via ``harchoc/train_kwargs.py`` ALLOWED_TRAIN_KWARGS into ``model.train(**kwargs)``.
_BASELINE_DEFAULTS: dict[str, Any] = {
    "model": "yolov8m.pt",
    "epochs": 100,
    "imgsz": 1280,
    "batch": 1,
    "device": 0,
    "optimizer": "AdamW",
    "conf": 0.05,
    "iou": 0.3,
    "max_det": 3000,
    "mosaic": 0.1,
    "hsv_h": 0.02,
    "hsv_s": 0.3,
    "hsv_v": 0.3,
    "translate": 0.05,
    "scale": 0.15,
    "lr0": 0.0002,
    "lrf": 0.01,
    "momentum": 0.97,
    "weight_decay": 0.0005,
    "patience": 50,
    "workers": 2,
    "verbose": True,
}


def _merge_train_config(cfg_obj: dict[str, Any]) -> dict[str, Any]:
    merged = dict(_BASELINE_DEFAULTS)
    if "train" in cfg_obj and isinstance(cfg_obj["train"], dict):
        merged.update(cfg_obj["train"])
    for k, v in cfg_obj.items():
        if k in ("train", "eval", "dataset"):
            continue
        merged[k] = v
    return merged


def _split_entry_for_yaml(
    *, dataset_root: Path, split_source: Path, out_dir: Path, split_name: str
) -> str:
    """
    Ultralytics may use a directory or a .txt list. For split lists with paths relative
    to the dataset root, materialize a txt of absolute image paths (see eval.py).
    """
    if split_source.is_file() and split_source.suffix.lower() == ".txt":
        out_txt = out_dir / f"{split_name}_abs_paths.txt"
        materialize_abs_split_list(
            split_source=split_source, dataset_root=dataset_root, out_path=out_txt
        )
        return str(out_txt.resolve())

    try:
        return str(split_source.resolve().relative_to(dataset_root.resolve()))
    except Exception:
        return str(split_source.resolve())


def _prepare_ultralytics_data_yaml(
    *,
    source_yaml: str,
    train_split_file: str | Path | None = None,
    val_split_file: str | Path | None = None,
) -> str:
    """
    Materialize a data.yaml with absolute split sources so Ultralytics does not resolve
    list entries relative to the process cwd.
    """
    src = Path(source_yaml).resolve()
    obj = parse_minimal_yaml_flat(src)
    base_dir = src.parent
    path_val = obj.get("path")
    if path_val:
        pv = Path(path_val).expanduser()
        base_dir = (src.parent / pv).resolve() if not pv.is_absolute() else pv.resolve()

    splits_dir = Path(tempfile.mkdtemp(prefix="ultra_train_splits_"))

    def _resolve_split(key: str, default: str) -> Path:
        raw = obj.get(key, default)
        sp = Path(raw).expanduser()
        return (base_dir / sp).resolve() if not sp.is_absolute() else sp.resolve()

    train_source = Path(train_split_file).expanduser() if train_split_file else None
    if train_source is not None and not train_source.is_absolute():
        train_source = (Path.cwd() / train_source).resolve()
    val_source = Path(val_split_file).expanduser() if val_split_file else None
    if val_source is not None and not val_source.is_absolute():
        val_source = (Path.cwd() / val_source).resolve()

    train_entry = _split_entry_for_yaml(
        dataset_root=base_dir,
        split_source=train_source if train_source is not None else _resolve_split("train", "images/train"),
        out_dir=splits_dir,
        split_name="train",
    )
    val_entry = _split_entry_for_yaml(
        dataset_root=base_dir,
        split_source=val_source if val_source is not None else _resolve_split("val", "images/val"),
        out_dir=splits_dir,
        split_name="val",
    )

    lines = [
        f"path: {base_dir.as_posix()}",
        f"train: {train_entry}",
        f"val: {val_entry}",
    ]
    if "nc" in obj:
        lines.append(f"nc: {obj['nc']}")

    tmp = tempfile.NamedTemporaryFile("w", suffix=".yaml", prefix="ultra_train_data_", delete=False)
    tmp.write("\n".join(lines) + "\n")
    tmp.close()
    return tmp.name


def _resolve_ultralytics_model(model: str) -> str:
    from harchoc.model_zoo import resolve_weights_ref

    res = resolve_weights_ref(backend="ultralytics", model=model)
    if res.kind == "ultralytics_id":
        if not res.exists or res.resolved_path is None:
            raise SystemExit(
                f"Ultralytics weights not cached: {model!r} (expected {res.cache_path}). "
                "Run: PYTHONPATH=. python scripts/check_weights_cache.py"
            )
        return str(res.resolved_path)
    if res.kind == "file_path":
        if not res.exists or res.resolved_path is None:
            raise SystemExit(f"Weights file not found: {model}")
        return str(res.resolved_path)
    return model


def _runtime_versions(*, warnings: list[str] | None = None) -> dict[str, str | None]:
    out: dict[str, str | None] = {"torch": None, "ultralytics": None, "cuda": None}
    with capture_failure("import torch") as cap:
        import torch

        out["torch"] = getattr(torch, "__version__", None)
        if torch.cuda.is_available():
            out["cuda"] = getattr(getattr(torch, "version", None), "cuda", None)
    append_capture_warning(warnings, cap)
    with capture_failure("import ultralytics") as cap:
        import ultralytics

        out["ultralytics"] = getattr(ultralytics, "__version__", None)
    append_capture_warning(warnings, cap)
    return out


def _val_metrics_summary(metrics: object | None) -> dict[str, Any] | None:
    if not isinstance(metrics, dict) or not metrics:
        return None
    summary: dict[str, Any] = {}
    for k, v in metrics.items():
        ks = str(k)
        if ks.startswith("metrics/") or "mAP" in ks or ks == "fitness":
            summary[ks] = v
    return summary or None


def _train_split_roles(*, repo_root: Path, test_split_file: Path | None) -> dict[str, Any]:
    return {
        "ultralytics_val": {
            "role": "validation",
            "description": "Early-stop and in-training val metrics (data.yaml val: images/val).",
        },
        "post_train_eval": {
            "role": "test",
            "description": "Reported detection metrics via scripts/eval.py (test split only).",
            "split_file": str(test_split_file.resolve()) if test_split_file is not None else None,
        },
        "repo_split_lists": str((repo_root / "data" / "splits").resolve()),
    }


def _resolve_test_split_file(*, repo_root: Path, dataset_root: Path) -> Path | None:
    for p in (repo_root / "data" / "splits" / "test.txt", dataset_root / "data" / "splits" / "test.txt"):
        if p.is_file():
            return p.resolve()
    return None


def _find_ultralytics_weights_dir(
    *,
    results: object | None,
    cwd: Path,
    run_name: str,
    out_dir: Path,
    warnings: list[str] | None = None,
) -> Path | None:
    # Best-effort across ultralytics versions and `project`/`out-dir` usage.
    save_dir = getattr(results, "save_dir", None) if results is not None else None
    if save_dir:
        with capture_failure("ultralytics results.save_dir") as cap:
            sd = Path(str(save_dir)).expanduser()
            if not sd.is_absolute():
                sd = (cwd / sd).resolve()
            wdir = (sd / "weights").resolve()
            if wdir.exists():
                return wdir
        append_capture_warning(warnings, cap)

    # Common defaults
    candidates = [
        out_dir / run_name / "weights",
        out_dir / "detect" / run_name / "weights",
        cwd / "runs" / "detect" / run_name / "weights",
    ]
    for wdir in candidates:
        if wdir.is_dir():
            return wdir.resolve()
    return None


def _write_run_jsons(*, run_dir: Path, config: dict[str, Any], meta: dict[str, Any], metrics: dict[str, Any]) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False) + "\n", "utf-8")
    (run_dir / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", "utf-8")
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", "utf-8")


def main(argv: list[str] | None = None) -> int:
    require_conda_env()
    p = argparse.ArgumentParser(description="Config-driven training entrypoint (ultralytics imported lazily).")
    add_dataset_args(p)
    add_dry_run_arg(p)
    p.add_argument("--out-dir", default="runs", help="Base output directory for runs.")
    p.add_argument("--name", default=None, help="Run name (directory name under --out-dir).")
    p.add_argument("--config", default=None, help="Optional path to a JSON config file (free-form).")
    p.add_argument(
        "--aug-config",
        default=None,
        help="Optional aug recipe YAML (e.g. configs/aug/robustness_minimal.yaml); merges ultralytics: keys.",
    )
    p.add_argument(
        "--train-split-file",
        default=None,
        help="Override train split list (.txt, one image path per line).",
    )
    p.add_argument(
        "--val-split-file",
        default=None,
        help="Override val split list for Ultralytics early-stop (.txt).",
    )
    p.add_argument(
        "--skip-eval",
        action="store_true",
        help="Do not run test-split eval after training.",
    )
    p.add_argument(
        "--eval-out",
        default=None,
        help="Where to write post-train eval JSON (default: reports/eval_<name>.json).",
    )
    args = p.parse_args(argv)

    if not args.dry_run:
        from harchoc.gpu_exclusive import adhoc_train_blocked, gpu_exclusive_message

        if adhoc_train_blocked():
            print(gpu_exclusive_message(), file=sys.stderr)
            return 2

    spec = resolve_dataset(
        manifest_path=args.manifest,
        default_dataset_name=args.default_dataset_name,
        dataset_name=args.dataset_name,
        dataset_root=args.dataset_root,
        yolo_data_yaml=args.yolo_data_yaml,
    )

    name = (args.name or "run").strip() or "run"
    run_dir = Path(args.out_dir) / name

    repo_root = Path(__file__).resolve().parents[1]
    cfg_obj: dict[str, Any] = {}
    if args.config:
        cfg_path = Path(args.config).expanduser()
        if not cfg_path.is_absolute():
            cfg_path = (repo_root / cfg_path).resolve()
        if cfg_path.is_file():
            cfg_obj = json.loads(cfg_path.read_text("utf-8"))
            if isinstance(cfg_obj, dict):
                cfg_obj = resolve_train_config_extends(
                    cfg_obj, repo_root=repo_root, config_path=cfg_path
                )
            else:
                raise SystemExit(f"Config must be a JSON object: {cfg_path}")
        else:
            raise SystemExit(f"Config file not found: {cfg_path}")

    train_cfg = _merge_train_config(cfg_obj)
    config_path_label = str(Path(args.config).resolve()) if args.config else "cli"
    validate_rtdetr_query_cap(
        model=str(train_cfg.get("model", _BASELINE_DEFAULTS["model"])),
        train_json=train_cfg,
        train_json_path=config_path_label,
        fail=True,
    )
    if args.aug_config:
        train_cfg = dict(train_cfg)
        train_cfg["aug_config"] = args.aug_config
    aug_spec = train_cfg.get("aug_config")
    aug_path_resolved: Path | None = None
    if aug_spec:
        aug_path = Path(str(aug_spec)).expanduser()
        if not aug_path.is_absolute():
            aug_path = (repo_root / aug_path).resolve()
        aug_path_resolved = aug_path
    train_cfg = effective_train_aug_merged(train_cfg, repo_root=repo_root)
    train_split_override = args.train_split_file or train_cfg.get("train_split_file")
    val_split_override = args.val_split_file or train_cfg.get("val_split_file")
    test_split_file = _resolve_test_split_file(repo_root=repo_root, dataset_root=spec.root)
    ml_warnings = ml_warnings_sink()
    _, freeze_policy, freeze_warnings = resolve_freeze_policy(train_cfg)
    for w in freeze_warnings:
        print(f"Warning: {w}", file=sys.stderr)
    if ml_warnings is not None:
        ml_warnings.extend(freeze_warnings)
    effective_train_cfg = effective_train_cfg_with_freeze(train_cfg)

    meta = {
        "status": "dry-run" if args.dry_run else "ok",
        "script": "train",
        "dataset": {"description": describe_dataset(spec)},
        "run_dir": str(run_dir),
        "freeze_policy": freeze_policy,
        "ultralytics_train": {"forwarded_keys": forwarded_keys_from_train_cfg(train_cfg)},
        "resources": snapshot_from_train_cfg(train_cfg),
        "split_roles": _train_split_roles(repo_root=repo_root, test_split_file=test_split_file),
        "train_split_file": str(train_split_override) if train_split_override else None,
        "val_split_file": str(val_split_override) if val_split_override else None,
        "run_metadata": collect_run_metadata(
            repo_root=repo_root,
            dataset_manifest=Path(args.manifest),
            extra_files={
                **({"config": Path(args.config)} if args.config else {}),
                **({"aug_config": aug_path_resolved} if aug_path_resolved else {}),
            }
            or None,
            include_repo_splits=True,
            warnings=ml_warnings,
        ),
        "created_at_unix_s": int(time.time()),
    }
    if ml_warnings is not None:
        meta["warnings"] = ml_warnings
    meta = with_schema_version(meta, schema_version="train_meta.v1")

    config_out = {
        "status": "dry-run" if args.dry_run else "ok",
        "script": "train",
        "name": name,
        "out_dir": str(Path(args.out_dir)),
        "dataset": {"description": describe_dataset(spec)},
        "train": train_cfg,
        "config_file": str(Path(args.config)) if args.config else None,
        "aug_config": str(aug_path_resolved) if aug_path_resolved else None,
    }
    config_out = with_schema_version(config_out, schema_version="train_config.v1")
    metrics_out: dict[str, Any] = {
        "status": "dry-run" if args.dry_run else "ok",
        "script": "train",
        "name": name,
        "metrics": None,
        "weights": None,
        "ultralytics_run": None,
    }
    metrics_out = with_schema_version(metrics_out, schema_version="train_metrics.v1")

    if args.dry_run:
        _write_run_jsons(run_dir=run_dir, config=config_out, meta=meta, metrics=metrics_out)
        return 0

    warn_train_schedule_close_mosaic(
        train_cfg, repo_root=repo_root, label=config_path_label
    )

    require_existing_dir(spec.root, what="Dataset root", hint="Export DATASET_ROOT=/path/to/extracted/dataset")

    epochs = int(train_cfg.get("epochs", _BASELINE_DEFAULTS["epochs"]))
    imgsz = int(train_cfg.get("imgsz", _BASELINE_DEFAULTS["imgsz"]))
    batch = int(train_cfg.get("batch", _BASELINE_DEFAULTS["batch"]))
    enforce_budget(epochs=epochs, imgsz=imgsz, batch=batch)

    model_id = _resolve_ultralytics_model(str(train_cfg.get("model", _BASELINE_DEFAULTS["model"])))
    source_yaml = ensure_data_yaml(dataset_root=spec.root, yolo_data_yaml=spec.yolo_data_yaml)
    data_yaml = _prepare_ultralytics_data_yaml(
        source_yaml=source_yaml,
        train_split_file=train_split_override,
        val_split_file=val_split_override,
    )

    t0 = time.perf_counter()
    model = load_ultralytics_train_model(model_id)
    train_kwargs = ultralytics_train_kwargs(
        effective_train_cfg, data_yaml=data_yaml, run_name=name
    )
    if "seed" in train_cfg:
        train_kwargs["seed"] = int(train_cfg["seed"])
    # Try to keep all artifacts under the caller's --out-dir.
    train_kwargs.setdefault("project", str(Path(args.out_dir)))
    train_kwargs.setdefault("exist_ok", True)
    meta["ultralytics_train"] = {"forwarded_keys": forwarded_train_keys(train_kwargs)}

    results = model.train(**train_kwargs)
    runtime_s = float(time.perf_counter() - t0)

    eval_section = cfg_obj.get("eval") if isinstance(cfg_obj.get("eval"), dict) else {}
    from harchoc.post_train_eval import release_cuda_after_train

    release_cuda_after_train(model)
    model = None  # noqa: F841 — released for post-train eval VRAM

    weights_dir = _find_ultralytics_weights_dir(
        results=results,
        cwd=Path.cwd(),
        run_name=name,
        out_dir=Path(args.out_dir).resolve(),
        warnings=ml_warnings,
    )
    best: Path | None = None
    if weights_dir is not None:
        for cand in (weights_dir / "best.pt", weights_dir / "last.pt"):
            if cand.is_file():
                best = cand.resolve()
                break
    weights_out = run_dir / "weights"
    weights_out.mkdir(parents=True, exist_ok=True)
    produced_weights: str | None = str(best) if best is not None else None
    recorded_weights: str | None = None
    if best is not None:
        dest = weights_out / best.name
        shutil.copy2(best, dest)
        recorded_weights = str(dest.resolve())

    raw_metrics = getattr(results, "results_dict", None) if results is not None else None
    metrics_out.update(
        {
            "metrics": raw_metrics,
            "val_metrics_summary": _val_metrics_summary(raw_metrics),
            "versions": _runtime_versions(warnings=ml_warnings),
            "weights": recorded_weights,
            "produced_weights": produced_weights,
            "ultralytics_run": str(weights_dir.parent) if weights_dir is not None else None,
            "runtime_s": runtime_s,
            "model": model_id,
        }
    )
    if ml_warnings is not None:
        meta["warnings"] = ml_warnings
    _write_run_jsons(run_dir=run_dir, config=config_out, meta=meta, metrics=metrics_out)
    print(f"Training finished in {runtime_s:.1f}s")
    if recorded_weights:
        print(f"Weights: {recorded_weights}")

    from harchoc.post_train_eval import (
        build_post_train_eval_argv,
        post_train_eval_skipped,
        resolve_post_train_eval_device,
    )

    if post_train_eval_skipped(cli_skip=bool(args.skip_eval), eval_section=eval_section):
        return 0

    if not recorded_weights:
        print("Skipping eval: no best.pt found after training.")
        return 0

    from scripts.eval import main as eval_main

    split_file = test_split_file
    eval_out = args.eval_out or str(repo_root / "reports" / f"eval_{name}.json")
    if split_file is None:
        print(
            "Warning: no test split file found (data/splits/test.txt or DATASET_ROOT/data/splits/test.txt); "
            "eval may refuse to run."
        )

    eval_device = resolve_post_train_eval_device(eval_section)
    print(f"Running post-train eval -> {eval_out} (device={eval_device})")
    eval_argv = build_post_train_eval_argv(
        recorded_weights=recorded_weights,
        eval_out=eval_out,
        manifest=str(args.manifest),
        default_dataset_name=args.default_dataset_name,
        dataset_name=args.dataset_name,
        dataset_root=args.dataset_root,
        yolo_data_yaml=args.yolo_data_yaml,
        split_file=str(split_file) if split_file is not None else None,
        eval_section=eval_section,
        train_imgsz=int(train_cfg.get("imgsz", _BASELINE_DEFAULTS["imgsz"])),
    )
    rc = eval_main(eval_argv)
    return int(rc)


if __name__ == "__main__":
    raise SystemExit(main())
