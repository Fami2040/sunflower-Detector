from __future__ import annotations

import argparse
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import sys; from pathlib import Path; _r = Path(__file__).resolve().parent.parent; (str(_r) not in sys.path) and sys.path.insert(0, str(_r)); from harchoc.script_entry import bootstrap_repo_imports; bootstrap_repo_imports()

from harchoc.post_train_eval import build_post_train_eval_argv, post_train_eval_skipped
from harchoc.bench_config import (
    BenchConfig,
    _bench_eval_max_det,
    _bench_run_name,
    _bench_to_train_config,
    _infer_imgsz,
    _load_committed_train_bench_json,
    _resolve_bench_train_config_path,
    bench_matrix_metadata,
    is_bench_row_config,
    load_bench_config,
    select_backend,
)
from harchoc.datasets import describe_dataset, resolve_dataset
from harchoc.experiment_config import load_config_json, merge_experiment_config
from harchoc.bench_assets import resolve_asset_ref
from harchoc.model_zoo import backend_availability as mz_backend_availability
from harchoc.resource_snapshot import snapshot_from_train_cfg
from harchoc.sahi_matrix import (
    SahiEvalParams,
    expand_bench_configs_with_sahi,
    parse_sahi_rows_config,
    sahi_eval_status,
    sahi_matrix_metadata,
)
from harchoc.strict_ml import capture_failure, fail_or_warn
from harchoc.training_budget import _budget_limit_int
from scripts._common_cli import (
    add_dataset_args,
    add_dry_run_arg,
    cli_print,
    extend_dataset_argv,
    require_conda_env,
    write_json,
)


def _backend_availability(backend: str, cfg: BenchConfig | None = None) -> tuple[bool, str | None]:
    if cfg is not None and backend == "external":
        return mz_backend_availability(
            "external",
            model_id=cfg.model_id,
            source_id=cfg.source_id,
        )
    return mz_backend_availability(backend)  # type: ignore[arg-type]


def build_run_record(
    *,
    cfg: BenchConfig,
    dataset_description: str,
    dataset_root: Path,
    yolo_data_yaml: Path | None,
    dry_run: bool,
    would_train: bool,
    would_eval: bool,
    sahi: SahiEvalParams | None = None,
    sahi_eval: bool = False,
    repo_root: Path | None = None,
) -> dict[str, object]:
    backend = select_backend(cfg)
    available, missing_reason = _backend_availability(backend, cfg)
    imgsz = _infer_imgsz(cfg)
    weights = resolve_asset_ref(cfg, backend=backend)
    # Best-effort planned resource snapshot for provenance (no torch required).
    planned_resources = snapshot_from_train_cfg(
        {
            "device": 0,
            "workers": 2,
            "batch": None,
            "imgsz": imgsz,
        }
        ,
        include_torch=False,
    )
    eval_status, eval_reason = sahi_eval_status(
        dry_run=dry_run,
        would_eval=would_eval,
        sahi_eval=sahi_eval and sahi is not None,
    )
    run_name = _bench_run_name(cfg)
    if sahi is not None:
        run_name = f"{run_name}__{sahi.run_suffix()}"
    sahi_meta = sahi_matrix_metadata(cfg, sahi)
    rtdetr_gate: str | None = None
    if repo_root is not None:
        from harchoc.rtdetr_zoo_gate import zoo_core_rtdetr_gate_skip_reason

        rtdetr_gate = zoo_core_rtdetr_gate_skip_reason(
            repo_root=repo_root,
            bench_path=cfg.path,
            groups=cfg.groups,
            model=cfg.model,
        )
    planned_train = would_train and rtdetr_gate is None
    planned_eval = would_eval and rtdetr_gate is None
    return {
        "schema_version": "benchmark_run.v1",
        "config": {"path": str(cfg.path), "name": cfg.name, "groups": list(cfg.groups)},
        "rtdetr_15ep_smoke_gate": rtdetr_gate,
        "planned": {
            "task": cfg.task,
            "backend": cfg.backend,
            "model_id": cfg.model_id,
            "model": cfg.model,
            "infer": cfg.infer,
            "train": {"epochs": cfg.epochs, "patience": cfg.patience, "seed": cfg.seed},
            "notes": cfg.notes,
            "run_name": run_name,
            "sahi": sahi.to_json() if sahi is not None else None,
        },
        "resolved": {
            "backend": backend,
            "backend_available": available,
            "backend_missing_reason": missing_reason,
            "infer": {"imgsz": imgsz},
            "resources": planned_resources,
            "weights": weights,
            "dataset": {
                "description": dataset_description,
                "root": str(dataset_root.resolve()),
                "yolo_data_yaml": str(yolo_data_yaml) if yolo_data_yaml is not None else None,
            },
        },
        "execution": {
            "dry_run": dry_run,
            "would_train": planned_train,
            "would_eval": planned_eval,
            "blocked_by_rtdetr_15ep_gate": rtdetr_gate is not None,
        },
        "eval": {
            "status": eval_status,
            "protocol": sahi_meta["eval_protocol"],
            "metrics": None,
            "reason": eval_reason,
        },
        "matrix_metadata": {**bench_matrix_metadata(cfg), **sahi_meta},
    }


def build_summary(
    *,
    configs: list[BenchConfig],
    dry_run: bool,
    dataset_description: str,
    dataset_root: Path,
    yolo_data_yaml: Path | None,
    would_train: bool,
    would_eval: bool,
    selected_groups: list[str] | None = None,
    sahi_eval: bool = False,
    sahi_rows: list[SahiEvalParams] | None = None,
    repo_root: Path | None = None,
) -> dict[str, object]:
    generated_at = datetime.now(timezone.utc).isoformat()
    expanded = expand_bench_configs_with_sahi(
        configs,
        matrix_rows=sahi_rows,
        sahi_eval=sahi_eval,
    )
    schema_version = "sahi_matrix_eval.v1" if sahi_eval else "benchmark_matrix.v1"
    notes = (
        "Benchmark matrix harness: dry-run writes a stable plan; "
        "non-dry-run can train (ultralytics, cached weights) and/or eval-only when weights exist."
    )
    if sahi_eval:
        notes = (
            "SAHI matrix eval protocol (scaffold): dry-run expands bench rows with slice params; "
            "GPU sliced test eval is not implemented yet."
        )
    return {
        "schema_version": schema_version,
        "status": "plan" if dry_run else "run",
        "dry_run": dry_run,
        "eval_protocol": "sahi" if sahi_eval else "ultralytics_val",
        "generated_at": generated_at,
        "dataset": {
            "description": dataset_description,
            "root": str(dataset_root.resolve()),
            "yolo_data_yaml": str(yolo_data_yaml) if yolo_data_yaml is not None else None,
        },
        "selection": {"groups": selected_groups or []},
        "sahi_rows": [r.to_json() for r in sahi_rows] if sahi_rows else None,
        "runs": [
            build_run_record(
                cfg=c,
                dataset_description=dataset_description,
                dataset_root=dataset_root,
                yolo_data_yaml=yolo_data_yaml,
                dry_run=dry_run,
                would_train=would_train,
                would_eval=would_eval,
                sahi=sahi,
                sahi_eval=sahi_eval,
                repo_root=repo_root,
            )
            for c, sahi in expanded
        ],
        "notes": notes,
    }


def _cached_ultralytics_weights(cfg: BenchConfig) -> Path | None:
    """Resolved cache path when the identifier is present on disk; never downloads."""
    if select_backend(cfg) != "ultralytics" or not cfg.model:
        return None
    wref = resolve_asset_ref(cfg, backend="ultralytics")
    if not wref.get("exists"):
        return None
    raw = wref.get("resolved_path")
    if raw is None:
        return None
    p = Path(str(raw))
    return p if p.is_file() else None



def _invoke_test_eval_for_bench(
    *,
    cfg: BenchConfig,
    weights: str | Path,
    manifest: str,
    default_dataset_name: str,
    dataset_env: dict[str, str] | None,
    train_doc: dict[str, Any],
    eval_out: Path | None = None,
) -> dict[str, object]:
    from scripts.eval import main as eval_main
    from scripts.train import _resolve_test_split_file

    eval_section = train_doc.get("eval") if isinstance(train_doc.get("eval"), dict) else {}
    if post_train_eval_skipped(cli_skip=False, eval_section=eval_section):
        return {
            "status": "skipped",
            "reason": "eval.skip",
            "split": "test",
            "mAP50": None,
            "mAP50_95": None,
            "eval_out": str(eval_out) if eval_out is not None else None,
        }

    spec = resolve_dataset(
        manifest_path=manifest,
        default_dataset_name=default_dataset_name,
        environ=dataset_env,
    )
    repo_root = Path(__file__).resolve().parents[1]
    split_file = _resolve_test_split_file(repo_root=repo_root, dataset_root=spec.root)
    imgsz = _infer_imgsz(cfg)
    if imgsz is None:
        imgsz = int(train_doc.get("imgsz") or 1280)
    if eval_out is None:
        eval_out = repo_root / "reports" / f"bench_eval_{cfg.name}.json"

    env = dataset_env or {}
    argv = build_post_train_eval_argv(
        recorded_weights=str(weights),
        eval_out=str(eval_out),
        manifest=manifest,
        default_dataset_name=default_dataset_name,
        dataset_name=env.get("DATASET_NAME"),
        dataset_root=env.get("DATASET_ROOT"),
        yolo_data_yaml=env.get("YOLO_DATA_YAML"),
        split_file=str(split_file) if split_file is not None else None,
        eval_section=eval_section,
        train_imgsz=int(imgsz),
    )

    cuda_visible_prior = os.environ.get("CUDA_VISIBLE_DEVICES")
    try:
        rc = int(eval_main(argv))
    finally:
        from harchoc.post_train_eval import restore_cuda_visible_devices_after_ultralytics_cpu

        restore_cuda_visible_devices_after_ultralytics_cpu(cuda_visible_prior)
    map50: float | None = None
    map50_95: float | None = None
    if eval_out is not None and eval_out.is_file():
        with capture_failure(f"parse eval JSON {eval_out}") as cap:
            obj = json.loads(eval_out.read_text("utf-8"))
            if isinstance(obj.get("mAP50"), (int, float)):
                map50 = float(obj["mAP50"])
            if isinstance(obj.get("mAP50_95"), (int, float)):
                map50_95 = float(obj["mAP50_95"])
        if cap.failed:
            fail_or_warn(f"{cap.context}: {cap.exc_type}: {cap.exc_msg}")

    return {
        "status": "ok" if rc == 0 else "failed",
        "returncode": rc,
        "split": "test",
        "mAP50": map50,
        "mAP50_95": map50_95,
        "eval_out": str(eval_out) if eval_out is not None else None,
    }


def _load_bench_train_recipe(cfg: BenchConfig) -> dict[str, Any]:
    committed = _resolve_bench_train_config_path(cfg)
    if committed is None:
        return {}
    recipe = _load_committed_train_bench_json(committed)
    if recipe.get("batch") is not None:
        batch = int(recipe["batch"])
        max_batch = _budget_limit_int("HARCHOC_MAX_BATCH", default=16)
        if batch > max_batch:
            raise SystemExit(
                f"batch={batch} in {committed} exceeds HARCHOC_MAX_BATCH={max_batch}"
            )
    return recipe


def _invoke_supergradients_train_for_bench(
    *,
    cfg: BenchConfig,
    dataset_root: Path,
    runs_dir: Path,
    recipe: dict[str, Any],
) -> dict[str, object]:
    from harchoc.supergradients_train import train_bench_run

    run_name = _bench_run_name(cfg)
    model_id = str(cfg.model_id or recipe.get("model_id") or "yolo_nas_s")
    imgsz = _infer_imgsz(cfg) or int(recipe.get("imgsz") or 1280)
    epochs = int(cfg.epochs if cfg.epochs is not None else recipe.get("epochs") or 100)
    batch = int(recipe.get("batch") or 1)
    seed = int(cfg.seed if cfg.seed is not None else recipe.get("seed") or 0)

    result = train_bench_run(
        model_id=model_id,
        dataset_root=dataset_root,
        runs_dir=runs_dir,
        run_name=run_name,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        seed=seed,
    )
    out: dict[str, object] = {
        "status": result.get("status", "failed"),
        "returncode": result.get("returncode", 1),
        "backend": "supergradients",
        "config_path": str(cfg.path),
        "name": cfg.name,
        "run_name": run_name,
        "run_dir": result.get("run_dir"),
        "weights": result.get("weights"),
        "model_id": model_id,
    }
    if result.get("reason"):
        out["reason"] = result["reason"]
    if result.get("val_metrics"):
        out["val_metrics"] = result["val_metrics"]
    return out


def _invoke_external_train_for_bench(
    *,
    cfg: BenchConfig,
    dataset_root: Path,
    runs_dir: Path,
) -> dict[str, object]:
    from harchoc.external_detector_train import train_bench_run

    run_name = _bench_run_name(cfg)
    imgsz = _infer_imgsz(cfg) or 1280
    epochs = int(cfg.epochs if cfg.epochs is not None else 100)
    seed = int(cfg.seed if cfg.seed is not None else 0)
    source_id = str(cfg.source_id or cfg.model_id or "").strip()

    result = train_bench_run(
        source_id=source_id,
        model_id=cfg.model_id,
        dataset_root=dataset_root,
        runs_dir=runs_dir,
        run_name=run_name,
        epochs=epochs,
        imgsz=imgsz,
        seed=seed,
    )
    out: dict[str, object] = {
        "status": result.get("status", "failed"),
        "returncode": result.get("returncode", 1),
        "backend": "external",
        "config_path": str(cfg.path),
        "name": cfg.name,
        "run_name": run_name,
        "source_id": source_id,
        "weights": result.get("weights"),
    }
    for key in ("reason", "log_path", "train_stack", "checkpoint", "coco_export"):
        if result.get(key) is not None:
            out[key] = result[key]
    return out


def _invoke_test_eval_for_external(
    *,
    cfg: BenchConfig,
    weights: str | Path,
    dataset_root: Path,
    runs_dir: Path,
    run_name: str,
    eval_out: Path | None = None,
    max_det: int | None = None,
    repo_root: Path | None = None,
) -> dict[str, object]:
    from harchoc.external_detector_eval import eval_hsp_for_bench

    imgsz = _infer_imgsz(cfg) or 1280
    source_id = str(cfg.source_id or cfg.model_id or "").strip()
    return eval_hsp_for_bench(
        source_id=source_id,
        model_id=cfg.model_id,
        weights=weights,
        dataset_root=dataset_root,
        run_dir=(runs_dir / run_name).resolve(),
        eval_out=eval_out,
        max_det=max_det,
        imgsz=int(imgsz),
        repo_root=repo_root,
    )


def _invoke_test_eval_for_supergradients(
    *,
    cfg: BenchConfig,
    weights: str | Path,
    dataset_root: Path,
    eval_out: Path | None = None,
    max_det: int | None = None,
) -> dict[str, object]:
    from harchoc.supergradients_eval import eval_test_for_bench

    return eval_test_for_bench(
        weights=weights,
        dataset_root=dataset_root,
        eval_out=eval_out,
        max_det=max_det,
        config_path=str(cfg.path),
    )


def _invoke_train_for_bench(
    *,
    cfg: BenchConfig,
    manifest: str,
    default_dataset_name: str,
    dataset_env: dict[str, str] | None,
    runs_dir: Path,
    train_doc: dict[str, Any] | None = None,
    dataset_root: Path | None = None,
) -> dict[str, object]:
    from scripts.train import main as train_main

    backend = select_backend(cfg)
    available, missing_reason = _backend_availability(backend, cfg)
    if not available:
        return {
            "status": "skipped",
            "reason": missing_reason,
            "backend": backend,
            "config_path": str(cfg.path),
            "name": cfg.name,
        }
    if backend == "supergradients":
        if dataset_root is None:
            spec = resolve_dataset(
                manifest_path=manifest,
                default_dataset_name=default_dataset_name,
                environ=dataset_env,
            )
            dataset_root = spec.root
        recipe = _load_bench_train_recipe(cfg)
        return _invoke_supergradients_train_for_bench(
            cfg=cfg,
            dataset_root=dataset_root,
            runs_dir=runs_dir,
            recipe=recipe,
        )
    if backend == "external":
        if dataset_root is None:
            spec = resolve_dataset(
                manifest_path=manifest,
                default_dataset_name=default_dataset_name,
                environ=dataset_env,
            )
            dataset_root = spec.root
        return _invoke_external_train_for_bench(
            cfg=cfg,
            dataset_root=dataset_root,
            runs_dir=runs_dir,
        )
    if backend != "ultralytics":
        return {
            "status": "skipped",
            "reason": "backend_not_implemented",
            "backend": backend,
            "config_path": str(cfg.path),
            "name": cfg.name,
        }

    weights = _cached_ultralytics_weights(cfg)
    if weights is None:
        wref = resolve_asset_ref(cfg, backend=backend)
        return {
            "status": "skipped",
            "reason": "weights_not_cached",
            "backend": backend,
            "config_path": str(cfg.path),
            "name": cfg.name,
            "model": cfg.model,
            "weights": wref,
        }

    run_name = _bench_run_name(cfg)
    if train_doc is None:
        train_doc = _bench_to_train_config(cfg, weights_path=str(weights))
    tmp_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as tmp:
            json.dump(train_doc, tmp)
            tmp.flush()
            tmp_path = tmp.name

        argv = extend_dataset_argv(
            [
                "--name",
                run_name,
                "--config",
                tmp_path,
                "--out-dir",
                str(runs_dir),
                "--skip-eval",
            ],
            manifest=manifest,
            default_dataset_name=default_dataset_name,
            dataset_env=dataset_env,
        )

        rc = int(train_main(argv))
    finally:
        if tmp_path:
            Path(tmp_path).unlink(missing_ok=True)

    run_dir = (runs_dir / run_name).resolve()
    recorded_weights = run_dir / "weights" / "best.pt"
    return {
        "status": "ok" if rc == 0 else "failed",
        "returncode": rc,
        "backend": backend,
        "config_path": str(cfg.path),
        "name": cfg.name,
        "run_name": run_name,
        "run_dir": str(run_dir),
        "weights_cache": str(weights),
        "weights": str(recorded_weights) if recorded_weights.is_file() else None,
    }


def _pick_existing_weights(cfg: BenchConfig) -> Path | None:
    """
    For ultralytics eval-only, we only run when a weights file path already exists.
    We intentionally do NOT trigger hub downloads in CI / scripts.
    """
    if not cfg.model:
        return None
    # Only accept explicit file paths for eval; identifiers are not auto-downloaded here.
    if ("/" not in cfg.model) and ("\\" not in cfg.model) and (not cfg.model.startswith((".", "~"))):
        return None
    p = Path(cfg.model).expanduser()
    if not p.is_absolute():
        p = (Path.cwd() / p)
    p = p.resolve()
    return p if p.exists() else None


def _resolve_yolo_data_yaml(*, dataset_root: Path, explicit_yaml: Path | None) -> Path:
    if explicit_yaml is not None:
        return explicit_yaml
    p = (dataset_root / "data.yaml").resolve()
    if p.exists():
        return p
    raise SystemExit(
        "Could not find ultralytics data.yaml.\n"
        "Fix by exporting YOLO_DATA_YAML=/path/to/data.yaml, or place data.yaml at DATASET_ROOT."
    )


def _ultralytics_eval_one(
    *,
    cfg: BenchConfig,
    dataset_yaml: Path,
    weights: Path,
) -> dict[str, object]:
    from harchoc.ultralytics_eval import run_val

    imgsz = _infer_imgsz(cfg)
    max_det = cfg.infer.get("max_det")
    metrics = run_val(
        weights,
        dataset_yaml,
        imgsz=imgsz if isinstance(imgsz, int) else None,
        max_det=int(max_det) if isinstance(max_det, int) else None,
        split="val",
    )

    results_dict: dict[str, object] | None = None
    results_dict_error: dict[str, str] | None = None
    with capture_failure("ultralytics metrics.results_dict") as cap:
        rd = getattr(metrics, "results_dict", None)
        if isinstance(rd, dict):
            results_dict = {str(k): v for k, v in rd.items()}
    if cap.failed:
        fail_or_warn(f"{cap.context}: {cap.exc_type}: {cap.exc_msg}")
        results_dict_error = {"type": cap.exc_type or "Error", "msg": cap.exc_msg or ""}

    out: dict[str, object] = {
        "status": "ok",
        "backend": "ultralytics",
        "config_path": str(cfg.path),
        "name": cfg.name,
        "weights": str(weights),
        "dataset_yaml": str(dataset_yaml),
        "imgsz": imgsz,
        "results": results_dict,
    }
    if results_dict_error is not None:
        out["results_dict_error"] = results_dict_error
    return out


def main(argv: list[str] | None = None) -> int:
    require_conda_env()
    p = argparse.ArgumentParser(description="Benchmark matrix harness (scaffold).")
    p.add_argument(
        "--config",
        action="append",
        default=[],
        help="Path to JSON experiment config. Can be repeated; later entries override earlier.",
    )
    add_dataset_args(p)
    p.add_argument(
        "--bench-config",
        action="append",
        default=[],
        help="Path to a bench YAML/JSON config (repeatable). If omitted, uses --bench-dir.",
    )
    p.add_argument("--bench-dir", default="configs/bench", help="Directory of bench config YAMLs.")
    p.add_argument("--pattern", default="*.yaml", help="Glob pattern within --bench-dir.")
    p.add_argument("--limit", type=int, default=0, help="Limit number of configs (0 = no limit).")
    p.add_argument(
        "--group",
        action="append",
        default=[],
        help="Filter configs by group/tag (repeatable). Matches values in YAML `groups:`/`group:` (comma/space separated).",
    )
    p.add_argument(
        "--list-groups",
        action="store_true",
        help="List known groups from selected configs and exit (writes JSON to --out).",
    )
    p.add_argument("--out", default="reports/benchmarks/matrix.json", help="Where to write JSON summary.")
    # Default remains "plan-only" behavior. Use --no-dry-run to actually run.
    g = p.add_mutually_exclusive_group()
    g.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse args and write plan JSON only (default).",
    )
    g.add_argument(
        "--no-dry-run",
        action="store_true",
        help="Allow executing eval-only (and later, training).",
    )
    p.set_defaults(dry_run=True, no_dry_run=False)
    p.add_argument("--no-train", action="store_true", help="Skip training step (when not dry-run).")
    p.add_argument("--no-eval", action="store_true", help="Skip evaluation step (when not dry-run).")
    p.add_argument(
        "--eval-out",
        default="reports/benchmarks/matrix_eval.json",
        help="Where to write aggregated eval JSON (when eval runs).",
    )
    p.add_argument(
        "--train-out",
        default="reports/benchmarks/matrix_train.json",
        help="Where to write aggregated train JSON (when training runs).",
    )
    p.add_argument(
        "--runs-dir",
        default="runs/bench",
        help="Base output directory for matrix-invoked training runs.",
    )
    p.add_argument(
        "--aggregate-seeds",
        action="store_true",
        help="Read --train-out JSON and write per-model seed comparison (no train/eval).",
    )
    p.add_argument(
        "--seed-stats-out",
        default="",
        help="Output path for --aggregate-seeds (default: <train-out> with .seed_stats.json suffix).",
    )
    p.add_argument(
        "--count-mae-json",
        action="append",
        default=[],
        help="Per-run count MAE artifact for --aggregate-seeds: run_name=path/to/error_* or threshold_* JSON.",
    )
    p.add_argument(
        "--sahi-eval",
        action="store_true",
        help="SAHI matrix eval protocol: expand plan rows with slice params (GPU eval scaffold only).",
    )
    p.add_argument(
        "--scaffold-zoo",
        action="store_true",
        help="Generate/update bench YAML and train_bench JSON from configs/zoo/matrix_rows.v1.json "
        "(skips rows with scaffold_bench/scaffold_train false).",
    )
    p.add_argument(
        "--validate-zoo",
        action="store_true",
        help="Validate configs/bench and train_bench JSON against the zoo matrix manifest; exit 1 on mismatch.",
    )
    p.add_argument(
        "--zoo-manifest",
        default="configs/zoo/matrix_rows.v1.json",
        help="Zoo matrix rows manifest for --scaffold-zoo / --validate-zoo.",
    )
    args = p.parse_args(argv)

    if args.scaffold_zoo or args.validate_zoo:
        from harchoc.zoo_matrix_scaffold import scaffold_zoo_matrix, validate_zoo_matrix

        manifest = Path(str(args.zoo_manifest)).expanduser()
        if not manifest.is_absolute():
            manifest = (Path(__file__).resolve().parents[1] / manifest).resolve()
        if args.scaffold_zoo and args.validate_zoo:
            raise SystemExit("Use only one of --scaffold-zoo or --validate-zoo")
        if args.scaffold_zoo:
            report = scaffold_zoo_matrix(manifest_path=manifest, write=True)
        else:
            report = validate_zoo_matrix(manifest_path=manifest)
        out_zoo = str(args.out)
        if out_zoo and out_zoo != "reports/benchmarks/matrix.json":
            out_path = Path(out_zoo)
        else:
            out_path = Path("reports/benchmarks/zoo_scaffold_report.json")
        written = write_json(out_path, report.to_json())
        cli_print(f"Wrote {written}")
        if report.errors:
            for err in report.errors:
                cli_print(f"ERROR: {err}")
            raise SystemExit(1)
        mode = report.mode
        if mode == "scaffold":
            cli_print(
                f"zoo scaffold: bench generated={report.bench_generated} updated={report.bench_updated} "
                f"skipped={report.bench_skipped}; train generated={report.train_generated} "
                f"updated={report.train_updated} skipped={report.train_skipped}"
            )
        else:
            from harchoc.zoo_matrix_scaffold import load_matrix_rows_manifest

            n_rows = len(load_matrix_rows_manifest(manifest).get("rows", []))
            cli_print(f"zoo validate: {n_rows} rows ok (validated-only, no files written)")
        for w in report.warnings:
            cli_print(f"WARN: {w}")
        return 0

    # Merge all --config entries (left-to-right). Shape:
    # - dataset: {manifest, default_dataset_name}
    # - benchmark: {bench_dir, pattern, limit, out, eval_out, dry_run/no_dry_run, ...}
    # - experiments.v1: {schema_version, dataset, run:{kind,...}} → flattened below
    config_obj: dict[str, Any] = {}
    for raw in args.config:
        cfg = load_config_json(raw)
        config_obj = merge_experiment_config(config=config_obj, cli=cfg)

    if config_obj.get("schema_version") == "experiments.v1":
        from harchoc.experiment_config import script_section_from_config

        run_section = script_section_from_config(config_obj, "benchmark_matrix")
        dataset_section = config_obj.get("dataset") if isinstance(config_obj.get("dataset"), dict) else {}
        bench_section = config_obj.get("benchmark") if isinstance(config_obj.get("benchmark"), dict) else {}
        config_obj = {
            "dataset": dataset_section,
            "benchmark": merge_experiment_config(config=bench_section, cli=run_section),
        }

    dataset_cfg = config_obj.get("dataset")
    dataset_cfg_obj = dataset_cfg if isinstance(dataset_cfg, dict) else {}
    bench_cfg = config_obj.get("benchmark")
    bench_cfg_obj = bench_cfg if isinstance(bench_cfg, dict) else {}

    # Prefer config over defaults; prefer explicit CLI over both.
    def _pick(name: str, *, default: object) -> object:
        cli_v = getattr(args, name, default)
        if cli_v != default:
            return cli_v
        if name in bench_cfg_obj:
            return bench_cfg_obj[name]
        return default

    def _pick_dataset(name: str, *, default: object) -> object:
        cli_v = getattr(args, name, default)
        if cli_v != default:
            return cli_v
        if name in dataset_cfg_obj:
            return dataset_cfg_obj[name]
        return default

    manifest = str(_pick_dataset("manifest", default="data/manifest.json"))
    default_dataset_name = str(_pick_dataset("default_dataset_name", default="sunflower-cvat-2500"))
    dataset_name = _pick_dataset("dataset_name", default=None)
    dataset_root = _pick_dataset("dataset_root", default=None)
    yolo_data_yaml = _pick_dataset("yolo_data_yaml", default=None)

    dataset_env: dict[str, str] = {}
    if dataset_name is not None and str(dataset_name).strip():
        dataset_env["DATASET_NAME"] = str(dataset_name).strip()
    if dataset_root is not None and str(dataset_root).strip():
        dataset_env["DATASET_ROOT"] = str(dataset_root).strip()
    if yolo_data_yaml is not None and str(yolo_data_yaml).strip():
        dataset_env["YOLO_DATA_YAML"] = str(yolo_data_yaml).strip()

    spec = resolve_dataset(
        manifest_path=manifest,
        default_dataset_name=default_dataset_name,
        environ=dataset_env or None,
    )
    dataset_description = describe_dataset(spec)

    bench_config_cli = list(args.bench_config or [])
    bench_config_cfg = bench_cfg_obj.get("bench_config") if isinstance(bench_cfg_obj.get("bench_config"), list) else []
    bench_config = bench_config_cli if bench_config_cli else [str(x) for x in bench_config_cfg]

    bench_dir = str(_pick("bench_dir", default="configs/bench"))
    pattern = str(_pick("pattern", default="*.yaml"))
    limit = int(_pick("limit", default=0) or 0)
    out = str(_pick("out", default="reports/benchmarks/matrix.json"))
    eval_out = str(_pick("eval_out", default="reports/benchmarks/matrix_eval.json"))
    train_out = str(_pick("train_out", default="reports/benchmarks/matrix_train.json"))
    runs_dir = Path(str(_pick("runs_dir", default="runs/bench")))
    aggregate_seeds = bool(_pick("aggregate_seeds", default=False))
    seed_stats_out = str(_pick("seed_stats_out", default="") or "")

    if aggregate_seeds:
        from harchoc.matrix_seed_stats import (
            BENCHMARK_MATRIX_SEED_STATS_V1,
            build_matrix_seed_stats_v1,
            parse_count_mae_json_args,
        )

        train_path = Path(train_out)
        if not train_path.is_file():
            raise SystemExit(f"--aggregate-seeds requires existing train JSON: {train_path}")
        train_doc = json.loads(train_path.read_text(encoding="utf-8"))
        repo_root = Path(__file__).resolve().parents[1]
        count_mae_paths = parse_count_mae_json_args(list(args.count_mae_json or []))
        payload = build_matrix_seed_stats_v1(
            train_doc,
            source_train_out=train_path,
            count_mae_paths=count_mae_paths or None,
            repo_root=repo_root,
            schema_version=BENCHMARK_MATRIX_SEED_STATS_V1,
        )
        out_stats = Path(seed_stats_out) if seed_stats_out.strip() else train_path.with_suffix(".seed_stats.json")

        written = write_json(out_stats, payload)
        cli_print(f"Wrote {written}")
        return 0

    dry_run_flag = bool(_pick("dry_run", default=True))
    no_dry_run_flag = bool(_pick("no_dry_run", default=False))
    no_train = bool(_pick("no_train", default=False))
    no_eval = bool(_pick("no_eval", default=False))
    sahi_eval = bool(_pick("sahi_eval", default=False))
    sahi_rows_raw = _pick("sahi_rows", default=None)
    sahi_rows = parse_sahi_rows_config(sahi_rows_raw) if sahi_eval and sahi_rows_raw is not None else None
    selected_groups = [str(g) for g in (_pick("group", default=[]) or []) if str(g).strip()]
    list_groups = bool(_pick("list_groups", default=False))

    if bench_config:
        cfg_paths = [Path(x) for x in bench_config]
    else:
        cfg_paths = sorted(Path(bench_dir).glob(pattern))
        if pattern == "*.yaml":
            cfg_paths.extend(sorted(Path(bench_dir).glob("*.json")))

    cfg_paths = sorted({p.resolve() for p in cfg_paths if is_bench_row_config(p)})
    if limit and limit > 0:
        cfg_paths = cfg_paths[:limit]

    configs = [load_bench_config(pth) for pth in cfg_paths]

    if selected_groups:
        want = set(selected_groups)
        configs = [c for c in configs if want.intersection(set(c.groups))]

    if list_groups:
        groups: dict[str, int] = {}
        for c in configs:
            for g in c.groups:
                groups[g] = groups.get(g, 0) + 1
        payload = {
            "schema_version": "benchmark_groups.v1",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "groups": [{"name": k, "count": v} for k, v in sorted(groups.items(), key=lambda kv: kv[0])],
        }
        write_json(out, payload)
        return 0

    dry_run = bool(dry_run_flag) and not bool(no_dry_run_flag)
    would_train = (not dry_run) and (not bool(no_train))
    would_eval = (not dry_run) and (not bool(no_eval))
    yolo_data_yaml: Path | None = None
    if not dry_run and (not no_eval):
        yolo_data_yaml = _resolve_yolo_data_yaml(dataset_root=spec.root, explicit_yaml=spec.yolo_data_yaml)
    repo_root = Path(__file__).resolve().parents[1]
    payload = build_summary(
        configs=configs,
        dry_run=dry_run,
        dataset_description=dataset_description,
        dataset_root=spec.root,
        yolo_data_yaml=yolo_data_yaml,
        would_train=would_train,
        would_eval=would_eval,
        selected_groups=selected_groups,
        sahi_eval=sahi_eval,
        sahi_rows=sahi_rows,
        repo_root=repo_root,
    )
    out_path = write_json(out, payload)

    if not dry_run and would_train:
        from harchoc.queue_notify import notify_matrix_row
        from harchoc.queue_skip_gates import existing_bench_run_weights, matrix_run_is_complete
        from harchoc.rtdetr_zoo_gate import zoo_core_rtdetr_gate_skip_reason

        parent_job_id = (os.environ.get("HARCHOC_GPU_QUEUE_JOB_ID") or "").strip() or None
        matrix_group = selected_groups[0] if len(selected_groups) == 1 else None

        def _notify_row(tr: dict[str, object], cfg: BenchConfig) -> None:
            if os.environ.get("HARCHOC_NOTIFY_DISABLE", "").strip().lower() in ("1", "true", "yes"):
                return
            st = str(tr.get("status") or "unknown")
            mae = tr.get("test_count_mae")
            detail = str(tr.get("reason") or tr.get("detail") or "") or None
            notify_matrix_row(
                repo_root=repo_root,
                row_name=str(tr.get("name") or cfg.name),
                status=st,
                matrix_group=matrix_group,
                test_count_mae=float(mae) if mae is not None else None,
                detail=detail,
                parent_job_id=parent_job_id,
            )

        train_runs: list[dict[str, object]] = []
        existing_train_doc: dict[str, object] | None = None
        train_path = Path(train_out)
        if train_path.is_file():
            try:
                raw = json.loads(train_path.read_text(encoding="utf-8"))
                if isinstance(raw, dict):
                    existing_train_doc = raw
            except Exception:
                existing_train_doc = None

        def _prior_matrix_run(cfg: BenchConfig) -> dict[str, object] | None:
            if not existing_train_doc:
                return None
            cfg_path = str(cfg.path.resolve())
            for r in existing_train_doc.get("runs") or []:
                if not isinstance(r, dict):
                    continue
                if str(r.get("config_path") or "") == cfg_path or str(r.get("name") or "") == cfg.name:
                    return r
            return None

        for cfg in configs:
            prior = _prior_matrix_run(cfg)
            if prior is not None and matrix_run_is_complete(repo_root, prior):
                row = dict(prior)
                train_runs.append(row)
                _notify_row(row, cfg)
                continue
            gate_reason = zoo_core_rtdetr_gate_skip_reason(
                repo_root=repo_root,
                bench_path=cfg.path,
                groups=cfg.groups,
                model=cfg.model,
            )
            if gate_reason:
                row = {
                    "status": "skipped",
                    "reason": "rtdetr_15ep_smoke_gate",
                    "detail": gate_reason,
                    "config_path": str(cfg.path),
                    "name": cfg.name,
                    "model": cfg.model,
                    "matrix_metadata": bench_matrix_metadata(cfg),
                }
                train_runs.append(row)
                _notify_row(row, cfg)
                continue
            existing_w = existing_bench_run_weights(cfg, runs_dir)
            if existing_w is not None:
                run_name = _bench_run_name(cfg)
                tr: dict[str, object] = {
                    "status": "ok",
                    "reason": "weights_exist",
                    "backend": select_backend(cfg),
                    "config_path": str(cfg.path),
                    "name": cfg.name,
                    "run_name": run_name,
                    "run_dir": str((runs_dir / run_name).resolve()),
                    "weights": str(existing_w),
                }
            else:
                tr = _invoke_train_for_bench(
                    cfg=cfg,
                    manifest=manifest,
                    default_dataset_name=default_dataset_name,
                    dataset_env=dataset_env or None,
                    runs_dir=runs_dir,
                    dataset_root=spec.root,
                )
            tr["matrix_metadata"] = bench_matrix_metadata(cfg)
            if would_eval and tr.get("status") == "ok" and tr.get("weights"):
                run_name = str(tr.get("run_name") or _bench_run_name(cfg))
                eval_path = runs_dir / run_name / "test_eval.json"
                backend = select_backend(cfg)
                if backend == "external":
                    max_det = _bench_eval_max_det(cfg, {})
                    ev = _invoke_test_eval_for_external(
                        cfg=cfg,
                        weights=str(tr["weights"]),
                        dataset_root=spec.root,
                        runs_dir=runs_dir,
                        run_name=run_name,
                        eval_out=eval_path,
                        max_det=max_det,
                        repo_root=repo_root,
                    )
                    tr["test_eval"] = ev
                    if ev.get("test_count_mae") is not None:
                        tr["test_count_mae"] = ev["test_count_mae"]
                    if ev.get("error_json"):
                        tr["error_test_report"] = ev["error_json"]
                    train_runs.append(tr)
                    _notify_row(tr, cfg)
                    continue
                    train_doc = {"eval": recipe.get("eval") if isinstance(recipe.get("eval"), dict) else {}}
                else:
                    cached = _cached_ultralytics_weights(cfg)
                    train_doc = _bench_to_train_config(
                        cfg, weights_path=str(cached or tr["weights"])
                    )
                eval_section = (
                    train_doc.get("eval") if isinstance(train_doc.get("eval"), dict) else {}
                )
                if post_train_eval_skipped(cli_skip=False, eval_section=eval_section):
                    train_runs.append(tr)
                    _notify_row(tr, cfg)
                    continue
                if backend == "supergradients":
                    max_det = _bench_eval_max_det(cfg, train_doc)
                    ev = _invoke_test_eval_for_supergradients(
                        cfg=cfg,
                        weights=str(tr["weights"]),
                        dataset_root=spec.root,
                        eval_out=eval_path,
                        max_det=max_det,
                    )
                else:
                    ev = _invoke_test_eval_for_bench(
                        cfg=cfg,
                        weights=str(tr["weights"]),
                        manifest=manifest,
                        default_dataset_name=default_dataset_name,
                        dataset_env=dataset_env or None,
                        train_doc=train_doc,
                        eval_out=eval_path,
                    )
                tr["test_eval"] = ev
                tr["mAP50"] = ev.get("mAP50")
                tr["mAP50_95"] = ev.get("mAP50_95")
            train_runs.append(tr)
            _notify_row(tr, cfg)
        train_payload: dict[str, object] = {
            "schema_version": "benchmark_matrix_train.v1",
            "status": "train",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "dataset": {"description": dataset_description},
            "runs_dir": str(runs_dir.resolve()),
            "runs": train_runs,
        }
        write_json(train_out, train_payload)

    if not dry_run and would_eval and not would_train:
        assert yolo_data_yaml is not None
        eval_runs: list[dict[str, object]] = []
        for cfg in configs:
            backend = select_backend(cfg)
            available, missing_reason = _backend_availability(backend, cfg)
            if not available:
                eval_runs.append(
                    {
                        "status": "skipped",
                        "reason": missing_reason,
                        "backend": backend,
                        "config_path": str(cfg.path),
                        "name": cfg.name,
                        "model_id": cfg.model_id,
                        "model": cfg.model,
                    }
                )
                continue

            if backend != "ultralytics":
                eval_runs.append(
                    {
                        "status": "skipped",
                        "reason": "backend_not_implemented",
                        "backend": backend,
                        "config_path": str(cfg.path),
                        "name": cfg.name,
                        "model_id": cfg.model_id,
                        "model": cfg.model,
                    }
                )
                continue

            weights = _pick_existing_weights(cfg)
            if weights is None:
                eval_runs.append(
                    {
                        "status": "skipped",
                        "reason": "weights_not_found",
                        "backend": "ultralytics",
                        "config_path": str(cfg.path),
                        "name": cfg.name,
                        "model": cfg.model,
                    }
                )
                continue
            eval_runs.append(_ultralytics_eval_one(cfg=cfg, dataset_yaml=yolo_data_yaml, weights=weights))

        eval_payload: dict[str, object] = {
            "status": "eval",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "dataset": {"description": dataset_description, "yolo_data_yaml": str(yolo_data_yaml)},
            "runs": eval_runs,
        }
        write_json(eval_out, eval_payload)
    cli_print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

