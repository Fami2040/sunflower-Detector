from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from harchoc.datasets import resolve_dataset
from harchoc.yaml_minimal import parse_minimal_yaml as _parse_minimal_yaml


def load_config(path: str | os.PathLike) -> dict[str, Any]:
    """
    Load a single experiment config.

    Preferred format is JSON. For legacy bench YAML, supports a minimal YAML subset
    (enough for existing bench configs) without adding new dependencies.
    """
    p = Path(path)
    suf = p.suffix.lower()
    if suf == ".json":
        obj = json.loads(p.read_text("utf-8"))
        if not isinstance(obj, dict):
            raise TypeError(f"Expected JSON object at top-level: {p}")
        return obj
    if suf in (".yaml", ".yml"):
        obj = _parse_minimal_yaml(p)
        return {str(k): v for k, v in obj.items()}
    raise ValueError(f"Unsupported config extension {p.suffix!r} for {p}")


def _repo_abspath(repo_root: Path, raw: str | os.PathLike | None) -> Path | None:
    if raw is None:
        return None
    s = str(raw).strip()
    if not s:
        return None
    p = Path(s).expanduser()
    if p.is_absolute():
        return p.resolve()
    return (repo_root / p).resolve()


def _resolve_split_source(
    *,
    cfg: dict[str, Any],
    repo_root: Path,
    dataset_root: Path,
    split: str,
) -> tuple[dict[str, object], Path]:
    """
    Resolve split source for a named split ("train"/"val"/"test").

    Supported config fields (checked in this order):
    - split_sources.<split> = {"kind": "...", "path": "..."}
    - split_file / split_dir (applies to "test" by default for backwards compatibility)
    - split_file_<split> / split_dir_<split>
    - default: repo_root/data/splits/<split>.txt if present, else dataset_root/images/<split>
    """
    sources = cfg.get("split_sources")
    if isinstance(sources, dict):
        item = sources.get(split)
        if isinstance(item, dict):
            kind = str(item.get("kind") or "").strip()
            raw_path = item.get("path")
            p = _repo_abspath(repo_root, raw_path)
            if kind in ("split_file", "dir") and p is not None:
                return ({"kind": kind, "path": str(p)}, p)

    def _pick_key(base: str) -> str | None:
        k1 = f"{base}_{split}"
        if k1 in cfg:
            return k1
        if split == "test" and base in cfg:
            return base
        return None

    if (k := _pick_key("split_file")) is not None:
        p = _repo_abspath(repo_root, cfg.get(k))
        if p is not None:
            return ({"kind": "split_file", "path": str(p)}, p)
    if (k := _pick_key("split_dir")) is not None:
        p = _repo_abspath(repo_root, cfg.get(k))
        if p is not None:
            return ({"kind": "dir", "path": str(p)}, p)

    default_split_file = (repo_root / "data" / "splits" / f"{split}.txt").resolve()
    if default_split_file.is_file():
        return ({"kind": "split_file", "path": str(default_split_file)}, default_split_file)

    p = (dataset_root / "images" / split).resolve()
    return ({"kind": "dir", "path": str(p)}, p)


def normalize_config(cfg: dict[str, Any], repo_root: str | os.PathLike) -> dict[str, Any]:
    """
    Fill defaults and resolve dataset + split sources.

    Returns a JSON-serializable dict ("ResolvedConfig") suitable for reports.
    This function is intentionally lightweight and safe to import in CI.
    """
    rr = Path(repo_root).expanduser().resolve()
    manifest_path = _repo_abspath(rr, cfg.get("manifest")) or (rr / "data" / "manifest.json").resolve()
    default_dataset_name = str(cfg.get("default_dataset_name") or "sunflower-cvat-2500")

    dataset_env = cfg.get("dataset_env")
    environ = dataset_env if isinstance(dataset_env, dict) else None
    spec = resolve_dataset(
        manifest_path=manifest_path,
        default_dataset_name=default_dataset_name,
        environ={str(k): str(v) for k, v in environ.items()} if environ is not None else None,
    )

    resolved: dict[str, Any] = dict(cfg)
    resolved["repo_root"] = str(rr)
    resolved["manifest"] = str(manifest_path)
    resolved["default_dataset_name"] = default_dataset_name
    resolved["dataset"] = {
        "root": str(spec.root),
        "yolo_data_yaml": str(spec.yolo_data_yaml) if spec.yolo_data_yaml is not None else None,
        "name": spec.name,
        "manifest_path": str(spec.manifest_path),
    }

    splits: dict[str, Any] = {}
    for split in ("train", "val", "test"):
        src, p = _resolve_split_source(cfg=cfg, repo_root=rr, dataset_root=spec.root, split=split)
        splits[split] = {"source": src, "path": str(p)}
    resolved["splits"] = splits

    # Back-compat convenience for eval-centric code paths.
    resolved["split_source"] = splits["test"]["source"]
    resolved["split_path"] = splits["test"]["path"]

    return json.loads(json.dumps(resolved, default=str))


def normalize_experiment_spec(cfg: dict[str, Any], repo_root: str | os.PathLike) -> dict[str, Any]:
    """
    Normalize a canonical experiment spec (configs/experiments/*.json).

    Schema (experiments.v1):
      {
        "schema_version": "experiments.v1",
        "dataset": {
          "manifest": "data/manifest.json",
          "default_dataset_name": "sunflower-cvat-2500",
          "dataset_env": {"DATASET_ROOT": "...", "YOLO_DATA_YAML": "...", "DATASET_NAME": "..."}  // optional
        },
        "run": {
          "kind": "eval" | "benchmark_matrix" | "split_drift" | "threshold_sweep" | "error_analysis" | "cv_eval",
          "dry_run": true,
          ... script-specific keys ...
        }
      }

    This is intentionally dependency-light (safe to import in CI).
    """
    rr = Path(repo_root).expanduser().resolve()
    schema_version = str(cfg.get("schema_version") or "").strip()
    if schema_version and schema_version != "experiments.v1":
        raise ValueError(f"Unsupported schema_version={schema_version!r} (expected 'experiments.v1')")

    dataset_cfg = cfg.get("dataset")
    dataset_obj: dict[str, Any] = dataset_cfg if isinstance(dataset_cfg, dict) else {}

    # Reuse legacy normalizer by passing a flattened view.
    flattened: dict[str, Any] = {
        "manifest": dataset_obj.get("manifest"),
        "default_dataset_name": dataset_obj.get("default_dataset_name"),
        "dataset_env": dataset_obj.get("dataset_env"),
        # Allow canonical configs to optionally set split sources using the same keys.
        "split_sources": dataset_obj.get("split_sources") or cfg.get("split_sources"),
        "split_file": dataset_obj.get("split_file") or cfg.get("split_file"),
        "split_dir": dataset_obj.get("split_dir") or cfg.get("split_dir"),
        "split_file_train": dataset_obj.get("split_file_train") or cfg.get("split_file_train"),
        "split_file_val": dataset_obj.get("split_file_val") or cfg.get("split_file_val"),
        "split_file_test": dataset_obj.get("split_file_test") or cfg.get("split_file_test"),
        "split_dir_train": dataset_obj.get("split_dir_train") or cfg.get("split_dir_train"),
        "split_dir_val": dataset_obj.get("split_dir_val") or cfg.get("split_dir_val"),
        "split_dir_test": dataset_obj.get("split_dir_test") or cfg.get("split_dir_test"),
    }
    legacy_resolved = normalize_config({k: v for k, v in flattened.items() if v is not None}, repo_root=rr)

    run_cfg = cfg.get("run")
    run_obj: dict[str, Any] = run_cfg if isinstance(run_cfg, dict) else {}
    kind = str(run_obj.get("kind") or "").strip()
    dry_run = bool(run_obj.get("dry_run", True))

    def _norm_path(key: str) -> str | None:
        p = _repo_abspath(rr, run_obj.get(key))
        return str(p) if p is not None else None

    normalized_run: dict[str, Any] = dict(run_obj)
    normalized_run["kind"] = kind
    normalized_run["dry_run"] = dry_run

    # Normalize a few common filesystem args as repo-root absolute paths.
    for k in (
        "out",
        "eval_out",
        "bench_dir",
        "weights",
        "split_file",
        "gt_json",
        "preds_json",
        "csv_out",
        "report",
        "locked_conf_from",
        "write_fold_splits",
    ):
        if k in normalized_run:
            normalized_run[k] = _norm_path(k)

    fold_metrics = run_obj.get("fold_metrics")
    if isinstance(fold_metrics, list):
        normalized_run["fold_metrics"] = [
            str(p) for x in fold_metrics if (p := _repo_abspath(rr, x)) is not None
        ]
    elif fold_metrics is not None and str(fold_metrics).strip():
        p = _repo_abspath(rr, fold_metrics)
        if p is not None:
            normalized_run["fold_metrics"] = [str(p)]

    # Canonicalize bench_config list entries (paths).
    if isinstance(run_obj.get("bench_config"), list):
        normalized_run["bench_config"] = [str(_repo_abspath(rr, p)) for p in run_obj["bench_config"] if _repo_abspath(rr, p) is not None]

    # Keep dataset-root-relative args as-is (e.g. split_drift.splits_dir) since they are not repo paths.
    resolved: dict[str, Any] = {
        "schema_version": "experiments.v1",
        "repo_root": legacy_resolved["repo_root"],
        "dataset": legacy_resolved["dataset"],
        "splits": legacy_resolved["splits"],
        "run": normalized_run,
    }
    return json.loads(json.dumps(resolved, default=str))


def _deep_merge_dict(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """
    Deep-merge mapping values.
    - dict values merge recursively
    - non-dict values replace
    """
    out: dict[str, Any] = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge_dict(out[k], v)  # type: ignore[arg-type]
        else:
            out[k] = v
    return out


def load_config_json(raw: str | None) -> dict[str, Any]:
    """
    Load config from either:
    - an inline JSON object string, or
    - a path to a JSON file.
    """
    if raw is None:
        return {}
    s = str(raw).strip()
    if not s:
        return {}
    if s.startswith("{") or s.startswith("["):
        obj = json.loads(s)
    else:
        p = Path(s).expanduser()
        obj = json.loads(p.read_text("utf-8"))
    if not isinstance(obj, dict):
        raise SystemExit("--config must be a JSON object (or a path to a JSON object file).")
    return obj


def merge_experiment_config(*, config: dict[str, Any], cli: dict[str, Any]) -> dict[str, Any]:
    """
    Merge config JSON with CLI-provided fields.
    Precedence: CLI overrides config.
    """
    if not config:
        return dict(cli)
    return _deep_merge_dict(config, cli)


def script_section_from_config(cfg: dict[str, Any], section: str) -> dict[str, Any]:
    """
    Return script kwargs from a flat ``section`` block or ``experiments.v1`` ``run``.

    When ``schema_version`` is ``experiments.v1`` and ``run.kind`` matches ``section``,
    the ``run`` dict is returned (excluding ``kind`` / ``dry_run`` handled by callers).
    """
    if cfg.get("schema_version") == "experiments.v1":
        run = cfg.get("run")
        if isinstance(run, dict):
            kind = str(run.get("kind") or "").strip()
            if kind == "sahi_matrix_eval":
                merged = {k: v for k, v in run.items() if k not in ("kind",)}
                merged["sahi_eval"] = True
                return merged
            if kind == section:
                return {k: v for k, v in run.items() if k not in ("kind",)}
    block = cfg.get(section)
    return block if isinstance(block, dict) else {}

