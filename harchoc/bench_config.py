from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from harchoc.rtdetr_limits import (
    is_rtdetr_model,
    rtdetr_fields_from_train_json,
    validate_rtdetr_infer_max_det,
    validate_rtdetr_query_cap,
)
from harchoc.training_budget import _budget_limit_int
from harchoc.train_config import load_train_config_json
from harchoc.yaml_minimal import parse_minimal_yaml


@dataclass(frozen=True)
class BenchConfig:
    path: Path
    name: str | None
    task: str | None
    backend: str | None
    model_id: str | None
    model: str | None  # weights id / file path (backend-specific)
    source_id: str | None  # keys configs/external/detector_sources.v1.json for backend=external
    groups: tuple[str, ...]
    infer: dict[str, object]
    imgsz: int | None  # legacy alias for infer.imgsz
    epochs: int | None
    patience: int | None
    seed: int | None
    train_config: str | None  # optional path to committed train JSON (see configs/experiments/train_bench_*.json)
    notes: str | None


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def is_bench_row_config(path: Path) -> bool:
    """True for matrix row configs (excludes ``_defaults.yaml`` include anchors)."""
    return path.suffix.lower() in (".yaml", ".json") and not path.name.startswith("_")


def select_backend(cfg: BenchConfig) -> str:
    if cfg.backend in ("ultralytics", "supergradients", "external"):
        return cfg.backend
    if cfg.model_id and not cfg.model:
        return "supergradients"
    return "ultralytics"


def _bench_model_stem(cfg: BenchConfig) -> str | None:
    if cfg.model and str(cfg.model).strip():
        return Path(str(cfg.model).strip()).stem or None
    if cfg.model_id and str(cfg.model_id).strip():
        return str(cfg.model_id).strip().replace(" ", "_") or None
    return None


def _infer_imgsz(cfg: BenchConfig) -> int | None:
    v = cfg.infer.get("imgsz")
    if isinstance(v, int):
        return v
    return cfg.imgsz


def _load_matrix_rows() -> dict[str, Any]:
    path = _repo_root() / "configs/zoo/matrix_rows.v1.json"
    return json.loads(path.read_text(encoding="utf-8"))


_RUNTIME_TRAIN_BENCH_COMMITTED_STEMS = frozenset(
    {"rtdetr-l", "rtdetr-l_nq1024", "rtdetr-x", "yolo_nas_s"}
)


def _matrix_row_for_cfg(cfg: BenchConfig) -> dict[str, Any] | None:
    stem = _bench_model_stem(cfg)
    if not stem:
        return None
    for row in _load_matrix_rows().get("rows") or []:
        if str(row.get("id") or "") == stem or str(row.get("train_config_stem") or "") == stem:
            return row
    return None


def _runtime_train_bench_raw(cfg: BenchConfig) -> dict[str, Any] | None:
    row = _matrix_row_for_cfg(cfg)
    if row is None:
        return None
    stem = str(row.get("train_config_stem") or row.get("id") or _bench_model_stem(cfg) or "")
    if not stem or stem in _RUNTIME_TRAIN_BENCH_COMMITTED_STEMS:
        return None
    overlay: dict[str, Any] = {
        "extends": "configs/experiments/train_bench_base.json",
        "batch": 1,
        "cache": False,
        "notes": f"Matrix/bench training recipe for {stem} @ 1280 (batch=1 on 8GB-class GPUs).",
    }
    if cfg.model and str(cfg.model).strip():
        overlay["model"] = str(cfg.model).strip()
    elif cfg.model_id and str(cfg.model_id).strip():
        overlay["model_id"] = str(cfg.model_id).strip()
    return overlay


def _load_bench_train_raw(cfg: BenchConfig) -> dict[str, Any]:
    committed = _resolve_bench_train_config_path(cfg)
    if committed is not None and committed.is_file():
        return _load_committed_train_bench_json(committed)
    runtime = _runtime_train_bench_raw(cfg)
    if runtime is not None:
        from harchoc.train_config import resolve_train_config_extends

        return resolve_train_config_extends(runtime, repo_root=_repo_root())
    raise FileNotFoundError(
        f"no committed or runtime train bench config for {cfg.path} "
        f"(stem={_bench_model_stem(cfg)!r})"
    )


def _resolve_bench_train_config_path(cfg: BenchConfig) -> Path | None:
    if cfg.train_config and str(cfg.train_config).strip():
        p = Path(str(cfg.train_config).strip()).expanduser()
        if not p.is_absolute():
            p = (_repo_root() / p).resolve()
        return p if p.is_file() else None
    stem = _bench_model_stem(cfg)
    if stem:
        p = (_repo_root() / "configs" / "experiments" / f"train_bench_{stem}.json").resolve()
        if p.is_file():
            return p
    return None


def _load_committed_train_bench_json(path: Path) -> dict[str, Any]:
    return load_train_config_json(path, repo_root=_repo_root())


def _merge_bench_dicts(base: dict[str, object], overlay: dict[str, object]) -> dict[str, object]:
    out: dict[str, object] = dict(base)
    for key, val in overlay.items():
        if key == "include":
            continue
        if key in out and isinstance(out[key], dict) and isinstance(val, dict):
            inner = dict(out[key])  # type: ignore[arg-type]
            inner.update(val)  # type: ignore[arg-type]
            out[key] = inner
        else:
            out[key] = val
    return out


def _resolve_bench_includes(obj: dict[str, object], path: Path) -> dict[str, object]:
    include = obj.get("include")
    if not isinstance(include, str) or not str(include).strip():
        return obj
    include_path = (path.parent / str(include).strip()).resolve()
    if not include_path.is_file():
        raise FileNotFoundError(
            f"bench include not found: {include!r} (from {path}, tried {include_path})"
        )
    base = parse_minimal_yaml(include_path)
    base = _resolve_bench_includes(base, include_path)
    return _merge_bench_dicts(base, obj)


def validate_bench_config(cfg: BenchConfig) -> None:
    """
    Fail fast on malformed bench configs.

    Keeps CI lightweight: validation is stdlib-only and runs during dry-run.
    """
    backend = select_backend(cfg)
    if cfg.backend is not None and cfg.backend not in ("ultralytics", "supergradients", "external"):
        raise SystemExit(
            f"Invalid backend={cfg.backend!r} in {cfg.path} (expected ultralytics|supergradients|external)"
        )

    if backend == "ultralytics":
        if not cfg.model or not str(cfg.model).strip():
            raise SystemExit(f"Missing required field `model` for ultralytics in {cfg.path}")
    if backend == "external":
        sid = (cfg.source_id or cfg.model_id or "").strip()
        if not sid:
            raise SystemExit(
                f"Missing required field `source_id` (or `model_id`) for external in {cfg.path}"
            )
        from harchoc.detector_sources import entry_for_bench

        if entry_for_bench(model_id=cfg.model_id, source_id=cfg.source_id) is None:
            raise SystemExit(
                f"Unknown external source_id={sid!r} in {cfg.path} "
                f"(see configs/external/detector_sources.v1.json)"
            )
    if backend == "supergradients":
        if (not cfg.model_id or not str(cfg.model_id).strip()) and (not cfg.model or not str(cfg.model).strip()):
            raise SystemExit(f"Missing `model_id` (or `model`) for supergradients in {cfg.path}")

    if cfg.epochs is not None:
        if cfg.epochs <= 0:
            raise SystemExit(f"epochs must be > 0 in {cfg.path} (got {cfg.epochs})")
        max_epochs = _budget_limit_int("HARCHOC_MAX_EPOCHS", default=500)
        if cfg.epochs > max_epochs:
            raise SystemExit(f"epochs={cfg.epochs} exceeds HARCHOC_MAX_EPOCHS={max_epochs} in {cfg.path}")

    if cfg.patience is not None:
        if cfg.patience < 0:
            raise SystemExit(f"patience must be >= 0 in {cfg.path} (got {cfg.patience})")
        if cfg.epochs is not None and cfg.patience >= cfg.epochs:
            raise SystemExit(f"patience={cfg.patience} must be < epochs={cfg.epochs} in {cfg.path}")

    imgsz = _infer_imgsz(cfg)
    if imgsz is not None:
        if imgsz <= 0:
            raise SystemExit(f"infer.imgsz must be > 0 in {cfg.path} (got {imgsz})")
        max_imgsz = _budget_limit_int("HARCHOC_MAX_IMGSZ", default=2048)
        if imgsz > max_imgsz:
            raise SystemExit(f"infer.imgsz={imgsz} exceeds HARCHOC_MAX_IMGSZ={max_imgsz} in {cfg.path}")

    try:
        raw = _load_bench_train_raw(cfg)
    except FileNotFoundError:
        raw = None
    if raw is not None:
        train_path = _resolve_bench_train_config_path(cfg)
        path_label = str(train_path) if train_path is not None else str(cfg.path)
        if raw.get("batch") is not None:
            batch = int(raw["batch"])
            max_batch = _budget_limit_int("HARCHOC_MAX_BATCH", default=16)
            if batch > max_batch:
                raise SystemExit(
                    f"batch={batch} in {path_label} exceeds HARCHOC_MAX_BATCH={max_batch}"
                )
        validate_rtdetr_query_cap(
            model=cfg.model,
            train_json=raw,
            train_json_path=path_label,
            fail=True,
        )
        infer_max_det = cfg.infer.get("max_det")
        infer_max_det_int = int(infer_max_det) if isinstance(infer_max_det, int) else None
        validate_rtdetr_infer_max_det(
            model=cfg.model,
            infer_max_det=infer_max_det_int,
            train_json=raw,
            train_json_path=path_label,
            cfg_path=str(cfg.path),
            fail=True,
        )


def load_bench_config(path: Path) -> BenchConfig:
    if path.suffix.lower() == ".json":
        raw = json.loads(path.read_text("utf-8"))
        if not isinstance(raw, dict):
            raise TypeError(f"Expected top-level JSON object: {path}")
        obj: dict[str, object] = {str(k): v for k, v in raw.items()}
    else:
        obj = parse_minimal_yaml(path)
        if obj.get("include"):
            obj = _resolve_bench_includes(obj, path)
    infer = obj.get("infer") if isinstance(obj.get("infer"), dict) else {}
    assert isinstance(infer, dict)
    infer_imgsz = infer.get("imgsz")
    imgsz = int(infer_imgsz) if isinstance(infer_imgsz, int) else (int(obj["imgsz"]) if isinstance(obj.get("imgsz"), int) else None)
    raw_groups = obj.get("groups", obj.get("group"))
    groups: tuple[str, ...] = ()
    if isinstance(raw_groups, str):
        # Keep YAML subset parser list-free. Accept comma/space separated strings.
        parts = [p.strip() for p in raw_groups.replace(" ", ",").split(",")]
        groups = tuple([p for p in parts if p])
    cfg = BenchConfig(
        path=path,
        name=str(obj.get("name")) if obj.get("name") is not None else None,
        task=str(obj.get("task")) if obj.get("task") is not None else None,
        backend=str(obj.get("backend")) if obj.get("backend") is not None else None,
        model_id=str(obj.get("model_id")) if obj.get("model_id") is not None else None,
        model=str(obj.get("model")) if obj.get("model") is not None else None,
        source_id=str(obj.get("source_id")) if obj.get("source_id") is not None else None,
        groups=groups,
        infer=infer,
        imgsz=imgsz,
        epochs=int(obj["epochs"]) if isinstance(obj.get("epochs"), int) else None,
        patience=int(obj["patience"]) if isinstance(obj.get("patience"), int) else None,
        seed=int(obj["seed"]) if isinstance(obj.get("seed"), int) else None,
        train_config=str(obj["train_config"]).strip()
        if obj.get("train_config") is not None and str(obj.get("train_config")).strip()
        else None,
        notes=str(obj.get("notes")) if obj.get("notes") is not None else None,
    )
    validate_bench_config(cfg)
    return cfg


def _bench_run_name(cfg: BenchConfig) -> str:
    """
    EXPERIMENTS.md pattern: {model}_e{N}_s{seed} when model/model_id + epochs + seed exist.
    Optional env HARCHOC_BENCH_USE_LEGACY_NAME=1 forces cfg.name / path stem.
    """
    if os.getenv("HARCHOC_BENCH_USE_LEGACY_NAME", "").strip() in ("1", "true", "yes"):
        if cfg.name and str(cfg.name).strip():
            return str(cfg.name).strip().replace(" ", "_")
        return cfg.path.stem

    stem = _bench_model_stem(cfg)
    if stem and cfg.epochs is not None and cfg.seed is not None:
        return f"{stem}_e{int(cfg.epochs)}_s{int(cfg.seed)}"
    if cfg.name and str(cfg.name).strip():
        return str(cfg.name).strip().replace(" ", "_")
    return cfg.path.stem


def _bench_to_train_config(cfg: BenchConfig, *, weights_path: str) -> dict[str, Any]:
    try:
        raw = _load_bench_train_raw(cfg)
    except FileNotFoundError:
        raw = None
    if raw is not None:
        from scripts.train import _merge_train_config

        merged = _merge_train_config(raw)
        merged["model"] = weights_path
        imgsz = _infer_imgsz(cfg)
        if imgsz is not None:
            merged["imgsz"] = imgsz
        if cfg.epochs is not None:
            merged["epochs"] = int(cfg.epochs)
        if cfg.patience is not None:
            merged["patience"] = int(cfg.patience)
        if cfg.seed is not None:
            merged["seed"] = int(cfg.seed)
        # Bench infer.max_det is an eval/infer cap only; do not overwrite train max_det
        # from committed train_bench_*.json (typically 3000 for dense trays).
        infer_max_det = cfg.infer.get("max_det")
        out: dict[str, Any] = {"train": merged}
        eval_section = raw.get("eval") if isinstance(raw.get("eval"), dict) else {}
        eval_out = dict(eval_section)
        if isinstance(infer_max_det, int):
            eval_out.setdefault("max_det", int(infer_max_det))
        if eval_out:
            out["eval"] = eval_out
        return out

    imgsz = _infer_imgsz(cfg) or 1280
    train: dict[str, Any] = {
        "model": weights_path,
        "epochs": int(cfg.epochs) if cfg.epochs is not None else 100,
        "imgsz": imgsz,
    }
    if cfg.patience is not None:
        train["patience"] = int(cfg.patience)
    if cfg.seed is not None:
        train["seed"] = int(cfg.seed)
    infer_max_det = cfg.infer.get("max_det")
    out: dict[str, Any] = {"train": train}
    if isinstance(infer_max_det, int):
        out["eval"] = {"max_det": int(infer_max_det)}
    return out


def _bench_eval_max_det(cfg: BenchConfig, train_doc: dict[str, Any]) -> int | None:
    eval_section = train_doc.get("eval")
    if isinstance(eval_section, dict):
        v = eval_section.get("max_det")
        if isinstance(v, int):
            return v
    max_det = cfg.infer.get("max_det")
    return int(max_det) if isinstance(max_det, int) else None


def bench_external_provenance(cfg: BenchConfig) -> dict[str, object] | None:
    """Provenance block for backend=external bench rows (matrix + weights report)."""
    from harchoc.detector_sources import entry_for_bench, external_entry_provenance

    if select_backend(cfg) != "external":
        return None
    entry = entry_for_bench(model_id=cfg.model_id, source_id=cfg.source_id)
    if entry is None:
        return None
    return external_entry_provenance(entry)


def bench_matrix_metadata(cfg: BenchConfig) -> dict[str, object]:
    """
    Provenance fields for matrix plan rows and ``matrix_train.json`` runs.

    RT-DETR query-cap policy is read from the resolved committed train JSON
    (typically ``train_bench_rtdetr-l.json``).
    """
    backend = select_backend(cfg)
    train_raw: dict[str, Any] = {}
    train_json_path = ""
    try:
        train_raw = _load_bench_train_raw(cfg)
        committed = _resolve_bench_train_config_path(cfg)
        train_json_path = str(committed) if committed is not None else str(cfg.path)
    except FileNotFoundError:
        pass

    model_ref = cfg.model or train_raw.get("model") or train_raw.get("model_id")
    nms_free = is_rtdetr_model(str(model_ref) if model_ref is not None else None)

    train_max_det_raw = train_raw.get("max_det")
    train_max_det = int(train_max_det_raw) if train_max_det_raw is not None else None

    infer_max_det_raw = cfg.infer.get("max_det")
    infer_max_det = int(infer_max_det_raw) if isinstance(infer_max_det_raw, int) else None

    rect_raw = train_raw.get("rect")
    rect = bool(rect_raw) if rect_raw is not None else False

    meta: dict[str, object] = {
        "backend": backend,
        "nms_free": nms_free,
        "rect": rect,
        "infer_max_det": infer_max_det,
        "train_max_det": train_max_det,
        "num_queries": None,
        "accept_rtdetr_query_truncation": None,
    }
    if nms_free:
        fields = rtdetr_fields_from_train_json(
            train_raw,
            path=train_json_path or str(cfg.path),
        )
        meta["num_queries"] = int(fields["num_queries"])
        meta["accept_rtdetr_query_truncation"] = bool(fields["accept_rtdetr_query_truncation"])
    return meta
