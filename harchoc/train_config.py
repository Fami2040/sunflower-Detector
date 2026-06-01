from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

from harchoc.aug_config import merge_aug_yaml
from harchoc.experiment_config import _deep_merge_dict

_EXTENDS_KEY = "extends"

# Train parity: bench recipes may differ from baseline only on these keys (plus inline aug vs aug_config).
BENCH_PARITY_ALLOWED_DIFF_KEYS = frozenset({"model", "batch", "notes", "aug_config", "amp", "grad_clip"})
# Nested ``eval`` keys stripped during bench iso-config parity (RT-DETR uses query-aligned max_det).
BENCH_EVAL_PARITY_ALLOWED_DIFF_KEYS = frozenset({"notes", "device", "max_det"})
BASELINE_INLINE_AUG_KEYS = frozenset(
    {"mosaic", "hsv_h", "hsv_s", "hsv_v", "translate", "scale"},
)
BENCH_AUG_CONFIG_RELPATH = "configs/aug/robustness_minimal.yaml"

# P1-AUG-MOSAIC / P1-AUG-CLOSE sweep defaults (docs/EXPERIMENTS.md § P1-AUG sweeps).
MOSAIC_SWEEP_VALUES: tuple[float, ...] = (0.0, 0.1, 0.3)
CLOSE_MOSAIC_SWEEP_100EP: tuple[int, ...] = (10, 15, 25)
CLOSE_MOSAIC_PRODUCTION_EPOCHS: int = 100
CLOSE_MOSAIC_PRODUCTION_DEFAULT: int = 15
# Smoke epoch tiers: micro = CI/unittest; rank = GPU aug / RT-DETR smokes (see docs/training_budget.md).
SMOKE_EPOCHS_MICRO: int = 3
SMOKE_EPOCHS_RANK: int = 15
CLOSE_MOSAIC_SMOKE_EPOCHS: int = SMOKE_EPOCHS_RANK
# Pre-scaled close_mosaic values for 15-ep smoke/sweep YAMLs (skip epoch rescaling).
CLOSE_MOSAIC_SMOKE_15EP_EFFECTIVE: frozenset[int] = frozenset({2, 3, 4})
# Architecture-specific keys allowed only on named bench overlays (not in baseline).
TRAIN_BENCH_MODEL_EXTRA_KEYS: dict[str, frozenset[str]] = {
    "train_bench_rtdetr-l.json": frozenset(
        {
            "num_queries",
            "documented_peak_gt_boxes_per_image",
            "accept_rtdetr_query_truncation",
        }
    ),
    "train_bench_rtdetr-l_nq1024.json": frozenset(
        {
            "num_queries",
            "documented_peak_gt_boxes_per_image",
            "accept_rtdetr_query_truncation",
        }
    ),
    "train_bench_rtdetr-x.json": frozenset(
        {
            "num_queries",
            "documented_peak_gt_boxes_per_image",
            "accept_rtdetr_query_truncation",
        }
    ),
}


def resolve_train_config_extends(
    cfg: dict[str, Any],
    *,
    repo_root: Path,
    config_path: Path | None = None,
    _stack: tuple[Path, ...] = (),
) -> dict[str, Any]:
    """
    Merge ``extends`` base JSON into a flat train config (overlay wins).

    ``extends`` is a repo-relative or absolute path to another JSON object.
    The ``extends`` key is stripped from the result. Cycles are rejected.
    """
    extends = cfg.get(_EXTENDS_KEY)
    if extends is None or (isinstance(extends, str) and not str(extends).strip()):
        return dict(cfg)

    base_spec = str(extends).strip()
    rel = Path(base_spec).expanduser()
    candidates: list[Path] = []
    if rel.is_absolute():
        candidates.append(rel.resolve())
    else:
        candidates.append((repo_root / rel).resolve())
        if config_path is not None:
            candidates.append((config_path.parent / rel).resolve())

    base_path = next((c for c in candidates if c.is_file()), None)
    if base_path is None:
        tried = ", ".join(str(c) for c in candidates) or base_spec
        raise FileNotFoundError(f"train config extends not found: {tried}")

    if base_path in _stack:
        chain = " -> ".join(str(p) for p in (*_stack, base_path))
        raise ValueError(f"cyclic {_EXTENDS_KEY} in train config: {chain}")

    base_raw = json.loads(base_path.read_text("utf-8"))
    if not isinstance(base_raw, dict):
        raise TypeError(f"Expected top-level JSON object: {base_path}")

    base_merged = resolve_train_config_extends(
        base_raw,
        repo_root=repo_root,
        config_path=base_path,
        _stack=(*_stack, base_path),
    )
    overlay = {k: v for k, v in cfg.items() if k != _EXTENDS_KEY}
    return _deep_merge_dict(base_merged, overlay)


def inline_aug_keys_for_bench_parity(repo_root: Path) -> frozenset[str]:
    """Ultralytics aug keys from the committed bench aug YAML plus baseline inline keys."""
    from harchoc.yaml_minimal import parse_minimal_yaml

    aug_path = repo_root / BENCH_AUG_CONFIG_RELPATH
    ultra = parse_minimal_yaml(aug_path).get("ultralytics")
    if isinstance(ultra, dict):
        return BASELINE_INLINE_AUG_KEYS | frozenset(str(k) for k in ultra)
    return BASELINE_INLINE_AUG_KEYS


def normalize_train_config_for_bench_parity(
    cfg: dict[str, Any],
    *,
    repo_root: Path,
    bench_config_name: str | None = None,
) -> dict[str, Any]:
    """Drop allowed bench-vs-baseline diffs and aug representation keys for equality checks."""
    strip = BENCH_PARITY_ALLOWED_DIFF_KEYS | inline_aug_keys_for_bench_parity(repo_root)
    if bench_config_name:
        strip = strip | TRAIN_BENCH_MODEL_EXTRA_KEYS.get(bench_config_name, frozenset())
    out = {k: v for k, v in cfg.items() if k not in strip}
    eval_cfg = out.get("eval")
    if isinstance(eval_cfg, dict):
        eval_norm = {
            k: v for k, v in eval_cfg.items() if k not in BENCH_EVAL_PARITY_ALLOWED_DIFF_KEYS
        }
        if eval_norm:
            out["eval"] = eval_norm
        else:
            out.pop("eval", None)
    return out


def effective_train_aug_merged(cfg: dict[str, Any], *, repo_root: Path) -> dict[str, Any]:
    """Train kwargs after ``aug_config`` YAML merge (same order as ``scripts/train.py``)."""
    merged = dict(cfg)
    aug_spec = merged.get("aug_config")
    if aug_spec:
        aug_path = Path(str(aug_spec)).expanduser()
        if not aug_path.is_absolute():
            aug_path = (repo_root / aug_path).resolve()
        merged = merge_aug_yaml(merged, aug_path, repo_root=repo_root)
    return apply_close_mosaic_epoch_scale(merged)


# Keys that define a unique 15-ep smoke train schedule (for duplicate detection).
EFFECTIVE_RECIPE_KEYS: tuple[str, ...] = (
    "epochs",
    "patience",
    "seed",
    "model",
    "mosaic",
    "close_mosaic",
    "translate",
    "hsv_s",
    "hsv_v",
    "erasing",
    "fliplr",
    "scale",
    "mixup",
    "lr0",
    "optimizer",
    "amp",
    "batch",
    "imgsz",
)


def effective_train_recipe_fingerprint(cfg: dict[str, Any], *, repo_root: Path) -> str:
    """Stable hash of merged train+aug schedule (P1-AUG-DUP-MAE dedup)."""
    import hashlib

    merged = effective_train_aug_merged(cfg, repo_root=repo_root)
    subset = {k: merged.get(k) for k in EFFECTIVE_RECIPE_KEYS}
    payload = json.dumps(subset, sort_keys=True, default=str).encode()
    return hashlib.sha256(payload).hexdigest()[:16]


def job_train_recipe_fingerprint(
    *,
    repo_root: Path,
    train_config: str,
    aug_config: str | None = None,
) -> str:
    cfg = load_train_config_json(repo_root / train_config, repo_root=repo_root)
    if aug_config:
        cfg = dict(cfg)
        cfg["aug_config"] = aug_config
    return effective_train_recipe_fingerprint(cfg, repo_root=repo_root)


def apply_close_mosaic_epoch_scale(
    cfg: dict[str, Any],
    *,
    production_epochs: int = CLOSE_MOSAIC_PRODUCTION_EPOCHS,
) -> dict[str, Any]:
    """
    Scale 100-ep ``close_mosaic`` values for short smokes; cap at early-stop runway.

    Skips values already at or below the rank-smoke tier (e.g. ``close_mosaic=3`` @ 15 ep).
    """
    epochs_raw = cfg.get("epochs")
    close_raw = cfg.get("close_mosaic")
    if epochs_raw is None or close_raw is None:
        return cfg
    epochs = int(epochs_raw)
    close_mosaic = int(close_raw)
    if close_mosaic <= 0 or epochs >= production_epochs or epochs > SMOKE_EPOCHS_RANK:
        return cfg
    if epochs <= SMOKE_EPOCHS_RANK and close_mosaic in CLOSE_MOSAIC_SMOKE_15EP_EFFECTIVE:
        return cfg
    smoke_tier = scale_close_mosaic_for_epochs(epochs, production_epochs=production_epochs)
    if close_mosaic <= smoke_tier:
        return cfg
    scaled = scale_close_mosaic_for_epochs(
        epochs,
        production_epochs=production_epochs,
        production_close_mosaic=close_mosaic,
    )
    patience_raw = cfg.get("patience")
    if patience_raw is not None:
        runway = epochs - int(patience_raw)
        if runway <= 0:
            scaled = 0
        else:
            scaled = min(scaled, runway)
    if scaled == close_mosaic:
        return cfg
    out = dict(cfg)
    out["close_mosaic"] = scaled
    return out


def scale_close_mosaic_for_epochs(
    epochs: int,
    *,
    production_epochs: int = CLOSE_MOSAIC_PRODUCTION_EPOCHS,
    production_close_mosaic: int = CLOSE_MOSAIC_PRODUCTION_DEFAULT,
) -> int:
    """
    Scale ``close_mosaic`` for short smokes to preserve the production tail fraction.

    Example: ``scale_close_mosaic_for_epochs(15)`` → ``3`` when production is 15/100.
    """
    if epochs <= 0:
        raise ValueError(f"epochs must be positive, got {epochs}")
    if production_epochs <= 0:
        raise ValueError(f"production_epochs must be positive, got {production_epochs}")
    if production_close_mosaic <= 0:
        return 0
    return max(1, math.ceil(production_close_mosaic * epochs / production_epochs))


def resolve_close_mosaic(cfg: dict[str, Any], *, repo_root: Path) -> int:
    """``close_mosaic`` from inline fields or merged ``aug_config`` ultralytics block."""
    effective = effective_train_aug_merged(cfg, repo_root=repo_root)
    raw = effective.get("close_mosaic")
    if raw is None:
        raise ValueError(
            "close_mosaic missing: set inline or via aug_config ultralytics.close_mosaic"
        )
    return int(raw)


def validate_epochs_patience_close_mosaic(
    cfg: dict[str, Any],
    *,
    repo_root: Path,
    label: str = "",
) -> None:
    """
    Early stop must leave room for the mosaic-off tail (Ultralytics ``close_mosaic``).

    Requires ``epochs - patience >= close_mosaic`` after aug merge.
    """
    effective = effective_train_aug_merged(cfg, repo_root=repo_root)
    prefix = f"{label}: " if label else ""
    for key in ("epochs", "patience"):
        if effective.get(key) is None:
            raise ValueError(f"{prefix}missing required train field {key!r}")
    epochs = int(effective["epochs"])
    patience = int(effective["patience"])
    close_mosaic = resolve_close_mosaic(cfg, repo_root=repo_root)
    runway = epochs - patience
    if runway < close_mosaic:
        raise ValueError(
            f"{prefix}epochs ({epochs}) - patience ({patience}) = {runway} "
            f"must be >= close_mosaic ({close_mosaic})"
        )


def warn_train_schedule_close_mosaic(
    cfg: dict[str, Any],
    *,
    repo_root: Path,
    label: str = "",
    smoke_epochs_max: int = SMOKE_EPOCHS_RANK,
) -> None:
    """
    Runtime guard for ``scripts/train.py``: validate schedule after aug merge.

    Short smokes (``epochs <= smoke_epochs_max``) that violate
    ``epochs - patience >= close_mosaic`` emit a stderr warning and continue.
    Longer schedules re-raise so misconfigured production runs fail loudly.
    """
    try:
        validate_epochs_patience_close_mosaic(cfg, repo_root=repo_root, label=label)
    except ValueError as exc:
        effective = effective_train_aug_merged(cfg, repo_root=repo_root)
        epochs = int(effective["epochs"])
        if epochs <= smoke_epochs_max:
            print(f"Warning: {exc}", file=sys.stderr)
            return
        raise SystemExit(str(exc)) from exc


def validate_train_bench_raw_cache(raw: dict[str, Any], *, path_label: str) -> None:
    """Each committed ``train_bench_*.json`` must set ``cache: false`` explicitly."""
    if "cache" not in raw:
        raise ValueError(f"{path_label}: missing explicit top-level 'cache'")
    if raw["cache"] is not False:
        raise ValueError(f"{path_label}: cache must be false (got {raw['cache']!r})")


def load_train_config_json(path: Path, *, repo_root: Path) -> dict[str, Any]:
    """Load a train JSON file and resolve ``extends`` overlays."""
    raw = json.loads(path.read_text("utf-8"))
    if not isinstance(raw, dict):
        raise TypeError(f"Expected top-level JSON object: {path}")
    return resolve_train_config_extends(
        {str(k): v for k, v in raw.items()},
        repo_root=repo_root,
        config_path=path.resolve(),
    )
