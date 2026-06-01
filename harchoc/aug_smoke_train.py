"""Runtime aug-smoke train config resolution (index + base template)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

DEFAULT_AUG_SMOKE_TRAIN_BASE = "configs/experiments/train_smoke_rank_15ep.json"

# Committed train JSON exceptions (non-default extends / model / schedule).
AUG_SMOKE_COMMITTED_TRAIN_STEMS = frozenset(
    {
        "train_aug_s9_no_aug_yaml_smoke",
        "train_aug_s10_yolo11s_smoke",
        "train_aug_s11_musgd_smoke",
        "train_aug_s12_amp_off_smoke",
        "train_aug_s13_patience5_smoke",
        "train_aug_close10_sweep_smoke_15ep",
        "train_aug_close25_sweep_smoke_15ep",
        "train_aug_close10_100ep",
        "train_aug_close25_100ep",
        "train_aug_mosaic_sweep_smoke_15ep",
        "train_aug_mosaic_sweep_template",
        "train_aug_schedule_patience25_100ep",
    }
)


def _entry_train_config_spec(entry: dict[str, Any]) -> str | None:
    tc = entry.get("train_config")
    if tc is None:
        return None
    spec = str(tc).strip()
    return spec or None


def is_committed_aug_smoke_train_config(path: str | Path) -> bool:
    stem = Path(str(path)).name
    if stem.endswith(".json"):
        stem = stem[: -len(".json")]
    return stem in AUG_SMOKE_COMMITTED_TRAIN_STEMS


def resolve_aug_smoke_train_config_path(
    entry: dict[str, Any],
    *,
    repo_root: Path,
    job_train_config: str | None = None,
) -> str:
    """Return ``--config`` path for an aug smoke job or index entry."""
    for spec in (_entry_train_config_spec(entry), job_train_config):
        if not spec:
            continue
        p = repo_root / spec
        if p.is_file() and is_committed_aug_smoke_train_config(spec):
            return spec
        if p.is_file():
            return spec
    return DEFAULT_AUG_SMOKE_TRAIN_BASE


def resolve_aug_smoke_aug_config(
    entry: dict[str, Any],
    *,
    repo_root: Path,
    job_aug_config: str | None = None,
    train_config_path: str | None = None,
) -> str | None:
    """Aug YAML path for train/eval (index is source of truth for runtime smokes)."""
    if job_aug_config is not None:
        return str(job_aug_config) if job_aug_config else None
    if "aug_config" in entry:
        ac = entry.get("aug_config")
        return str(ac) if ac else None
    tc = train_config_path or resolve_aug_smoke_train_config_path(entry, repo_root=repo_root)
    if tc != DEFAULT_AUG_SMOKE_TRAIN_BASE:
        from harchoc.train_config import load_train_config_json

        merged = load_train_config_json(repo_root / tc, repo_root=repo_root)
        ac = merged.get("aug_config")
        return str(ac) if ac else None
    return None


def resolve_aug_smoke_train_raw(
    entry: dict[str, Any],
    *,
    repo_root: Path,
    job_train_config: str | None = None,
    job_aug_config: str | None = None,
) -> dict[str, Any]:
    """Fully merged train config dict (for recipe fingerprint / parity tests)."""
    from harchoc.train_config import load_train_config_json

    tc = resolve_aug_smoke_train_config_path(
        entry, repo_root=repo_root, job_train_config=job_train_config
    )
    cfg = load_train_config_json(repo_root / tc, repo_root=repo_root)
    aug = resolve_aug_smoke_aug_config(
        entry,
        repo_root=repo_root,
        job_aug_config=job_aug_config,
        train_config_path=tc,
    )
    if aug is not None or "aug_config" in entry:
        cfg = dict(cfg)
        cfg["aug_config"] = aug
    return cfg
