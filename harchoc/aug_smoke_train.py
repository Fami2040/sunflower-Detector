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


_DEFAULT_INDEX = "configs/experiments/aug_smoke_index.json"
_RUNTIME_ONLY_SMOKE_IDS = frozenset({f"S{i}" for i in range(9)} | {"S14"})
_COMMITTED_SMOKE_IDS = frozenset({f"S{i}" for i in range(9, 14)})
_EQUIVALENCE_SWEEP_ALIASES = frozenset({"CLOSE25", "CLOSE10"})


def _aug_path_exists(repo_root: Path, spec: str | None) -> bool:
    if spec is None:
        return True
    s = str(spec).strip()
    if not s:
        return True
    return (repo_root / s).is_file()


def _validate_train_config_ref(
    repo_root: Path,
    spec: str,
    *,
    label: str,
    errors: list[str],
) -> None:
    path = repo_root / spec
    if not path.is_file():
        errors.append(f"{label}: missing train_config {spec!r}")
        return
    stem = path.stem
    if stem not in AUG_SMOKE_COMMITTED_TRAIN_STEMS:
        errors.append(f"{label}: train_config stem {stem!r} not in committed set")
        return
    try:
        from harchoc.train_config import load_train_config_json

        load_train_config_json(path, repo_root=repo_root)
    except (FileNotFoundError, ValueError, TypeError) as exc:
        errors.append(f"{label}: train_config {spec!r} failed to load: {exc}")


def validate_aug_smoke_configs(
    repo_root: Path | None = None,
    *,
    index_path: str | Path | None = None,
) -> list[str]:
    """Return config consistency errors (empty list = OK). No ML deps."""
    from harchoc.aug_smoke_runner import find_smoke_entry, load_aug_smoke_index
    from harchoc.train_config import validate_epochs_patience_close_mosaic

    root = (repo_root or Path(__file__).resolve().parents[1]).resolve()
    idx_rel = str(index_path or _DEFAULT_INDEX)
    idx_path = root / idx_rel
    errors: list[str] = []

    if not idx_path.is_file():
        return [f"missing aug smoke index: {idx_rel}"]

    index = load_aug_smoke_index(idx_path)
    smoke_ids = {str(e.get("id") or "").upper() for e in index.get("smokes") or []}

    for entry in index.get("smokes") or []:
        if not isinstance(entry, dict):
            errors.append("smokes[] entry is not an object")
            continue
        sid = str(entry.get("id") or "").upper()
        label = f"smoke {sid}"
        has_tc = "train_config" in entry and bool(str(entry.get("train_config") or "").strip())

        if sid in _RUNTIME_ONLY_SMOKE_IDS:
            if has_tc:
                errors.append(f"{label}: runtime-only smoke must not set train_config")
        elif sid in _COMMITTED_SMOKE_IDS:
            tc = entry.get("train_config")
            if not tc:
                errors.append(f"{label}: committed smoke requires train_config")
            else:
                _validate_train_config_ref(root, str(tc), label=label, errors=errors)
        elif has_tc:
            _validate_train_config_ref(root, str(entry["train_config"]), label=label, errors=errors)

        try:
            tc_path = resolve_aug_smoke_train_config_path(entry, repo_root=root)
            if not (root / tc_path).is_file():
                errors.append(f"{label}: resolved train config missing: {tc_path!r}")
        except (ValueError, TypeError) as exc:
            errors.append(f"{label}: resolve train config failed: {exc}")

        if "aug_config" in entry:
            ac = entry.get("aug_config")
            if ac is not None and not _aug_path_exists(root, str(ac)):
                errors.append(f"{label}: missing aug_config {ac!r}")
            try:
                resolved_aug = resolve_aug_smoke_aug_config(entry, repo_root=root)
            except (FileNotFoundError, ValueError, TypeError) as exc:
                errors.append(f"{label}: resolve aug_config failed: {exc}")
                resolved_aug = None
            index_aug = str(ac) if ac else None
            resolved_aug_s = str(resolved_aug) if resolved_aug else None
            if index_aug != resolved_aug_s:
                errors.append(
                    f"{label}: index aug_config {index_aug!r} != resolved {resolved_aug_s!r}"
                )

        try:
            raw = resolve_aug_smoke_train_raw(entry, repo_root=root)
            if sid != "S9" and raw.get("aug_config") is not None:
                validate_epochs_patience_close_mosaic(raw, repo_root=root, label=label)
        except (ValueError, TypeError) as exc:
            errors.append(f"{label}: schedule guard failed: {exc}")

    def _check_arm(arm: dict[str, Any], *, prefix: str) -> None:
        tc = arm.get("train_config")
        if tc:
            _validate_train_config_ref(root, str(tc), label=prefix, errors=errors)
        ac = arm.get("aug_config")
        if ac is not None and not _aug_path_exists(root, str(ac)):
            errors.append(f"{prefix}: missing aug_config {ac!r}")

    sweeps_15 = index.get("sweeps_15ep")
    if isinstance(sweeps_15, dict):
        for arm in sweeps_15.get("arms") or []:
            if isinstance(arm, dict):
                _check_arm(arm, prefix=f"sweep15 {arm.get('id')}")

    sweeps_100 = index.get("sweeps_100ep")
    if isinstance(sweeps_100, dict):
        for arm in sweeps_100.get("arms") or []:
            if isinstance(arm, dict):
                _check_arm(arm, prefix=f"sweep100 {arm.get('id')}")

    sched = index.get("schedule_smoke_100ep")
    if isinstance(sched, dict) and sched.get("train_config"):
        _validate_train_config_ref(
            root, str(sched["train_config"]), label="schedule_smoke_100ep", errors=errors
        )

    exp_dir = root / "configs" / "experiments"
    for path in sorted(exp_dir.glob("train_aug_s*_smoke.json")):
        if path.stem not in AUG_SMOKE_COMMITTED_TRAIN_STEMS:
            errors.append(f"orphan aug smoke train JSON: {path.relative_to(root)}")

    equiv = index.get("equivalence_classes")
    if isinstance(equiv, dict):
        for cls in equiv.get("classes") or []:
            if not isinstance(cls, dict):
                continue
            canon = str(cls.get("canonical_smoke_id") or "").upper()
            if canon and canon not in smoke_ids and canon not in _EQUIVALENCE_SWEEP_ALIASES:
                errors.append(f"equivalence canonical_smoke_id {canon!r} unknown")
            for raw_id in cls.get("smoke_ids") or []:
                eid = str(raw_id).upper()
                if eid in smoke_ids or eid in _EQUIVALENCE_SWEEP_ALIASES:
                    continue
                errors.append(f"equivalence smoke_ids entry {eid!r} unknown")

    for stem in sorted(AUG_SMOKE_COMMITTED_TRAIN_STEMS):
        if stem.startswith("train_aug_s") and stem.endswith("_smoke"):
            path = exp_dir / f"{stem}.json"
            if path.is_file():
                try:
                    from harchoc.train_config import load_train_config_json

                    load_train_config_json(path, repo_root=root)
                except (FileNotFoundError, ValueError, TypeError) as exc:
                    errors.append(f"committed {stem}: load failed: {exc}")

    return errors
