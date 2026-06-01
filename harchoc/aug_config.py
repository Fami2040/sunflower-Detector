from __future__ import annotations

from pathlib import Path
from typing import Any

from harchoc.yaml_minimal import parse_minimal_yaml


def _coerce_aug_scalar(v: object) -> object:
    if isinstance(v, str):
        s = v.strip()
        if s.isdigit():
            return int(s)
        try:
            return float(s)
        except ValueError:
            return v
    return v


def resolve_aug_yaml(aug_path: Path, *, repo_root: Path | None = None) -> dict[str, Any]:
    """Load aug YAML; merge ``extends`` parent ``ultralytics`` before child overrides."""
    path = aug_path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Aug config not found: {path}")
    obj = parse_minimal_yaml(path)
    if not isinstance(obj, dict):
        raise TypeError(f"Aug config must be a mapping: {path}")
    extends = obj.get("extends")
    if extends:
        base_rel = Path(str(extends))
        if repo_root is not None and not base_rel.is_absolute():
            base_path = (repo_root / base_rel).resolve()
        elif base_rel.is_absolute():
            base_path = base_rel.resolve()
        else:
            base_path = (path.parent / base_rel).resolve()
        base_obj = resolve_aug_yaml(base_path, repo_root=repo_root)
        base_ultra = dict(base_obj.get("ultralytics") or {})
        delta_ultra = obj.get("ultralytics")
        if isinstance(delta_ultra, dict):
            for k, v in delta_ultra.items():
                if v is not None and v != "":
                    base_ultra[k] = _coerce_aug_scalar(v)
        obj = {
            k: v
            for k, v in obj.items()
            if k not in ("extends", "ultralytics")
        }
        obj = {**base_obj, **obj}
        obj["ultralytics"] = base_ultra
    return obj


def merge_aug_yaml(
    train_cfg: dict[str, Any],
    aug_path: Path,
    *,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    """Merge ``ultralytics:`` keys from configs/aug/*.yaml into train kwargs."""
    path = aug_path.expanduser().resolve()
    obj = resolve_aug_yaml(path, repo_root=repo_root)
    ultra = obj.get("ultralytics")
    if not isinstance(ultra, dict):
        raise ValueError(f"Aug config must define top-level ultralytics: mapping: {path}")
    merged = dict(train_cfg)
    for k, v in ultra.items():
        if v is not None and v != "":
            merged[k] = _coerce_aug_scalar(v)
    merged["aug_config"] = str(path.resolve())
    return merged
