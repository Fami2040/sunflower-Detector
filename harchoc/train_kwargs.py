"""Ultralytics ``model.train()`` kwargs whitelist and forwarding helpers."""

from __future__ import annotations

from typing import Any

# Keys forwarded from merged train config into ``YOLO(...).train(**kwargs)``.
# ``data`` and ``name`` are always set by ``ultralytics_train_kwargs``; ``project`` / ``exist_ok``
# may be added in ``scripts/train.py``.
ALLOWED_TRAIN_KWARGS: frozenset[str] = frozenset(
    {
        "epochs",
        "imgsz",
        "batch",
        "device",
        "optimizer",
        "conf",
        "iou",
        "max_det",
        "mosaic",
        "close_mosaic",
        "mixup",
        "cutmix",
        "hsv_h",
        "hsv_s",
        "hsv_v",
        "translate",
        "scale",
        "degrees",
        "shear",
        "perspective",
        "fliplr",
        "flipud",
        "erasing",
        "lr0",
        "lrf",
        "momentum",
        "weight_decay",
        "patience",
        "workers",
        "verbose",
        "seed",
        "project",
        "exist_ok",
        "amp",
        "nbs",
        "cache",
        "freeze",
    }
)

# Documented in train JSON / ``rtdetr_limits``; not a valid ``model.train()`` arg (Ultralytics).
TRAIN_POLICY_ONLY_KEYS: frozenset[str] = frozenset({"num_queries", "grad_clip"})

# Transfer-policy keys merged from ``configs/transfer/*.yaml``; not forwarded to Ultralytics.
FREEZE_POLICY_ONLY_KEYS: frozenset[str] = frozenset({"freeze_backbone", "unfreeze_epoch"})

# Gandhi/TFA heuristic when ``freeze_backbone: true`` without explicit ``freeze`` (YOLOv8 layer index).
DEFAULT_FREEZE_WHEN_BACKBONE = 10

_TRAIN_META_RESERVED: frozenset[str] = frozenset({"data", "name", "project", "exist_ok"})


def resolve_freeze_policy(train_cfg: dict[str, Any]) -> tuple[Any | None, dict[str, Any], list[str]]:
    """
    Resolve Ultralytics ``freeze`` and policy metadata from merged train config.

    ``freeze_backbone: true`` without explicit ``freeze`` defaults to
    ``DEFAULT_FREEZE_WHEN_BACKBONE``. ``unfreeze_epoch`` is recorded only;
    Ultralytics ``model.train()`` does not support mid-run staged unfreeze.
    """
    warnings: list[str] = []
    freeze_backbone = train_cfg.get("freeze_backbone")
    unfreeze_epoch = train_cfg.get("unfreeze_epoch")
    explicit_freeze = train_cfg.get("freeze")

    freeze_value: Any | None = None
    if explicit_freeze is not None:
        freeze_value = explicit_freeze
    elif freeze_backbone:
        freeze_value = DEFAULT_FREEZE_WHEN_BACKBONE

    policy: dict[str, Any] = {
        "freeze_backbone": freeze_backbone,
        "unfreeze_epoch": unfreeze_epoch,
        "freeze": freeze_value,
        "ultralytics_freeze_honored": freeze_value is not None,
        "unfreeze_epoch_honored": False,
    }
    if freeze_backbone and explicit_freeze is None and freeze_value is not None:
        policy["default_freeze"] = DEFAULT_FREEZE_WHEN_BACKBONE
    if unfreeze_epoch is not None:
        warnings.append(
            f"unfreeze_epoch={unfreeze_epoch} recorded in freeze_policy only; "
            "Ultralytics model.train() does not support staged unfreeze in a single pass "
            "(use two-stage train or callback — not implemented)."
        )
        policy["note"] = (
            "unfreeze_epoch requires a second train stage or callback; not applied automatically."
        )
    return freeze_value, policy, warnings


def effective_train_cfg_with_freeze(train_cfg: dict[str, Any]) -> dict[str, Any]:
    """Copy train config with resolved ``freeze`` when policy implies it."""
    freeze_value, _, _ = resolve_freeze_policy(train_cfg)
    if freeze_value is None or train_cfg.get("freeze") is not None:
        return train_cfg
    out = dict(train_cfg)
    out["freeze"] = freeze_value
    return out


def ultralytics_train_kwargs(train_cfg: dict[str, Any], *, data_yaml: str, run_name: str) -> dict[str, Any]:
    kwargs: dict[str, Any] = {"data": data_yaml, "name": run_name}
    for k in ALLOWED_TRAIN_KWARGS:
        if k in train_cfg and train_cfg[k] is not None:
            kwargs[k] = train_cfg[k]
    return kwargs


def forwarded_keys_from_train_cfg(train_cfg: dict[str, Any]) -> list[str]:
    """Keys from ``train_cfg`` that would be forwarded (dry-run meta without data yaml)."""
    effective = effective_train_cfg_with_freeze(train_cfg)
    return sorted(
        k for k in ALLOWED_TRAIN_KWARGS if k in effective and effective[k] is not None
    )


def forwarded_train_keys(train_kwargs: dict[str, Any]) -> list[str]:
    """Sorted Ultralytics train keys recorded in meta (excludes runtime-only entries)."""
    return sorted(k for k in train_kwargs if k not in _TRAIN_META_RESERVED)


def load_ultralytics_train_model(model: str):
    """Instantiate Ultralytics YOLO or RT-DETR trainer for a weights path or architecture YAML."""
    try:
        from ultralytics import YOLO  # type: ignore
    except Exception as ex:
        raise SystemExit(
            f"Ultralytics is required for training, but could not be imported: {ex}"
        ) from ex
    from harchoc.rtdetr_limits import is_rtdetr_model

    ref = str(model)
    if is_rtdetr_model(ref):
        from ultralytics import RTDETR  # type: ignore

        return RTDETR(ref)
    return YOLO(ref)
