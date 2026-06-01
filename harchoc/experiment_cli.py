"""Shared CLI + experiments.v1 config merge for legacy script entrypoints."""

from __future__ import annotations

from argparse import Namespace
from typing import Any


def merge_config_objects(config_paths: list[str]) -> dict[str, Any]:
    """Merge left-to-right ``--config`` JSON paths or inline objects."""
    from harchoc.experiment_config import load_config_json, merge_experiment_config

    config_obj: dict[str, Any] = {}
    for raw in config_paths:
        cfg = load_config_json(raw)
        config_obj = merge_experiment_config(config=config_obj, cli=cfg)
    return config_obj


def pick_cli_or_section(
    args: Namespace,
    name: str,
    *,
    section_cfg: dict[str, Any],
    default: object,
) -> object:
    """CLI wins when it differs from the argparse default; else config section; else default."""
    cli_v = getattr(args, name, default)
    if cli_v != default:
        return cli_v
    if name in section_cfg:
        return section_cfg[name]
    return default


def pick_cli_or_dataset(
    args: Namespace,
    name: str,
    *,
    dataset_cfg: dict[str, Any],
    default: object,
) -> object:
    cli_v = getattr(args, name, default)
    if cli_v != default:
        return cli_v
    if name in dataset_cfg:
        return dataset_cfg[name]
    return default


def section_and_dataset_from_config(
    config_obj: dict[str, Any],
    section: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    from harchoc.experiment_config import script_section_from_config

    dataset_cfg = config_obj.get("dataset")
    dataset_cfg_obj: dict[str, Any] = dataset_cfg if isinstance(dataset_cfg, dict) else {}
    section_cfg = script_section_from_config(config_obj, section)
    return section_cfg, dataset_cfg_obj


def apply_dataset_args(
    args: Namespace,
    dataset_cfg_obj: dict[str, Any],
    *,
    manifest_default: str = "data/manifest.json",
    name_default: str = "sunflower-cvat-1093",
) -> None:
    from harchoc.config_coerce import optional_str

    args.manifest = str(
        pick_cli_or_dataset(
            args, "manifest", dataset_cfg=dataset_cfg_obj, default=manifest_default
        )
    )
    args.default_dataset_name = str(
        pick_cli_or_dataset(
            args,
            "default_dataset_name",
            dataset_cfg=dataset_cfg_obj,
            default=name_default,
        )
    )
    args.dataset_name = optional_str(
        pick_cli_or_dataset(args, "dataset_name", dataset_cfg=dataset_cfg_obj, default=None)
    )
    args.dataset_root = optional_str(
        pick_cli_or_dataset(args, "dataset_root", dataset_cfg=dataset_cfg_obj, default=None)
    )
    args.yolo_data_yaml = optional_str(
        pick_cli_or_dataset(args, "yolo_data_yaml", dataset_cfg=dataset_cfg_obj, default=None)
    )
