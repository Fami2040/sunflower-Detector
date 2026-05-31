from __future__ import annotations

from pathlib import Path
from typing import Any

from harchoc.datasets import DatasetSpec, describe_dataset, resolve_dataset
from harchoc.schemas import with_schema_version


def _path_str(value: str | Path) -> str:
    return str(Path(value))


def resolve_dataset_args(args: Any) -> DatasetSpec:
    return resolve_dataset(
        manifest_path=args.manifest,
        default_dataset_name=args.default_dataset_name,
        dataset_name=args.dataset_name,
        dataset_root=args.dataset_root,
        yolo_data_yaml=args.yolo_data_yaml,
    )


def build_versioned_dry_run_payload(
    *,
    script: str,
    schema_version: str,
    out: str | Path,
    **fields: Any,
) -> dict[str, Any]:
    """Build schema-versioned dry-run JSON for dataset scaffold scripts."""
    payload: dict[str, Any] = {
        "status": "dry-run",
        "script": script,
        "out": _path_str(out),
    }
    for key, value in fields.items():
        payload[key] = _path_str(value) if isinstance(value, (str, Path)) else value
    return with_schema_version(payload, schema_version=schema_version)


def build_versioned_scaffold_payload(
    *,
    schema_version: str,
    dataset_spec: DatasetSpec,
    notes: str,
    **fields: Any,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "status": "scaffold",
        "dataset": {"description": describe_dataset(dataset_spec)},
        "notes": notes,
    }
    for key, value in fields.items():
        payload[key] = _path_str(value) if isinstance(value, (str, Path)) else value
    return with_schema_version(payload, schema_version=schema_version)


def build_plan_dry_run_payload(plan: dict[str, Any], **fields: Any) -> dict[str, Any]:
    """Extend a pre-built plan dict with dry-run status and path fields."""
    payload = {**plan, "status": "dry-run"}
    for key, value in fields.items():
        payload[key] = _path_str(value) if isinstance(value, (str, Path)) else value
    return payload


def build_plan_scaffold_payload(plan: dict[str, Any], **fields: Any) -> dict[str, Any]:
    payload = {**plan, "status": "scaffold"}
    for key, value in fields.items():
        payload[key] = _path_str(value) if isinstance(value, (str, Path)) else value
    return payload
