from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DatasetSpec:
    root: Path
    yolo_data_yaml: Path | None
    name: str | None
    manifest_path: Path


def dataset_root_from_manifest(
    *,
    manifest_path: str | os.PathLike = "data/manifest.json",
    dataset_name: str,
) -> Path:
    mp = Path(manifest_path)
    obj = json.loads(mp.read_text("utf-8"))
    for ds in obj.get("datasets", []):
        if ds.get("name") == dataset_name:
            extracted = ds.get("extracted_paths") or []
            if not extracted:
                raise RuntimeError(f"{dataset_name}: extracted_paths empty in {mp}")
            # Use first extracted path by convention.
            p = Path(extracted[0]).expanduser()
            if p.is_absolute():
                return p.resolve()
            # Convention: extracted paths in the tracked manifest are repo-root relative.
            # For the default manifest at data/manifest.json, repo root is mp.parent.parent.
            return (mp.parent.parent / p).resolve()
    raise RuntimeError(f"Dataset {dataset_name!r} not found in {mp}")


def resolve_dataset(
    *,
    manifest_path: str | os.PathLike = "data/manifest.json",
    default_dataset_name: str = "sunflower-cvat-1093",
    dataset_name: str | None = None,
    dataset_root: str | os.PathLike | None = None,
    yolo_data_yaml: str | os.PathLike | None = None,
    environ: dict[str, str] | None = None,
) -> DatasetSpec:
    env = os.environ if environ is None else environ

    dataset_root_env = (env.get("DATASET_ROOT") or "").strip()
    yolo_data_yaml_env = (env.get("YOLO_DATA_YAML") or "").strip()
    dataset_name_env = (env.get("DATASET_NAME") or "").strip()

    mp = Path(manifest_path)

    # CLI / caller overrides win over env.
    dataset_name_final = (dataset_name or "").strip() or dataset_name_env or default_dataset_name

    if yolo_data_yaml is not None and str(yolo_data_yaml).strip():
        yolo_data_yaml_path = Path(yolo_data_yaml).expanduser().resolve()
    else:
        yolo_data_yaml_path = Path(yolo_data_yaml_env).expanduser().resolve() if yolo_data_yaml_env else None

    if dataset_root is not None and str(dataset_root).strip():
        root = Path(dataset_root).expanduser().resolve()
    elif dataset_root_env:
        root = Path(dataset_root_env).expanduser().resolve()
    elif yolo_data_yaml_path is not None:
        # If a user points directly at data.yaml, treat its parent as a best-effort "root".
        root = yolo_data_yaml_path.parent
    else:
        root = dataset_root_from_manifest(manifest_path=mp, dataset_name=dataset_name_final)

    name_field: str | None = None
    if (dataset_name or "").strip() or dataset_name_env:
        name_field = dataset_name_final

    return DatasetSpec(root=root, yolo_data_yaml=yolo_data_yaml_path, name=name_field, manifest_path=mp)


def describe_dataset(spec: DatasetSpec) -> str:
    parts = [f"root={spec.root}"]
    if spec.yolo_data_yaml is not None:
        parts.append(f"yolo_data_yaml={spec.yolo_data_yaml}")
    if spec.name is not None:
        parts.append(f"name={spec.name}")
    parts.append(f"manifest={spec.manifest_path}")
    return ", ".join(parts)

