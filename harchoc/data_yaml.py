from __future__ import annotations

import os
import tempfile
from collections.abc import Sequence
from pathlib import Path

from harchoc.sunflower_dataset import CLASS_NAMES
from harchoc.yaml_minimal import parse_names_and_nc


def resolve_data_yaml_path(
    *,
    dataset_root: Path,
    explicit_yaml: Path | None = None,
    use_env: bool = True,
) -> Path | None:
    """
    Resolve an on-disk Ultralytics data.yaml.

    Precedence: explicit path (must exist) → YOLO_DATA_YAML (when use_env) →
    ``dataset_root / data.yaml``.
    """
    if explicit_yaml is not None and explicit_yaml.is_file():
        return explicit_yaml.resolve()

    if use_env:
        env_yaml = (os.getenv("YOLO_DATA_YAML") or "").strip()
        if env_yaml:
            p = Path(env_yaml).expanduser().resolve()
            if not p.is_file():
                raise FileNotFoundError(f"YOLO_DATA_YAML not found: {p}")
            return p

    p = (dataset_root / "data.yaml").resolve()
    if p.is_file():
        return p
    return None


def require_data_yaml_path(
    *,
    dataset_root: Path,
    explicit_yaml: Path | None = None,
) -> Path:
    """
    Resolve data.yaml for eval/benchmark paths.

    Unlike ``resolve_data_yaml_path``, a caller-provided ``explicit_yaml`` is returned
    even when the file is missing (dataset resolution may have set the path).
    """
    found = resolve_data_yaml_path(
        dataset_root=dataset_root, explicit_yaml=explicit_yaml, use_env=False
    )
    if found is not None:
        return found
    if explicit_yaml is not None:
        return explicit_yaml.resolve()
    raise FileNotFoundError(
        "Could not find ultralytics data.yaml.\n"
        "Fix by exporting YOLO_DATA_YAML=/path/to/data.yaml, or place data.yaml at DATASET_ROOT."
    )


def _infer_nc_from_labels(dataset_root: Path, split: str) -> int | None:
    labels_dir = dataset_root / "labels" / split
    if not labels_dir.is_dir():
        return None
    max_cls = -1
    for lp in labels_dir.glob("*.txt"):
        txt = lp.read_text("utf-8", errors="ignore").strip()
        if not txt:
            continue
        for line in txt.splitlines():
            head = (line.strip().split(maxsplit=1) or [""])[0]
            try:
                c = int(head)
            except ValueError:
                continue
            max_cls = max(max_cls, c)
    return (max_cls + 1) if max_cls >= 0 else None


def ensure_data_yaml(*, dataset_root: Path, yolo_data_yaml: Path | None = None) -> str:
    """
    Return a path to an Ultralytics-compatible data.yaml.

    Uses ``resolve_data_yaml_path`` when possible; otherwise writes a minimal yaml
    for standard ``images/{train,val}`` layout under ``dataset_root``.
    """
    found = resolve_data_yaml_path(
        dataset_root=dataset_root, explicit_yaml=yolo_data_yaml, use_env=True
    )
    if found is not None:
        return str(found)

    train_dir = dataset_root / "images" / "train"
    val_dir = dataset_root / "images" / "val"
    if not (train_dir.is_dir() and val_dir.is_dir()):
        raise FileNotFoundError(
            "Could not find data.yaml and dataset does not look like a YOLO dataset. "
            f"Expected {train_dir} and {val_dir}."
        )

    nc = _infer_nc_from_labels(dataset_root, "train") or _infer_nc_from_labels(dataset_root, "val") or 2
    names_block = "\n".join(f"  {i}: {name}" for i, name in enumerate(CLASS_NAMES))
    yaml_text = "\n".join(
        [
            f"path: {dataset_root.resolve().as_posix()}",
            "train: images/train",
            "val: images/val",
            f"nc: {int(nc)}",
            "names:",
            names_block,
            "",
        ]
    )
    tmp = tempfile.NamedTemporaryFile("w", suffix=".yaml", prefix="ultra_train_data_", delete=False)
    tmp.write(yaml_text)
    tmp.close()
    return tmp.name


def read_class_names(
    *,
    dataset_root: Path,
    data_yaml: Path | None = None,
    default: Sequence[str] | None = None,
) -> list[str]:
    """Load class names from data.yaml, or return *default* (two generic classes)."""
    if default is None:
        default = CLASS_NAMES

    path = data_yaml
    if path is None:
        path = resolve_data_yaml_path(dataset_root=dataset_root, use_env=False)
    if path is None or not path.is_file():
        return list(default)

    names, nc = parse_names_and_nc(path)
    if names:
        max_idx = max(names)
        return [names.get(i, f"class_{i}") for i in range(max_idx + 1)]

    if nc is not None and nc > 0:
        return [f"class_{i}" for i in range(nc)]

    return list(default)


def labels_path_for_image(*, dataset_root: Path, image_path: Path) -> Path:
    """Map ``images/<split>/foo.jpg`` → ``labels/<split>/foo.txt`` under *dataset_root*."""
    rel = image_path.resolve().relative_to(dataset_root.resolve())
    parts = list(rel.parts)
    if "images" in parts:
        idx = parts.index("images")
        parts[idx] = "labels"
        return dataset_root / Path(*parts).with_suffix(".txt")
    return dataset_root / "labels" / rel.with_suffix(".txt").name
