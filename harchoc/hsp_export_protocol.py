"""Manuscript counting-export defaults (conf / IoU / split / device); HSP = internal reports/hsp/ label."""

from __future__ import annotations

from pathlib import Path
from typing import Any

EXPORT_CONF: float = 0.001
EXPORT_IOU: float = 0.3
DEFAULT_EXPORT_MAX_DET: int = 3000
DEFAULT_SPLIT_FILE: str = "data/splits/test.txt"
DEFAULT_VAL_SPLIT_FILE: str = "data/splits/val.txt"
EXPORT_DEVICE: str = "cpu"


def split_file_for_repo(repo_root: Path, *, split: str = "test") -> Path:
    rel = DEFAULT_VAL_SPLIT_FILE if split == "val" else DEFAULT_SPLIT_FILE
    return (repo_root / rel).resolve()


def eval_export_cli_flags(
    *,
    conf: float | None = None,
    iou: float | None = None,
    max_det: int | None = None,
    device: str | None = None,
    split_file: str | None = None,
) -> list[str]:
    """``scripts/eval.py`` export flag argv fragments."""
    out = [
        "--export-conf",
        str(EXPORT_CONF if conf is None else float(conf)),
        "--export-iou",
        str(EXPORT_IOU if iou is None else float(iou)),
    ]
    if max_det is not None:
        out.extend(["--export-max-det", str(int(max_det))])
    dev = EXPORT_DEVICE if device is None else str(device)
    out.extend(["--export-device", dev])
    if split_file is not None:
        out.extend(["--split-file", str(split_file)])
    return out


def export_result_meta(
    *,
    conf: float | None = None,
    iou: float | None = None,
    max_det: int | None = None,
    device: str | None = None,
    **extra: Any,
) -> dict[str, Any]:
    """Metadata keys written beside GT/pred JSON exports."""
    meta: dict[str, Any] = {
        "export_conf": EXPORT_CONF if conf is None else float(conf),
        "export_iou": EXPORT_IOU if iou is None else float(iou),
    }
    if max_det is not None:
        meta["export_max_det"] = int(max_det)
    if device is not None:
        meta["export_device"] = device
    meta.update(extra)
    return meta
