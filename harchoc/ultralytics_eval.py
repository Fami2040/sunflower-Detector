from __future__ import annotations

from pathlib import Path
from typing import Any


def run_val(
    weights: str | Path,
    data_yaml: str | Path,
    *,
    imgsz: int | None = None,
    max_det: int | None = None,
    split: str | None = None,
    device: str | None = None,
) -> Any:
    """
    Run Ultralytics YOLO validation on ``data_yaml`` with ``weights``.

    Returns the Ultralytics metrics object from ``model.val()``.
    """
    from ultralytics import YOLO  # type: ignore

    kwargs: dict[str, Any] = {"data": str(data_yaml), "verbose": False}
    if imgsz is not None:
        kwargs["imgsz"] = int(imgsz)
    if max_det is not None:
        kwargs["max_det"] = int(max_det)
    if split is not None:
        kwargs["split"] = str(split)
    if device is not None:
        kwargs["device"] = str(device)

    model = YOLO(str(weights))
    return model.val(**kwargs)
