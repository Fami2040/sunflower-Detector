"""Shared SAHI sliced inference for deploy paths (bot + run_infer_once)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class SahiSliceConfig:
    slice_size: int = 500
    overlap: float = 0.35
    nms_iou: float = 0.50

    @classmethod
    def from_env(cls) -> SahiSliceConfig:
        return cls(
            slice_size=int(os.getenv("SLICE_SIZE", "500")),
            overlap=float(os.getenv("OVERLAP", "0.35")),
            nms_iou=float(os.getenv("NMS_IOU", "0.50")),
        )


def model_confidence_min_from_env() -> float:
    """Min per-class conf passed to SAHI model (weak class-1 boxes not dropped early)."""
    legacy = os.getenv("CONF_THR", "").strip()
    conf_fert = float(
        os.getenv("CONF_THR_FERTILIZED", legacy if legacy else "0.06")
    )
    conf_unfert = float(
        os.getenv("CONF_THR_UNFERTILIZED", legacy if legacy else "0.04")
    )
    return min(conf_fert, conf_unfert)


def load_ultralytics_detection_model(
    model_path: str,
    *,
    device: str,
    confidence_threshold: float | None = None,
) -> Any:
    from sahi import AutoDetectionModel

    thr = (
        confidence_threshold
        if confidence_threshold is not None
        else model_confidence_min_from_env()
    )
    return AutoDetectionModel.from_pretrained(
        model_type="ultralytics",
        model_path=model_path,
        confidence_threshold=thr,
        device=device,
    )


def run_sliced_prediction(
    image: str,
    detection_model: Any,
    config: SahiSliceConfig | None = None,
) -> Any:
    from sahi.predict import get_sliced_prediction

    cfg = config or SahiSliceConfig.from_env()
    return get_sliced_prediction(
        image=image,
        detection_model=detection_model,
        slice_height=cfg.slice_size,
        slice_width=cfg.slice_size,
        overlap_height_ratio=cfg.overlap,
        overlap_width_ratio=cfg.overlap,
        postprocess_type="NMS",
        postprocess_match_threshold=cfg.nms_iou,
    )
