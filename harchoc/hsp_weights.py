"""Canonical HSP detection checkpoint for eval, deploy parity, and dev/tests."""

from __future__ import annotations

import os
from pathlib import Path

# YOLOv8m manuscript baseline (HSP_* = internal reports/hsp/ naming), not deploy SAHI.
HSP_DETECTION_WEIGHTS = "models/best2.pt"


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def resolve_detection_weights(raw: str | None = None) -> Path:
    """
    Resolve detection weights path.

    Priority: explicit ``raw`` → ``DETECTION_MODEL`` env → ``HSP_DETECTION_WEIGHTS``.
    """
    if raw and str(raw).strip():
        return Path(str(raw).strip()).expanduser()
    env = (os.getenv("DETECTION_MODEL") or "").strip()
    if env:
        return Path(env).expanduser()
    return Path(HSP_DETECTION_WEIGHTS)


def detection_weights_str(raw: str | None = None) -> str:
    return str(resolve_detection_weights(raw))
