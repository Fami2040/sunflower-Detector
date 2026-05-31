"""Strict ML helpers: opt-in loud failures, structured warnings, and failure capture."""

from __future__ import annotations

import os
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Generator

__all__ = [
    "FailureCapture",
    "StrictWarnings",
    "append_capture_warning",
    "append_ml_error",
    "capture_failure",
    "fail_or_warn",
    "is_ml_strict",
    "ml_warnings_sink",
    "record_ml_failure",
    "require_cuda",
    "require_torch",
    "strict_ml_enabled",
]


def _env_truthy(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in ("1", "true", "yes")


def strict_ml_enabled() -> bool:
    """True when ``HARCHOC_STRICT_ML`` (canonical) or legacy ``HARCHOC_ML_STRICT`` is set."""
    return _env_truthy("HARCHOC_STRICT_ML") or _env_truthy("HARCHOC_ML_STRICT")


def is_ml_strict() -> bool:
    """Alias for :func:`strict_ml_enabled` (legacy callers)."""
    return strict_ml_enabled()


@dataclass
class FailureCapture:
    context: str
    exc_type: str | None = None
    exc_msg: str | None = None

    @property
    def failed(self) -> bool:
        return self.exc_type is not None


@contextmanager
def capture_failure(context: str) -> Generator[FailureCapture, None, None]:
    """Record the first exception in the block without re-raising."""
    cap = FailureCapture(context=context)
    try:
        yield cap
    except Exception as ex:
        cap.exc_type = type(ex).__name__
        cap.exc_msg = str(ex)


def append_capture_warning(warnings: list[str] | None, cap: FailureCapture) -> None:
    if warnings is None or not cap.failed:
        return
    warnings.append(f"{cap.context}: {cap.exc_type}: {cap.exc_msg}")


def fail_or_warn(message: str, *, strict: bool | None = None) -> None:
    """Raise ``RuntimeError`` when strict; otherwise print to stderr."""
    if strict is None:
        strict = strict_ml_enabled()
    if strict:
        raise RuntimeError(message)
    print(message, file=sys.stderr)


def require_torch() -> object:
    """Import torch or raise with a mamba re-exec hint."""
    from harchoc.gpu_probe import try_import_torch

    torch_mod, _, err = try_import_torch()
    if torch_mod is None:
        detail = err or "unknown import error"
        raise RuntimeError(
            f"PyTorch required but not available ({detail}). "
            "Run under the project env, e.g. mamba run -n harchoc python ..."
        )
    return torch_mod


def require_cuda(*, strict: bool = False) -> dict[str, Any]:
    """Return CUDA probe payload from :mod:`harchoc.gpu_probe`."""
    torch_mod = require_torch()
    from harchoc.gpu_probe import torch_cuda_payload

    payload = torch_cuda_payload(torch_mod)
    if strict and not payload.get("cuda_available"):
        err = payload.get("device_error")
        msg = "CUDA is not available for the current PyTorch build/runtime."
        if err:
            msg = f"{msg} ({err})"
        raise RuntimeError(msg)
    return payload


def ml_warnings_sink() -> list[str] | None:
    """Return a mutable warning list when strict ML is enabled, else ``None``."""
    if not strict_ml_enabled():
        return None
    return []


class StrictWarnings:
    """Accumulate structured warnings; optional raise when strict ML is on."""

    def __init__(self) -> None:
        self.items: list[dict[str, Any]] = []

    def warn(
        self,
        code: str,
        message: str,
        *,
        raise_if_strict: bool = False,
        **extra: Any,
    ) -> None:
        row: dict[str, Any] = {"code": code, "message": message}
        row.update(extra)
        self.items.append(row)
        if raise_if_strict and strict_ml_enabled():
            raise RuntimeError(f"{code}: {message}")

    def as_list(self) -> list[dict[str, Any]]:
        return list(self.items)


def append_ml_error(
    errors: list[dict[str, Any]],
    *,
    panel_index: int,
    exc: BaseException,
    max_entries: int = 5,
) -> None:
    """Append a structured error row; no-op when *errors* is already at *max_entries*."""
    if len(errors) >= max_entries:
        return
    errors.append(
        {
            "panel_index": int(panel_index),
            "error_type": type(exc).__name__,
            "message": str(exc),
        }
    )


def record_ml_failure(
    errors: list[dict[str, Any]],
    *,
    panel_index: int,
    exc: BaseException,
    max_entries: int = 5,
) -> None:
    """Record *exc*; re-raise when strict ML mode is enabled."""
    append_ml_error(errors, panel_index=panel_index, exc=exc, max_entries=max_entries)
    if strict_ml_enabled():
        raise exc
