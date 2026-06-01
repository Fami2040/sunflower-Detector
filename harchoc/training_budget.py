"""Training resource caps via HARCHOC_MAX_* environment variables."""

from __future__ import annotations

import os


def _budget_limit_int(env_key: str, *, default: int) -> int:
    raw = os.getenv(env_key)
    if raw is None or not str(raw).strip():
        return default
    try:
        v = int(str(raw).strip())
    except (ValueError, TypeError) as exc:
        raise SystemExit(f"{env_key} must be an integer (got {raw!r})") from exc
    if v <= 0:
        raise SystemExit(f"{env_key} must be > 0 (got {raw!r})")
    return v


def enforce_budget(*, epochs: int, imgsz: int, batch: int) -> None:
    max_epochs = _budget_limit_int("HARCHOC_MAX_EPOCHS", default=500)
    max_imgsz = _budget_limit_int("HARCHOC_MAX_IMGSZ", default=2048)
    max_batch = _budget_limit_int("HARCHOC_MAX_BATCH", default=16)
    if epochs > max_epochs:
        raise SystemExit(f"epochs={epochs} exceeds HARCHOC_MAX_EPOCHS={max_epochs}")
    if imgsz > max_imgsz:
        raise SystemExit(f"imgsz={imgsz} exceeds HARCHOC_MAX_IMGSZ={max_imgsz}")
    if batch > max_batch:
        raise SystemExit(f"batch={batch} exceeds HARCHOC_MAX_BATCH={max_batch}")
