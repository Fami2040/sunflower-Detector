"""Exclusive GPU lock: only gpu_queue (or explicit override) may run train.py."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_LOCK_PATH = Path("reports/gpu_queue/GPU_EXCLUSIVE.lock")


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def lock_path(repo_root: str | Path | None = None) -> Path:
    root = Path(repo_root or ".").resolve()
    return root / DEFAULT_LOCK_PATH


def acquire_gpu_exclusive(
    *,
    repo_root: str | Path | None = None,
    owner: str = "gpu_queue",
    pid: int | None = None,
) -> Path:
    path = lock_path(repo_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "owner": owner,
        "pid": pid if pid is not None else os.getpid(),
        "started_at": _utc_now(),
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def release_gpu_exclusive(*, repo_root: str | Path | None = None) -> None:
    path = lock_path(repo_root)
    if path.is_file():
        path.unlink()


def gpu_exclusive_message(repo_root: str | Path | None = None) -> str:
    path = lock_path(repo_root)
    if not path.is_file():
        return ""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        data = {}
    owner = data.get("owner") or "unknown"
    started = data.get("started_at") or "?"
    return (
        f"GPU exclusive lock active ({path}): owner={owner} started={started}. "
        "Stop ad-hoc trains; use ./scripts/run_gpu_queue.sh or remove the lock after killing strays."
    )


def adhoc_train_blocked(*, repo_root: str | Path | None = None) -> bool:
    """True when lock exists and caller is not gpu_queue subprocess."""
    if os.environ.get("HARCHOC_GPU_QUEUE_CHILD") == "1":
        return False
    if os.environ.get("HARCHOC_ALLOW_ADHOC_TRAIN") == "1":
        return False
    return lock_path(repo_root).is_file()
