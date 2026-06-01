from __future__ import annotations

import hashlib
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

from harchoc.strict_ml import append_capture_warning, capture_failure


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _file_record(path: Path, *, warnings: list[str] | None = None) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        return {"path": str(p), "exists": False}
    if not p.is_file():
        return {"path": str(p), "exists": True, "kind": "not-a-file"}
    rec: dict[str, Any] = {"path": str(p), "exists": True, "kind": "file", "sha256": _file_sha256(p)}
    with capture_failure(f"stat size_bytes {p}") as cap:
        rec["size_bytes"] = int(p.stat().st_size)
    append_capture_warning(warnings, cap)
    return rec


def _try_git_info(repo_root: Path, *, warnings: list[str] | None = None) -> dict[str, Any] | None:
    if not (repo_root / ".git").exists():
        return None

    commit: str | None = None
    with capture_failure("git rev-parse HEAD") as cap:
        commit = (
            subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(repo_root), stderr=subprocess.DEVNULL)
            .decode("utf-8", errors="replace")
            .strip()
        )
    if cap.failed:
        append_capture_warning(warnings, cap)
        return None

    branch: str | None = None
    with capture_failure("git rev-parse --abbrev-ref HEAD") as cap:
        branch = (
            subprocess.check_output(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=str(repo_root), stderr=subprocess.DEVNULL
            )
            .decode("utf-8", errors="replace")
            .strip()
        )
    append_capture_warning(warnings, cap)

    dirty: bool | None = None
    with capture_failure("git status --porcelain") as cap:
        dirty = bool(
            subprocess.check_output(["git", "status", "--porcelain"], cwd=str(repo_root), stderr=subprocess.DEVNULL)
            .decode("utf-8", errors="replace")
            .strip()
        )
    append_capture_warning(warnings, cap)

    return {"commit": commit, "branch": branch, "dirty": dirty}


def collect_repo_split_files(
    repo_root: str | Path, *, warnings: list[str] | None = None
) -> dict[str, Any]:
    """SHA256 records for tracked split lists under data/splits/."""
    rr = Path(repo_root).expanduser().resolve()
    splits_dir = rr / "data" / "splits"
    files: dict[str, Any] = {}
    for split in ("train", "val", "test"):
        files[split] = _file_record(splits_dir / f"{split}.txt", warnings=warnings)
    return {"splits_dir": str(splits_dir), "files": files}


def collect_run_metadata(
    *,
    repo_root: str | Path,
    dataset_manifest: str | Path | None = None,
    extra_files: dict[str, str | Path] | None = None,
    include_repo_splits: bool = False,
    warnings: list[str] | None = None,
) -> dict[str, Any]:
    """
    Lightweight run metadata for reproducibility without heavy ML deps.
    Intended to be embedded into reports/runs JSON.
    """
    rr = Path(repo_root).expanduser().resolve()
    files: dict[str, Any] = {}

    if dataset_manifest is not None:
        files["dataset_manifest"] = _file_record(Path(dataset_manifest), warnings=warnings)

    if extra_files:
        for k, v in extra_files.items():
            if isinstance(v, str) and not v.strip():
                continue
            files[str(k)] = _file_record(Path(v), warnings=warnings)

    out: dict[str, Any] = {
        "python": {"version": platform.python_version(), "executable": sys.executable},
        "platform": platform.platform(),
        "repo_root": str(rr),
        "git": _try_git_info(rr, warnings=warnings),
        "files": files,
    }
    if include_repo_splits:
        out["repo_splits"] = collect_repo_split_files(rr, warnings=warnings)
    return out

