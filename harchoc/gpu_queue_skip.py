"""GPU queue skip gates, GPU wait, and dry-run log cleanup."""

from __future__ import annotations

import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Callable

from harchoc.aug_smoke_runner import (
    extract_count_mae,
    find_smoke_entry,
    hsp_eval_artifacts_verified,
    load_aug_smoke_index,
    resolve_train_weights,
)
from harchoc.equivalence_index import summary_has_count_mae
from harchoc.json_io import load_json_dict
from harchoc.queue_skip_gates import (
    ensure_hsp_summary_from_artifacts,
    job_hsp_out_dir,
    job_needs_hsp_eval,
    job_summary_path,
    matrix_train_verified,
)

DEFAULT_MIN_FREE_MIB = 5500
DEFAULT_GPU_POLL_S = 30

_EVAL_COUNT_MAE_KINDS = frozenset(
    {"aug_smoke", "aug_sweep_15", "rtdetr_smoke", "train_compare", "amp_smoke", "sg_smoke"}
)
_REAL_TRAIN_LOG_MIN_BYTES = 1024
_KEEP_DRY_RUN_LOG_DIRS = frozenset({"preflight"})


def _job_log_dir_is_real_transcript(job_dir: Path) -> bool:
    if job_dir.name in _KEEP_DRY_RUN_LOG_DIRS:
        return True
    for pattern in ("train.log", "train_*.log"):
        for p in job_dir.glob(pattern):
            if p.is_file() and p.stat().st_size >= _REAL_TRAIN_LOG_MIN_BYTES:
                return True
    return False


def _prune_dry_run_log_stubs(log_root: Path) -> list[str]:
    removed: list[str] = []
    if not log_root.is_dir():
        return removed
    for job_dir in sorted(log_root.iterdir()):
        if not job_dir.is_dir() or _job_log_dir_is_real_transcript(job_dir):
            continue
        shutil.rmtree(job_dir)
        removed.append(job_dir.name)
    return removed


def _gpu_memory_mib() -> tuple[int | None, int | None]:
    try:
        proc = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return None, None
    if proc.returncode != 0 or not proc.stdout.strip():
        return None, None
    parts = [p.strip() for p in proc.stdout.strip().splitlines()[0].split(",")]
    if len(parts) != 2:
        return None, None
    return int(float(parts[0])), int(float(parts[1]))


def wait_gpu_free(
    *,
    min_free_mib: int = DEFAULT_MIN_FREE_MIB,
    poll_s: float = DEFAULT_GPU_POLL_S,
    timeout_s: float | None = None,
    dry_run: bool = False,
    log_fn: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Block until GPU free memory >= min_free_mib."""
    if dry_run:
        used, total = _gpu_memory_mib()
        return {
            "status": "dry_run",
            "used_mib": used,
            "total_mib": total,
            "min_free_mib": min_free_mib,
        }
    start = time.monotonic()
    while True:
        used, total = _gpu_memory_mib()
        if total is None:
            return {"status": "no_gpu", "used_mib": None, "total_mib": None}
        free = total - (used or 0)
        if log_fn is not None:
            log_fn(f"gpu_wait: used={used} total={total} free={free} need>={min_free_mib}")
        if free >= min_free_mib:
            return {
                "status": "ok",
                "used_mib": used,
                "total_mib": total,
                "free_mib": free,
                "waited_s": time.monotonic() - start,
            }
        if timeout_s is not None and (time.monotonic() - start) >= timeout_s:
            raise TimeoutError(f"GPU did not free {min_free_mib} MiB within {timeout_s}s")
        time.sleep(poll_s)


def _summary_is_verified_complete(
    obj: dict[str, Any],
    *,
    repo_root: Path,
    job: dict[str, Any],
) -> bool:
    if obj.get("status") != "complete":
        return False
    if "stages" in obj and not summary_has_count_mae(obj, repo_root):
        return False

    skip_if = job.get("skip_if") or {}
    eval_err = skip_if.get("eval_error_json")
    if eval_err:
        err_p = repo_root / str(eval_err)
        mae_val, _ = extract_count_mae(err_p) if err_p.is_file() else (None, None)
        if mae_val is None:
            return False

    kind = str(job.get("kind") or "")
    if kind in _EVAL_COUNT_MAE_KINDS and job.get("skip_eval") is not True:
        if not summary_has_count_mae(obj, repo_root):
            return False

    weights = (
        obj.get("weights")
        or (obj.get("train") or {}).get("weights")
        or (obj.get("context") or {}).get("weights")
    )
    if weights and "placeholder" in str(weights):
        return False

    if skip_if.get("require_weights"):
        run_name = str(job.get("run_name") or obj.get("run_name") or "")
        if run_name and resolve_train_weights(repo_root=repo_root, run_name=run_name) is None:
            return False

    return True


def should_skip_job(job: dict[str, Any], *, repo_root: Path) -> tuple[bool, str]:
    kind = str(job.get("kind") or "")
    if kind == "zoo_matrix_train":
        train_out = str(job.get("out") or "reports/hsp/matrix_train.json")
        matrix_group = str(job.get("matrix_group") or "")
        ok, reason = matrix_train_verified(repo_root, train_out, matrix_group)
        if ok:
            return True, reason

    skip_if = job.get("skip_if") or {}
    summary = skip_if.get("summary") or job_summary_path(job)
    if summary:
        sp = repo_root / str(summary)
        if sp.is_file():
            try:
                obj = load_json_dict(sp)
                if _summary_is_verified_complete(obj, repo_root=repo_root, job=job):
                    return True, f"summary complete: {summary}"
            except Exception:
                pass

    if job_needs_hsp_eval(job) or bool(job.get("eval_only")):
        run_name = str(job.get("run_name") or "")
        if run_name and hsp_eval_artifacts_verified(
            repo_root, run_name=run_name, out_dir=job_hsp_out_dir(job)
        ):
            if ensure_hsp_summary_from_artifacts(job, repo_root=repo_root):
                return True, f"HSP eval artifacts complete: {run_name}"

    index_status = skip_if.get("index_status")
    smoke_id = job.get("smoke_id")
    if index_status == "complete" and smoke_id:
        idx_path = job.get("aug_index") or "configs/experiments/aug_smoke_index.json"
        try:
            entry = find_smoke_entry(load_aug_smoke_index(repo_root / idx_path), str(smoke_id))
            if entry.get("status") == "complete":
                return True, f"aug index complete: {smoke_id}"
        except Exception:
            pass
    weights_run = skip_if.get("weights_run_name") or (
        job.get("run_name") if skip_if.get("skip_when_weights_exist") else None
    )
    if weights_run and not bool(job.get("eval_only")):
        if resolve_train_weights(repo_root=repo_root, run_name=str(weights_run)) is not None:
            return True, f"weights exist: {weights_run}"
    if job.get("skip") is True:
        return True, str(job.get("skip_reason") or "manifest skip=true")
    return False, ""
