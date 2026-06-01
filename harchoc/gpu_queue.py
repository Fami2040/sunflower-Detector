"""Sequential GPU backlog queue: manifest-driven jobs with staged logging."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from harchoc.aug_smoke_leaderboard import parse_equivalence_classes
from harchoc.aug_smoke_runner import (
    DEFAULT_LOCKED_CONF_FROM,
    DEFAULT_OUT_DIR,
    extract_count_mae,
    finalize_smoke_job,
    find_smoke_entry,
    hsp_eval_artifacts_verified,
    load_aug_smoke_index,
    resolve_train_weights,
    run_smoke_hsp_eval_chain,
)
from harchoc.queue_skip_gates import (
    ensure_hsp_summary_from_artifacts,
    job_hsp_out_dir,
    job_needs_hsp_eval,
    job_summary_path,
    matrix_train_verified,
)
from harchoc.gpu_exclusive import acquire_gpu_exclusive, release_gpu_exclusive
from harchoc.manuscript_repro import _format_cmd
from harchoc.queue_notify import (
    notify_queue_job,
    notify_queue_manifest_complete,
)

GPU_QUEUE_MANIFEST_SCHEMA = "gpu_queue_manifest.v1"
GPU_QUEUE_RUN_SCHEMA = "gpu_queue_run.v1"
DEFAULT_STATE_PATH = "reports/gpu_queue/run_state.json"
DEFAULT_LOG_ROOT = "reports/gpu_queue/logs"
DEFAULT_JOBS_ROOT = "reports/gpu_queue/jobs"
DEFAULT_SUMMARIES_ROOT = "reports/gpu_queue/summaries"
DEFAULT_EVAL_OUT_DIR = "reports/gpu_queue/eval"
DEFAULT_MIN_FREE_MIB = 5500
DEFAULT_GPU_POLL_S = 30

_EVAL_COUNT_MAE_KINDS = frozenset(
    {"aug_smoke", "aug_sweep_15", "rtdetr_smoke", "train_compare", "amp_smoke", "sg_smoke"}
)
AUG_SMOKE_PENDING_STATUSES = frozenset({"gpu_pending"})
_LEADERBOARD_JOB_KINDS = frozenset(
    {"aug_smoke", "aug_sweep_15", "aug_sweep_100", "amp_smoke", "sg_smoke"}
)
_REAL_TRAIN_LOG_MIN_BYTES = 1024
_KEEP_DRY_RUN_LOG_DIRS = frozenset({"preflight"})


class GpuQueueError(Exception):
    """Raised when a queue stage fails."""

    def __init__(self, *, job_id: str, stage_id: str, exit_code: int, log_path: str, hint: str = "") -> None:
        self.job_id = job_id
        self.stage_id = stage_id
        self.exit_code = exit_code
        self.log_path = log_path
        self.hint = hint
        super().__init__(f"job {job_id!r} stage {stage_id!r} failed (exit {exit_code}): {hint or log_path}")


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _job_log_dir_is_real_transcript(job_dir: Path) -> bool:
    """True when logs include a substantive train transcript (live run, not dry-run stub)."""
    if job_dir.name in _KEEP_DRY_RUN_LOG_DIRS:
        return True
    for pattern in ("train.log", "train_*.log"):
        for p in job_dir.glob(pattern):
            if p.is_file() and p.stat().st_size >= _REAL_TRAIN_LOG_MIN_BYTES:
                return True
    return False


def _prune_dry_run_log_stubs(log_root: Path) -> list[str]:
    """Drop per-job log dirs that only exist from dry-run (no real train transcript)."""
    removed: list[str] = []
    if not log_root.is_dir():
        return removed
    for job_dir in sorted(log_root.iterdir()):
        if not job_dir.is_dir() or _job_log_dir_is_real_transcript(job_dir):
            continue
        shutil.rmtree(job_dir)
        removed.append(job_dir.name)
    return removed


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2) + "\n", encoding="utf-8")


def _load_sg_train_recipe(repo_root: Path, train_config: str) -> dict[str, Any]:
    from harchoc.train_config import resolve_train_config_extends

    p = (repo_root / train_config).resolve()
    raw = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"train config must be object: {p}")
    return resolve_train_config_extends(raw, repo_root=repo_root, config_path=p)


def _sg_smoke_requires_supergradients(job: dict[str, Any], *, repo_root: Path) -> bool:
    """Train always needs SG; eval-only needs SG only when checkpoint is .pth."""
    if not bool(job.get("eval_only")):
        return True
    run_name = str(job.get("run_name") or "")
    weights = resolve_train_weights(repo_root=repo_root, run_name=run_name) if run_name else None
    if weights is None:
        return True
    return weights.suffix.lower() == ".pth"


def load_gpu_queue_manifest(
    path: str | Path,
    *,
    repo_root: str | Path | None = None,
) -> dict[str, Any]:
    p = Path(path).expanduser().resolve()
    obj = _read_json(p)
    schema = obj.get("schema_version")
    if schema != GPU_QUEUE_MANIFEST_SCHEMA:
        raise ValueError(
            f"unsupported gpu queue manifest schema: {schema!r} "
            f"(expected {GPU_QUEUE_MANIFEST_SCHEMA!r})"
        )
    out = dict(obj)
    out["manifest_path"] = str(p)
    jobs = list(out.get("jobs") or [])

    rr = Path(repo_root or p.parent.parent.parent).expanduser().resolve()
    idx = str(out.get("aug_smoke_index") or "configs/experiments/aug_smoke_index.json")
    if out.get("aug_smoke_from_index"):
        jobs = _merge_aug_smoke_jobs(jobs, repo_root=rr, index_path=idx)
    out["jobs"] = _filter_duplicate_preds_sha(
        _filter_duplicate_train_recipes(jobs, repo_root=rr),
        repo_root=rr,
        index_path=idx,
    )
    return out


def _job_train_recipe_fingerprint(job: dict[str, Any], *, repo_root: Path) -> str | None:
    kind = str(job.get("kind") or "")
    if kind not in ("aug_smoke", "aug_sweep_15", "aug_sweep_100", "amp_smoke"):
        return None
    if bool(job.get("eval_only")):
        return None
    tc = job.get("train_config")
    aug = job.get("aug_config")
    if kind == "aug_smoke" and (not tc or not aug):
        try:
            idx = load_aug_smoke_index(
                repo_root / (job.get("aug_index") or "configs/experiments/aug_smoke_index.json")
            )
            entry = find_smoke_entry(idx, str(job.get("smoke_id") or ""))
            if not tc:
                tc = entry.get("train_config")
            if not aug:
                aug = entry.get("aug_config")
        except Exception:
            pass
    if not tc:
        return None
    from harchoc.train_config import job_train_recipe_fingerprint

    return job_train_recipe_fingerprint(
        repo_root=repo_root,
        train_config=str(tc),
        aug_config=str(aug) if aug else None,
    )


def _preds_sha_from_summary_obj(obj: dict[str, Any]) -> str | None:
    arts = obj.get("artifacts") or {}
    preds = arts.get("preds_json") or {}
    sha = preds.get("sha256")
    return str(sha) if sha else None


_PREDS_DEDUP_TRAIN_KINDS = frozenset({"aug_smoke", "aug_sweep_15", "aug_sweep_100"})


def _index_preds_sha_by_smoke_id(index: dict[str, Any]) -> dict[str, str]:
    """Map smoke_id -> preds_sha256 from index equivalence_classes (audit metadata)."""
    out: dict[str, str] = {}
    equiv = index.get("equivalence_classes") or {}
    for cls in equiv.get("classes") or []:
        sha = cls.get("preds_sha256")
        if not sha:
            continue
        for sid in cls.get("smoke_ids") or []:
            out[str(sid).upper()] = str(sha)
    return out


def _sweep_dedup_id_from_arm(arm: dict[str, Any]) -> str:
    aid = str(arm.get("id") or "").upper()
    return aid


def _job_dedup_id(
    job: dict[str, Any],
    *,
    index: dict[str, Any] | None = None,
    repo_root: Path | None = None,
) -> str:
    """Smoke or sweep label used for preds dedup (e.g. S6, CLOSE25)."""
    sid = str(job.get("smoke_id") or "").upper()
    if sid:
        return sid
    kind = str(job.get("kind") or "")
    if kind not in ("aug_sweep_15", "aug_sweep_100"):
        return ""
    job_id = str(job.get("id") or "")
    if index and job_id:
        for section in ("sweeps_15ep", "sweeps_100ep"):
            for arm in (index.get(section) or {}).get("arms") or []:
                if str(arm.get("queue_job_id") or "") == job_id:
                    return _sweep_dedup_id_from_arm(arm)
    if repo_root is not None:
        summary = _job_summary_path_for_preds_dedup(job, dedup_id="")
        if summary:
            sp = repo_root / summary
            if sp.is_file():
                try:
                    obj = _read_json(sp)
                    from_summary = str(obj.get("smoke_id") or "").upper()
                    if from_summary:
                        return from_summary
                except Exception:
                    pass
    return ""


def _job_summary_path_for_preds_dedup(job: dict[str, Any], *, dedup_id: str) -> str | None:
    summary = job.get("summary_path")
    if summary:
        return str(summary)
    skip_if = job.get("skip_if") or {}
    summary = skip_if.get("summary")
    if summary:
        return str(summary)
    if dedup_id.startswith("S"):
        return f"reports/aug_smoke/{dedup_id.lower()}_summary.json"
    return None


def _preds_sha_from_verified_summary_path(repo_root: Path, summary_rel: str) -> str | None:
    sp = repo_root / summary_rel
    if not sp.is_file():
        return None
    try:
        obj = _read_json(sp)
        if obj.get("status") != "complete" or not _summary_has_count_mae(obj, repo_root):
            return None
        return _preds_sha_from_summary_obj(obj)
    except Exception:
        return None


def _register_preds_sha_owner(
    owners: dict[str, str],
    *,
    owner_id: str,
    summary_rel: str,
    repo_root: Path,
) -> None:
    sha = _preds_sha_from_verified_summary_path(repo_root, summary_rel)
    if sha:
        owners.setdefault(sha, owner_id)


def _complete_preds_sha_owners(
    *,
    repo_root: Path,
    index_path: str = "configs/experiments/aug_smoke_index.json",
) -> dict[str, str]:
    """Map preds_json sha256 -> first complete smoke/sweep id with verified summary."""
    owners: dict[str, str] = {}
    index = load_aug_smoke_index(repo_root / index_path)
    for entry in index.get("smokes") or []:
        if str(entry.get("status") or "") != "complete":
            continue
        sid = str(entry.get("id") or "").upper()
        summary = str(entry.get("summary") or f"reports/aug_smoke/{sid.lower()}_summary.json")
        _register_preds_sha_owner(owners, owner_id=sid, summary_rel=summary, repo_root=repo_root)
    for section in ("sweeps_15ep", "sweeps_100ep"):
        for arm in (index.get(section) or {}).get("arms") or []:
            if str(arm.get("status") or "") != "complete":
                continue
            sid = _sweep_dedup_id_from_arm(arm)
            summary = str(arm.get("summary") or "")
            if not sid or not summary:
                continue
            _register_preds_sha_owner(owners, owner_id=sid, summary_rel=summary, repo_root=repo_root)
    return owners


def _preds_dup_skip_reason(owner: str, sha: str) -> str:
    label = "complete run" if not str(owner).startswith("S") else "complete smoke"
    return f"preds duplicate of {label} {owner} (sha={sha[:12]}...)"


def _job_preds_sha_for_dedup(
    job: dict[str, Any],
    *,
    repo_root: Path,
    index_preds: dict[str, str],
    index: dict[str, Any] | None = None,
) -> str | None:
    """Known test preds sha for aug smoke/sweep train jobs (index audit or verified summary)."""
    kind = str(job.get("kind") or "")
    if kind not in _PREDS_DEDUP_TRAIN_KINDS or bool(job.get("eval_only")):
        return None
    dedup_id = _job_dedup_id(job, index=index, repo_root=repo_root)
    if not dedup_id:
        return None
    if dedup_id in index_preds:
        return index_preds[dedup_id]
    summary = _job_summary_path_for_preds_dedup(job, dedup_id=dedup_id)
    if summary:
        return _preds_sha_from_verified_summary_path(repo_root, summary)
    return None


def _filter_duplicate_preds_sha(
    jobs: list[dict[str, Any]],
    *,
    repo_root: Path,
    index_path: str = "configs/experiments/aug_smoke_index.json",
) -> list[dict[str, Any]]:
    index = load_aug_smoke_index(repo_root / index_path)
    owners = _complete_preds_sha_owners(repo_root=repo_root, index_path=index_path)
    index_preds = _index_preds_sha_by_smoke_id(index)
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for job in jobs:
        if job.get("skip") is True:
            out.append(job)
            continue
        sha = _job_preds_sha_for_dedup(
            job, repo_root=repo_root, index_preds=index_preds, index=index
        )
        if sha is None:
            out.append(job)
            continue
        dedup_id = _job_dedup_id(job, index=index, repo_root=repo_root)
        owner = owners.get(sha)
        if owner and dedup_id and dedup_id != owner:
            job = dict(job)
            job["skip"] = True
            job["skip_reason"] = _preds_dup_skip_reason(owner, sha)
        elif sha in seen:
            job = dict(job)
            job["skip"] = True
            job["skip_reason"] = f"duplicate preds in manifest (sha={sha[:12]}...)"
        else:
            seen.add(sha)
        out.append(job)
    return out


def _complete_recipe_owners(
    *,
    repo_root: Path,
    index_path: str = "configs/experiments/aug_smoke_index.json",
) -> dict[str, str]:
    """Map recipe fingerprint -> first complete smoke_id with verified summary."""
    from harchoc.train_config import job_train_recipe_fingerprint

    owners: dict[str, str] = {}
    index = load_aug_smoke_index(repo_root / index_path)
    for entry in index.get("smokes") or []:
        if str(entry.get("status") or "") != "complete":
            continue
        sid = str(entry.get("id") or "").upper()
        tc = entry.get("train_config")
        if not tc:
            continue
        summary = str(entry.get("summary") or f"reports/aug_smoke/{sid.lower()}_summary.json")
        sp = repo_root / summary
        if not sp.is_file():
            continue
        try:
            obj = _read_json(sp)
            if obj.get("status") != "complete" or not _summary_has_count_mae(obj, repo_root):
                continue
        except Exception:
            continue
        fp = job_train_recipe_fingerprint(
            repo_root=repo_root,
            train_config=str(tc),
            aug_config=str(entry["aug_config"]) if entry.get("aug_config") else None,
        )
        owners.setdefault(fp, sid)
    return owners


def _filter_duplicate_train_recipes(
    jobs: list[dict[str, Any]],
    *,
    repo_root: Path,
) -> list[dict[str, Any]]:
    owners = _complete_recipe_owners(repo_root=repo_root)
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for job in jobs:
        fp = _job_train_recipe_fingerprint(job, repo_root=repo_root)
        if fp is None:
            out.append(job)
            continue
        owner = owners.get(fp)
        if owner and str(job.get("smoke_id") or "").upper() != owner:
            job = dict(job)
            job["skip"] = True
            job["skip_reason"] = f"recipe duplicate of complete smoke {owner} (fp={fp})"
        elif fp in seen:
            job = dict(job)
            job["skip"] = True
            job["skip_reason"] = f"duplicate recipe in manifest (fp={fp})"
        else:
            seen.add(fp)
        out.append(job)
    return out


def _audit_only_equivalence_skip_reason(*, canonical: str, preds_sha: str) -> str:
    return (
        f"audit-only equivalence class (canonical {canonical}; "
        f"preds_sha256={preds_sha[:12]}...)"
    )


def expand_aug_smoke_jobs_from_index(
    *,
    repo_root: Path,
    index_path: str = "configs/experiments/aug_smoke_index.json",
    statuses: frozenset[str] | None = None,
) -> list[dict[str, Any]]:
    """Build aug_smoke queue jobs for index rows matching *statuses* (default gpu_pending)."""
    pending = statuses or AUG_SMOKE_PENDING_STATUSES
    index = load_aug_smoke_index(repo_root / index_path)
    audit_only, _, audit_skip = parse_equivalence_classes(index)
    jobs: list[dict[str, Any]] = []
    for entry in index.get("smokes") or []:
        if str(entry.get("status") or "") not in pending:
            continue
        sid = str(entry.get("id") or "").strip().upper()
        if not sid:
            continue
        summary = str(entry.get("summary") or f"reports/aug_smoke/{sid.lower()}_summary.json")
        backlog: list[str] = ["P1-AUG"]
        if entry.get("negative_control"):
            backlog = ["P0-1", "P1-AUG"]
        job: dict[str, Any] = {
            "id": f"aug_smoke_{sid}",
            "kind": "aug_smoke",
            "backlog": backlog,
            "smoke_id": sid,
            "estimated_minutes": 15 if entry.get("eval_only") else 45,
            "env": {"HARCHOC_MAX_EPOCHS": "15"},
            "skip_if": {"summary": summary, "index_status": "complete"},
        }
        if entry.get("eval_only"):
            job["eval_only"] = True
        if entry.get("run_name"):
            job["run_name"] = entry["run_name"]
        if entry.get("weights_run_name"):
            job["weights_run_name"] = entry["weights_run_name"]
        if entry.get("max_det") is not None:
            job["max_det"] = entry["max_det"]
        if entry.get("aug_config"):
            job["aug_config"] = str(entry["aug_config"])
        note = entry.get("key_overrides") or entry.get("notes")
        if note:
            job["notes"] = str(note)
        if sid in audit_only:
            job["skip"] = True
            canonical, sha = audit_skip.get(sid, ("", ""))
            if canonical and sha:
                job["skip_reason"] = _audit_only_equivalence_skip_reason(
                    canonical=canonical, preds_sha=sha
                )
            elif canonical:
                job["skip_reason"] = f"audit-only equivalence class (canonical {canonical})"
            else:
                job["skip_reason"] = "audit-only equivalence class"
        jobs.append(job)
    return jobs


def _merge_aug_smoke_jobs(
    jobs: list[dict[str, Any]],
    *,
    repo_root: Path,
    index_path: str,
) -> list[dict[str, Any]]:
    """Replace inline aug_smoke jobs with index-expanded gpu_pending rows."""
    expanded = expand_aug_smoke_jobs_from_index(repo_root=repo_root, index_path=index_path)
    owners = _complete_recipe_owners(repo_root=repo_root, index_path=index_path)
    filtered: list[dict[str, Any]] = []
    for job in expanded:
        fp = _job_train_recipe_fingerprint(job, repo_root=repo_root)
        owner = owners.get(fp) if fp else None
        sid = str(job.get("smoke_id") or "").upper()
        if owner and sid != owner:
            continue
        filtered.append(job)
    without = [j for j in jobs if str(j.get("kind")) != "aug_smoke"]
    insert_at = 0
    for i, job in enumerate(without):
        if str(job.get("kind")) in ("preflight", "vram_probe"):
            insert_at = i + 1
    return without[:insert_at] + filtered + without[insert_at:]


def _mamba_env() -> str:
    return os.environ.get("HARCHOC_MAMBA_ENV", "harchoc")


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


def _summary_has_count_mae(obj: dict[str, Any], repo_root: Path) -> bool:
    mae = obj.get("test_count_mae")
    if mae is not None:
        return True
    err = (obj.get("test_eval") or {}).get("error_json")
    if err:
        path = Path(err)
        if not path.is_absolute():
            path = repo_root / err
        mae_val, _ = extract_count_mae(path)
        return mae_val is not None
    return False


def _summary_is_verified_complete(
    obj: dict[str, Any],
    *,
    repo_root: Path,
    job: dict[str, Any],
) -> bool:
    """True only when skip_if summary reflects real eval (not a dry-run job transcript)."""
    if obj.get("status") != "complete":
        return False
    # run_gpu_queue job transcripts list stages but lack HSP count MAE — not skip-worthy.
    if "stages" in obj and not _summary_has_count_mae(obj, repo_root):
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
        if not _summary_has_count_mae(obj, repo_root):
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
                obj = _read_json(sp)
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


def _resolve_job_hsp_max_det(job: dict[str, Any], *, repo_root: Path, train_config: str) -> int:
    """HSP eval max_det: job override, merged train eval section, or RT-DETR num_queries."""
    if job.get("max_det") is not None:
        return int(job["max_det"])
    from harchoc.rtdetr_limits import (
        is_rtdetr_model,
        rtdetr_eval_max_det,
        rtdetr_fields_from_train_json,
    )
    from harchoc.train_config import load_train_config_json

    cfg_path = (repo_root / train_config).resolve()
    merged = load_train_config_json(cfg_path, repo_root=repo_root)
    eval_section = merged.get("eval")
    if isinstance(eval_section, dict) and eval_section.get("max_det") is not None:
        return int(eval_section["max_det"])
    model = merged.get("model")
    if is_rtdetr_model(str(model) if model is not None else None):
        fields = rtdetr_fields_from_train_json(merged, path=str(cfg_path))
        return rtdetr_eval_max_det(int(fields["num_queries"]))
    return 3000


def _is_rtdetr_train_job(job: dict[str, Any], *, repo_root: Path) -> bool:
    from harchoc.rtdetr_limits import is_rtdetr_model
    from harchoc.train_config import load_train_config_json

    if str(job.get("kind") or "") == "rtdetr_smoke":
        return True
    cfg = job.get("train_config")
    if not cfg:
        return False
    merged = load_train_config_json((repo_root / str(cfg)).resolve(), repo_root=repo_root)
    return is_rtdetr_model(str(merged.get("model") or ""))


def _validate_job_files(job: dict[str, Any], repo_root: Path) -> None:
    kind = str(job.get("kind") or "")
    if kind in ("aug_smoke", "train_compare", "vram_probe", "rtdetr_smoke", "amp_smoke", "sg_smoke", "aug_sweep_15", "aug_sweep_100"):
        cfg = job.get("train_config")
        if cfg and not (repo_root / str(cfg)).is_file():
            raise FileNotFoundError(f"job {job.get('id')}: missing train_config {cfg}")
    if kind == "aug_smoke":
        smoke_id = job.get("smoke_id") or job.get("id", "").replace("aug_smoke_", "")
        idx = load_aug_smoke_index(repo_root / (job.get("aug_index") or "configs/experiments/aug_smoke_index.json"))
        entry = find_smoke_entry(idx, str(smoke_id))
        tc = entry.get("train_config") or job.get("train_config")
        if tc and not (repo_root / str(tc)).is_file():
            raise FileNotFoundError(f"aug_smoke {smoke_id}: missing train_config {tc}")


def _script_argv(script: str, tail: list[str]) -> list[str]:
    return [f"scripts/{script}" if not script.startswith("scripts/") else script, *tail]


def _smoke_weights_run_name(
    *,
    job: dict[str, Any],
    meta: dict[str, Any],
    run_name: str,
) -> str:
    """Ultralytics run dir for best.pt; defaults to eval artifact run_name."""
    return str(
        meta.get("weights_run_name")
        or job.get("weights_run_name")
        or run_name
    )


def _build_ultralytics_smoke_stages(
    stages: list[dict[str, Any]],
    *,
    cfg: str,
    name: str,
    locked: str,
    eval_only: bool = False,
    skip_eval: bool = False,
    train_skip_eval: bool = True,
    aug_config: str | None = None,
    eval_meta: dict[str, Any],
    summary_meta: dict[str, Any],
) -> None:
    """Append dry_run/train/eval/summary onto preflight *stages* (dry_run slot + gpu_wait)."""
    if not eval_only:
        dry_argv: list[str] = ["--config", cfg, "--name", name, "--dry-run"]
        train_argv: list[str] = ["--config", cfg, "--name", name]
        if aug_config:
            dry_argv.extend(["--aug-config", str(aug_config)])
            train_argv.extend(["--aug-config", str(aug_config)])
        if train_skip_eval:
            dry_argv.append("--skip-eval")
            train_argv.append("--skip-eval")
        stages[0] = {
            "stage_id": "dry_run",
            "argv": _script_argv("train.py", dry_argv),
            "mamba": True,
        }
        stages.append(
            {
                "stage_id": "train",
                "argv": _script_argv("train.py", train_argv),
                "mamba": True,
            }
        )
    if not skip_eval:
        stages.append(
            {
                "stage_id": "eval_test",
                "internal": "smoke_hsp_eval",
                "mamba": False,
                "meta": {
                    "run_name": name,
                    "train_config": cfg,
                    "locked_conf_from": locked,
                    **eval_meta,
                },
            }
        )
    stages.append(
        {
            "stage_id": "summary",
            "internal": "job_summary",
            "mamba": False,
            "meta": summary_meta,
        }
    )


def _finalize_job_summary(
    *,
    repo_root: Path,
    job: dict[str, Any],
    job_context: dict[str, Any],
    meta: dict[str, Any],
    dry_run: bool,
) -> int:
    """Build summary JSON, job transcript, and optional aug index patch."""
    summary_kind = str(meta.get("summary_kind") or "generic")
    run_name = str(job_context.get("run_name") or meta.get("run_name") or job.get("run_name") or "")
    if dry_run:
        print(f"# summary {job.get('id')} ({summary_kind})")
        return 0

    if summary_kind == "vram_probe":
        summary_path = str(
            meta.get("summary_path")
            or job.get("summary_path")
            or f"{DEFAULT_SUMMARIES_ROOT}/{job.get('id')}.json"
        )
        weights = job_context.get("weights") or resolve_train_weights(repo_root=repo_root, run_name=run_name)
        payload = {
            "schema_version": "gpu_queue_job.v1",
            "job_id": job.get("id"),
            "kind": job.get("kind"),
            "status": "complete" if weights else "failed",
            "finished_at": _utc_now(),
            "run_name": run_name,
            "weights": str(weights) if weights else None,
        }
        _write_json(repo_root / summary_path, payload)
        _write_json(repo_root / DEFAULT_JOBS_ROOT / f"{job.get('id')}.json", payload)
        return 0 if weights else 1

    weights = job_context.get("weights") or resolve_train_weights(repo_root=repo_root, run_name=run_name)
    if not weights and str(job.get("kind")) != "amp_smoke":
        return 1

    out_dir = str(job_context.get("eval_out_dir") or meta.get("out_dir") or DEFAULT_OUT_DIR)
    summary_path = str(
        meta.get("summary_path")
        or job.get("summary_path")
        or f"{DEFAULT_SUMMARIES_ROOT}/{job.get('id') or run_name}.json"
    )

    if job.get("skip_eval") or not job_context.get("eval_artifacts"):
        probe_payload = {
            "schema_version": "gpu_queue_job.v1",
            "job_id": job.get("id"),
            "kind": job.get("kind"),
            "status": "complete",
            "finished_at": _utc_now(),
            "run_name": run_name,
            "weights": str(weights) if weights else None,
        }
        _write_json(repo_root / summary_path, probe_payload)
        _write_json(repo_root / DEFAULT_JOBS_ROOT / f"{job.get('id')}.json", probe_payload)
        return 0 if weights or str(job.get("kind")) == "amp_smoke" else 1

    if not weights:
        return 1

    smoke_id = str(meta.get("smoke_id") or meta.get("job_id") or job.get("id") or run_name)
    index_path = str(
        meta.get("index_path") or job.get("aug_index") or "configs/experiments/aug_smoke_index.json"
    )
    payload = finalize_smoke_job(
        repo_root=repo_root,
        run_name=run_name,
        train_config=str(meta.get("train_config") or job.get("train_config") or ""),
        weights=Path(str(weights)),
        summary_path=summary_path,
        smoke_id=smoke_id,
        locked_conf_from=str(meta.get("locked_conf_from") or DEFAULT_LOCKED_CONF_FROM),
        out_dir=out_dir,
        arch_ticket=str(meta.get("arch_ticket") or ",".join(job.get("backlog") or [])),
        index_path=index_path,
        patch_index=summary_kind == "aug_smoke",
        train_runtime_s=job_context.get("train_runtime_s"),
        refresh_leaderboard=False,
    )
    prefix = f"{out_dir}/{run_name}"
    error_json = str((repo_root / f"{prefix}_error.json").resolve())
    transcript: dict[str, Any] = {
        "job_id": job.get("id"),
        "status": payload.get("status"),
        "test_count_mae": payload.get("test_count_mae"),
        "summary_path": summary_path,
        "weights": str(weights),
    }
    if summary_kind in ("generic", "rtdetr"):
        transcript["eval_error_json"] = error_json
    _write_json(repo_root / DEFAULT_JOBS_ROOT / f"{job.get('id')}.json", transcript)
    return 0 if payload.get("status") == "complete" else 1


def _maybe_refresh_aug_leaderboard(
    *,
    repo_root: Path,
    job: dict[str, Any],
    dry_run: bool,
) -> None:
    if dry_run or str(job.get("kind") or "") not in _LEADERBOARD_JOB_KINDS:
        return
    from harchoc.aug_smoke_leaderboard import refresh_aug_smoke_leaderboard

    refresh_aug_smoke_leaderboard(repo_root=repo_root)


def _run_smoke_hsp_eval_stage(
    *,
    job: dict[str, Any],
    meta: dict[str, Any],
    repo_root: Path,
    log_path: Path,
    dry_run: bool,
    job_context: dict[str, Any],
) -> int:
    run_name = str(meta.get("run_name") or job.get("run_name") or "")
    weights_run_name = _smoke_weights_run_name(job=job, meta=meta, run_name=run_name)
    out_dir = str(meta.get("out_dir") or DEFAULT_OUT_DIR)
    max_det = int(meta.get("max_det") or job.get("max_det") or 3000)
    model_id = str(meta.get("model_id") or job.get("model_id") or "yolo_nas_s")
    locked = str(meta.get("locked_conf_from") or DEFAULT_LOCKED_CONF_FROM)
    weights_for_dry = resolve_train_weights(
        repo_root=repo_root, run_name=weights_run_name
    ) or (repo_root / "runs/placeholder/weights/best.pt")

    with log_path.open("w", encoding="utf-8") as f:

        def _on(stage_id: str, argv: list[str]) -> None:
            if stage_id == "sg_export":
                f.write(f"# {stage_id}: backend auto from weights suffix\n")
            else:
                f.write(f"# {stage_id}: {_format_cmd(argv, mamba=True)}\n")

        if dry_run:
            run_smoke_hsp_eval_chain(
                repo_root=repo_root,
                run_name=run_name,
                weights=weights_for_dry,
                locked_conf_from=locked,
                out_dir=out_dir,
                max_det=max_det,
                model_id=model_id,
                dry_run=True,
                on_stage=_on,
            )
            return 0

        weights = resolve_train_weights(repo_root=repo_root, run_name=weights_run_name)
        if weights is None:
            f.write(f"weights not found for run {weights_run_name}\n")
            return 1
        job_context["weights"] = str(weights)
        job_context["run_name"] = run_name
        job_context["weights_run_name"] = weights_run_name
        job_context["eval_out_dir"] = out_dir
        try:
            artifacts = run_smoke_hsp_eval_chain(
                repo_root=repo_root,
                run_name=run_name,
                weights=weights,
                locked_conf_from=locked,
                out_dir=out_dir,
                max_det=max_det,
                model_id=model_id,
                dry_run=False,
                on_stage=_on,
            )
            job_context["eval_artifacts"] = artifacts
            job_context.update(meta)
        except RuntimeError as ex:
            f.write(str(ex) + "\n")
            return 1
    return 0


def build_job_stages(
    job: dict[str, Any],
    *,
    repo_root: Path,
    defaults: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Expand job into stage dicts: {stage_id, argv, mamba, internal?}."""
    kind = str(job.get("kind") or "")
    defs = defaults or {}
    locked = str(job.get("locked_conf_from") or defs.get("locked_conf_from") or DEFAULT_LOCKED_CONF_FROM)
    stages: list[dict[str, Any]] = []

    if kind == "preflight":
        stages.append(
            {
                "stage_id": "validate_splits",
                "argv": _script_argv("validate_splits.py", ["--require-test"]),
                "mamba": True,
            }
        )
        stages.append(
            {
                "stage_id": "check_gpu",
                "argv": _script_argv(
                    "check_gpu.py",
                    ["--json-out", str(job.get("gpu_check_out") or "reports/gpu_queue/gpu_check.json")],
                ),
                "mamba": True,
            }
        )
        return stages

    if kind == "gpu_wait_only":
        stages.append({"stage_id": "gpu_wait", "internal": "gpu_wait", "mamba": False})
        return stages

    # Common preflight for GPU jobs
    if kind not in ("preflight",):
        stages.append(
            {
                "stage_id": "dry_run",
                "argv": [],
                "mamba": True,
                "internal": "dry_run",
            }
        )
        stages.append({"stage_id": "gpu_wait", "internal": "gpu_wait", "mamba": False})

    if kind == "vram_probe":
        cfg = str(job.get("train_config") or "configs/experiments/train_batch_probe_rtdetr-l.json")
        name = str(job.get("run_name") or "batch_probe_rtdetr-l")
        stages[0] = {
            "stage_id": "dry_run",
            "argv": _script_argv(
                "train.py",
                ["--config", cfg, "--name", name, "--dry-run", "--skip-eval"],
            ),
            "mamba": True,
        }
        stages.append(
            {
                "stage_id": "train",
                "argv": _script_argv(
                    "train.py",
                    ["--config", cfg, "--name", name, "--skip-eval"],
                ),
                "mamba": True,
            }
        )
        stages.append(
            {
                "stage_id": "summary",
                "internal": "job_summary",
                "mamba": False,
                "meta": {
                    "summary_kind": "vram_probe",
                    "summary_path": str(
                        job.get("summary_path")
                        or f"{DEFAULT_SUMMARIES_ROOT}/{job.get('id') or 'vram_probe'}.json"
                    ),
                },
            }
        )
        return stages

    if kind == "aug_smoke":
        smoke_id = str(job.get("smoke_id") or "")
        idx_path = str(job.get("aug_index") or "configs/experiments/aug_smoke_index.json")
        entry = find_smoke_entry(load_aug_smoke_index(repo_root / idx_path), smoke_id)
        cfg = str(job.get("train_config") or entry.get("train_config") or "")
        name = str(job.get("run_name") or entry.get("run_name") or "")
        if not cfg or not name:
            raise ValueError(f"aug_smoke {smoke_id}: missing train_config or run_name")
        eval_only = bool(job.get("eval_only") or entry.get("eval_only"))
        weights_run_name = str(
            job.get("weights_run_name") or entry.get("weights_run_name") or name
        )
        max_det = int(job.get("max_det") or entry.get("max_det") or 3000)
        aug = job.get("aug_config") or entry.get("aug_config")
        summary_path = str(
            job.get("summary_path")
            or entry.get("summary")
            or f"{DEFAULT_OUT_DIR}/{smoke_id.lower()}_summary.json"
        )
        _build_ultralytics_smoke_stages(
            stages,
            cfg=cfg,
            name=name,
            locked=locked,
            eval_only=eval_only,
            aug_config=str(aug) if aug else None,
            eval_meta={
                "smoke_id": smoke_id,
                "weights_run_name": weights_run_name,
                "max_det": max_det,
                "out_dir": DEFAULT_OUT_DIR,
                "index_path": idx_path,
            },
            summary_meta={
                "summary_kind": "aug_smoke",
                "smoke_id": smoke_id,
                "run_name": name,
                "train_config": cfg,
                "locked_conf_from": locked,
                "index_path": idx_path,
                "summary_path": summary_path,
                "out_dir": DEFAULT_OUT_DIR,
            },
        )
        return stages

    if kind == "rtdetr_smoke":
        cfg = str(job.get("train_config") or "configs/experiments/train_rtdetr_queries_smoke_15ep.json")
        name = str(job.get("run_name") or "rtdetr_queries_smoke_15ep")
        out_dir = str(job.get("eval_out_dir") or DEFAULT_EVAL_OUT_DIR)
        summary_path = str(
            job.get("summary_path") or f"{DEFAULT_SUMMARIES_ROOT}/{job.get('id') or name}.json"
        )
        arch = ",".join(job.get("backlog") or [])
        _build_ultralytics_smoke_stages(
            stages,
            cfg=cfg,
            name=name,
            locked=locked,
            eval_meta={
                "max_det": int(job.get("max_det") or 1024),
                "out_dir": out_dir,
                "summary_path": summary_path,
                "arch_ticket": arch,
            },
            summary_meta={
                "summary_kind": "rtdetr",
                "run_name": name,
                "train_config": cfg,
                "locked_conf_from": locked,
                "out_dir": out_dir,
                "summary_path": summary_path,
                "arch_ticket": arch,
            },
        )
        return stages

    if kind == "amp_smoke":
        cfg = str(job.get("train_config") or "")
        name = str(job.get("run_name") or job.get("id") or "")
        eval_only = bool(job.get("eval_only"))
        skip_eval = bool(job.get("skip_eval"))
        out_dir = str(job.get("eval_out_dir") or "reports/hsp")
        summary_path = str(job.get("summary_path") or f"{out_dir}/{name}_summary.json")
        _build_ultralytics_smoke_stages(
            stages,
            cfg=cfg,
            name=name,
            locked=locked,
            eval_only=eval_only,
            skip_eval=skip_eval,
            eval_meta={"max_det": int(job.get("max_det") or 3000), "out_dir": out_dir},
            summary_meta={
                "summary_kind": "aug_sweep",
                "job_id": job.get("id"),
                "run_name": name,
                "train_config": cfg,
                "locked_conf_from": locked,
                "summary_path": summary_path,
                "out_dir": out_dir,
                "arch_ticket": ",".join(job.get("backlog") or ["P1-AMP-SMOKE"]),
            },
        )
        return stages

    if kind == "sg_smoke":
        cfg = str(job.get("train_config") or "")
        name = str(job.get("run_name") or job.get("id") or "")
        eval_only = bool(job.get("eval_only"))
        out_dir = str(job.get("eval_out_dir") or "reports/aug_smoke")
        summary_path = str(job.get("summary_path") or f"{out_dir}/{name}_summary.json")
        model_id = str(job.get("model_id") or "yolo_nas_s")
        skip_eval = bool(job.get("skip_eval"))
        if not eval_only:
            stages[0] = {
                "stage_id": "dry_run",
                "internal": "sg_train",
                "mamba": False,
                "meta": {
                    "train_config": cfg,
                    "run_name": name,
                    "model_id": model_id,
                    "dry_run": True,
                },
            }
            stages.append(
                {
                    "stage_id": "train",
                    "internal": "sg_train",
                    "mamba": False,
                    "meta": {
                        "train_config": cfg,
                        "run_name": name,
                        "model_id": model_id,
                    },
                }
            )
        if not skip_eval:
            stages.append(
                {
                    "stage_id": "eval_test",
                    "internal": "smoke_hsp_eval",
                    "mamba": False,
                    "meta": {
                        "run_name": name,
                        "train_config": cfg,
                        "locked_conf_from": locked,
                        "max_det": int(job.get("max_det") or 3000),
                        "out_dir": out_dir,
                        "model_id": model_id,
                    },
                }
            )
        stages.append(
            {
                "stage_id": "summary",
                "internal": "job_summary",
                "mamba": False,
                "meta": {
                    "summary_kind": "aug_sweep",
                    "job_id": job.get("id"),
                    "run_name": name,
                    "train_config": cfg,
                    "locked_conf_from": locked,
                    "summary_path": summary_path,
                    "out_dir": out_dir,
                    "arch_ticket": ",".join(job.get("backlog") or ["P1-SG"]),
                },
            }
        )
        return stages

    if kind == "train_compare":
        cfg = str(job.get("train_config") or "")
        name = str(job.get("run_name") or job.get("id") or "")
        skip_eval = bool(job.get("skip_eval", False))
        out_dir = str(job.get("eval_out_dir") or DEFAULT_EVAL_OUT_DIR)
        summary_path = str(
            job.get("summary_path") or f"{DEFAULT_SUMMARIES_ROOT}/{job.get('id') or name}.json"
        )
        arch = ",".join(job.get("backlog") or [])
        max_det = _resolve_job_hsp_max_det(job, repo_root=repo_root, train_config=cfg)
        summary_kind = "rtdetr" if _is_rtdetr_train_job(job, repo_root=repo_root) else "generic"
        _build_ultralytics_smoke_stages(
            stages,
            cfg=cfg,
            name=name,
            locked=locked,
            skip_eval=skip_eval,
            train_skip_eval=True,
            eval_meta={
                "max_det": max_det,
                "out_dir": out_dir,
                "summary_path": summary_path,
                "arch_ticket": arch,
            },
            summary_meta={
                "summary_kind": summary_kind,
                "run_name": name,
                "train_config": cfg,
                "locked_conf_from": locked,
                "out_dir": out_dir,
                "summary_path": summary_path,
                "arch_ticket": arch,
            },
        )
        return stages

    if kind in ("aug_sweep_15", "aug_sweep_100"):
        cfg = str(
            job.get("train_config") or "configs/experiments/train_aug_mosaic_sweep_smoke_15ep.json"
        )
        name = str(job.get("run_name") or job.get("id") or "")
        aug = job.get("aug_config")
        summary_path = str(job.get("summary_path") or f"reports/aug_smoke/{name}_summary.json")
        _build_ultralytics_smoke_stages(
            stages,
            cfg=cfg,
            name=name,
            locked=locked,
            aug_config=str(aug) if aug else None,
            eval_meta={"max_det": 3000, "out_dir": DEFAULT_OUT_DIR},
            summary_meta={
                "summary_kind": "aug_sweep",
                "job_id": job.get("id"),
                "run_name": name,
                "train_config": cfg,
                "locked_conf_from": locked,
                "summary_path": summary_path,
                "out_dir": DEFAULT_OUT_DIR,
                "arch_ticket": ",".join(job.get("backlog") or []),
            },
        )
        return stages

    if kind == "zoo_matrix_train":
        out = str(job.get("out") or "reports/hsp/matrix_train.json")
        matrix_group = str(job.get("matrix_group") or "").strip()
        bench_argv: list[str] = []
        if matrix_group:
            bench_argv.extend(["--group", matrix_group])
        stages = [
            {
                "stage_id": "dry_run",
                "argv": _script_argv(
                    "benchmark_matrix.py",
                    ["--dry-run", "--out", str(job.get("plan_out") or "reports/hsp/matrix_plan.json")]
                    + bench_argv,
                ),
                "mamba": True,
            },
            {
                "stage_id": "rtdetr_15ep_gate",
                "internal": "zoo_rtdetr_gate",
                "mamba": False,
                "meta": {"matrix_group": matrix_group},
            },
            {"stage_id": "gpu_wait", "internal": "gpu_wait", "mamba": False},
            {
                "stage_id": "train",
                "argv": _script_argv(
                    "benchmark_matrix.py",
                    [
                        "--no-dry-run",
                        "--runs-dir",
                        str(job.get("runs_dir") or "runs/hsp_zoo"),
                        "--train-out",
                        out,
                    ]
                    + bench_argv,
                ),
                "mamba": True,
            },
            {"stage_id": "summary", "internal": "job_summary", "mamba": False, "meta": {"summary_kind": "generic"}},
        ]
        return stages

    if kind == "cv_fold_train":
        folds = int(job.get("folds") or 5)
        splits_out = str(job.get("splits_out") or "reports/cv_folds")
        stages = [
            {
                "stage_id": "cv_splits",
                "argv": _script_argv(
                    "cv_eval.py",
                    ["--write-fold-splits", splits_out, "--folds", str(folds)],
                ),
                "mamba": False,
            },
        ]
        cfg = str(job.get("train_config") or "configs/experiments/train_yolov8m_baseline.json")
        for i in range(folds):
            name = f"cv_fold_{i}"
            stages.append({"stage_id": f"gpu_wait_fold_{i}", "internal": "gpu_wait", "mamba": False})
            stages.append(
                {
                    "stage_id": f"train_fold_{i}",
                    "argv": _script_argv(
                        "train.py",
                        ["--config", cfg, "--name", name, "--skip-eval"],
                    ),
                    "mamba": True,
                    "meta": {"fold": i, "note": "fold-specific data.yaml wiring deferred; uses repo splits"},
                }
            )
        stages.append({"stage_id": "summary", "internal": "job_summary", "mamba": False, "meta": {"summary_kind": "generic"}})
        return stages

    raise ValueError(f"unsupported job kind: {kind!r}")


def _tail_log(path: Path, n: int = 40) -> list[str]:
    if not path.is_file():
        return []
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return lines[-n:]


def _run_subprocess_stage(
    *,
    argv: list[str],
    mamba: bool,
    repo_root: Path,
    log_path: Path,
    env: dict[str, str],
    dry_run: bool,
) -> int:
    mamba_env = _mamba_env()
    if mamba:
        cmd = ["mamba", "run", "-n", mamba_env, "python", *argv]
    else:
        cmd = [sys.executable, *argv]
    line = _format_cmd([argv[0].replace("scripts/", "scripts/"), *argv[1:]], mamba=mamba)
    if dry_run:
        print(f"# {line}")
        return 0
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log_f:
        log_f.write(f"# started {_utc_now()}\n# cmd: {line}\n\n")
        log_f.flush()
        proc = subprocess.run(cmd, cwd=str(repo_root), env=env, stdout=log_f, stderr=subprocess.STDOUT)
        log_f.write(f"\n# exit_code={proc.returncode} finished {_utc_now()}\n")
    return int(proc.returncode)


def _run_internal_stage(
    stage: dict[str, Any],
    *,
    job: dict[str, Any],
    repo_root: Path,
    log_path: Path,
    dry_run: bool,
    job_context: dict[str, Any],
    min_free_mib: int,
) -> int:
    internal = stage.get("internal")
    if internal == "gpu_wait":
        with log_path.open("w", encoding="utf-8") as f:
            try:
                info = wait_gpu_free(min_free_mib=min_free_mib, dry_run=dry_run, log_fn=f.write)
                f.write(json.dumps(info, indent=2) + "\n")
                return 0
            except TimeoutError as ex:
                f.write(str(ex) + "\n")
                return 1

    if internal == "dry_run":
        stages = build_job_stages(job, repo_root=repo_root)
        for st in stages:
            if st.get("stage_id") == "dry_run" and st.get("argv"):
                return _run_subprocess_stage(
                    argv=st["argv"],
                    mamba=bool(st.get("mamba", True)),
                    repo_root=repo_root,
                    log_path=log_path,
                    env=dict(os.environ),
                    dry_run=dry_run,
                )
        return 0

    if internal == "zoo_rtdetr_gate":
        from harchoc.rtdetr_zoo_gate import (
            check_zoo_core_rtdetr_15ep_gates,
            format_zoo_core_rtdetr_gate_blockers,
        )

        matrix_group = str((stage.get("meta") or {}).get("matrix_group") or job.get("matrix_group") or "")
        with log_path.open("w", encoding="utf-8") as f:
            if matrix_group != "zoo_core":
                f.write(f"# zoo_rtdetr_gate: skipped (matrix_group={matrix_group!r})\n")
                return 0
            blockers = format_zoo_core_rtdetr_gate_blockers(repo_root)
            f.write(blockers + "\n")
            gates = check_zoo_core_rtdetr_15ep_gates(repo_root=repo_root)
            f.write(json.dumps({"gates": gates}, indent=2) + "\n")
            if dry_run:
                f.write("# dry_run: gate logged only; live train blocked when any gate fails\n")
                return 0
            if not all(g["passed"] for g in gates):
                return 1
        return 0

    meta = stage.get("meta") or {}

    if internal == "sg_train":
        cfg = str(meta.get("train_config") or job.get("train_config") or "")
        run_name = str(meta.get("run_name") or job.get("run_name") or "")
        model_id = str(meta.get("model_id") or "yolo_nas_s")
        is_cfg_dry = bool(meta.get("dry_run"))
        recipe = _load_sg_train_recipe(repo_root, cfg) if cfg else {}
        if dry_run or is_cfg_dry:
            with log_path.open("w", encoding="utf-8") as f:
                f.write(
                    f"# sg_train dry_run: model_id={model_id} run_name={run_name} "
                    f"epochs={recipe.get('epochs')} imgsz={recipe.get('imgsz')} batch={recipe.get('batch')}\n"
                )
            return 0
        from harchoc.datasets import resolve_dataset
        from harchoc.supergradients_train import train_bench_run

        spec = resolve_dataset(
            manifest_path=repo_root / "data/manifest.json",
            default_dataset_name="sunflower",
        )
        result = train_bench_run(
            model_id=str(recipe.get("model_id") or model_id),
            dataset_root=spec.root,
            runs_dir=repo_root / "runs",
            run_name=run_name,
            epochs=int(recipe.get("epochs") or 15),
            imgsz=int(recipe.get("imgsz") or 1280),
            batch=int(recipe.get("batch") or 1),
            seed=int(recipe.get("seed") or 0),
        )
        if result.get("status") != "ok":
            with log_path.open("w", encoding="utf-8") as f:
                f.write(json.dumps(result, indent=2) + "\n")
            return int(result.get("returncode") or 1)
        job_context["weights"] = result.get("weights")
        job_context["run_name"] = run_name
        return 0

    if internal == "smoke_hsp_eval":
        return _run_smoke_hsp_eval_stage(
            job=job,
            meta=meta,
            repo_root=repo_root,
            log_path=log_path,
            dry_run=dry_run,
            job_context=job_context,
        )

    if internal in (
        "job_summary",
        "aug_smoke_summary",
        "aug_sweep_summary",
        "generic_train_summary",
        "rtdetr_summary",
        "vram_probe_summary",
    ):
        stage_meta = dict(meta)
        if internal == "vram_probe_summary" and "summary_kind" not in stage_meta:
            stage_meta["summary_kind"] = "vram_probe"
        elif internal == "aug_smoke_summary":
            stage_meta.setdefault("summary_kind", "aug_smoke")
        elif internal == "aug_sweep_summary":
            stage_meta.setdefault("summary_kind", "aug_sweep")
        elif internal == "rtdetr_summary":
            stage_meta.setdefault("summary_kind", "rtdetr")
        elif internal == "generic_train_summary":
            stage_meta.setdefault("summary_kind", "generic")
        return _finalize_job_summary(
            repo_root=repo_root,
            job=job,
            job_context=job_context,
            meta=stage_meta,
            dry_run=dry_run,
        )

    with log_path.open("w") as f:
        f.write(f"unknown internal stage: {internal}\n")
    return 1


def run_job(
    job: dict[str, Any],
    *,
    repo_root: Path,
    defaults: dict[str, Any],
    dry_run: bool,
    min_free_mib: int,
    log_root: Path,
) -> dict[str, Any]:
    job_id = str(job.get("id") or "unknown")
    _validate_job_files(job, repo_root)
    stages = build_job_stages(job, repo_root=repo_root, defaults=defaults)
    job_env = {
        **dict(os.environ),
        **{k: str(v) for k, v in (defaults.get("env") or {}).items()},
        **{k: str(v) for k, v in (job.get("env") or {}).items()},
        "HARCHOC_GPU_QUEUE_CHILD": "1",
        "HARCHOC_GPU_QUEUE_JOB_ID": job_id,
    }
    job_context: dict[str, Any] = {"run_name": job.get("run_name")}
    stage_results: list[dict[str, Any]] = []
    train_start: float | None = None

    for stage in stages:
        stage_id = str(stage.get("stage_id") or "unknown")
        log_path = log_root / job_id / f"{stage_id}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        if stage.get("internal"):
            if stage_id == "train" or stage_id.startswith("train_"):
                train_start = time.monotonic()
            rc = _run_internal_stage(
                stage,
                job=job,
                repo_root=repo_root,
                log_path=log_path,
                dry_run=dry_run,
                job_context=job_context,
                min_free_mib=min_free_mib,
            )
            if stage_id == "train" or stage_id.startswith("train_"):
                if train_start is not None and not dry_run:
                    job_context["train_runtime_s"] = time.monotonic() - train_start
        else:
            argv = stage.get("argv") or []
            if stage_id == "train":
                train_start = time.monotonic()
                run_name = str(job_context.get("run_name") or job.get("run_name") or "")
                if (
                    not dry_run
                    and run_name
                    and resolve_train_weights(repo_root=repo_root, run_name=run_name) is not None
                ):
                    w = resolve_train_weights(repo_root=repo_root, run_name=run_name)
                    job_context["weights"] = str(w)
                    log_path.write_text(f"train skipped: weights exist ({w})\n", encoding="utf-8")
                    stage_results.append(
                        {"stage_id": stage_id, "exit_code": 0, "log_path": str(log_path), "skipped": True}
                    )
                    if train_start is not None:
                        job_context["train_runtime_s"] = time.monotonic() - train_start
                    continue
            rc = _run_subprocess_stage(
                argv=list(argv),
                mamba=bool(stage.get("mamba", True)),
                repo_root=repo_root,
                log_path=log_path,
                env=job_env,
                dry_run=dry_run,
            )
            if stage_id == "train" and train_start is not None and not dry_run:
                job_context["train_runtime_s"] = time.monotonic() - train_start

        stage_results.append({"stage_id": stage_id, "exit_code": rc, "log_path": str(log_path)})
        if rc != 0:
            hint = ""
            if "OOM" in "\n".join(_tail_log(log_path)):
                hint = "CUDA OOM — ensure exclusive GPU (wait for prior job)"
            raise GpuQueueError(
                job_id=job_id,
                stage_id=stage_id,
                exit_code=rc,
                log_path=str(log_path),
                hint=hint,
            )

    _maybe_refresh_aug_leaderboard(repo_root=repo_root, job=job, dry_run=dry_run)
    status = "dry_run_complete" if dry_run else "complete"
    return {"job_id": job_id, "status": status, "stages": stage_results, "context": job_context}


def load_run_state(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"schema_version": GPU_QUEUE_RUN_SCHEMA, "completed": [], "failed": None}
    return _read_json(path)


def repair_resume_state(
    state: dict[str, Any],
    *,
    manifest_path: Path,
    dry_run: bool,
) -> dict[str, Any]:
    """Drop stale fields when resuming a different manifest or switching dry_run → live."""
    manifest = str(manifest_path.resolve())
    out = dict(state)
    manifest_changed = str(out.get("manifest") or "") != manifest
    dry_mismatch = bool(out.get("dry_run")) != bool(dry_run)
    if manifest_changed:
        out["completed"] = []
        out["skipped"] = []
        out["failed"] = None
        out["started_at"] = _utc_now()
    if manifest_changed or dry_mismatch or out.get("finished_at"):
        out["manifest"] = manifest
        out["dry_run"] = dry_run
        out["finished_at"] = None
    return out


def save_run_state(path: Path, state: dict[str, Any]) -> None:
    state["schema_version"] = GPU_QUEUE_RUN_SCHEMA
    state["updated_at"] = _utc_now()
    _write_json(path, state)


def run_gpu_queue(
    manifest_path: str | Path,
    *,
    repo_root: str | Path | None = None,
    dry_run: bool = False,
    resume: bool = False,
    job_filter: str | None = None,
    state_path: str | Path = DEFAULT_STATE_PATH,
    min_free_mib: int = DEFAULT_MIN_FREE_MIB,
) -> int:
    rr = Path(repo_root or ".").expanduser().resolve()
    manifest = load_gpu_queue_manifest(manifest_path, repo_root=rr)
    defaults = manifest.get("defaults") or {}
    jobs: list[dict[str, Any]] = list(manifest.get("jobs") or [])
    if job_filter:
        jobs = [j for j in jobs if str(j.get("id")) == job_filter]

    st_path = rr / state_path
    if resume:
        state = repair_resume_state(
            load_run_state(st_path),
            manifest_path=Path(manifest_path).resolve(),
            dry_run=dry_run,
        )
    else:
        state = {
            "schema_version": GPU_QUEUE_RUN_SCHEMA,
            "manifest": str(Path(manifest_path).resolve()),
            "started_at": _utc_now(),
            "dry_run": dry_run,
            "completed": [],
            "skipped": [],
            "failed": None,
            "current_job": None,
        }
    if not resume:
        state["started_at"] = _utc_now()
        state["dry_run"] = dry_run

    log_root = rr / DEFAULT_LOG_ROOT
    completed_ids = set(state.get("completed") or [])

    if not dry_run:
        acquire_gpu_exclusive(repo_root=rr, owner="gpu_queue")

    try:
        for job in jobs:
            job_id = str(job.get("id") or "")
            if job_id in completed_ids:
                continue
            skip, reason = should_skip_job(job, repo_root=rr)
            if skip:
                skipped = state.setdefault("skipped", [])
                if not any(s.get("job_id") == job_id for s in skipped):
                    skipped.append({"job_id": job_id, "reason": reason})
                save_run_state(st_path, state)
                print(f"skip {job_id}: {reason}")
                continue
            state["skipped"] = [s for s in (state.get("skipped") or []) if s.get("job_id") != job_id]

            state["current_job"] = job_id
            save_run_state(st_path, state)
            if not dry_run:
                print(f"=== job {job_id} ({job.get('kind')}) ===")

            try:
                if str(job.get("kind")) == "sg_smoke" and _sg_smoke_requires_supergradients(job, repo_root=rr):
                    try:
                        import super_gradients  # noqa: F401
                    except ImportError:
                        state.setdefault("skipped", []).append(
                            {"job_id": job_id, "reason": "super_gradients not installed"}
                        )
                        save_run_state(st_path, state)
                        print(f"skip {job_id}: super_gradients not installed")
                        continue

                result = run_job(
                    job,
                    repo_root=rr,
                    defaults=defaults,
                    dry_run=dry_run,
                    min_free_mib=min_free_mib,
                    log_root=log_root,
                )
                if result.get("status") != "dry_run_complete":
                    state.setdefault("completed", []).append(job_id)
                state["current_job"] = None
                if not dry_run:
                    job_out = rr / DEFAULT_JOBS_ROOT / f"{job_id}.json"
                    _write_json(job_out, result)
                notify_queue_job(
                    repo_root=rr,
                    job=job,
                    status="complete",
                    dry_run=dry_run,
                )
                save_run_state(st_path, state)
            except GpuQueueError as ex:
                state["failed"] = {
                    "job_id": ex.job_id,
                    "stage_id": ex.stage_id,
                    "exit_code": ex.exit_code,
                    "log_path": ex.log_path,
                    "hint": ex.hint,
                    "tail": _tail_log(Path(ex.log_path)),
                }
                state["current_job"] = None
                fail_out = rr / DEFAULT_JOBS_ROOT / f"{ex.job_id}.json"
                _write_json(
                    fail_out,
                    {
                        "job_id": ex.job_id,
                        "status": "failed",
                        "stage_id": ex.stage_id,
                        "exit_code": ex.exit_code,
                        "log_path": ex.log_path,
                        "hint": ex.hint,
                    },
                )
                notify_queue_job(
                    repo_root=rr,
                    job=job,
                    status="failed",
                    dry_run=dry_run,
                    stage_id=ex.stage_id,
                    exit_code=ex.exit_code,
                    hint=ex.hint or None,
                )
                save_run_state(st_path, state)
                if not dry_run:
                    print(f"FAILED {ex.job_id} @ {ex.stage_id}: {ex.hint or ex.log_path}", file=sys.stderr)
                return 1
            except (FileNotFoundError, ValueError, KeyError) as ex:
                state["failed"] = {"job_id": job_id, "stage_id": "preflight", "error": str(ex)}
                notify_queue_job(
                    repo_root=rr,
                    job=job,
                    status="failed",
                    dry_run=dry_run,
                    stage_id="preflight",
                    hint=str(ex),
                )
                save_run_state(st_path, state)
                if not dry_run:
                    print(f"FAILED {job_id} preflight: {ex}", file=sys.stderr)
                return 1

        state["current_job"] = None
        state["finished_at"] = _utc_now()
        save_run_state(st_path, state)
        notify_queue_manifest_complete(
            repo_root=rr,
            manifest_path=str(manifest_path),
            completed=list(state.get("completed") or []),
            skipped=list(state.get("skipped") or []),
            dry_run=dry_run,
        )
        if dry_run:
            pruned = _prune_dry_run_log_stubs(log_root)
            if pruned:
                print(f"# pruned dry-run log stubs: {', '.join(pruned)}")
        return 0
    finally:
        if not dry_run:
            release_gpu_exclusive(repo_root=rr)
