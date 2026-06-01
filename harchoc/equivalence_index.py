"""Preds-SHA equivalence index and aug-smoke dedup helpers (queue + leaderboard)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from harchoc.aug_smoke_runner import load_aug_smoke_index
from harchoc.hsp_eval_chain import extract_count_mae
from harchoc.model_zoo import file_sha256

PREDS_DEDUP_TRAIN_KINDS = frozenset({"aug_smoke", "aug_sweep_15", "aug_sweep_100"})


def parse_equivalence_classes(
    index: dict[str, Any],
) -> tuple[set[str], dict[float, str], dict[str, tuple[str, str]]]:
    """Return audit-only smoke_ids, preds SHA by MAE, and audit sid -> (canonical, preds_sha)."""
    equiv = index.get("equivalence_classes") or {}
    audit_only: set[str] = set()
    verified_preds: dict[float, str] = {}
    audit_skip: dict[str, tuple[str, str]] = {}

    for cls in equiv.get("classes") or []:
        smoke_ids = [str(x).upper() for x in (cls.get("smoke_ids") or [])]
        if not smoke_ids:
            continue
        canonical = str(cls.get("canonical_smoke_id") or "").upper()
        if not canonical:
            canonical = "S1" if "S1" in smoke_ids else smoke_ids[0]
        sha = str(cls.get("preds_sha256") or "") if cls.get("preds_sha256") else ""
        for sid in smoke_ids:
            if sid != canonical:
                audit_only.add(sid)
                if sha:
                    audit_skip[sid] = (canonical, sha)
        mae = cls.get("test_count_mae")
        if sha and mae is not None:
            verified_preds[round(float(mae), 12)] = sha

    return audit_only, verified_preds, audit_skip


def audit_only_ids(index: dict[str, Any]) -> set[str]:
    audit_only, _, _ = parse_equivalence_classes(index)
    return audit_only


def preds_sha_from_summary_obj(obj: dict[str, Any]) -> str | None:
    arts = obj.get("artifacts") or {}
    preds = arts.get("preds_json") or {}
    sha = preds.get("sha256")
    return str(sha) if sha else None


def _load_summary(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def preds_sha_from_summary_path(repo_root: Path, summary_rel: str) -> str | None:
    sp = repo_root / summary_rel
    summary = _load_summary(sp)
    if summary is None:
        return None
    sha = preds_sha_from_summary_obj(summary)
    if sha:
        return sha
    run_name = str(summary.get("run_name") or "")
    if not run_name:
        return None
    preds_path = sp.parent / f"{run_name}_preds.json"
    if not preds_path.is_file():
        return None
    return file_sha256(preds_path)


def summary_has_count_mae(obj: dict[str, Any], repo_root: Path) -> bool:
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


def preds_sha_from_verified_summary_path(repo_root: Path, summary_rel: str) -> str | None:
    sp = repo_root / summary_rel
    if not sp.is_file():
        return None
    try:
        obj = _load_summary(sp)
        if obj is None or obj.get("status") != "complete" or not summary_has_count_mae(obj, repo_root):
            return None
        return preds_sha_from_summary_obj(obj)
    except Exception:
        return None


def index_preds_sha_by_smoke_id(index: dict[str, Any]) -> dict[str, str]:
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
    return str(arm.get("id") or "").upper()


def job_summary_path_for_preds_dedup(job: dict[str, Any], *, dedup_id: str) -> str | None:
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


def job_dedup_id(
    job: dict[str, Any],
    *,
    index: dict[str, Any] | None = None,
    repo_root: Path | None = None,
) -> str:
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
        summary = job_summary_path_for_preds_dedup(job, dedup_id="")
        if summary:
            sp = repo_root / summary
            if sp.is_file():
                try:
                    obj = _load_summary(sp)
                    if obj:
                        from_summary = str(obj.get("smoke_id") or "").upper()
                        if from_summary:
                            return from_summary
                except Exception:
                    pass
    return ""


def register_preds_sha_owner(
    owners: dict[str, str],
    *,
    owner_id: str,
    summary_rel: str,
    repo_root: Path,
) -> None:
    sha = preds_sha_from_verified_summary_path(repo_root, summary_rel)
    if sha:
        owners.setdefault(sha, owner_id)


def complete_preds_sha_owners(
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
        register_preds_sha_owner(owners, owner_id=sid, summary_rel=summary, repo_root=repo_root)
    for section in ("sweeps_15ep", "sweeps_100ep"):
        for arm in (index.get(section) or {}).get("arms") or []:
            if str(arm.get("status") or "") != "complete":
                continue
            sid = _sweep_dedup_id_from_arm(arm)
            summary = str(arm.get("summary") or "")
            if not sid or not summary:
                continue
            register_preds_sha_owner(owners, owner_id=sid, summary_rel=summary, repo_root=repo_root)
    return owners


def preds_sha_for_job(
    job: dict[str, Any],
    *,
    repo_root: Path,
    index_preds: dict[str, str],
    index: dict[str, Any] | None = None,
) -> str | None:
    """Known test preds sha for aug smoke/sweep train jobs (index audit or verified summary)."""
    kind = str(job.get("kind") or "")
    if kind not in PREDS_DEDUP_TRAIN_KINDS or bool(job.get("eval_only")):
        return None
    dedup_id = job_dedup_id(job, index=index, repo_root=repo_root)
    if not dedup_id:
        return None
    if dedup_id in index_preds:
        return index_preds[dedup_id]
    summary = job_summary_path_for_preds_dedup(job, dedup_id=dedup_id)
    if summary:
        return preds_sha_from_verified_summary_path(repo_root, summary)
    return None


def _preds_dup_skip_reason(owner: str, sha: str) -> str:
    label = "complete run" if not str(owner).startswith("S") else "complete smoke"
    return f"preds duplicate of {label} {owner} (sha={sha[:12]}...)"


def filter_duplicate_preds_sha(
    jobs: list[dict[str, Any]],
    *,
    repo_root: Path,
    index_path: str = "configs/experiments/aug_smoke_index.json",
) -> list[dict[str, Any]]:
    index = load_aug_smoke_index(repo_root / index_path)
    owners = complete_preds_sha_owners(repo_root=repo_root, index_path=index_path)
    index_preds = index_preds_sha_by_smoke_id(index)
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for job in jobs:
        if job.get("skip") is True:
            out.append(job)
            continue
        sha = preds_sha_for_job(job, repo_root=repo_root, index_preds=index_preds, index=index)
        if sha is None:
            out.append(job)
            continue
        dedup_id = job_dedup_id(job, index=index, repo_root=repo_root)
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
