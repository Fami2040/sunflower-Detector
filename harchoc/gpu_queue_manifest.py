"""GPU queue manifest load/expand (aug smoke index parity)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from harchoc.aug_smoke_runner import load_aug_smoke_index
from harchoc.equivalence_index import parse_equivalence_classes
from harchoc.gpu_queue_dedup import (
    audit_only_equivalence_skip_reason,
    complete_recipe_owners,
    filter_duplicate_preds_sha,
    filter_duplicate_train_recipes,
    job_train_recipe_fingerprint,
)
from harchoc.retrain_baseline_dedup import annotate_train_compare_dedup_skips

GPU_QUEUE_MANIFEST_SCHEMA = "gpu_queue_manifest.v1"
AUG_SMOKE_PENDING_STATUSES = frozenset({"gpu_pending"})


from harchoc.json_io import load_json_dict


def _resolve_repo_root_from_manifest(manifest_path: Path) -> Path:
    """Walk parents until repo markers (supports manifests under configs/experiments/archive/)."""
    cand = manifest_path.parent
    for _ in range(8):
        if (cand / "scripts" / "experiment.py").is_file() or (cand / "data" / "manifest.json").is_file():
            return cand.resolve()
        if cand.parent == cand:
            break
        cand = cand.parent
    return manifest_path.parent.parent.parent.resolve()


def expand_aug_smoke_jobs_from_index(
    *,
    repo_root: Path,
    index_path: str = "configs/experiments/aug_smoke_index.json",
    statuses: frozenset[str] | None = None,
) -> list[dict[str, Any]]:
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
                job["skip_reason"] = audit_only_equivalence_skip_reason(
                    canonical=canonical, preds_sha=sha
                )
            elif canonical:
                job["skip_reason"] = f"audit-only equivalence class (canonical {canonical})"
            else:
                job["skip_reason"] = "audit-only equivalence class"
        jobs.append(job)
    return jobs


def merge_aug_smoke_jobs(
    jobs: list[dict[str, Any]],
    *,
    repo_root: Path,
    index_path: str,
) -> list[dict[str, Any]]:
    expanded = expand_aug_smoke_jobs_from_index(repo_root=repo_root, index_path=index_path)
    owners = complete_recipe_owners(repo_root=repo_root, index_path=index_path)
    filtered: list[dict[str, Any]] = []
    for job in expanded:
        fp = job_train_recipe_fingerprint(job, repo_root=repo_root)
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


def load_gpu_queue_manifest(
    path: str | Path,
    *,
    repo_root: str | Path | None = None,
) -> dict[str, Any]:
    p = Path(path).expanduser().resolve()
    obj = load_json_dict(p)
    schema = obj.get("schema_version")
    if schema != GPU_QUEUE_MANIFEST_SCHEMA:
        raise ValueError(
            f"unsupported gpu queue manifest schema: {schema!r} "
            f"(expected {GPU_QUEUE_MANIFEST_SCHEMA!r})"
        )
    out = dict(obj)
    out["manifest_path"] = str(p)
    jobs = list(out.get("jobs") or [])

    if repo_root is not None:
        rr = Path(repo_root).expanduser().resolve()
    else:
        rr = _resolve_repo_root_from_manifest(p)
    idx = str(out.get("aug_smoke_index") or "configs/experiments/aug_smoke_index.json")
    if out.get("aug_smoke_from_index"):
        jobs = merge_aug_smoke_jobs(jobs, repo_root=rr, index_path=idx)
    defs = out.get("defaults") or {}
    jobs = filter_duplicate_train_recipes(jobs, repo_root=rr)
    jobs = annotate_train_compare_dedup_skips(jobs, repo_root=rr, defaults=defs)
    out["jobs"] = filter_duplicate_preds_sha(jobs, repo_root=rr, index_path=idx)
    return out
