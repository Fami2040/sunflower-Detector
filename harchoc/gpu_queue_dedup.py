"""GPU queue recipe dedup (preds-SHA dedup lives in ``equivalence_index``)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from harchoc.aug_smoke_runner import find_smoke_entry, load_aug_smoke_index
from harchoc.aug_smoke_train import resolve_aug_smoke_train_raw
from harchoc.equivalence_index import (
    filter_duplicate_preds_sha,
    parse_equivalence_classes,
    summary_has_count_mae,
)

__all__ = [
    "audit_only_equivalence_skip_reason",
    "complete_recipe_owners",
    "filter_duplicate_preds_sha",
    "filter_duplicate_train_recipes",
    "job_train_recipe_fingerprint",
]


from harchoc.json_io import load_json_dict
def job_train_recipe_fingerprint(job: dict[str, Any], *, repo_root: Path) -> str | None:
    kind = str(job.get("kind") or "")
    if kind not in ("aug_smoke", "aug_sweep_15", "aug_sweep_100", "amp_smoke"):
        return None
    if bool(job.get("eval_only")):
        return None
    tc = job.get("train_config")
    aug = job.get("aug_config")
    entry: dict[str, Any] | None = None
    if kind == "aug_smoke":
        try:
            idx = load_aug_smoke_index(
                repo_root / (job.get("aug_index") or "configs/experiments/aug_smoke_index.json")
            )
            entry = find_smoke_entry(idx, str(job.get("smoke_id") or ""))
        except Exception:
            entry = None
    if kind == "aug_smoke" and entry is not None:
        from harchoc.train_config import effective_train_recipe_fingerprint

        cfg = resolve_aug_smoke_train_raw(
            entry,
            repo_root=repo_root,
            job_train_config=str(tc) if tc else None,
            job_aug_config=str(aug) if aug else None,
        )
        return effective_train_recipe_fingerprint(cfg, repo_root=repo_root)
    if not tc:
        return None
    from harchoc.train_config import job_train_recipe_fingerprint

    return job_train_recipe_fingerprint(
        repo_root=repo_root,
        train_config=str(tc),
        aug_config=str(aug) if aug else None,
    )


def complete_recipe_owners(
    *,
    repo_root: Path,
    index_path: str = "configs/experiments/aug_smoke_index.json",
) -> dict[str, str]:
    from harchoc.train_config import effective_train_recipe_fingerprint

    owners: dict[str, str] = {}
    index = load_aug_smoke_index(repo_root / index_path)
    for entry in index.get("smokes") or []:
        if str(entry.get("status") or "") != "complete":
            continue
        sid = str(entry.get("id") or "").upper()
        summary = str(entry.get("summary") or f"reports/aug_smoke/{sid.lower()}_summary.json")
        sp = repo_root / summary
        if not sp.is_file():
            continue
        try:
            obj = load_json_dict(sp)
            if obj.get("status") != "complete" or not summary_has_count_mae(obj, repo_root):
                continue
        except Exception:
            continue
        fp = effective_train_recipe_fingerprint(
            resolve_aug_smoke_train_raw(entry, repo_root=repo_root),
            repo_root=repo_root,
        )
        owners.setdefault(fp, sid)
    return owners


def filter_duplicate_train_recipes(
    jobs: list[dict[str, Any]],
    *,
    repo_root: Path,
) -> list[dict[str, Any]]:
    owners = complete_recipe_owners(repo_root=repo_root)
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for job in jobs:
        fp = job_train_recipe_fingerprint(job, repo_root=repo_root)
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


def audit_only_equivalence_skip_reason(*, canonical: str, preds_sha: str) -> str:
    return (
        f"audit-only equivalence class (canonical {canonical}; "
        f"preds_sha256={preds_sha[:12]}...)"
    )
