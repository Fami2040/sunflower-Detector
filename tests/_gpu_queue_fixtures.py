"""Shared helpers for gpu_queue tests."""

from __future__ import annotations

import json
from pathlib import Path


def write_pending_fixture_index(
    repo: Path,
    tmp_dir: Path,
    pending_ids: tuple[str, ...],
    *,
    include_equivalence_classes: bool = True,
) -> str:
    """Clone production index with *pending_ids* as gpu_pending; return repo-relative path."""
    from harchoc.aug_smoke_runner import load_aug_smoke_index

    prod = load_aug_smoke_index(repo / "configs/experiments/aug_smoke_index.json")
    pending_set = {s.upper() for s in pending_ids}
    smokes: list[dict] = []
    for entry in prod.get("smokes") or []:
        e = dict(entry)
        sid = str(e.get("id") or "").upper()
        if sid in pending_set:
            e["status"] = "gpu_pending"
        smokes.append(e)
    obj: dict = {"schema_version": "aug_smoke_index.v1", "smokes": smokes}
    if include_equivalence_classes and prod.get("equivalence_classes"):
        obj["equivalence_classes"] = prod["equivalence_classes"]
    path = tmp_dir / "aug_smoke_index_fixture.json"
    path.write_text(json.dumps(obj), encoding="utf-8")
    return str(path.relative_to(repo))


def load_manifest_with_index(
    repo: Path,
    *,
    template: Path,
    index_rel: str,
    tmp_dir: Path,
) -> dict:
    from harchoc.gpu_queue import load_gpu_queue_manifest

    base = json.loads(template.read_text(encoding="utf-8"))
    base["aug_smoke_index"] = index_rel
    manifest = tmp_dir / "manifest.json"
    manifest.write_text(json.dumps(base), encoding="utf-8")
    return load_gpu_queue_manifest(manifest, repo_root=repo)


def index_entry_recipe_fingerprint(entry: dict, *, repo: Path) -> str:
    from harchoc.aug_smoke_train import resolve_aug_smoke_train_raw
    from harchoc.train_config import effective_train_recipe_fingerprint

    return effective_train_recipe_fingerprint(
        resolve_aug_smoke_train_raw(entry, repo_root=repo),
        repo_root=repo,
    )


def enrich_aug_smoke_job_from_index(job: dict, *, repo: Path) -> dict:
    from harchoc.aug_smoke_runner import find_smoke_entry, load_aug_smoke_index
    from harchoc.aug_smoke_train import (
        resolve_aug_smoke_aug_config,
        resolve_aug_smoke_train_config_path,
    )

    entry = find_smoke_entry(
        load_aug_smoke_index(repo / "configs/experiments/aug_smoke_index.json"),
        str(job.get("smoke_id") or ""),
    )
    enriched = dict(job)
    tc = resolve_aug_smoke_train_config_path(entry, repo_root=repo)
    enriched["train_config"] = tc
    aug = resolve_aug_smoke_aug_config(entry, repo_root=repo, train_config_path=tc)
    if aug:
        enriched["aug_config"] = aug
    return enriched
