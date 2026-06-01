from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from harchoc.detector_sources import (
    DetectorSourceEntry,
    default_detector_sources_path,
    load_detector_sources,
    load_train_stack_metadata,
)


EXTERNAL_REPOS_SCHEMA = "external_repos.v1"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def default_external_repos_path() -> Path:
    return _repo_root() / "configs" / "external" / "external_repos.v1.json"


def default_external_root() -> Path:
    raw = os.getenv("HARCHOC_EXTERNAL_ROOT", "external")
    p = Path(raw).expanduser()
    if not p.is_absolute():
        p = (_repo_root() / p)
    return p.resolve()


_STACK_ENV_OVERRIDE: dict[str, str] = {
    "deim": "HARCHOC_DEIM_REPO",
    "dfine": "HARCHOC_DFINE_REPO",
    "rtdetrv2_pytorch": "HARCHOC_RTDETR_REPO",
}


@dataclass(frozen=True)
class ExternalRepoSpec:
    key: str
    train_stack: str
    url: str
    ref: str
    cache_dirname: str
    train_script: str
    subdir: str | None
    source_ids: tuple[str, ...]
    commit_pin: str | None

    def clone_root(self, *, external_root: Path | None = None) -> Path:
        root = external_root if external_root is not None else default_external_root()
        return (root / self.cache_dirname).resolve()

    def working_dir(self, *, external_root: Path | None = None) -> Path:
        base = self.clone_root(external_root=external_root)
        if self.subdir:
            return (base / self.subdir).resolve()
        return base


def _normalize_git_url(url: str) -> str:
    u = url.strip().rstrip("/")
    if u.endswith(".git"):
        return u
    return f"{u}.git"


def derive_external_repo_specs(
    *,
    registry: dict[str, DetectorSourceEntry] | None = None,
    sources_path: Path | None = None,
) -> dict[str, ExternalRepoSpec]:
    """
    Build clone specs from unique ``train_stack`` in detector_sources plus ``train_stacks`` metadata.
    """
    path = sources_path if sources_path is not None else default_detector_sources_path()
    registry = registry if registry is not None else load_detector_sources(path)
    stack_meta = load_train_stack_metadata(path)

    by_stack: dict[str, list[str]] = {}
    stack_urls: dict[str, str] = {}
    for source_id, entry in registry.items():
        stack = entry.train_stack
        if not stack:
            continue
        by_stack.setdefault(stack, []).append(source_id)
        if stack not in stack_urls:
            primary = str(entry.repos.get("primary") or "").strip()
            if primary:
                stack_urls[stack] = primary

    specs: dict[str, ExternalRepoSpec] = {}
    for train_stack in sorted(by_stack.keys()):
        meta = stack_meta.get(train_stack)
        if not isinstance(meta, dict):
            raise ValueError(
                f"Missing train_stacks.{train_stack} in {path} "
                f"(required for source_ids={by_stack[train_stack]})"
            )
        url = str(meta.get("url") or stack_urls.get(train_stack) or "").strip()
        if not url:
            raise ValueError(f"No repo url for train_stack={train_stack!r} in {path}")
        subdir_raw = meta.get("subdir")
        subdir: str | None = str(subdir_raw).strip() if subdir_raw else None
        if not subdir:
            for sid in by_stack[train_stack]:
                entry = registry[sid]
                sub = str(entry.repos.get("subdir") or "").strip()
                if sub:
                    subdir = sub
                    break
        pin = meta.get("commit_pin")
        specs[train_stack] = ExternalRepoSpec(
            key=train_stack,
            train_stack=train_stack,
            url=_normalize_git_url(url),
            ref=str(meta.get("ref") or "main"),
            cache_dirname=str(meta.get("cache_dirname") or train_stack),
            train_script=str(meta.get("train_script") or "train.py"),
            subdir=subdir,
            source_ids=tuple(sorted(by_stack[train_stack])),
            commit_pin=str(pin).strip() if pin else None,
        )
    return specs


def write_external_repos_manifest(
    path: Path | None = None,
    *,
    sources_path: Path | None = None,
) -> Path:
    """Write generated external_repos.v1.json mirror from detector_sources (optional artifact)."""
    out = path if path is not None else default_external_repos_path()
    specs = derive_external_repo_specs(sources_path=sources_path)
    payload: dict[str, object] = {
        "schema_version": EXTERNAL_REPOS_SCHEMA,
        "notes": (
            "GENERATED from configs/external/detector_sources.v1.json train_stacks + entries. "
            "Do not edit by hand; refresh via: "
            "python scripts/check_weights_cache.py --sync-repos-manifest"
        ),
        "repos": {
            spec.key: {
                "train_stack": spec.train_stack,
                "url": spec.url,
                "ref": spec.ref,
                "cache_dirname": spec.cache_dirname,
                "train_script": spec.train_script,
                **({"subdir": spec.subdir} if spec.subdir else {}),
                "source_ids": list(spec.source_ids),
                **({"commit_pin": spec.commit_pin} if spec.commit_pin else {}),
            }
            for spec in sorted(specs.values(), key=lambda s: s.key)
        },
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return out


def load_external_repo_specs(path: Path | None = None) -> dict[str, ExternalRepoSpec]:
    return derive_external_repo_specs()


def spec_for_train_stack(
    train_stack: str,
    *,
    specs: dict[str, ExternalRepoSpec] | None = None,
) -> ExternalRepoSpec | None:
    specs = specs if specs is not None else load_external_repo_specs()
    for spec in specs.values():
        if spec.train_stack == train_stack:
            return spec
    return None


def resolve_external_repo_path(
    train_stack: str,
    *,
    external_root: Path | None = None,
    specs: dict[str, ExternalRepoSpec] | None = None,
) -> Path | None:
    env_key = _STACK_ENV_OVERRIDE.get(train_stack)
    if env_key:
        raw = (os.getenv(env_key) or "").strip()
        if raw:
            p = Path(raw).expanduser().resolve()
            if p.is_dir():
                return p
    spec = spec_for_train_stack(train_stack, specs=specs)
    if spec is None:
        return None
    work = spec.working_dir(external_root=external_root)
    return work if work.is_dir() else None


def _git_head(repo_dir: Path) -> str | None:
    if not (repo_dir / ".git").exists():
        return None
    proc = subprocess.run(
        ["git", "-C", str(repo_dir), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        return None
    return proc.stdout.strip() or None


def validate_repo_layout(
    spec: ExternalRepoSpec,
    *,
    registry: dict[str, DetectorSourceEntry] | None = None,
    external_root: Path | None = None,
    working_dir: Path | None = None,
) -> list[str]:
    issues: list[str] = []
    work = (
        working_dir.resolve()
        if working_dir is not None
        else spec.working_dir(external_root=external_root)
    )
    if not work.is_dir():
        issues.append(f"missing_repo_dir:{work}")
        return issues
    train_py = work / spec.train_script
    if not train_py.is_file():
        issues.append(f"missing_train_script:{train_py}")
    registry = registry if registry is not None else load_detector_sources()
    for sid in spec.source_ids:
        entry = registry.get(sid)
        if entry is None:
            issues.append(f"unknown_source_id:{sid}")
            continue
        cfg = work / entry.config_relpath
        if not cfg.is_file():
            issues.append(f"missing_config:{cfg}")
    return issues


def ensure_external_repo(
    spec: ExternalRepoSpec,
    *,
    download: bool,
    external_root: Path | None = None,
    registry: dict[str, DetectorSourceEntry] | None = None,
    expected_commit: str | None = None,
) -> dict[str, Any]:
    root = external_root if external_root is not None else default_external_root()
    clone_root = spec.clone_root(external_root=root)
    work = spec.working_dir(external_root=root)
    row: dict[str, Any] = {
        "repo_key": spec.key,
        "train_stack": spec.train_stack,
        "url": spec.url,
        "ref": spec.ref,
        "clone_root": str(clone_root),
        "working_dir": str(work),
        "train_script": spec.train_script,
        "source_ids": list(spec.source_ids),
    }

    issues = validate_repo_layout(spec, registry=registry, external_root=root)
    if not issues and work.is_dir():
        head = _git_head(clone_root)
        row["commit"] = head
        row["exists"] = True
        pin = expected_commit or spec.commit_pin
        if pin and head and head != pin:
            issues.append(f"commit_mismatch:want={pin},got={head}")
        row["valid"] = not issues
        if issues:
            row["validation_issues"] = issues
        return row

    if not download:
        row["exists"] = False
        row["valid"] = False
        row["validation_issues"] = issues or [f"missing_repo_dir:{work}"]
        return row

    clone_root.parent.mkdir(parents=True, exist_ok=True)
    if clone_root.exists():
        import shutil

        shutil.rmtree(clone_root)
    cmd = [
        "git",
        "clone",
        "--depth",
        "1",
        "--branch",
        spec.ref,
        spec.url,
        str(clone_root),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    row["clone"] = {
        "downloaded": proc.returncode == 0,
        "returncode": proc.returncode,
        "stderr": (proc.stderr or "")[-2000:],
    }
    if proc.returncode != 0:
        row["exists"] = False
        row["valid"] = False
        row["validation_issues"] = [f"git_clone_failed:{proc.returncode}"]
        return row

    issues = validate_repo_layout(spec, registry=registry, external_root=root)
    head = _git_head(clone_root)
    row["commit"] = head
    row["exists"] = work.is_dir()
    pin = expected_commit or spec.commit_pin
    if pin and head and head != pin:
        issues.append(f"commit_mismatch:want={pin},got={head}")
    row["valid"] = row["exists"] and not issues
    if issues:
        row["validation_issues"] = issues
    return row


def ensure_external_repos_for_registry(
    *,
    download: bool,
    source_ids: list[str] | None = None,
    bench_dir: Path | None = None,
    bench_config_paths: list[Path] | None = None,
    external_root: Path | None = None,
    manifest_repos: dict[str, Any] | None = None,
) -> dict[str, object]:
    specs = load_external_repo_specs()
    registry = load_detector_sources()
    want_stacks: set[str] = set()
    if source_ids:
        for sid in source_ids:
            entry = registry.get(sid)
            if entry is not None:
                want_stacks.add(entry.train_stack)
    elif bench_dir is not None:
        want_stacks = stacks_required_by_bench(
            bench_dir=bench_dir, bench_config_paths=bench_config_paths
        )
    else:
        for spec in specs.values():
            want_stacks.add(spec.train_stack)

    entries: list[dict[str, Any]] = []
    for spec in sorted(specs.values(), key=lambda s: s.key):
        if spec.train_stack not in want_stacks:
            continue
        expected: str | None = None
        manifest_key = spec.key
        if manifest_repos and isinstance(manifest_repos.get(manifest_key), dict):
            mc = manifest_repos[manifest_key].get("commit")
            if isinstance(mc, str) and mc.strip():
                expected = mc.strip()
        entries.append(
            ensure_external_repo(
                spec,
                download=download,
                external_root=external_root,
                registry=registry,
                expected_commit=expected,
            )
        )
    missing = sum(1 for e in entries if not e.get("valid"))
    return {
        "external_root": str(external_root or default_external_root()),
        "repos_manifest": str(default_detector_sources_path()),
        "repos_manifest_generated": str(default_external_repos_path()),
        "entries": entries,
        "summary": {
            "total": len(entries),
            "valid": len(entries) - missing,
            "missing": missing,
        },
    }


def stacks_required_by_bench(
    *,
    bench_dir: Path,
    bench_config_paths: list[Path] | None = None,
) -> set[str]:
    from harchoc.bench_config import load_bench_config, select_backend
    from harchoc.detector_sources import collect_bench_external_source_ids

    by_id = collect_bench_external_source_ids(
        bench_dir=bench_dir, bench_config_paths=bench_config_paths
    )
    registry = load_detector_sources()
    stacks: set[str] = set()
    for sid in by_id:
        entry = registry.get(sid)
        if entry is not None:
            stacks.add(entry.train_stack)
    return stacks
