from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from harchoc.bench_config import BenchConfig, bench_external_provenance, select_backend
from harchoc.detector_sources import (
    collect_bench_external_source_ids,
    default_detector_sources_path,
    default_external_weights_dir,
    download_external_checkpoint,
    load_detector_sources,
)
from harchoc.external_repos import ensure_external_repos_for_registry
from harchoc.model_zoo import (
    _default_cache_dir,
    collect_ultralytics_identifiers,
    default_weights_manifest_path,
    download_hints_for_cache_path,
    download_ultralytics_weight,
    file_sha256,
    identifiers_missing_from_manifest,
    load_weights_manifest,
    resolve_weights_ref,
    sync_weights_manifest_from_prep_report,
)
from harchoc.strict_ml import capture_failure

WEIGHTS_PREP_REPORT_SCHEMA = "weights_cache_report.v1"


@dataclass(frozen=True)
class RequiredAssets:
    """Assets referenced by a bench directory or explicit config list."""

    ultralytics_ids: dict[str, list[str]]
    external_source_ids: dict[str, list[str]]
    train_stacks: frozenset[str]


def _is_bench_row_path(p: Path) -> bool:
    """Skip include-only fragments (e.g. ``_defaults.yaml``)."""
    return not p.name.startswith("_")


def iter_bench_config_paths(
    *,
    bench_dir: Path,
    pattern: str = "*.yaml",
    bench_config_paths: list[Path] | None = None,
) -> list[Path]:
    if bench_config_paths is None:
        paths = sorted(bench_dir.glob(pattern))
        if pattern == "*.yaml":
            paths.extend(sorted(bench_dir.glob("*.json")))
        return sorted({p.resolve() for p in paths if _is_bench_row_path(p)})
    return [p.resolve() for p in bench_config_paths]


def collect_required_assets(
    bench_dir: Path,
    *,
    pattern: str = "*.yaml",
    bench_config_paths: list[Path] | None = None,
) -> RequiredAssets:
    """
    Collect ultralytics weight ids, external source_ids, and train stacks for a bench set.
    """
    paths = iter_bench_config_paths(
        bench_dir=bench_dir, pattern=pattern, bench_config_paths=bench_config_paths
    )
    ultralytics = collect_ultralytics_identifiers(
        bench_dir=bench_dir, pattern=pattern, bench_config_paths=paths
    )
    external = collect_bench_external_source_ids(
        bench_dir=bench_dir, bench_config_paths=paths
    )
    registry = load_detector_sources()
    stacks: set[str] = set()
    for sid in external:
        entry = registry.get(sid)
        if entry is not None:
            stacks.add(entry.train_stack)
    return RequiredAssets(
        ultralytics_ids=ultralytics,
        external_source_ids=external,
        train_stacks=frozenset(stacks),
    )


def resolve_asset_ref(cfg: BenchConfig, *, backend: str | None = None) -> dict[str, object]:
    """
    Stable asset resolution for matrix / prep (no downloads).

    Unifies ultralytics, supergradients, and external checkpoint refs.
    """
    backend = backend or select_backend(cfg)
    if backend in ("ultralytics", "supergradients"):
        res = resolve_weights_ref(backend=backend, model=cfg.model, model_id=cfg.model_id)  # type: ignore[arg-type]
        return res.to_json()
    if backend == "external":
        prov = bench_external_provenance(cfg)
        if prov is None:
            requested = cfg.source_id or cfg.model_id
            return {
                "backend": backend,
                "requested": requested,
                "resolved_path": None,
                "exists": False,
                "resolution": "unknown_source",
            }
        exists = bool(prov.get("exists"))
        return {
            "backend": backend,
            "requested": prov.get("source_id"),
            "kind": "external_checkpoint",
            "cache_path": prov["cache_path"],
            "resolved_path": prov["cache_path"] if exists else None,
            "exists": exists,
            "resolution": "cached" if exists else "not_cached",
            "train_stack": prov["train_stack"],
            "config_relpath": prov["config_relpath"],
            "checkpoint_url": prov["checkpoint_url"],
            "repos": prov["repos"],
        }
    requested: str | None = cfg.model or cfg.model_id
    return {"requested": requested, "resolved_path": None, "exists": None, "resolution": "not_applicable"}


def download_missing_ultralytics_weights(
    *,
    by_id: dict[str, list[str]],
) -> list[dict[str, object]]:
    results: list[dict[str, object]] = []
    for ident in sorted(by_id.keys()):
        res = resolve_weights_ref(backend="ultralytics", model=ident)
        if res.exists and res.cache_path is not None:
            results.append(
                {
                    "identifier": ident,
                    "downloaded": False,
                    "cache_path": str(res.cache_path),
                    "sha256": file_sha256(res.cache_path),
                    "size_bytes": res.cache_path.stat().st_size,
                }
            )
            continue
        if res.cache_path is None:
            results.append({"identifier": ident, "downloaded": False, "error": "no_cache_path"})
            continue
        with capture_failure(f"download:{ident}") as cap:
            info = download_ultralytics_weight(identifier=ident, cache_path=res.cache_path)
        if cap.failed:
            results.append(
                {
                    "identifier": ident,
                    "downloaded": False,
                    "error": cap.exc_msg,
                    "error_type": cap.exc_type,
                    "download_hints": download_hints_for_cache_path(
                        cache_path=res.cache_path, identifier=ident
                    ),
                }
            )
        else:
            results.append(info)
    return results


def build_external_cache_report(
    *,
    by_id: dict[str, list[str]] | None = None,
    bench_dir: Path | None = None,
    bench_config_paths: list[Path] | None = None,
    download: bool = False,
) -> dict[str, object]:
    if by_id is None:
        if bench_dir is None:
            raise ValueError("bench_dir or by_id required")
        by_id = collect_bench_external_source_ids(
            bench_dir=bench_dir, bench_config_paths=bench_config_paths
        )
    registry = load_detector_sources()
    entries: list[dict[str, object]] = []
    for source_id in sorted(by_id.keys()):
        entry = registry.get(source_id)
        if entry is None:
            entries.append(
                {
                    "source_id": source_id,
                    "bench_configs": by_id[source_id],
                    "exists": False,
                    "error": "unknown_source_id",
                }
            )
            continue
        exists = entry.cache_path.is_file() and entry.cache_path.stat().st_size > 0
        row: dict[str, object] = {
            "source_id": source_id,
            "label": entry.label,
            "train_stack": entry.train_stack,
            "bench_configs": by_id[source_id],
            "cache_path": str(entry.cache_path),
            "exists": exists,
            "checkpoint_url": entry.checkpoint_url,
            "repos": entry.repos,
            "config_relpath": entry.config_relpath,
        }
        if download and not exists:
            with capture_failure(f"download_external:{source_id}") as cap:
                dl = download_external_checkpoint(entry)
            if cap.failed:
                row["download"] = {
                    "downloaded": False,
                    "error": cap.exc_msg,
                    "error_type": cap.exc_type,
                }
            else:
                row["download"] = dl
                exists = entry.cache_path.is_file()
                row["exists"] = exists
                if exists:
                    row["size_bytes"] = entry.cache_path.stat().st_size
                    row["sha256"] = file_sha256(entry.cache_path)
        elif exists:
            row["sha256"] = file_sha256(entry.cache_path)
            row["size_bytes"] = entry.cache_path.stat().st_size
        entries.append(row)
    missing = sum(1 for e in entries if not e.get("exists"))
    return {
        "cache_dir": str(default_external_weights_dir()),
        "sources_registry": str(default_detector_sources_path()),
        "entries": entries,
        "summary": {"total": len(entries), "cached": len(entries) - missing, "missing": missing},
    }


def build_ultralytics_cache_report(
    *,
    by_id: dict[str, list[str]],
    download: bool = False,
) -> tuple[list[dict[str, object]], dict[str, int]]:
    download_results: list[dict[str, object]] | None = None
    if download:
        download_results = download_missing_ultralytics_weights(by_id=by_id)

    entries: list[dict[str, object]] = []
    for ident in sorted(by_id.keys()):
        res = resolve_weights_ref(backend="ultralytics", model=ident)
        exists = bool(res.exists)
        entry: dict[str, object] = {
            "identifier": ident,
            "bench_configs": by_id[ident],
            "cache_path": str(res.cache_path) if res.cache_path is not None else None,
            "exists": exists,
            "resolution": res.resolution,
        }
        if res.cache_path is not None and exists:
            entry["sha256"] = file_sha256(res.cache_path)
            entry["size_bytes"] = res.cache_path.stat().st_size
        if res.cache_path is not None and not exists:
            entry["download_hints"] = download_hints_for_cache_path(
                cache_path=res.cache_path, identifier=ident
            )
        if download_results is not None:
            dl = next((d for d in download_results if d.get("identifier") == ident), None)
            if dl is not None:
                entry["download"] = dl
                if dl.get("downloaded") and res.cache_path is not None:
                    entry["exists"] = res.cache_path.is_file()
                    if entry["exists"]:
                        entry["sha256"] = file_sha256(res.cache_path)
                        entry["size_bytes"] = res.cache_path.stat().st_size
        entries.append(entry)

    missing = sum(1 for e in entries if not e.get("exists"))
    summary = {
        "total": len(entries),
        "cached": len(entries) - missing,
        "missing": missing,
    }
    return entries, summary


def missing_from_manifest(
    *,
    required: RequiredAssets,
    manifest_path: Path,
) -> tuple[list[str], list[str]]:
    """Return (ultralytics_ids, external_source_ids) missing from manifest entries."""
    manifest = load_weights_manifest(manifest_path)
    entries = manifest.get("entries")
    if not isinstance(entries, dict):
        entries = {}
    ul_missing = sorted(i for i in required.ultralytics_ids if i not in entries)
    ext_missing = sorted(s for s in required.external_source_ids if s not in entries)
    return ul_missing, ext_missing


def build_weights_prep_report(
    *,
    bench_dir: Path,
    pattern: str = "*.yaml",
    bench_config_paths: list[Path] | None = None,
    download: bool = False,
    manifest_path: Path | None = None,
    check_manifest: bool = False,
    include_external: bool = True,
) -> dict[str, object]:
    """
    Prep report: Ultralytics cache, optional external checkpoints, and upstream repos.
    """
    required = collect_required_assets(
        bench_dir, pattern=pattern, bench_config_paths=bench_config_paths
    )
    cache_dir = _default_cache_dir()
    identifiers, ul_summary = build_ultralytics_cache_report(
        by_id=required.ultralytics_ids, download=download
    )

    manifest_missing_ul: list[str] = []
    manifest_missing_ext: list[str] = []
    if check_manifest and manifest_path is not None:
        manifest_missing_ul, manifest_missing_ext = missing_from_manifest(
            required=required, manifest_path=manifest_path
        )

    report: dict[str, object] = {
        "schema_version": WEIGHTS_PREP_REPORT_SCHEMA,
        "cache_dir": str(cache_dir),
        "ultralytics_cache_dir": str(cache_dir / "ultralytics"),
        "bench_dir": str(bench_dir.resolve()),
        "required": {
            "ultralytics_ids": sorted(required.ultralytics_ids.keys()),
            "external_source_ids": sorted(required.external_source_ids.keys()),
            "train_stacks": sorted(required.train_stacks),
        },
        "identifiers": identifiers,
        "summary": {
            **ul_summary,
            "missing_from_manifest": len(manifest_missing_ul) + len(manifest_missing_ext),
        },
        "would_download": bool(download),
    }

    missing_manifest_rows: list[dict[str, object]] = []
    for ident in manifest_missing_ul:
        missing_manifest_rows.append(
            {"kind": "ultralytics", "key": ident, "bench_configs": required.ultralytics_ids[ident]}
        )
    for sid in manifest_missing_ext:
        missing_manifest_rows.append(
            {
                "kind": "external",
                "key": sid,
                "bench_configs": required.external_source_ids[sid],
            }
        )
    if missing_manifest_rows:
        report["missing_from_manifest"] = missing_manifest_rows
        report["weights_manifest_path"] = str(manifest_path.resolve())  # type: ignore[union-attr]

    if include_external:
        external_source_ids = sorted(required.external_source_ids.keys())
        manifest_repos: dict[str, Any] = {}
        if manifest_path is not None and manifest_path.is_file():
            loaded = load_weights_manifest(manifest_path)
            er = loaded.get("external_repos")
            if isinstance(er, dict):
                manifest_repos = er
        report["external_repos"] = ensure_external_repos_for_registry(
            download=download,
            source_ids=external_source_ids,
            manifest_repos=manifest_repos,
        )
        report["external"] = build_external_cache_report(
            by_id=required.external_source_ids,
            download=download,
        )
        ext_summary = report["external"]["summary"]  # type: ignore[index]
        if isinstance(ext_summary, dict):
            report["summary"] = {  # type: ignore[assignment]
                **report["summary"],  # type: ignore[misc]
                "external_missing": int(ext_summary.get("missing", 0)),
            }

    if manifest_path is not None and download:
        sync_weights_manifest_from_prep_report(manifest_path, report)
        report["manifest_path"] = str(manifest_path.resolve())

    return report


# Back-compat alias for tests and callers.
build_report = build_weights_prep_report

__all__ = [
    "RequiredAssets",
    "WEIGHTS_PREP_REPORT_SCHEMA",
    "build_external_cache_report",
    "build_report",
    "build_ultralytics_cache_report",
    "build_weights_prep_report",
    "collect_required_assets",
    "default_weights_manifest_path",
    "download_missing_ultralytics_weights",
    "iter_bench_config_paths",
    "missing_from_manifest",
    "resolve_asset_ref",
]
