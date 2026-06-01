from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

WEIGHTS_MANIFEST_SCHEMA = "weights_manifest.v1"


Backend = Literal["ultralytics", "supergradients", "external"]
WeightsKind = Literal["file_path", "ultralytics_id", "supergradients_id", "missing"]


def _default_cache_dir() -> Path:
    # Intentionally under data/ (gitignored) to avoid committing heavy artifacts.
    # Allow override for shared caches / cluster paths.
    raw = os.getenv("WEIGHTS_CACHE_DIR", "data/weights")
    p = Path(raw).expanduser()
    if not p.is_absolute():
        p = (Path.cwd() / p)
    return p.resolve()


def required_module(backend: Backend) -> str:
    if backend == "ultralytics":
        return "ultralytics"
    if backend == "external":
        return ""
    # pip package: super-gradients; import module: super_gradients
    return "super_gradients"


def backend_availability(
    backend: Backend,
    *,
    model_id: str | None = None,
    source_id: str | None = None,
) -> tuple[bool, str | None]:
    """
    Returns (available, reason_if_missing).

    Uses `find_spec` to avoid importing heavy ML stacks in CI/dry-run.
    External backend requires ``source_id`` or ``model_id`` for row-level checks.
    """
    if backend == "external":
        sid = (source_id or model_id or "").strip()
        if not sid:
            return False, "external_requires_source_id"
        from harchoc.external_detector_train import external_bench_availability

        return external_bench_availability(model_id=model_id, source_id=source_id)
    mod = required_module(backend)
    if importlib.util.find_spec(mod) is None:
        return False, f"missing_dependency:{mod}"
    return True, None


@dataclass(frozen=True)
class WeightsResolution:
    backend: Backend
    requested: str | None
    kind: WeightsKind
    # cache_path is deterministic for identifier-based refs.
    cache_path: Path | None
    resolved_path: Path | None
    exists: bool | None
    resolution: str

    def to_json(self) -> dict[str, object]:
        return {
            "backend": self.backend,
            "requested": self.requested,
            "kind": self.kind,
            "cache_path": str(self.cache_path) if self.cache_path is not None else None,
            "resolved_path": str(self.resolved_path) if self.resolved_path is not None else None,
            "exists": self.exists,
            "resolution": self.resolution,
            "cache_dir": str(_default_cache_dir()),
            "would_download": False,
        }


def _looks_like_path(s: str) -> bool:
    # Heuristic: if it contains a path separator, treat it as a file path.
    return ("/" in s) or ("\\" in s)


def resolve_weights_ref(
    *,
    backend: Backend,
    model: str | None,
    model_id: str | None = None,
) -> WeightsResolution:
    """
    Resolve a weights reference without downloading.

    Ultralytics:
    - If `model` is an existing file path (absolute or relative), record it.
    - Else treat it as an identifier and map to a deterministic cache path:
      `WEIGHTS_CACHE_DIR/ultralytics/<identifier>`.

    SuperGradients:
    - `model_id` identifies the architecture; weights handling is backend-specific.
      We record the request but do not resolve to a path here.
    """
    cache_dir = _default_cache_dir()

    if backend == "supergradients":
        requested = model_id or model
        if not requested:
            return WeightsResolution(
                backend=backend,
                requested=None,
                kind="missing",
                cache_path=None,
                resolved_path=None,
                exists=False,
                resolution="missing",
            )
        return WeightsResolution(
            backend=backend,
            requested=str(requested),
            kind="supergradients_id",
            cache_path=None,
            resolved_path=None,
            exists=None,
            resolution="not_applicable",
        )

    # ultralytics
    if not model:
        return WeightsResolution(
            backend=backend,
            requested=None,
            kind="missing",
            cache_path=None,
            resolved_path=None,
            exists=False,
            resolution="missing",
        )

    requested = str(model)
    if _looks_like_path(requested) or requested.startswith((".", "~")):
        p = Path(requested).expanduser()
        if not p.is_absolute():
            p = (Path.cwd() / p)
        p = p.resolve()
        return WeightsResolution(
            backend=backend,
            requested=requested,
            kind="file_path",
            cache_path=None,
            resolved_path=p,
            exists=p.exists(),
            resolution="existing_file" if p.exists() else "missing_file",
        )

    # Treat as identifier, e.g. "yolov8s.pt", "yolo11s.pt", "rtdetr-l.pt".
    cache_path = (cache_dir / "ultralytics" / requested).resolve()
    return WeightsResolution(
        backend=backend,
        requested=requested,
        kind="ultralytics_id",
        cache_path=cache_path,
        resolved_path=cache_path,
        exists=cache_path.exists(),
        resolution="cached" if cache_path.exists() else "not_cached",
    )


# Ultralytics release assets (best-effort hint URLs; never fetched by this repo).
_DEFAULT_ULTRALYTICS_ASSETS_TAG = "v8.4.0"


def ultralytics_assets_url(identifier: str, *, assets_tag: str | None = None) -> str:
    """Public GitHub assets URL for a standard Ultralytics pretrained weight file."""
    name = identifier.strip()
    if not name:
        raise ValueError("identifier must be non-empty")
    tag = assets_tag or os.getenv("HARCHOC_ULTRALYTICS_ASSETS_TAG", _DEFAULT_ULTRALYTICS_ASSETS_TAG)
    return (
        "https://github.com/ultralytics/assets/releases/download/"
        f"{tag}/{name}"
    )


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def download_ultralytics_weight(
    *,
    identifier: str,
    cache_path: Path | None = None,
    assets_tags: tuple[str, ...] | None = None,
) -> dict[str, object]:
    """
    Download a pretrained Ultralytics weight file into the deterministic cache path.

    Tries ``assets_tags`` in order (default: env tag, then v8.3.0, v8.4.0).
    """
    if cache_path is None:
        cache_path = (_default_cache_dir() / "ultralytics" / identifier).resolve()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if cache_path.is_file() and cache_path.stat().st_size > 0:
        return {
            "identifier": identifier,
            "cache_path": str(cache_path),
            "downloaded": False,
            "sha256": file_sha256(cache_path),
            "size_bytes": cache_path.stat().st_size,
        }

    tags = assets_tags
    if tags is None:
        primary = os.getenv("HARCHOC_ULTRALYTICS_ASSETS_TAG", _DEFAULT_ULTRALYTICS_ASSETS_TAG)
        tags = tuple(dict.fromkeys([primary, "v8.4.0", "v8.3.0"]))

    last_err: Exception | None = None
    used_url: str | None = None
    for tag in tags:
        url = ultralytics_assets_url(identifier, assets_tag=tag)
        used_url = url
        try:
            tmp = cache_path.with_suffix(cache_path.suffix + ".part")
            urllib.request.urlretrieve(url, tmp)  # noqa: S310
            tmp.replace(cache_path)
            return {
                "identifier": identifier,
                "cache_path": str(cache_path),
                "downloaded": True,
                "url": url,
                "assets_tag": tag,
                "sha256": file_sha256(cache_path),
                "size_bytes": cache_path.stat().st_size,
            }
        except (OSError, urllib.error.URLError) as exc:
            last_err = exc
            if cache_path.exists():
                cache_path.unlink(missing_ok=True)

    raise RuntimeError(
        f"Failed to download {identifier!r} (last url={used_url!r}): {last_err}"
    ) from last_err


def collect_ultralytics_identifiers(
    *,
    bench_dir: Path,
    pattern: str = "*.yaml",
    bench_config_paths: list[Path] | None = None,
) -> dict[str, list[str]]:
    """
    Return {identifier: [bench_config_basenames,...]} for ultralytics zoo ids.
    """
    from harchoc.bench_config import load_bench_config, select_backend

    if bench_config_paths is None:
        from harchoc.bench_assets import iter_bench_config_paths

        paths = iter_bench_config_paths(bench_dir=bench_dir, pattern=pattern)
    else:
        paths = [p.resolve() for p in bench_config_paths]

    by_id: dict[str, list[str]] = {}
    for pth in paths:
        cfg = load_bench_config(pth)
        if select_backend(cfg) != "ultralytics" or not cfg.model:
            continue
        res = resolve_weights_ref(backend="ultralytics", model=cfg.model)
        if res.kind != "ultralytics_id" or not res.requested:
            continue
        ident = str(res.requested)
        by_id.setdefault(ident, []).append(pth.name)
    return by_id


def default_weights_manifest_path() -> Path:
    return _default_cache_dir() / "weights_manifest.json"


def load_weights_manifest(path: Path) -> dict[str, object]:
    if not path.is_file():
        return {"schema_version": WEIGHTS_MANIFEST_SCHEMA, "entries": {}}
    obj = json.loads(path.read_text("utf-8"))
    if not isinstance(obj, dict):
        return {"schema_version": WEIGHTS_MANIFEST_SCHEMA, "entries": {}}
    entries = obj.get("entries")
    if not isinstance(entries, dict):
        obj["entries"] = {}
    return obj


def write_weights_manifest(path: Path, manifest: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", "utf-8")


def _manifest_entry_from_ultralytics_row(e: dict[str, object]) -> dict[str, object]:
    return {
        "asset_kind": "ultralytics",
        "cache_path": e.get("cache_path"),
        "sha256": e.get("sha256"),
        "size_bytes": e.get("size_bytes"),
        "bench_configs": e.get("bench_configs"),
    }


def _manifest_entry_from_external_row(e: dict[str, object]) -> dict[str, object]:
    return {
        "asset_kind": "external",
        "source_id": e.get("source_id"),
        "cache_path": e.get("cache_path"),
        "sha256": e.get("sha256"),
        "size_bytes": e.get("size_bytes"),
        "bench_configs": e.get("bench_configs"),
        "train_stack": e.get("train_stack"),
    }


def sync_weights_manifest_from_report_entries(
    manifest_path: Path,
    *,
    entries: list[dict[str, object]],
    external_entries: list[dict[str, object]] | None = None,
) -> None:
    """Merge cached ultralytics (and optional external) rows into the manifest."""
    cache_dir = _default_cache_dir()
    manifest = load_weights_manifest(manifest_path)
    manifest_entries = manifest.setdefault("entries", {})
    assert isinstance(manifest_entries, dict)
    for e in entries:
        if not e.get("exists"):
            continue
        ident = str(e["identifier"])
        manifest_entries[ident] = _manifest_entry_from_ultralytics_row(e)
    for e in external_entries or []:
        if not e.get("exists"):
            continue
        sid = str(e.get("source_id") or "")
        if not sid:
            continue
        manifest_entries[sid] = _manifest_entry_from_external_row(e)
    manifest["ultralytics_cache_dir"] = str(cache_dir / "ultralytics")
    from harchoc.detector_sources import default_external_weights_dir

    manifest["external_cache_dir"] = str(default_external_weights_dir())
    manifest["supergradients_note"] = (
        "YOLO-NAS checkpoints are resolved at train time under the run directory "
        "(models.get pretrained_weights=coco); not pre-cached here."
    )
    write_weights_manifest(manifest_path, manifest)


def sync_weights_manifest_from_prep_report(
    manifest_path: Path,
    report: dict[str, object],
) -> None:
    """Merge a weights_cache_report.v1 prep report into weights_manifest.json."""
    from harchoc.detector_sources import default_detector_sources_path

    ultralytics = report.get("identifiers")
    external_block = report.get("external")
    ext_entries: list[dict[str, object]] = []
    if isinstance(external_block, dict):
        raw = external_block.get("entries")
        if isinstance(raw, list):
            ext_entries = [e for e in raw if isinstance(e, dict)]
    sync_weights_manifest_from_report_entries(
        manifest_path,
        entries=[e for e in ultralytics if isinstance(e, dict)] if isinstance(ultralytics, list) else [],
        external_entries=ext_entries,
    )
    manifest = load_weights_manifest(manifest_path)
    ext_repos = report.get("external_repos")
    if isinstance(ext_repos, dict):
        repo_entries = ext_repos.get("entries")
        if isinstance(repo_entries, list):
            manifest["external_repos"] = {
                str(row["repo_key"]): {
                    "train_stack": row.get("train_stack"),
                    "url": row.get("url"),
                    "ref": row.get("ref"),
                    "commit": row.get("commit"),
                    "working_dir": row.get("working_dir"),
                    "clone_root": row.get("clone_root"),
                }
                for row in repo_entries
                if isinstance(row, dict) and row.get("valid")
            }
    manifest["external_repos_note"] = (
        "Cloned by check_weights_cache.py from configs/external/external_repos.v1.json; "
        "git-ignored under external/."
    )
    manifest["external_sources_registry"] = str(default_detector_sources_path())
    write_weights_manifest(manifest_path, manifest)


def identifiers_missing_from_manifest(
    *,
    identifiers: list[str],
    manifest_path: Path,
    source_ids: list[str] | None = None,
) -> list[str]:
    """Return sorted manifest keys absent (ultralytics ids and/or external source_ids)."""
    manifest = load_weights_manifest(manifest_path)
    entries = manifest.get("entries")
    if not isinstance(entries, dict):
        entries = {}
    missing = sorted(ident for ident in identifiers if ident not in entries)
    if source_ids:
        missing.extend(sorted(sid for sid in source_ids if sid not in entries))
    return sorted(set(missing))


def verify_weights_manifest(manifest_path: Path | None = None) -> list[str]:
    """
    Return human-readable issues for a weights manifest and its cached files.

    Checks that each entry's cache file exists and matches the recorded SHA256 when present.
    """
    path = manifest_path if manifest_path is not None else default_weights_manifest_path()
    issues: list[str] = []
    if not path.is_file():
        issues.append(
            "missing weights manifest "
            f"(prep: python scripts/check_weights_cache.py --download): {path}"
        )
        return issues
    manifest = load_weights_manifest(path)
    entries = manifest.get("entries")
    if not isinstance(entries, dict) or not entries:
        issues.append(f"weights manifest has no entries: {path}")
        return issues
    for ident, meta in entries.items():
        if not isinstance(meta, dict):
            continue
        cache_path = meta.get("cache_path")
        if not cache_path:
            issues.append(f"weights {ident}: cache_path missing in manifest")
            continue
        p = Path(str(cache_path)).expanduser()
        if not p.is_absolute():
            p = (Path.cwd() / p).resolve()
        else:
            p = p.resolve()
        if not p.is_file():
            issues.append(f"weights {ident}: cache file missing: {p}")
            continue
        expected_sha = meta.get("sha256")
        if expected_sha:
            actual = file_sha256(p)
            if actual != str(expected_sha):
                issues.append(
                    f"weights {ident}: sha256 mismatch "
                    f"(expected {expected_sha}, got {actual})"
                )
    return issues


def download_hints_for_cache_path(*, cache_path: Path, identifier: str) -> dict[str, str]:
    """
    Manual fetch commands (no network I/O). Override tag via HARCHOC_ULTRALYTICS_ASSETS_TAG.
    """
    url = ultralytics_assets_url(identifier)
    dest = str(cache_path)
    return {
        "url": url,
        "wget": f"mkdir -p {cache_path.parent} && wget -O {dest} {url}",
        "curl": f"mkdir -p {cache_path.parent} && curl -L -o {dest} {url}",
        "note": (
            "Or run once in a Python env with ultralytics installed: "
            f"python -c \"from ultralytics import YOLO; YOLO('{identifier}')\""
        ),
    }

