from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DETECTOR_SOURCES_SCHEMA = "detector_sources.v1"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def default_detector_sources_path() -> Path:
    return _repo_root() / "configs" / "external" / "detector_sources.v1.json"


def default_external_weights_dir() -> Path:
    raw = os.getenv("EXTERNAL_WEIGHTS_CACHE_DIR", "data/weights/external")
    p = Path(raw).expanduser()
    if not p.is_absolute():
        p = (_repo_root() / p)
    return p.resolve()


@dataclass(frozen=True)
class DetectorSourceEntry:
    source_id: str
    label: str
    family: str
    train_stack: str
    repos: dict[str, str]
    config_relpath: str
    checkpoint_url: str
    checkpoint_cache_name: str
    gdown_id: str | None
    baseline_source_id: str | None

    @property
    def cache_path(self) -> Path:
        return default_external_weights_dir() / self.checkpoint_cache_name

    def to_json(self) -> dict[str, object]:
        return {
            "source_id": self.source_id,
            "label": self.label,
            "family": self.family,
            "train_stack": self.train_stack,
            "repos": dict(self.repos),
            "config_relpath": self.config_relpath,
            "checkpoint": {
                "url": self.checkpoint_url,
                "cache_path": str(self.cache_path),
                "cache_name": self.checkpoint_cache_name,
                "gdown_id": self.gdown_id,
            },
            "baseline_source_id": self.baseline_source_id,
        }


def load_train_stack_metadata(path: Path | None = None) -> dict[str, dict[str, Any]]:
    p = path if path is not None else default_detector_sources_path()
    raw = json.loads(p.read_text(encoding="utf-8"))
    if raw.get("schema_version") != DETECTOR_SOURCES_SCHEMA:
        raise ValueError(f"Unsupported detector sources schema in {p}")
    stacks = raw.get("train_stacks")
    if not isinstance(stacks, dict):
        raise ValueError(f"Missing train_stacks in {p}")
    out: dict[str, dict[str, Any]] = {}
    for key, obj in stacks.items():
        if isinstance(obj, dict):
            out[str(key)] = {str(k): v for k, v in obj.items()}
    return out


def load_detector_sources(path: Path | None = None) -> dict[str, DetectorSourceEntry]:
    p = path if path is not None else default_detector_sources_path()
    raw = json.loads(p.read_text(encoding="utf-8"))
    if raw.get("schema_version") != DETECTOR_SOURCES_SCHEMA:
        raise ValueError(f"Unsupported detector sources schema in {p}")
    entries = raw.get("entries")
    if not isinstance(entries, dict):
        raise ValueError(f"Missing entries in {p}")
    out: dict[str, DetectorSourceEntry] = {}
    for source_id, obj in entries.items():
        if not isinstance(obj, dict):
            continue
        ckpt = obj.get("checkpoint")
        if not isinstance(ckpt, dict):
            raise ValueError(f"checkpoint block required for {source_id}")
        url = str(ckpt.get("url") or ckpt.get("direct_url") or "").strip()
        cache_name = str(ckpt.get("cache_name") or "").strip()
        if not url or not cache_name:
            raise ValueError(f"checkpoint url/cache_name required for {source_id}")
        gdown_raw = ckpt.get("gdown_id")
        gdown_id = str(gdown_raw).strip() if gdown_raw else None
        repos = obj.get("repos")
        if not isinstance(repos, dict):
            repos = {}
        baseline = obj.get("baseline_source_id")
        out[str(source_id)] = DetectorSourceEntry(
            source_id=str(source_id),
            label=str(obj.get("label") or source_id),
            family=str(obj.get("family") or ""),
            train_stack=str(obj.get("train_stack") or ""),
            repos={str(k): str(v) for k, v in repos.items()},
            config_relpath=str(obj.get("config_relpath") or ""),
            checkpoint_url=url,
            checkpoint_cache_name=cache_name,
            gdown_id=gdown_id or None,
            baseline_source_id=str(baseline).strip() if baseline else None,
        )
    return out


def resolve_source_id(*, model_id: str | None, source_id: str | None) -> str | None:
    sid = (source_id or model_id or "").strip()
    return sid or None


def entry_for_bench(*, model_id: str | None, source_id: str | None) -> DetectorSourceEntry | None:
    sid = resolve_source_id(model_id=model_id, source_id=source_id)
    if not sid:
        return None
    return load_detector_sources().get(sid)


def external_entry_provenance(entry: DetectorSourceEntry) -> dict[str, object]:
    """Shared provenance fields for matrix weights and weights-cache report rows."""
    exists = entry.cache_path.is_file() and entry.cache_path.stat().st_size > 0
    return {
        "source_id": entry.source_id,
        "label": entry.label,
        "train_stack": entry.train_stack,
        "cache_path": str(entry.cache_path),
        "exists": exists,
        "checkpoint_url": entry.checkpoint_url,
        "repos": dict(entry.repos),
        "config_relpath": entry.config_relpath,
    }


def _download_http(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    urllib.request.urlretrieve(url, tmp)  # noqa: S310
    tmp.replace(dest)


def _download_gdrive(gdown_id: str, dest: Path) -> None:
    try:
        import gdown  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "gdown is required for Google Drive checkpoints "
            f"(pip install gdown); id={gdown_id}"
        ) from exc
    dest.parent.mkdir(parents=True, exist_ok=True)
    url = f"https://drive.google.com/uc?id={gdown_id}"
    gdown.download(url, str(dest), quiet=False)


def download_external_checkpoint(entry: DetectorSourceEntry) -> dict[str, object]:
    dest = entry.cache_path
    if dest.is_file() and dest.stat().st_size > 0:
        return {
            "source_id": entry.source_id,
            "downloaded": False,
            "cache_path": str(dest),
            "size_bytes": dest.stat().st_size,
        }
    if entry.gdown_id:
        _download_gdrive(entry.gdown_id, dest)
        method = "gdown"
    elif entry.checkpoint_url.startswith("https://github.com/"):
        _download_http(entry.checkpoint_url, dest)
        method = "http"
    else:
        raise RuntimeError(
            f"No direct download for {entry.source_id}; install gdown or fetch manually: "
            f"{entry.checkpoint_url}"
        )
    return {
        "source_id": entry.source_id,
        "downloaded": True,
        "cache_path": str(dest),
        "size_bytes": dest.stat().st_size,
        "method": method,
    }


def collect_bench_external_source_ids(
    *,
    bench_dir: Path,
    bench_config_paths: list[Path] | None = None,
) -> dict[str, list[str]]:
    from harchoc.bench_config import load_bench_config, select_backend

    if bench_config_paths is None:
        from harchoc.bench_assets import iter_bench_config_paths

        paths = iter_bench_config_paths(bench_dir=bench_dir, pattern="*.yaml")
    else:
        paths = bench_config_paths
    by_id: dict[str, list[str]] = {}
    for pth in paths:
        cfg = load_bench_config(pth)
        if select_backend(cfg) != "external":
            continue
        sid = resolve_source_id(model_id=cfg.model_id, source_id=cfg.source_id)
        if not sid:
            continue
        by_id.setdefault(sid, []).append(pth.name)
    return by_id
