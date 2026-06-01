from __future__ import annotations

from pathlib import Path
from typing import Any

from harchoc.schemas import with_schema_version

DOMAIN_EVAL_SCHEMA = "domain_eval.v1"
CATALOG_RUN_SCHEMA = "eval_domains_run.v1"


def catalog_record_domains(catalog: dict[str, Any]) -> list[dict[str, Any]]:
    """Return domain records from a nested catalog blob or top-level domains list."""
    nested = catalog.get("catalog")
    if isinstance(nested, dict):
        domains = nested.get("domains")
        if isinstance(domains, list):
            return [d for d in domains if isinstance(d, dict)]
    domains = catalog.get("domains")
    if isinstance(domains, list):
        return [d for d in domains if isinstance(d, dict)]
    return []


def tray_keys_from_catalog_blob(catalog: dict[str, Any]) -> list[str]:
    keys: list[str] = []
    for rec in catalog_record_domains(catalog):
        key = str(rec.get("tray_key") or "").strip()
        if key and key not in keys:
            keys.append(key)
    return keys


def domain_split_file(*, tray_key: str, split: str = "test", domains_dir: str | Path = "data/domains") -> str:
    return str(Path(domains_dir) / f"{split}_{tray_key}.txt")


def scaffold_domain_eval_entries(
    tray_keys: list[str],
    *,
    domains_dir: str | Path = "data/domains",
    split: str = "test",
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for key in tray_keys:
        out.append(
            {
                "tray_key": key,
                "split_file": domain_split_file(tray_key=key, split=split, domains_dir=domains_dir),
                "metrics": None,
                "delta_vs_canonical": None,
            }
        )
    return out


def planned_tray_eval_entries(
    tray_keys: list[str],
    *,
    domains_dir: str | Path = "data/domains",
    split: str = "test",
    weights: str,
    device: str = "cpu",
    locked_conf_from: str | None = None,
) -> list[dict[str, Any]]:
    """Planned per-tray eval.py invocations (--run-all-trays dry-run)."""
    out: list[dict[str, Any]] = []
    for key in tray_keys:
        entry: dict[str, Any] = {
            "tray_key": key,
            "split_file": domain_split_file(tray_key=key, split=split, domains_dir=domains_dir),
            "weights": str(weights),
            "device": str(device),
        }
        if locked_conf_from:
            entry["locked_conf_from"] = str(locked_conf_from)
        out.append(entry)
    return out


def build_domain_eval_payload(
    *,
    status: str,
    script: str,
    out: str | Path,
    weights: str,
    catalog_path: str | Path | None,
    domains_dir: str | Path,
    canonical_split_file: str,
    tray_keys: list[str],
    catalog: dict[str, Any] | None = None,
    domain_metadata_tags: dict[str, Any] | None = None,
    notes: str | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "status": status,
        "script": script,
        "out": str(Path(out)),
        "weights": str(weights),
        "catalog_path": str(Path(catalog_path)) if catalog_path is not None else None,
        "domains_dir": str(Path(domains_dir)),
        "canonical_test": {
            "split_file": canonical_split_file,
            "metrics": None,
        },
        "domains": scaffold_domain_eval_entries(tray_keys, domains_dir=domains_dir),
    }
    if catalog is not None:
        payload["catalog"] = catalog
    if domain_metadata_tags is not None:
        payload["domain_metadata_tags"] = domain_metadata_tags
    if notes:
        payload["notes"] = notes
    return with_schema_version(payload, schema_version=DOMAIN_EVAL_SCHEMA)
