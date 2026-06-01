"""Rank weak trays from domain_count_mae / domain_eval artifacts (CPU-only)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from harchoc.json_io import load_json_dict
from harchoc.schemas import with_schema_version

WEAK_TRAY_PLAN_SCHEMA = "weak_tray_plan.v1"


def _per_tray_from_count_mae(path: Path) -> list[dict[str, Any]]:
    raw = load_json_dict(path)
    domains = raw.get("domains")
    if isinstance(domains, list) and domains:
        return [d for d in domains if isinstance(d, dict)]
    summary = raw.get("summary")
    if isinstance(summary, dict):
        per = summary.get("per_tray")
        if isinstance(per, list):
            return [d for d in per if isinstance(d, dict)]
    return []


def _per_tray_from_domain_eval(path: Path) -> list[dict[str, Any]]:
    raw = load_json_dict(path)
    domains = raw.get("domains")
    if not isinstance(domains, list):
        return []
    out: list[dict[str, Any]] = []
    for rec in domains:
        if not isinstance(rec, dict):
            continue
        metrics = rec.get("metrics")
        mae = None
        if isinstance(metrics, dict):
            mae = metrics.get("count_mae")
        out.append(
            {
                "tray_key": rec.get("tray_key"),
                "count_mae": mae,
                "mAP50": metrics.get("mAP50") if isinstance(metrics, dict) else None,
                "source": "domain_eval.v1",
            }
        )
    return out


def build_weak_tray_plan(
    *,
    count_mae_path: Path | None,
    domain_eval_path: Path | None,
    top_k: int = 3,
    global_mae: float | None = 61.3,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    if count_mae_path is not None and count_mae_path.is_file():
        rows = _per_tray_from_count_mae(count_mae_path)
        source = str(count_mae_path)
    elif domain_eval_path is not None and domain_eval_path.is_file():
        rows = _per_tray_from_domain_eval(domain_eval_path)
        source = str(domain_eval_path)
    else:
        return with_schema_version(
            {
                "status": "pending",
                "notes": "Run eval_domains --merge-tray-count-mae first.",
                "recommended_tray_keys": [],
            },
            schema_version=WEAK_TRAY_PLAN_SCHEMA,
        )

    ranked: list[dict[str, Any]] = []
    for rec in rows:
        key = str(rec.get("tray_key") or "").strip()
        mae = rec.get("count_mae")
        if not key or mae is None:
            continue
        mae_f = float(mae)
        ranked.append(
            {
                "tray_key": key,
                "count_mae": mae_f,
                "delta_vs_global": (mae_f - float(global_mae)) if global_mae is not None else None,
                "mAP50": rec.get("mAP50"),
            }
        )
    ranked.sort(key=lambda x: float(x["count_mae"]), reverse=True)
    top = ranked[: max(1, int(top_k))]

    return with_schema_version(
        {
            "status": "ok" if ranked else "empty",
            "source": source,
            "global_mae_reference": global_mae,
            "n_trays_ranked": len(ranked),
            "recommended_tray_keys": [t["tray_key"] for t in top],
            "top_trays": top,
            "finetune_hint": (
                f"mamba run -n harchoc python scripts/finetune.py --tray-key {top[0]['tray_key']} "
                "--train-mode tray_adapt --dataset-root $DATASET_ROOT"
                if top
                else None
            ),
            "docs": "docs/FINETUNE_WEAK_TRAYS.md",
        },
        schema_version=WEAK_TRAY_PLAN_SCHEMA,
    )


def write_weak_tray_plan(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path
