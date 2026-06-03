"""Record finetune stage-1 base weights after joined close3 retrains vs production best2."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from harchoc.finetune_pipeline import DEFAULT_FINETUNE_BASE_WEIGHTS, DEFAULT_GLOBAL_MAE_REF
from harchoc.json_io import load_json_dict
from harchoc.schemas import with_schema_version

FINETUNE_BASE_SELECTION_SCHEMA = "finetune_base_selection.v1"
JOINED_STUDY_SUMMARY_SCHEMA = "joined_close3_study_summary.v1"
DEFAULT_SELECTION_PATH = "reports/hsp/finetune_base_selection.json"
DEFAULT_JOINED_STUDY_PATH = "reports/gpu_queue/joined_close3_study_summary.json"
DEFAULT_JOINED_SUMMARIES_GLOB = "reports/gpu_queue/summaries/joined_close3_*_100ep.json"


def _rel(repo_root: Path, path: Path) -> str:
    try:
        return str(path.resolve().relative_to(repo_root.resolve()))
    except ValueError:
        return str(path.resolve())


def load_aug_smoke_summary(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        doc = load_json_dict(path)
    except Exception:
        return None
    if str(doc.get("schema_version") or "") != "aug_smoke_summary.v1":
        return None
    if doc.get("status") != "complete":
        return None
    mae = doc.get("test_count_mae")
    if mae is None:
        return None
    return doc


def collect_joined_close3_candidates(
    repo_root: Path,
    *,
    summaries_glob: str = DEFAULT_JOINED_SUMMARIES_GLOB,
) -> list[dict[str, Any]]:
    """Candidates from completed joined_close3 gpu_queue summaries."""
    out: list[dict[str, Any]] = []
    for path in sorted(repo_root.glob(summaries_glob)):
        doc = load_aug_smoke_summary(path)
        if not doc:
            continue
        weights = (doc.get("train") or {}).get("weights") or (doc.get("artifacts") or {}).get(
            "weights", {}
        ).get("path")
        out.append(
            {
                "id": str(doc.get("smoke_id") or path.stem),
                "source": "joined_close3_100ep",
                "summary_path": _rel(repo_root, path),
                "run_name": str(doc.get("run_name") or ""),
                "weights": str(weights) if weights else None,
                "test_count_mae": float(doc["test_count_mae"]),
                "test_count_mae_ci": doc.get("test_count_mae_ci"),
            }
        )
    return out


def pick_finetune_base_candidate(
    candidates: list[dict[str, Any]],
    *,
    anchor_mae: float = DEFAULT_GLOBAL_MAE_REF,
    production_weights: str = DEFAULT_FINETUNE_BASE_WEIGHTS,
) -> dict[str, Any]:
    """
    Stage-1 finetune base: production best2 unless a retrain beats the anchor on test MAE.
    """
    anchor = {
        "id": "best2",
        "source": "production",
        "weights": production_weights,
        "test_count_mae": float(anchor_mae),
        "beats_anchor": False,
    }
    retrains = [c for c in candidates if c.get("weights") and c.get("test_count_mae") is not None]
    if not retrains:
        winner = dict(anchor)
        winner["reason"] = "no_retrain_candidates"
        return winner

    best = min(retrains, key=lambda c: float(c["test_count_mae"]))
    best_mae = float(best["test_count_mae"])
    if best_mae < float(anchor_mae):
        w = dict(best)
        w["beats_anchor"] = True
        w["reason"] = f"retrain_mae_{best_mae:.4f}_lt_anchor_{anchor_mae}"
        return w

    winner = dict(anchor)
    winner["reason"] = (
        f"keep_production_best_mae_{anchor_mae}_best_retrain_{best_mae:.4f}_"
        f"({best.get('id')})"
    )
    winner["best_retrain"] = {
        "id": best.get("id"),
        "weights": best.get("weights"),
        "test_count_mae": best_mae,
        "summary_path": best.get("summary_path"),
    }
    return winner


def build_finetune_base_selection(
    repo_root: Path,
    *,
    anchor_mae: float = DEFAULT_GLOBAL_MAE_REF,
    production_weights: str = DEFAULT_FINETUNE_BASE_WEIGHTS,
    summaries_glob: str = DEFAULT_JOINED_SUMMARIES_GLOB,
    aug_confirm_summary: str = "reports/aug_smoke/aug_confirm_winner_100ep_summary.json",
) -> dict[str, Any]:
    joined = collect_joined_close3_candidates(repo_root, summaries_glob=summaries_glob)
    extra: list[dict[str, Any]] = []
    confirm_path = repo_root / aug_confirm_summary
    confirm_doc = load_aug_smoke_summary(confirm_path)
    if confirm_doc:
        w = (confirm_doc.get("train") or {}).get("weights")
        extra.append(
            {
                "id": "aug_confirm_winner_100ep",
                "source": "aug_confirm_100ep",
                "summary_path": _rel(repo_root, confirm_path),
                "run_name": str(confirm_doc.get("run_name") or ""),
                "weights": str(w) if w else None,
                "test_count_mae": float(confirm_doc["test_count_mae"]),
            }
        )

    all_retrains = joined + [e for e in extra if e.get("weights")]
    winner = pick_finetune_base_candidate(
        all_retrains,
        anchor_mae=anchor_mae,
        production_weights=production_weights,
    )
    payload: dict[str, Any] = {
        "anchor_test_count_mae": float(anchor_mae),
        "production_weights": production_weights,
        "joined_close3_candidates": joined,
        "other_retrain_candidates": extra,
        "selected": winner,
        "stage1_base_weights": str(winner.get("weights") or production_weights),
    }
    return with_schema_version(payload, schema_version=FINETUNE_BASE_SELECTION_SCHEMA)


def build_joined_close3_study_summary(
    repo_root: Path,
    *,
    summaries_glob: str = DEFAULT_JOINED_SUMMARIES_GLOB,
    matrix_train_path: str = "reports/hsp/matrix_train.json",
    anchor_mae: float = DEFAULT_GLOBAL_MAE_REF,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for path in sorted(repo_root.glob(summaries_glob)):
        doc = load_aug_smoke_summary(path)
        if not doc:
            continue
        mae = float(doc["test_count_mae"])
        rows.append(
            {
                "job_id": path.stem.replace("_100ep", ""),
                "run_name": doc.get("run_name"),
                "test_count_mae": mae,
                "delta_vs_anchor": round(mae - float(anchor_mae), 4),
                "summary_path": _rel(repo_root, path),
            }
        )
    rows.sort(key=lambda r: r["test_count_mae"])

    zoo: list[dict[str, Any]] = []
    mt_path = repo_root / matrix_train_path
    if mt_path.is_file():
        try:
            mt = load_json_dict(mt_path)
            for run in mt.get("runs") or []:
                if run.get("test_count_mae") is None:
                    continue
                zoo.append(
                    {
                        "name": run.get("name"),
                        "test_count_mae": float(run["test_count_mae"]),
                    }
                )
            zoo.sort(key=lambda r: r["test_count_mae"])
        except Exception:
            zoo = []

    best_joined = rows[0] if rows else None
    payload: dict[str, Any] = {
        "anchor_test_count_mae": float(anchor_mae),
        "aug_config": "configs/aug/robustness_smoke_close3.yaml",
        "joined_runs": rows,
        "zoo_yolo_only_baselines": zoo,
        "conclusions": [],
    }
    if best_joined:
        if best_joined["test_count_mae"] < float(anchor_mae):
            payload["conclusions"].append("joined_best_beats_best2")
        else:
            payload["conclusions"].append("none_beat_best2")
        if len(rows) >= 2 and rows[0]["test_count_mae"] == rows[-1]["test_count_mae"]:
            payload["conclusions"].append("all_joined_same_mae")
    return with_schema_version(payload, schema_version=JOINED_STUDY_SUMMARY_SCHEMA)


def write_finetune_base_selection(
    repo_root: Path,
    path: str | Path = DEFAULT_SELECTION_PATH,
    **kwargs: Any,
) -> Path:
    out = repo_root / path
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = build_finetune_base_selection(repo_root, **kwargs)
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return out


def write_joined_close3_study_summary(
    repo_root: Path,
    path: str | Path = DEFAULT_JOINED_STUDY_PATH,
    **kwargs: Any,
) -> Path:
    out = repo_root / path
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = build_joined_close3_study_summary(repo_root, **kwargs)
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return out
