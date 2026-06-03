"""Post-train aug smoke eval chain: test export → error_analysis → summary.v1."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from harchoc.hsp_eval_chain import (
    DEFAULT_LOCKED_CONF_FROM,
    build_ultralytics_hsp_stages,
    extract_count_mae,
    hsp_eval_artifacts_verified,
    hsp_eval_prefix_paths,
    infer_smoke_eval_backend,
    run_hsp_eval_chain,
)
from harchoc.model_zoo import file_sha256

AUG_SMOKE_SUMMARY_SCHEMA = "aug_smoke_summary.v1"
DEFAULT_OUT_DIR = "reports/aug_smoke"


def load_aug_smoke_index(path: str | Path) -> dict[str, Any]:
    p = Path(path).expanduser().resolve()
    obj = json.loads(p.read_text(encoding="utf-8"))
    if obj.get("schema_version") != "aug_smoke_index.v1":
        raise ValueError(f"unsupported aug smoke index schema: {obj.get('schema_version')!r}")
    return obj


def find_smoke_entry(index: dict[str, Any], smoke_id: str) -> dict[str, Any]:
    sid = str(smoke_id).strip().upper()
    for entry in index.get("smokes") or []:
        if str(entry.get("id") or "").upper() == sid:
            return entry
    raise KeyError(f"smoke_id {smoke_id!r} not in aug_smoke_index")


def resolve_train_weights(
    *,
    repo_root: Path,
    run_name: str,
    out_dir: str | Path | None = None,
) -> Path | None:
    """Locate best.pt from a completed Ultralytics run."""
    candidates: list[Path] = []
    od = Path(out_dir) if out_dir else repo_root / "runs"
    for base in (od, repo_root / "runs", repo_root / "runs" / "detect"):
        candidates.extend(
            [
                base / run_name / "weights" / "best.pt",
                base / "detect" / run_name / "weights" / "best.pt",
                base / run_name / "weights" / "last.pt",
            ]
        )
    for c in candidates:
        if c.is_file():
            return c.resolve()
    return None


def build_aug_smoke_eval_stages(
    *,
    repo_root: Path,
    run_name: str,
    weights: str | Path,
    locked_conf_from: str = DEFAULT_LOCKED_CONF_FROM,
    out_dir: str = DEFAULT_OUT_DIR,
    max_det: int = 3000,
    imgsz: int = 1280,
) -> list[tuple[str, list[str], bool]]:
    """Return (stage_id, argv relative to repo, use_mamba) for test eval chain."""
    stages = build_ultralytics_hsp_stages(
        repo_root=repo_root,
        run_name=run_name,
        weights=weights,
        locked_conf_from=locked_conf_from,
        out_dir=out_dir,
        max_det=max_det,
        imgsz=imgsz,
    )
    return [(sid, argv, True) for sid, argv in stages]


def artifact_fingerprints(
    *,
    weights: str | Path,
    preds_json: str | Path | None = None,
    error_json: str | Path | None = None,
) -> dict[str, Any]:
    """SHA256/size records for post-hoc duplicate-MAE audits (P1-AUG-DUP-MAE)."""
    out: dict[str, Any] = {}
    wp = Path(weights)
    if wp.is_file():
        out["weights"] = {
            "path": str(wp.resolve()),
            "sha256": file_sha256(wp),
            "size_bytes": wp.stat().st_size,
        }
    for key, path in (("preds_json", preds_json), ("error_json", error_json)):
        if not path:
            continue
        pp = Path(path)
        if pp.is_file():
            out[key] = {
                "path": str(pp.resolve()),
                "sha256": file_sha256(pp),
                "size_bytes": pp.stat().st_size,
            }
    return out


def build_aug_smoke_summary(
    *,
    smoke_id: str,
    run_name: str,
    train_config: str,
    weights: str | Path,
    train_runtime_s: float | None,
    locked_conf_from: str,
    out_dir: str,
    error_json: str,
    eval_json: str,
    val_map50_95: float | None = None,
    arch_ticket: str = "P1-AUG",
) -> dict[str, Any]:
    mae, mae_ci = extract_count_mae(error_json)
    preds_path = Path(error_json).with_name(
        Path(error_json).name.replace("_error.json", "_preds.json")
    )
    return {
        "schema_version": AUG_SMOKE_SUMMARY_SCHEMA,
        "smoke_id": smoke_id,
        "run_name": run_name,
        "arch_ticket": arch_ticket,
        "train_config": train_config,
        "status": "complete" if mae is not None else "failed",
        "train": {
            "epochs": None,
            "runtime_s": train_runtime_s,
            "weights": str(Path(weights).resolve()),
            "val_map50_95": val_map50_95,
        },
        "test_eval": {
            "split": "test",
            "device": "cpu",
            "locked_conf_from": locked_conf_from,
            "eval_json": eval_json,
            "error_json": error_json,
        },
        "artifacts": artifact_fingerprints(
            weights=weights,
            preds_json=preds_path,
            error_json=error_json,
        ),
        "test_count_mae": mae,
        "test_count_mae_ci": mae_ci,
        "n_test_images": 109,
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }


def write_aug_smoke_summary(path: str | Path, payload: dict[str, Any]) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return p.resolve()


def patch_aug_smoke_index_entry(
    index_path: str | Path,
    smoke_id: str,
    *,
    status: str,
    summary: str,
    test_count_mae: float | None,
) -> None:
    p = Path(index_path).expanduser().resolve()
    obj = load_aug_smoke_index(p)
    sid = str(smoke_id).strip().upper()
    for entry in obj.get("smokes") or []:
        if str(entry.get("id") or "").upper() == sid:
            entry["status"] = status
            entry["summary"] = summary
            if test_count_mae is not None:
                entry["test_count_mae"] = test_count_mae
            break
    else:
        raise KeyError(smoke_id)
    p.write_text(json.dumps(obj, indent=2) + "\n", encoding="utf-8")


def run_smoke_hsp_eval_chain(
    *,
    repo_root: str | Path,
    run_name: str,
    weights: str | Path,
    locked_conf_from: str = DEFAULT_LOCKED_CONF_FROM,
    out_dir: str = DEFAULT_OUT_DIR,
    max_det: int = 3000,
    backend: str = "auto",
    model_id: str = "yolo_nas_s",
    dry_run: bool = False,
    on_stage: Callable[[str, list[str]], None] | None = None,
    env: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Run HSP test export + error_analysis; backend auto-detects from weights suffix."""
    return run_hsp_eval_chain(
        repo_root=repo_root,
        run_name=run_name,
        weights=weights,
        locked_conf_from=locked_conf_from,
        out_dir=out_dir,
        max_det=max_det,
        backend=backend,
        model_id=model_id,
        dry_run=dry_run,
        on_stage=on_stage,
        env=env,
    )


def run_aug_smoke_eval_chain(
    *,
    repo_root: str | Path,
    run_name: str,
    weights: str | Path,
    locked_conf_from: str = DEFAULT_LOCKED_CONF_FROM,
    out_dir: str = DEFAULT_OUT_DIR,
    max_det: int = 3000,
    dry_run: bool = False,
    on_stage: Callable[[str, list[str]], None] | None = None,
    env: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Run eval export + error_analysis; return artifact paths."""
    return run_hsp_eval_chain(
        repo_root=repo_root,
        run_name=run_name,
        weights=weights,
        locked_conf_from=locked_conf_from,
        out_dir=out_dir,
        max_det=max_det,
        backend="ultralytics",
        dry_run=dry_run,
        on_stage=on_stage,
        env=env,
    )


def finalize_smoke_job(
    *,
    repo_root: str | Path,
    run_name: str,
    train_config: str,
    weights: str | Path,
    summary_path: str,
    smoke_id: str,
    locked_conf_from: str = DEFAULT_LOCKED_CONF_FROM,
    out_dir: str = DEFAULT_OUT_DIR,
    arch_ticket: str = "P1-AUG",
    index_path: str | Path = "configs/experiments/aug_smoke_index.json",
    patch_index: bool = False,
    train_runtime_s: float | None = None,
    refresh_leaderboard: bool = True,
    hsp_artifacts: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Write aug_smoke_summary.v1; optionally patch index and refresh leaderboard."""
    rr = Path(repo_root).resolve()
    sid = str(smoke_id).strip().upper()
    prefix = f"{out_dir}/{run_name}"
    if hsp_artifacts:
        error_json = str(hsp_artifacts.get("error") or (rr / f"{prefix}_error.json").resolve())
        eval_json = str(hsp_artifacts.get("eval") or (rr / f"{prefix}_eval.json").resolve())
    else:
        error_json = str((rr / f"{prefix}_error.json").resolve())
        eval_json = str((rr / f"{prefix}_eval.json").resolve())

    payload = build_aug_smoke_summary(
        smoke_id=sid,
        run_name=run_name,
        train_config=train_config,
        weights=weights,
        train_runtime_s=train_runtime_s,
        locked_conf_from=locked_conf_from,
        out_dir=out_dir,
        error_json=error_json,
        eval_json=eval_json,
        arch_ticket=arch_ticket,
    )
    write_aug_smoke_summary(rr / summary_path, payload)
    if patch_index and payload.get("status") == "complete":
        patch_aug_smoke_index_entry(
            index_path,
            sid,
            status="complete",
            summary=summary_path,
            test_count_mae=payload.get("test_count_mae"),
        )
    if refresh_leaderboard:
        from harchoc.aug_smoke_leaderboard import refresh_aug_smoke_leaderboard

        refresh_aug_smoke_leaderboard(repo_root=rr, index_path=index_path, out_dir=out_dir)
    return payload


def finalize_aug_smoke(
    *,
    repo_root: str | Path,
    smoke_id: str,
    index_path: str | Path = "configs/experiments/aug_smoke_index.json",
    train_config: str,
    run_name: str,
    weights: str | Path,
    train_runtime_s: float | None = None,
    locked_conf_from: str = DEFAULT_LOCKED_CONF_FROM,
    out_dir: str = DEFAULT_OUT_DIR,
    summary_path: str | None = None,
    refresh_leaderboard: bool = True,
) -> dict[str, Any]:
    """Write summary JSON and patch aug_smoke_index."""
    sid = str(smoke_id).strip().upper()
    summary_out = summary_path or f"{out_dir}/{sid.lower()}_summary.json"
    return finalize_smoke_job(
        repo_root=repo_root,
        smoke_id=sid,
        run_name=run_name,
        train_config=train_config,
        weights=weights,
        summary_path=summary_out,
        locked_conf_from=locked_conf_from,
        out_dir=out_dir,
        index_path=index_path,
        patch_index=True,
        train_runtime_s=train_runtime_s,
        refresh_leaderboard=refresh_leaderboard,
    )


def aug_smoke_index_queue_parity_errors(
    *,
    repo_root: str | Path,
    index_path: str | Path = "configs/experiments/aug_smoke_index.json",
    manifest_path: str | Path = "configs/experiments/archive/gpu_queue_full.json",
) -> list[str]:
    """Every ``gpu_pending`` smoke index row must have a matching ``aug_smoke`` queue job."""
    rr = Path(repo_root).resolve()
    index = load_aug_smoke_index(rr / index_path)
    raw = json.loads((rr / manifest_path).read_text(encoding="utf-8"))
    queue_smoke_ids: set[str] = set()
    if raw.get("aug_smoke_from_index"):
        from harchoc.gpu_queue import expand_aug_smoke_jobs_from_index

        idx = str(raw.get("aug_smoke_index") or index_path)
        for job in expand_aug_smoke_jobs_from_index(repo_root=rr, index_path=idx):
            sid = str(job.get("smoke_id") or "").strip().upper()
            if sid:
                queue_smoke_ids.add(sid)
    else:
        for job in raw.get("jobs") or []:
            if str(job.get("kind") or "") != "aug_smoke":
                continue
            sid = str(job.get("smoke_id") or "").strip().upper()
            if sid:
                queue_smoke_ids.add(sid)
    errors: list[str] = []
    for entry in index.get("smokes") or []:
        if str(entry.get("status") or "") != "gpu_pending":
            continue
        sid = str(entry.get("id") or "").strip().upper()
        if sid and sid not in queue_smoke_ids:
            errors.append(
                f"{sid} is gpu_pending in index but missing aug_smoke job in {manifest_path}"
            )
    return errors
