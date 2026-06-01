"""Backfill HSP test summaries for GPU queue jobs with train Done but no summary JSON."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from harchoc.aug_smoke_runner import (
    DEFAULT_LOCKED_CONF_FROM,
    DEFAULT_OUT_DIR,
    finalize_smoke_job,
    load_aug_smoke_index,
    resolve_train_weights,
    run_smoke_hsp_eval_chain,
)


@dataclass(frozen=True)
class QueueSummaryBackfill:
    job_id: str
    run_name: str
    train_config: str
    summary_path: str
    smoke_id: str
    out_dir: str = DEFAULT_OUT_DIR
    max_det: int = 3000
    model_id: str = "yolo_nas_s"
    arch_ticket: str = "P1-AUG"
    sweep_index_id: str | None = None  # sweeps_15ep.arms[].id when set


# Tier-2 jobs in gpu_queue_full.json where train weights exist but summary was never written.
GPU_QUEUE_SUMMARY_BACKFILLS: tuple[QueueSummaryBackfill, ...] = (
    QueueSummaryBackfill(
        job_id="amp_smoke_15ep_on_hsp_eval",
        run_name="amp_on_smoke_15ep",
        train_config="configs/experiments/train_amp_on_15ep_smoke.json",
        summary_path="reports/hsp/amp_on_smoke_15ep_summary.json",
        smoke_id="AMP_ON_15EP",
        out_dir="reports/hsp",
        arch_ticket="P1-AMP-HSP-EVAL",
    ),
    QueueSummaryBackfill(
        job_id="sg_yolo_nas_s_hsp_eval",
        run_name="sg_yolo_nas_s_smoke_15ep",
        train_config="configs/experiments/train_sg_yolo_nas_s_smoke_15ep.json",
        summary_path="reports/aug_smoke/sg_yolo_nas_s_smoke_15ep_summary.json",
        smoke_id="SG_YOLO_NAS_S",
        arch_ticket="P1-SG-HSP-EVAL",
    ),
    QueueSummaryBackfill(
        job_id="aug_sweep_15_close10",
        run_name="aug_sweep_close10_15ep",
        train_config="configs/experiments/train_aug_close10_sweep_smoke_15ep.json",
        summary_path="reports/aug_smoke/sweep_close10_15ep_summary.json",
        smoke_id="CLOSE10",
        arch_ticket="P1-AUG-CLOSE",
        sweep_index_id="close10",
    ),
    QueueSummaryBackfill(
        job_id="aug_sweep_15_close25",
        run_name="aug_sweep_close25_15ep",
        train_config="configs/experiments/train_aug_close25_sweep_smoke_15ep.json",
        summary_path="reports/aug_smoke/sweep_close25_15ep_summary.json",
        smoke_id="CLOSE25",
        arch_ticket="P1-AUG-CLOSE",
        sweep_index_id="close25",
    ),
)


def _run_hsp_eval_or_error_only(
    spec: QueueSummaryBackfill,
    *,
    repo_root: Path,
    weights: Path,
    locked_conf_from: str,
) -> None:
    rr = repo_root.resolve()
    prefix = rr / spec.out_dir / spec.run_name
    gt = prefix.with_name(prefix.name + "_gt.json")
    preds = prefix.with_name(prefix.name + "_preds.json")
    err = prefix.with_name(prefix.name + "_error.json")
    if gt.is_file() and preds.is_file() and not err.is_file():
        import os
        import subprocess

        mamba_env = os.environ.get("HARCHOC_MAMBA_ENV", "harchoc")
        rel_gt = str(gt.relative_to(rr))
        rel_preds = str(preds.relative_to(rr))
        rel_err = str(err.relative_to(rr))
        cmd = [
            "mamba",
            "run",
            "-n",
            mamba_env,
            "python",
            "scripts/error_analysis.py",
            "--gt-json",
            rel_gt,
            "--preds-json",
            rel_preds,
            "--locked-conf-from",
            locked_conf_from,
            "--out",
            rel_err,
        ]
        proc = subprocess.run(cmd, cwd=str(rr), env={**os.environ})
        if proc.returncode != 0:
            raise RuntimeError(f"error_analysis failed for {spec.job_id} exit {proc.returncode}")
        return

    run_smoke_hsp_eval_chain(
        repo_root=rr,
        run_name=spec.run_name,
        weights=weights,
        locked_conf_from=locked_conf_from,
        out_dir=spec.out_dir,
        max_det=spec.max_det,
        model_id=spec.model_id,
        dry_run=False,
    )


def _patch_sweep_arm(
    index_path: Path,
    *,
    arm_id: str,
    summary: str,
    test_count_mae: float | None,
) -> None:
    obj = load_aug_smoke_index(index_path)
    arms = (obj.get("sweeps_15ep") or {}).get("arms") or []
    for arm in arms:
        if str(arm.get("id") or "") == arm_id:
            arm["status"] = "complete"
            arm["summary"] = summary
            if test_count_mae is not None:
                arm["test_count_mae"] = test_count_mae
            break
    else:
        raise KeyError(f"sweep arm {arm_id!r} not in sweeps_15ep")
    index_path.write_text(json.dumps(obj, indent=2) + "\n", encoding="utf-8")


def backfill_queue_summary(
    spec: QueueSummaryBackfill,
    *,
    repo_root: Path,
    locked_conf_from: str = DEFAULT_LOCKED_CONF_FROM,
    skip_if_complete: bool = True,
    index_path: str | Path = "configs/experiments/aug_smoke_index.json",
) -> dict[str, Any]:
    rr = repo_root.resolve()
    summary_p = rr / spec.summary_path
    if skip_if_complete and summary_p.is_file():
        try:
            existing = json.loads(summary_p.read_text(encoding="utf-8"))
            if existing.get("status") == "complete" and existing.get("test_count_mae") is not None:
                return {"job_id": spec.job_id, "status": "skipped", "reason": "summary already complete"}
        except Exception:
            pass

    weights = resolve_train_weights(repo_root=rr, run_name=spec.run_name)
    if weights is None:
        raise FileNotFoundError(f"weights not found for run_name={spec.run_name!r}")

    _run_hsp_eval_or_error_only(
        spec,
        repo_root=rr,
        weights=weights,
        locked_conf_from=locked_conf_from,
    )

    payload = finalize_smoke_job(
        repo_root=rr,
        run_name=spec.run_name,
        train_config=spec.train_config,
        weights=weights,
        summary_path=spec.summary_path,
        smoke_id=spec.smoke_id,
        locked_conf_from=locked_conf_from,
        out_dir=spec.out_dir,
        arch_ticket=spec.arch_ticket,
        patch_index=False,
        refresh_leaderboard=False,
    )

    if spec.sweep_index_id and payload.get("status") == "complete":
        _patch_sweep_arm(
            rr / index_path,
            arm_id=spec.sweep_index_id,
            summary=spec.summary_path,
            test_count_mae=payload.get("test_count_mae"),
        )

    return {
        "job_id": spec.job_id,
        "status": payload.get("status"),
        "test_count_mae": payload.get("test_count_mae"),
        "summary_path": spec.summary_path,
    }


def backfill_gpu_queue_summaries(
    *,
    repo_root: str | Path,
    specs: tuple[QueueSummaryBackfill, ...] = GPU_QUEUE_SUMMARY_BACKFILLS,
    skip_if_complete: bool = True,
) -> list[dict[str, Any]]:
    from harchoc.aug_smoke_leaderboard import refresh_aug_smoke_leaderboard

    rr = Path(repo_root).resolve()
    results: list[dict[str, Any]] = []
    for spec in specs:
        results.append(
            backfill_queue_summary(spec, repo_root=rr, skip_if_complete=skip_if_complete)
        )
    refresh_aug_smoke_leaderboard(repo_root=rr)
    return results
