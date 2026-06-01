"""GPU queue job stage expansion (manifest job -> stage list)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from harchoc.aug_smoke_runner import (
    DEFAULT_LOCKED_CONF_FROM,
    DEFAULT_OUT_DIR,
    find_smoke_entry,
    load_aug_smoke_index,
)
from harchoc.aug_smoke_train import (
    resolve_aug_smoke_aug_config,
    resolve_aug_smoke_train_config_path,
)

DEFAULT_EVAL_OUT_DIR = "reports/gpu_queue/eval"
DEFAULT_SUMMARIES_ROOT = "reports/gpu_queue/summaries"

__all__ = ["build_job_stages", "validate_job_files"]

def _resolve_job_hsp_max_det(job: dict[str, Any], *, repo_root: Path, train_config: str) -> int:
    """HSP eval max_det: job override, merged train eval section, or RT-DETR num_queries."""
    if job.get("max_det") is not None:
        return int(job["max_det"])
    from harchoc.rtdetr_limits import (
        is_rtdetr_model,
        rtdetr_eval_max_det,
        rtdetr_fields_from_train_json,
    )
    from harchoc.train_config import load_train_config_json

    cfg_path = (repo_root / train_config).resolve()
    merged = load_train_config_json(cfg_path, repo_root=repo_root)
    eval_section = merged.get("eval")
    if isinstance(eval_section, dict) and eval_section.get("max_det") is not None:
        return int(eval_section["max_det"])
    model = merged.get("model")
    if is_rtdetr_model(str(model) if model is not None else None):
        fields = rtdetr_fields_from_train_json(merged, path=str(cfg_path))
        return rtdetr_eval_max_det(int(fields["num_queries"]))
    return 3000


def _is_rtdetr_train_job(job: dict[str, Any], *, repo_root: Path) -> bool:
    from harchoc.rtdetr_limits import is_rtdetr_model
    from harchoc.train_config import load_train_config_json

    if str(job.get("kind") or "") == "rtdetr_smoke":
        return True
    cfg = job.get("train_config")
    if not cfg:
        return False
    merged = load_train_config_json((repo_root / str(cfg)).resolve(), repo_root=repo_root)
    return is_rtdetr_model(str(merged.get("model") or ""))


def validate_job_files(job: dict[str, Any], repo_root: Path) -> None:
    kind = str(job.get("kind") or "")
    if kind in ("aug_smoke", "train_compare", "vram_probe", "rtdetr_smoke", "amp_smoke", "sg_smoke", "aug_sweep_15", "aug_sweep_100"):
        cfg = job.get("train_config")
        if cfg and not (repo_root / str(cfg)).is_file():
            raise FileNotFoundError(f"job {job.get('id')}: missing train_config {cfg}")
    if kind == "aug_smoke":
        smoke_id = job.get("smoke_id") or job.get("id", "").replace("aug_smoke_", "")
        idx = load_aug_smoke_index(repo_root / (job.get("aug_index") or "configs/experiments/aug_smoke_index.json"))
        entry = find_smoke_entry(idx, str(smoke_id))
        tc = resolve_aug_smoke_train_config_path(
            entry,
            repo_root=repo_root,
            job_train_config=str(job.get("train_config") or "") or None,
        )
        if not (repo_root / tc).is_file():
            raise FileNotFoundError(f"aug_smoke {smoke_id}: missing train_config {tc}")

def _script_argv(script: str, tail: list[str]) -> list[str]:
    return [f"scripts/{script}" if not script.startswith("scripts/") else script, *tail]


def _smoke_weights_run_name(
    *,
    job: dict[str, Any],
    meta: dict[str, Any],
    run_name: str,
) -> str:
    """Ultralytics run dir for best.pt; defaults to eval artifact run_name."""
    return str(
        meta.get("weights_run_name")
        or job.get("weights_run_name")
        or run_name
    )


def _build_ultralytics_smoke_stages(
    stages: list[dict[str, Any]],
    *,
    cfg: str,
    name: str,
    locked: str,
    eval_only: bool = False,
    skip_eval: bool = False,
    train_skip_eval: bool = True,
    aug_config: str | None = None,
    eval_meta: dict[str, Any],
    summary_meta: dict[str, Any],
) -> None:
    """Append dry_run/train/eval/summary onto preflight *stages* (dry_run slot + gpu_wait)."""
    if not eval_only:
        dry_argv: list[str] = ["--config", cfg, "--name", name, "--dry-run"]
        train_argv: list[str] = ["--config", cfg, "--name", name]
        if aug_config:
            dry_argv.extend(["--aug-config", str(aug_config)])
            train_argv.extend(["--aug-config", str(aug_config)])
        if train_skip_eval:
            dry_argv.append("--skip-eval")
            train_argv.append("--skip-eval")
        stages[0] = {
            "stage_id": "dry_run",
            "argv": _script_argv("train.py", dry_argv),
            "mamba": True,
        }
        stages.append(
            {
                "stage_id": "train",
                "argv": _script_argv("train.py", train_argv),
                "mamba": True,
            }
        )
    if not skip_eval:
        stages.append(
            {
                "stage_id": "eval_test",
                "internal": "smoke_hsp_eval",
                "mamba": False,
                "meta": {
                    "run_name": name,
                    "train_config": cfg,
                    "locked_conf_from": locked,
                    **eval_meta,
                },
            }
        )
    stages.append(
        {
            "stage_id": "summary",
            "internal": "job_summary",
            "mamba": False,
            "meta": summary_meta,
        }
    )

def build_job_stages(
    job: dict[str, Any],
    *,
    repo_root: Path,
    defaults: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Expand job into stage dicts: {stage_id, argv, mamba, internal?}."""
    kind = str(job.get("kind") or "")
    defs = defaults or {}
    locked = str(job.get("locked_conf_from") or defs.get("locked_conf_from") or DEFAULT_LOCKED_CONF_FROM)
    stages: list[dict[str, Any]] = []

    if kind == "preflight":
        stages.append(
            {
                "stage_id": "validate_splits",
                "argv": _script_argv("validate_splits.py", ["--require-test"]),
                "mamba": True,
            }
        )
        stages.append(
            {
                "stage_id": "check_gpu",
                "argv": _script_argv(
                    "check_gpu.py",
                    ["--json-out", str(job.get("gpu_check_out") or "reports/gpu_queue/gpu_check.json")],
                ),
                "mamba": True,
            }
        )
        return stages

    if kind == "gpu_wait_only":
        stages.append({"stage_id": "gpu_wait", "internal": "gpu_wait", "mamba": False})
        return stages

    # Common preflight for GPU jobs
    if kind not in ("preflight",):
        stages.append(
            {
                "stage_id": "dry_run",
                "argv": [],
                "mamba": True,
                "internal": "dry_run",
            }
        )
        stages.append({"stage_id": "gpu_wait", "internal": "gpu_wait", "mamba": False})

    if kind == "vram_probe":
        cfg = str(job.get("train_config") or "configs/experiments/train_batch_probe_rtdetr-l.json")
        name = str(job.get("run_name") or "batch_probe_rtdetr-l")
        stages[0] = {
            "stage_id": "dry_run",
            "argv": _script_argv(
                "train.py",
                ["--config", cfg, "--name", name, "--dry-run", "--skip-eval"],
            ),
            "mamba": True,
        }
        stages.append(
            {
                "stage_id": "train",
                "argv": _script_argv(
                    "train.py",
                    ["--config", cfg, "--name", name, "--skip-eval"],
                ),
                "mamba": True,
            }
        )
        stages.append(
            {
                "stage_id": "summary",
                "internal": "job_summary",
                "mamba": False,
                "meta": {
                    "summary_kind": "vram_probe",
                    "summary_path": str(
                        job.get("summary_path")
                        or f"{DEFAULT_SUMMARIES_ROOT}/{job.get('id') or 'vram_probe'}.json"
                    ),
                },
            }
        )
        return stages

    if kind == "aug_smoke":
        smoke_id = str(job.get("smoke_id") or "")
        idx_path = str(job.get("aug_index") or "configs/experiments/aug_smoke_index.json")
        entry = find_smoke_entry(load_aug_smoke_index(repo_root / idx_path), smoke_id)
        cfg = resolve_aug_smoke_train_config_path(
            entry,
            repo_root=repo_root,
            job_train_config=str(job.get("train_config") or "") or None,
        )
        name = str(job.get("run_name") or entry.get("run_name") or "")
        if not cfg or not name:
            raise ValueError(f"aug_smoke {smoke_id}: missing train_config or run_name")
        eval_only = bool(job.get("eval_only") or entry.get("eval_only"))
        weights_run_name = str(
            job.get("weights_run_name") or entry.get("weights_run_name") or name
        )
        max_det = int(job.get("max_det") or entry.get("max_det") or 3000)
        aug = resolve_aug_smoke_aug_config(
            entry,
            repo_root=repo_root,
            job_aug_config=str(job.get("aug_config")) if job.get("aug_config") else None,
            train_config_path=cfg,
        )
        if aug is not None:
            aug = str(aug)
        summary_path = str(
            job.get("summary_path")
            or entry.get("summary")
            or f"{DEFAULT_OUT_DIR}/{smoke_id.lower()}_summary.json"
        )
        _build_ultralytics_smoke_stages(
            stages,
            cfg=cfg,
            name=name,
            locked=locked,
            eval_only=eval_only,
            aug_config=aug,
            eval_meta={
                "smoke_id": smoke_id,
                "weights_run_name": weights_run_name,
                "max_det": max_det,
                "out_dir": DEFAULT_OUT_DIR,
                "index_path": idx_path,
            },
            summary_meta={
                "summary_kind": "aug_smoke",
                "smoke_id": smoke_id,
                "run_name": name,
                "train_config": cfg,
                "locked_conf_from": locked,
                "index_path": idx_path,
                "summary_path": summary_path,
                "out_dir": DEFAULT_OUT_DIR,
            },
        )
        return stages

    if kind == "rtdetr_smoke":
        cfg = str(job.get("train_config") or "configs/experiments/train_rtdetr_queries_smoke_15ep.json")
        name = str(job.get("run_name") or "rtdetr_queries_smoke_15ep")
        out_dir = str(job.get("eval_out_dir") or DEFAULT_EVAL_OUT_DIR)
        summary_path = str(
            job.get("summary_path") or f"{DEFAULT_SUMMARIES_ROOT}/{job.get('id') or name}.json"
        )
        arch = ",".join(job.get("backlog") or [])
        _build_ultralytics_smoke_stages(
            stages,
            cfg=cfg,
            name=name,
            locked=locked,
            eval_meta={
                "max_det": int(job.get("max_det") or 1024),
                "out_dir": out_dir,
                "summary_path": summary_path,
                "arch_ticket": arch,
            },
            summary_meta={
                "summary_kind": "rtdetr",
                "run_name": name,
                "train_config": cfg,
                "locked_conf_from": locked,
                "out_dir": out_dir,
                "summary_path": summary_path,
                "arch_ticket": arch,
            },
        )
        return stages

    if kind == "amp_smoke":
        cfg = str(job.get("train_config") or "")
        name = str(job.get("run_name") or job.get("id") or "")
        eval_only = bool(job.get("eval_only"))
        skip_eval = bool(job.get("skip_eval"))
        out_dir = str(job.get("eval_out_dir") or "reports/hsp")
        summary_path = str(job.get("summary_path") or f"{out_dir}/{name}_summary.json")
        _build_ultralytics_smoke_stages(
            stages,
            cfg=cfg,
            name=name,
            locked=locked,
            eval_only=eval_only,
            skip_eval=skip_eval,
            eval_meta={"max_det": int(job.get("max_det") or 3000), "out_dir": out_dir},
            summary_meta={
                "summary_kind": "aug_sweep",
                "job_id": job.get("id"),
                "run_name": name,
                "train_config": cfg,
                "locked_conf_from": locked,
                "summary_path": summary_path,
                "out_dir": out_dir,
                "arch_ticket": ",".join(job.get("backlog") or ["P1-AMP-SMOKE"]),
            },
        )
        return stages

    if kind == "sg_smoke":
        cfg = str(job.get("train_config") or "")
        name = str(job.get("run_name") or job.get("id") or "")
        eval_only = bool(job.get("eval_only"))
        out_dir = str(job.get("eval_out_dir") or "reports/aug_smoke")
        summary_path = str(job.get("summary_path") or f"{out_dir}/{name}_summary.json")
        model_id = str(job.get("model_id") or "yolo_nas_s")
        skip_eval = bool(job.get("skip_eval"))
        if not eval_only:
            stages[0] = {
                "stage_id": "dry_run",
                "internal": "sg_train",
                "mamba": False,
                "meta": {
                    "train_config": cfg,
                    "run_name": name,
                    "model_id": model_id,
                    "dry_run": True,
                },
            }
            stages.append(
                {
                    "stage_id": "train",
                    "internal": "sg_train",
                    "mamba": False,
                    "meta": {
                        "train_config": cfg,
                        "run_name": name,
                        "model_id": model_id,
                    },
                }
            )
        if not skip_eval:
            stages.append(
                {
                    "stage_id": "eval_test",
                    "internal": "smoke_hsp_eval",
                    "mamba": False,
                    "meta": {
                        "run_name": name,
                        "train_config": cfg,
                        "locked_conf_from": locked,
                        "max_det": int(job.get("max_det") or 3000),
                        "out_dir": out_dir,
                        "model_id": model_id,
                    },
                }
            )
        stages.append(
            {
                "stage_id": "summary",
                "internal": "job_summary",
                "mamba": False,
                "meta": {
                    "summary_kind": "aug_sweep",
                    "job_id": job.get("id"),
                    "run_name": name,
                    "train_config": cfg,
                    "locked_conf_from": locked,
                    "summary_path": summary_path,
                    "out_dir": out_dir,
                    "arch_ticket": ",".join(job.get("backlog") or ["P1-SG"]),
                },
            }
        )
        return stages

    if kind == "train_compare":
        cfg = str(job.get("train_config") or "")
        name = str(job.get("run_name") or job.get("id") or "")
        skip_eval = bool(job.get("skip_eval", False))
        out_dir = str(job.get("eval_out_dir") or DEFAULT_EVAL_OUT_DIR)
        summary_path = str(
            job.get("summary_path") or f"{DEFAULT_SUMMARIES_ROOT}/{job.get('id') or name}.json"
        )
        arch = ",".join(job.get("backlog") or [])
        max_det = _resolve_job_hsp_max_det(job, repo_root=repo_root, train_config=cfg)
        summary_kind = "rtdetr" if _is_rtdetr_train_job(job, repo_root=repo_root) else "generic"
        _build_ultralytics_smoke_stages(
            stages,
            cfg=cfg,
            name=name,
            locked=locked,
            skip_eval=skip_eval,
            train_skip_eval=True,
            eval_meta={
                "max_det": max_det,
                "out_dir": out_dir,
                "summary_path": summary_path,
                "arch_ticket": arch,
            },
            summary_meta={
                "summary_kind": summary_kind,
                "run_name": name,
                "train_config": cfg,
                "locked_conf_from": locked,
                "out_dir": out_dir,
                "summary_path": summary_path,
                "arch_ticket": arch,
            },
        )
        return stages

    if kind in ("aug_sweep_15", "aug_sweep_100"):
        cfg = str(
            job.get("train_config") or "configs/experiments/train_aug_mosaic_sweep_smoke_15ep.json"
        )
        name = str(job.get("run_name") or job.get("id") or "")
        aug = job.get("aug_config")
        summary_path = str(job.get("summary_path") or f"reports/aug_smoke/{name}_summary.json")
        _build_ultralytics_smoke_stages(
            stages,
            cfg=cfg,
            name=name,
            locked=locked,
            aug_config=str(aug) if aug else None,
            eval_meta={"max_det": 3000, "out_dir": DEFAULT_OUT_DIR},
            summary_meta={
                "summary_kind": "aug_sweep",
                "job_id": job.get("id"),
                "run_name": name,
                "train_config": cfg,
                "locked_conf_from": locked,
                "summary_path": summary_path,
                "out_dir": DEFAULT_OUT_DIR,
                "arch_ticket": ",".join(job.get("backlog") or []),
            },
        )
        return stages

    if kind == "zoo_matrix_train":
        out = str(job.get("out") or "reports/hsp/matrix_train.json")
        matrix_group = str(job.get("matrix_group") or "").strip()
        bench_argv: list[str] = []
        if matrix_group:
            bench_argv.extend(["--group", matrix_group])
        stages = [
            {
                "stage_id": "dry_run",
                "argv": _script_argv(
                    "benchmark_matrix.py",
                    ["--dry-run", "--out", str(job.get("plan_out") or "reports/hsp/matrix_plan.json")]
                    + bench_argv,
                ),
                "mamba": True,
            },
            {
                "stage_id": "rtdetr_15ep_gate",
                "internal": "zoo_rtdetr_gate",
                "mamba": False,
                "meta": {"matrix_group": matrix_group},
            },
            {"stage_id": "gpu_wait", "internal": "gpu_wait", "mamba": False},
            {
                "stage_id": "train",
                "argv": _script_argv(
                    "benchmark_matrix.py",
                    [
                        "--no-dry-run",
                        "--runs-dir",
                        str(job.get("runs_dir") or "runs/hsp_zoo"),
                        "--train-out",
                        out,
                    ]
                    + bench_argv,
                ),
                "mamba": True,
            },
            {"stage_id": "summary", "internal": "job_summary", "mamba": False, "meta": {"summary_kind": "generic"}},
        ]
        return stages

    if kind == "cv_fold_train":
        folds = int(job.get("folds") or 5)
        splits_out = str(job.get("splits_out") or "reports/cv_folds")
        stages = [
            {
                "stage_id": "cv_splits",
                "argv": _script_argv(
                    "cv_eval.py",
                    ["--write-fold-splits", splits_out, "--folds", str(folds)],
                ),
                "mamba": False,
            },
        ]
        cfg = str(job.get("train_config") or "configs/experiments/train_yolov8m_baseline.json")
        for i in range(folds):
            name = f"cv_fold_{i}"
            stages.append({"stage_id": f"gpu_wait_fold_{i}", "internal": "gpu_wait", "mamba": False})
            stages.append(
                {
                    "stage_id": f"train_fold_{i}",
                    "argv": _script_argv(
                        "train.py",
                        ["--config", cfg, "--name", name, "--skip-eval"],
                    ),
                    "mamba": True,
                    "meta": {"fold": i, "note": "fold-specific data.yaml wiring deferred; uses repo splits"},
                }
            )
        stages.append({"stage_id": "summary", "internal": "job_summary", "mamba": False, "meta": {"summary_kind": "generic"}})
        return stages

    raise ValueError(f"unsupported job kind: {kind!r}")
