"""GPU queue job runner: subprocess stages, state, orchestration."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from harchoc.aug_smoke_runner import (
    DEFAULT_LOCKED_CONF_FROM,
    DEFAULT_OUT_DIR,
    finalize_smoke_job,
    resolve_train_weights,
    run_smoke_hsp_eval_chain,
)
from harchoc.gpu_exclusive import acquire_gpu_exclusive, release_gpu_exclusive
from harchoc.gpu_queue_manifest import load_gpu_queue_manifest
from harchoc.gpu_queue_skip import (
    _prune_dry_run_log_stubs,
    should_skip_job,
    wait_gpu_free,
)
from harchoc.gpu_queue_stages import build_job_stages, validate_job_files
from harchoc.gpu_queue_stages import _smoke_weights_run_name
from harchoc.json_io import load_json_dict
from harchoc.manuscript_repro import _format_cmd
from harchoc.queue_notify import (
    notify_queue_job,
    notify_queue_manifest_complete,
)

GPU_QUEUE_RUN_SCHEMA = "gpu_queue_run.v1"
DEFAULT_STATE_PATH = "reports/gpu_queue/run_state.json"
DEFAULT_LOG_ROOT = "reports/gpu_queue/logs"
DEFAULT_JOBS_ROOT = "reports/gpu_queue/jobs"
DEFAULT_SUMMARIES_ROOT = "reports/gpu_queue/summaries"
DEFAULT_EVAL_OUT_DIR = "reports/gpu_queue/eval"
DEFAULT_MIN_FREE_MIB = 5500

_LEADERBOARD_JOB_KINDS = frozenset(
    {"aug_smoke", "aug_sweep_15", "aug_sweep_100", "amp_smoke", "sg_smoke"}
)

__all__ = [
    "DEFAULT_EVAL_OUT_DIR",
    "DEFAULT_JOBS_ROOT",
    "DEFAULT_LOG_ROOT",
    "DEFAULT_MIN_FREE_MIB",
    "DEFAULT_STATE_PATH",
    "DEFAULT_SUMMARIES_ROOT",
    "GPU_QUEUE_RUN_SCHEMA",
    "GpuQueueError",
    "load_run_state",
    "repair_resume_state",
    "run_gpu_queue",
    "run_job",
    "save_run_state",
]

class GpuQueueError(Exception):
    """Raised when a queue stage fails."""

    def __init__(self, *, job_id: str, stage_id: str, exit_code: int, log_path: str, hint: str = "") -> None:
        self.job_id = job_id
        self.stage_id = stage_id
        self.exit_code = exit_code
        self.log_path = log_path
        self.hint = hint
        super().__init__(f"job {job_id!r} stage {stage_id!r} failed (exit {exit_code}): {hint or log_path}")


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
def _write_json(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2) + "\n", encoding="utf-8")


def _load_sg_train_recipe(repo_root: Path, train_config: str) -> dict[str, Any]:
    from harchoc.train_config import resolve_train_config_extends

    p = (repo_root / train_config).resolve()
    raw = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"train config must be object: {p}")
    return resolve_train_config_extends(raw, repo_root=repo_root, config_path=p)


def _sg_smoke_requires_supergradients(job: dict[str, Any], *, repo_root: Path) -> bool:
    """Train always needs SG; eval-only needs SG only when checkpoint is .pth."""
    if not bool(job.get("eval_only")):
        return True
    run_name = str(job.get("run_name") or "")
    weights = resolve_train_weights(repo_root=repo_root, run_name=run_name) if run_name else None
    if weights is None:
        return True
    return weights.suffix.lower() == ".pth"


def _mamba_env() -> str:
    return os.environ.get("HARCHOC_MAMBA_ENV", "harchoc")


def _finalize_finetune_job_summary(
    *,
    repo_root: Path,
    job: dict[str, Any],
    meta: dict[str, Any],
    dry_run: bool,
) -> int:
    """Transcript for finetune_tray jobs (finetune_run.v1 already at finetune_out)."""
    job_id = str(job.get("id") or "finetune_tray")
    finetune_out = str(
        meta.get("finetune_out")
        or job.get("out")
        or job.get("finetune_out")
        or f"reports/transfer/finetune_queue_{job_id}.json"
    )
    if dry_run:
        print(f"# finetune summary {job_id} -> {finetune_out}")
        return 0

    fin_path = repo_root / finetune_out
    if not fin_path.is_file():
        return 1
    fin_obj = load_json_dict(fin_path)
    ok = fin_obj.get("status") == "ok"
    outcome = fin_obj.get("finetune_outcome") or {}
    transcript: dict[str, Any] = {
        "schema_version": "gpu_queue_job.v1",
        "job_id": job_id,
        "kind": job.get("kind"),
        "status": "complete" if ok else "failed",
        "finished_at": _utc_now(),
        "tray_key": job.get("tray_key"),
        "stage": job.get("stage"),
        "finetune_out": finetune_out,
        "weights": fin_obj.get("weights"),
        "finetune_outcome": outcome,
        "canonical_gate_passed": (outcome.get("canonical_gate") or {}).get("passed"),
    }
    hold = outcome.get("tray_holdout") or []
    if hold and isinstance(hold[0], dict):
        transcript["tray_count_mae_after"] = hold[0].get("count_mae_after")
    _write_json(repo_root / DEFAULT_JOBS_ROOT / f"{job_id}.json", transcript)
    summary_path = str(meta.get("summary_path") or finetune_out)
    if summary_path != finetune_out:
        _write_json(repo_root / summary_path, transcript)
    return 0 if ok else 1


def _finalize_job_summary(
    *,
    repo_root: Path,
    job: dict[str, Any],
    job_context: dict[str, Any],
    meta: dict[str, Any],
    dry_run: bool,
) -> int:
    """Build summary JSON, job transcript, and optional aug index patch."""
    summary_kind = str(meta.get("summary_kind") or "generic")
    run_name = str(job_context.get("run_name") or meta.get("run_name") or job.get("run_name") or "")
    if dry_run:
        print(f"# summary {job.get('id')} ({summary_kind})")
        return 0

    if summary_kind == "vram_probe":
        summary_path = str(
            meta.get("summary_path")
            or job.get("summary_path")
            or f"{DEFAULT_SUMMARIES_ROOT}/{job.get('id')}.json"
        )
        weights = job_context.get("weights") or resolve_train_weights(repo_root=repo_root, run_name=run_name)
        payload = {
            "schema_version": "gpu_queue_job.v1",
            "job_id": job.get("id"),
            "kind": job.get("kind"),
            "status": "complete" if weights else "failed",
            "finished_at": _utc_now(),
            "run_name": run_name,
            "weights": str(weights) if weights else None,
        }
        _write_json(repo_root / summary_path, payload)
        _write_json(repo_root / DEFAULT_JOBS_ROOT / f"{job.get('id')}.json", payload)
        return 0 if weights else 1

    weights = job_context.get("weights") or resolve_train_weights(repo_root=repo_root, run_name=run_name)
    if not weights and str(job.get("kind")) != "amp_smoke":
        return 1

    out_dir = str(job_context.get("eval_out_dir") or meta.get("out_dir") or DEFAULT_OUT_DIR)
    summary_path = str(
        meta.get("summary_path")
        or job.get("summary_path")
        or f"{DEFAULT_SUMMARIES_ROOT}/{job.get('id') or run_name}.json"
    )

    if job.get("skip_eval") or not job_context.get("eval_artifacts"):
        probe_payload = {
            "schema_version": "gpu_queue_job.v1",
            "job_id": job.get("id"),
            "kind": job.get("kind"),
            "status": "complete",
            "finished_at": _utc_now(),
            "run_name": run_name,
            "weights": str(weights) if weights else None,
        }
        _write_json(repo_root / summary_path, probe_payload)
        _write_json(repo_root / DEFAULT_JOBS_ROOT / f"{job.get('id')}.json", probe_payload)
        return 0 if weights or str(job.get("kind")) == "amp_smoke" else 1

    if not weights:
        return 1

    smoke_id = str(meta.get("smoke_id") or meta.get("job_id") or job.get("id") or run_name)
    index_path = str(
        meta.get("index_path") or job.get("aug_index") or "configs/experiments/aug_smoke_index.json"
    )
    payload = finalize_smoke_job(
        repo_root=repo_root,
        run_name=run_name,
        train_config=str(meta.get("train_config") or job.get("train_config") or ""),
        weights=Path(str(weights)),
        summary_path=summary_path,
        smoke_id=smoke_id,
        locked_conf_from=str(meta.get("locked_conf_from") or DEFAULT_LOCKED_CONF_FROM),
        out_dir=out_dir,
        arch_ticket=str(meta.get("arch_ticket") or ",".join(job.get("backlog") or [])),
        index_path=index_path,
        patch_index=summary_kind == "aug_smoke",
        train_runtime_s=job_context.get("train_runtime_s"),
        refresh_leaderboard=False,
    )
    prefix = f"{out_dir}/{run_name}"
    error_json = str((repo_root / f"{prefix}_error.json").resolve())
    transcript: dict[str, Any] = {
        "job_id": job.get("id"),
        "status": payload.get("status"),
        "test_count_mae": payload.get("test_count_mae"),
        "summary_path": summary_path,
        "weights": str(weights),
    }
    if summary_kind in ("generic", "rtdetr"):
        transcript["eval_error_json"] = error_json
    _write_json(repo_root / DEFAULT_JOBS_ROOT / f"{job.get('id')}.json", transcript)
    return 0 if payload.get("status") == "complete" else 1


def _maybe_refresh_aug_leaderboard(
    *,
    repo_root: Path,
    job: dict[str, Any],
    dry_run: bool,
) -> None:
    if dry_run or str(job.get("kind") or "") not in _LEADERBOARD_JOB_KINDS:
        return
    from harchoc.aug_smoke_leaderboard import refresh_aug_smoke_leaderboard

    refresh_aug_smoke_leaderboard(repo_root=repo_root)


def _run_smoke_hsp_eval_stage(
    *,
    job: dict[str, Any],
    meta: dict[str, Any],
    repo_root: Path,
    log_path: Path,
    dry_run: bool,
    job_context: dict[str, Any],
) -> int:
    run_name = str(meta.get("run_name") or job.get("run_name") or "")
    weights_run_name = _smoke_weights_run_name(job=job, meta=meta, run_name=run_name)
    out_dir = str(meta.get("out_dir") or DEFAULT_OUT_DIR)
    max_det = int(meta.get("max_det") or job.get("max_det") or 3000)
    model_id = str(meta.get("model_id") or job.get("model_id") or "yolo_nas_s")
    locked = str(meta.get("locked_conf_from") or DEFAULT_LOCKED_CONF_FROM)
    weights_for_dry = resolve_train_weights(
        repo_root=repo_root, run_name=weights_run_name
    ) or (repo_root / "runs/placeholder/weights/best.pt")

    with log_path.open("w", encoding="utf-8") as f:

        def _on(stage_id: str, argv: list[str]) -> None:
            if stage_id == "sg_export":
                f.write(f"# {stage_id}: backend auto from weights suffix\n")
            else:
                f.write(f"# {stage_id}: {_format_cmd(argv, mamba=True)}\n")

        if dry_run:
            run_smoke_hsp_eval_chain(
                repo_root=repo_root,
                run_name=run_name,
                weights=weights_for_dry,
                locked_conf_from=locked,
                out_dir=out_dir,
                max_det=max_det,
                model_id=model_id,
                dry_run=True,
                on_stage=_on,
            )
            return 0

        weights = resolve_train_weights(repo_root=repo_root, run_name=weights_run_name)
        if weights is None:
            f.write(f"weights not found for run {weights_run_name}\n")
            return 1
        job_context["weights"] = str(weights)
        job_context["run_name"] = run_name
        job_context["weights_run_name"] = weights_run_name
        job_context["eval_out_dir"] = out_dir
        try:
            artifacts = run_smoke_hsp_eval_chain(
                repo_root=repo_root,
                run_name=run_name,
                weights=weights,
                locked_conf_from=locked,
                out_dir=out_dir,
                max_det=max_det,
                model_id=model_id,
                dry_run=False,
                on_stage=_on,
            )
            job_context["eval_artifacts"] = artifacts
            job_context.update(meta)
        except RuntimeError as ex:
            f.write(str(ex) + "\n")
            return 1
    return 0
def _tail_log(path: Path, n: int = 40) -> list[str]:
    if not path.is_file():
        return []
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return lines[-n:]


def _run_subprocess_stage(
    *,
    argv: list[str],
    mamba: bool,
    repo_root: Path,
    log_path: Path,
    env: dict[str, str],
    dry_run: bool,
) -> int:
    mamba_env = _mamba_env()
    if mamba:
        cmd = ["mamba", "run", "-n", mamba_env, "python", *argv]
    else:
        cmd = [sys.executable, *argv]
    line = _format_cmd([argv[0].replace("scripts/", "scripts/"), *argv[1:]], mamba=mamba)
    if dry_run:
        print(f"# {line}")
        return 0
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log_f:
        log_f.write(f"# started {_utc_now()}\n# cmd: {line}\n\n")
        log_f.flush()
        proc = subprocess.run(cmd, cwd=str(repo_root), env=env, stdout=log_f, stderr=subprocess.STDOUT)
        log_f.write(f"\n# exit_code={proc.returncode} finished {_utc_now()}\n")
    return int(proc.returncode)


def _run_internal_stage(
    stage: dict[str, Any],
    *,
    job: dict[str, Any],
    repo_root: Path,
    log_path: Path,
    dry_run: bool,
    job_context: dict[str, Any],
    min_free_mib: int,
) -> int:
    internal = stage.get("internal")
    if internal == "gpu_wait":
        with log_path.open("w", encoding="utf-8") as f:
            try:
                def _log(msg: str) -> None:
                    f.write(msg)

                info = wait_gpu_free(min_free_mib=min_free_mib, dry_run=dry_run, log_fn=_log)
                f.write(json.dumps(info, indent=2) + "\n")
                return 0
            except TimeoutError as ex:
                f.write(str(ex) + "\n")
                return 1

    if internal == "dry_run":
        stages = build_job_stages(job, repo_root=repo_root)
        for st in stages:
            if st.get("stage_id") == "dry_run" and st.get("argv"):
                return _run_subprocess_stage(
                    argv=st["argv"],
                    mamba=bool(st.get("mamba", True)),
                    repo_root=repo_root,
                    log_path=log_path,
                    env=dict(os.environ),
                    dry_run=dry_run,
                )
        return 0

    if internal == "zoo_rtdetr_gate":
        from harchoc.rtdetr_zoo_gate import (
            check_zoo_core_rtdetr_15ep_gates,
            format_zoo_core_rtdetr_gate_blockers,
        )

        matrix_group = str((stage.get("meta") or {}).get("matrix_group") or job.get("matrix_group") or "")
        with log_path.open("w", encoding="utf-8") as f:
            if matrix_group != "zoo_core":
                f.write(f"# zoo_rtdetr_gate: skipped (matrix_group={matrix_group!r})\n")
                return 0
            blockers = format_zoo_core_rtdetr_gate_blockers(repo_root)
            f.write(blockers + "\n")
            gates = check_zoo_core_rtdetr_15ep_gates(repo_root=repo_root)
            f.write(json.dumps({"gates": gates}, indent=2) + "\n")
            if dry_run:
                f.write("# dry_run: gate logged only; live train blocked when any gate fails\n")
                return 0
            if not all(g["passed"] for g in gates):
                return 1
        return 0

    meta = stage.get("meta") or {}

    if internal == "sg_train":
        cfg = str(meta.get("train_config") or job.get("train_config") or "")
        run_name = str(meta.get("run_name") or job.get("run_name") or "")
        model_id = str(meta.get("model_id") or "yolo_nas_s")
        is_cfg_dry = bool(meta.get("dry_run"))
        recipe = _load_sg_train_recipe(repo_root, cfg) if cfg else {}
        if dry_run or is_cfg_dry:
            with log_path.open("w", encoding="utf-8") as f:
                f.write(
                    f"# sg_train dry_run: model_id={model_id} run_name={run_name} "
                    f"epochs={recipe.get('epochs')} imgsz={recipe.get('imgsz')} batch={recipe.get('batch')}\n"
                )
            return 0
        from harchoc.datasets import resolve_dataset
        from harchoc.supergradients_train import train_bench_run

        spec = resolve_dataset(
            manifest_path=repo_root / "data/manifest.json",
            default_dataset_name="sunflower",
        )
        result = train_bench_run(
            model_id=str(recipe.get("model_id") or model_id),
            dataset_root=spec.root,
            runs_dir=repo_root / "runs",
            run_name=run_name,
            epochs=int(recipe.get("epochs") or 15),
            imgsz=int(recipe.get("imgsz") or 1280),
            batch=int(recipe.get("batch") or 1),
            seed=int(recipe.get("seed") or 0),
        )
        if result.get("status") != "ok":
            with log_path.open("w", encoding="utf-8") as f:
                f.write(json.dumps(result, indent=2) + "\n")
            return int(result.get("returncode") or 1)
        job_context["weights"] = result.get("weights")
        job_context["run_name"] = run_name
        return 0

    if internal == "smoke_hsp_eval":
        return _run_smoke_hsp_eval_stage(
            job=job,
            meta=meta,
            repo_root=repo_root,
            log_path=log_path,
            dry_run=dry_run,
            job_context=job_context,
        )

    if internal == "finetune_summary":
        return _finalize_finetune_job_summary(
            repo_root=repo_root,
            job=job,
            meta=meta,
            dry_run=dry_run,
        )

    if internal in (
        "job_summary",
        "aug_smoke_summary",
        "aug_sweep_summary",
        "generic_train_summary",
        "rtdetr_summary",
        "vram_probe_summary",
    ):
        stage_meta = dict(meta)
        if internal == "vram_probe_summary" and "summary_kind" not in stage_meta:
            stage_meta["summary_kind"] = "vram_probe"
        elif internal == "aug_smoke_summary":
            stage_meta.setdefault("summary_kind", "aug_smoke")
        elif internal == "aug_sweep_summary":
            stage_meta.setdefault("summary_kind", "aug_sweep")
        elif internal == "rtdetr_summary":
            stage_meta.setdefault("summary_kind", "rtdetr")
        elif internal == "generic_train_summary":
            stage_meta.setdefault("summary_kind", "generic")
        return _finalize_job_summary(
            repo_root=repo_root,
            job=job,
            job_context=job_context,
            meta=stage_meta,
            dry_run=dry_run,
        )

    with log_path.open("w") as f:
        f.write(f"unknown internal stage: {internal}\n")
    return 1


def run_job(
    job: dict[str, Any],
    *,
    repo_root: Path,
    defaults: dict[str, Any],
    dry_run: bool,
    min_free_mib: int,
    log_root: Path,
) -> dict[str, Any]:
    job_id = str(job.get("id") or "unknown")
    validate_job_files(job, repo_root)
    stages = build_job_stages(job, repo_root=repo_root, defaults=defaults)
    job_env = {
        **dict(os.environ),
        **{k: str(v) for k, v in (defaults.get("env") or {}).items()},
        **{k: str(v) for k, v in (job.get("env") or {}).items()},
        "HARCHOC_GPU_QUEUE_CHILD": "1",
        "HARCHOC_GPU_QUEUE_JOB_ID": job_id,
    }
    job_context: dict[str, Any] = {"run_name": job.get("run_name")}
    stage_results: list[dict[str, Any]] = []
    train_start: float | None = None

    for stage in stages:
        stage_id = str(stage.get("stage_id") or "unknown")
        log_path = log_root / job_id / f"{stage_id}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        if stage.get("internal"):
            if stage_id == "train" or stage_id.startswith("train_"):
                train_start = time.monotonic()
            rc = _run_internal_stage(
                stage,
                job=job,
                repo_root=repo_root,
                log_path=log_path,
                dry_run=dry_run,
                job_context=job_context,
                min_free_mib=min_free_mib,
            )
            if stage_id == "train" or stage_id.startswith("train_"):
                if train_start is not None and not dry_run:
                    job_context["train_runtime_s"] = time.monotonic() - train_start
        else:
            argv = stage.get("argv") or []
            if stage_id == "train":
                train_start = time.monotonic()
                run_name = str(job_context.get("run_name") or job.get("run_name") or "")
                if (
                    not dry_run
                    and run_name
                    and resolve_train_weights(repo_root=repo_root, run_name=run_name) is not None
                ):
                    w = resolve_train_weights(repo_root=repo_root, run_name=run_name)
                    job_context["weights"] = str(w)
                    log_path.write_text(f"train skipped: weights exist ({w})\n", encoding="utf-8")
                    stage_results.append(
                        {"stage_id": stage_id, "exit_code": 0, "log_path": str(log_path), "skipped": True}
                    )
                    if train_start is not None:
                        job_context["train_runtime_s"] = time.monotonic() - train_start
                    continue
            rc = _run_subprocess_stage(
                argv=list(argv),
                mamba=bool(stage.get("mamba", True)),
                repo_root=repo_root,
                log_path=log_path,
                env=job_env,
                dry_run=dry_run,
            )
            if stage_id == "train" and train_start is not None and not dry_run:
                job_context["train_runtime_s"] = time.monotonic() - train_start

        stage_results.append({"stage_id": stage_id, "exit_code": rc, "log_path": str(log_path)})
        if rc != 0:
            hint = ""
            if "OOM" in "\n".join(_tail_log(log_path)):
                hint = "CUDA OOM — ensure exclusive GPU (wait for prior job)"
            raise GpuQueueError(
                job_id=job_id,
                stage_id=stage_id,
                exit_code=rc,
                log_path=str(log_path),
                hint=hint,
            )

    _maybe_refresh_aug_leaderboard(repo_root=repo_root, job=job, dry_run=dry_run)
    status = "dry_run_complete" if dry_run else "complete"
    return {"job_id": job_id, "status": status, "stages": stage_results, "context": job_context}


def load_run_state(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"schema_version": GPU_QUEUE_RUN_SCHEMA, "completed": [], "failed": None}
    return load_json_dict(path)


def repair_resume_state(
    state: dict[str, Any],
    *,
    manifest_path: Path,
    dry_run: bool,
) -> dict[str, Any]:
    """Drop stale fields when resuming a different manifest or switching dry_run → live."""
    manifest = str(manifest_path.resolve())
    out = dict(state)
    manifest_changed = str(out.get("manifest") or "") != manifest
    dry_mismatch = bool(out.get("dry_run")) != bool(dry_run)
    if manifest_changed:
        out["completed"] = []
        out["skipped"] = []
        out["failed"] = None
        out["started_at"] = _utc_now()
    if manifest_changed or dry_mismatch or out.get("finished_at"):
        out["manifest"] = manifest
        out["dry_run"] = dry_run
        out["finished_at"] = None
    return out


def save_run_state(path: Path, state: dict[str, Any]) -> None:
    state["schema_version"] = GPU_QUEUE_RUN_SCHEMA
    state["updated_at"] = _utc_now()
    _write_json(path, state)


def run_gpu_queue(
    manifest_path: str | Path,
    *,
    repo_root: str | Path | None = None,
    dry_run: bool = False,
    resume: bool = False,
    job_filter: str | None = None,
    state_path: str | Path = DEFAULT_STATE_PATH,
    min_free_mib: int = DEFAULT_MIN_FREE_MIB,
) -> int:
    rr = Path(repo_root or ".").expanduser().resolve()
    manifest = load_gpu_queue_manifest(manifest_path, repo_root=rr)
    defaults = manifest.get("defaults") or {}
    jobs: list[dict[str, Any]] = list(manifest.get("jobs") or [])
    if job_filter:
        jobs = [j for j in jobs if str(j.get("id")) == job_filter]

    st_path = rr / state_path
    if resume:
        state = repair_resume_state(
            load_run_state(st_path),
            manifest_path=Path(manifest_path).resolve(),
            dry_run=dry_run,
        )
    else:
        state = {
            "schema_version": GPU_QUEUE_RUN_SCHEMA,
            "manifest": str(Path(manifest_path).resolve()),
            "started_at": _utc_now(),
            "dry_run": dry_run,
            "completed": [],
            "skipped": [],
            "failed": None,
            "current_job": None,
        }
    if not resume:
        state["started_at"] = _utc_now()
        state["dry_run"] = dry_run

    log_root = rr / DEFAULT_LOG_ROOT
    completed_ids = set(state.get("completed") or [])

    if not dry_run:
        acquire_gpu_exclusive(repo_root=rr, owner="gpu_queue")

    try:
        for job in jobs:
            job_id = str(job.get("id") or "")
            if job_id in completed_ids:
                continue
            skip, reason = should_skip_job(job, repo_root=rr)
            if skip:
                skipped = state.setdefault("skipped", [])
                if not any(s.get("job_id") == job_id for s in skipped):
                    skipped.append({"job_id": job_id, "reason": reason})
                save_run_state(st_path, state)
                print(f"skip {job_id}: {reason}")
                continue
            state["skipped"] = [s for s in (state.get("skipped") or []) if s.get("job_id") != job_id]

            state["current_job"] = job_id
            save_run_state(st_path, state)
            if not dry_run:
                print(f"=== job {job_id} ({job.get('kind')}) ===")

            try:
                if str(job.get("kind")) == "sg_smoke" and _sg_smoke_requires_supergradients(job, repo_root=rr):
                    try:
                        import super_gradients  # noqa: F401
                    except ImportError:
                        state.setdefault("skipped", []).append(
                            {"job_id": job_id, "reason": "super_gradients not installed"}
                        )
                        save_run_state(st_path, state)
                        print(f"skip {job_id}: super_gradients not installed")
                        continue

                result = run_job(
                    job,
                    repo_root=rr,
                    defaults=defaults,
                    dry_run=dry_run,
                    min_free_mib=min_free_mib,
                    log_root=log_root,
                )
                if result.get("status") != "dry_run_complete":
                    state.setdefault("completed", []).append(job_id)
                state["current_job"] = None
                if not dry_run:
                    job_out = rr / DEFAULT_JOBS_ROOT / f"{job_id}.json"
                    _write_json(job_out, result)
                notify_queue_job(
                    repo_root=rr,
                    job=job,
                    status="complete",
                    dry_run=dry_run,
                )
                save_run_state(st_path, state)
            except GpuQueueError as ex:
                state["failed"] = {
                    "job_id": ex.job_id,
                    "stage_id": ex.stage_id,
                    "exit_code": ex.exit_code,
                    "log_path": ex.log_path,
                    "hint": ex.hint,
                    "tail": _tail_log(Path(ex.log_path)),
                }
                state["current_job"] = None
                fail_out = rr / DEFAULT_JOBS_ROOT / f"{ex.job_id}.json"
                _write_json(
                    fail_out,
                    {
                        "job_id": ex.job_id,
                        "status": "failed",
                        "stage_id": ex.stage_id,
                        "exit_code": ex.exit_code,
                        "log_path": ex.log_path,
                        "hint": ex.hint,
                    },
                )
                notify_queue_job(
                    repo_root=rr,
                    job=job,
                    status="failed",
                    dry_run=dry_run,
                    stage_id=ex.stage_id,
                    exit_code=ex.exit_code,
                    hint=ex.hint or None,
                )
                save_run_state(st_path, state)
                if not dry_run:
                    print(f"FAILED {ex.job_id} @ {ex.stage_id}: {ex.hint or ex.log_path}", file=sys.stderr)
                return 1
            except (FileNotFoundError, ValueError, KeyError) as ex:
                state["failed"] = {"job_id": job_id, "stage_id": "preflight", "error": str(ex)}
                notify_queue_job(
                    repo_root=rr,
                    job=job,
                    status="failed",
                    dry_run=dry_run,
                    stage_id="preflight",
                    hint=str(ex),
                )
                save_run_state(st_path, state)
                if not dry_run:
                    print(f"FAILED {job_id} preflight: {ex}", file=sys.stderr)
                return 1

        state["current_job"] = None
        state["finished_at"] = _utc_now()
        save_run_state(st_path, state)
        notify_queue_manifest_complete(
            repo_root=rr,
            manifest_path=str(manifest_path),
            completed=list(state.get("completed") or []),
            skipped=list(state.get("skipped") or []),
            dry_run=dry_run,
        )
        if dry_run:
            pruned = _prune_dry_run_log_stubs(log_root)
            if pruned:
                print(f"# pruned dry-run log stubs: {', '.join(pruned)}")
        return 0
    finally:
        if not dry_run:
            release_gpu_exclusive(repo_root=rr)
