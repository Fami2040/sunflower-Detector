"""Runnable smoke verification for ``docs/manuscript/now_todos.md`` pending work."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from harchoc.json_io import load_json_dict


@dataclass
class StageResult:
    stage_id: str
    status: str  # ok | skip | fail
    detail: str
    exit_code: int | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.stage_id,
            "status": self.status,
            "detail": self.detail,
            "exit_code": self.exit_code,
        }


def load_smoke_bundle(path: str | Path) -> dict[str, Any]:
    p = Path(path).expanduser().resolve()
    doc = load_json_dict(p)
    if str(doc.get("schema_version") or "") != "now_todos_smoke_bundle.v1":
        raise ValueError(f"Unsupported bundle schema at {p}")
    return doc


def _repo_paths_exist(repo_root: Path, rel_paths: list[str], *, stage_id: str) -> StageResult:
    missing = [r for r in rel_paths if not (repo_root / r).is_file()]
    if missing:
        return StageResult(
            stage_id,
            "fail",
            f"missing: {', '.join(missing)}",
            exit_code=1,
        )
    return StageResult(stage_id, "ok", f"{len(rel_paths)} config paths exist")


def _matrix_train_gate(
    repo_root: Path,
    spec: dict[str, Any],
) -> StageResult:
    from harchoc.queue_skip_gates import matrix_train_verified

    train_out = str(spec.get("train_out") or "reports/hsp/matrix_train.json")
    group = str(spec.get("matrix_group") or "zoo_yolo_only")
    accept = bool(spec.get("accept_skipped_no_weights", False))
    expect_blocked = bool(spec.get("expect_blocked", False))
    ok, detail = matrix_train_verified(
        repo_root,
        train_out,
        group,
        accept_skipped_no_weights=accept,
    )
    sid = str(spec.get("id") or "matrix_train_gate")
    if expect_blocked:
        if ok:
            return StageResult(sid, "fail", f"expected gate blocked but verified: {detail}")
        return StageResult(sid, "ok", f"gate blocked as expected ({detail or 'incomplete matrix'})")
    if ok:
        return StageResult(sid, "ok", detail or "matrix_train verified")
    return StageResult(sid, "fail", detail or "matrix_train not verified", exit_code=1)


def _run_experiment(
    repo_root: Path,
    command: str,
    *,
    config: str | None = None,
    kwargs: dict[str, Any] | None = None,
    env: dict[str, str] | None = None,
) -> tuple[int, str]:
    script = repo_root / "scripts" / "experiment.py"
    argv = [sys.executable, str(script)]
    if config:
        argv.extend(["--config", str(config)])
    argv.append(command)
    kw = dict(kwargs or {})
    if kw.get("dry_run"):
        argv.append("--dry-run")
    for key, val in kw.items():
        if key in ("dry_run",):
            continue
        flag = key.replace("_", "-")
        if isinstance(val, bool):
            if val:
                argv.append(f"--{flag}")
        else:
            argv.extend([f"--{flag}", str(val)])
    run_env = os.environ.copy()
    run_env.setdefault("PYTHONPATH", str(repo_root))
    run_env.setdefault("HARCHOC_ALLOW_BASE_PYTHON", "1")
    if env:
        run_env.update({str(k): str(v) for k, v in env.items()})
    proc = subprocess.run(
        argv,
        cwd=str(repo_root),
        env=run_env,
        capture_output=True,
        text=True,
    )
    tail = (proc.stderr or proc.stdout or "").strip()[-500:]
    return proc.returncode, tail


def _stage_skip_missing(repo_root: Path, spec: dict[str, Any]) -> StageResult | None:
    for rel in spec.get("skip_if_missing") or []:
        if not (repo_root / str(rel)).is_file():
            sid = str(spec.get("id") or "unknown")
            return StageResult(sid, "skip", f"missing prerequisite: {rel}")
    for rel in spec.get("require_artifacts") or []:
        if not (repo_root / str(rel)).is_file():
            sid = str(spec.get("id") or "unknown")
            return StageResult(sid, "skip", f"missing artifact: {rel}")
    return None


def run_stage(repo_root: Path, spec: dict[str, Any]) -> StageResult:
    sid = str(spec.get("id") or spec.get("type") or "stage")
    skipped = _stage_skip_missing(repo_root, spec)
    if skipped is not None:
        return skipped

    stype = str(spec.get("type") or "").strip()
    if stype == "config_paths_exist":
        return _repo_paths_exist(
            repo_root, [str(p) for p in spec.get("paths") or []], stage_id=sid
        )
    if stype == "matrix_train_gate":
        return _matrix_train_gate(repo_root, spec)
    if stype == "experiment":
        rc, tail = _run_experiment(
            repo_root,
            str(spec["command"]),
            config=spec.get("config"),
            kwargs=spec.get("kwargs") if isinstance(spec.get("kwargs"), dict) else None,
            env=spec.get("env") if isinstance(spec.get("env"), dict) else None,
        )
        if rc == 0:
            return StageResult(sid, "ok", tail or "exit 0")
        return StageResult(sid, "fail", tail or f"exit {rc}", exit_code=rc)
    if stype == "gpu_queue_dry":
        from harchoc.gpu_queue import run_gpu_queue

        manifest = str(spec.get("manifest") or "configs/experiments/gpu_queue_post_zoo_smoke.json")
        rc = run_gpu_queue(
            manifest,
            repo_root=repo_root,
            dry_run=True,
            resume=False,
        )
        if rc == 0:
            return StageResult(sid, "ok", f"dry-run plan ok ({manifest})")
        return StageResult(sid, "fail", f"gpu queue dry-run exit {rc}", exit_code=rc)
    if stype == "gpu_queue_run":
        if os.environ.get("HARCHOC_RUN_GPU_SMOKE", "").strip() not in ("1", "true", "yes"):
            return StageResult(sid, "skip", "set HARCHOC_RUN_GPU_SMOKE=1 to run live GPU queue job")
        from harchoc.gpu_queue import run_gpu_queue

        manifest = str(spec["manifest"])
        job_filter = spec.get("job_filter")
        rc = run_gpu_queue(
            manifest,
            repo_root=repo_root,
            dry_run=False,
            resume=True,
            job_filter=str(job_filter) if job_filter else None,
        )
        if rc == 0:
            return StageResult(sid, "ok", f"queue run ok ({manifest})")
        return StageResult(sid, "fail", f"gpu queue exit {rc}", exit_code=rc)
    if stype == "finetune_stages_dry":
        from harchoc.gpu_queue import build_job_stages, load_gpu_queue_manifest

        manifest = load_gpu_queue_manifest(
            str(spec.get("manifest") or "configs/experiments/gpu_queue_post_zoo_smoke.json"),
            repo_root=repo_root,
        )
        job_id = str(spec.get("job_id") or "finetune_tray_smoke_1ep")
        job = next((j for j in manifest.get("jobs") or [] if j.get("id") == job_id), None)
        if not isinstance(job, dict):
            return StageResult(sid, "fail", f"job {job_id!r} not in manifest")
        stages = build_job_stages(job, repo_root=repo_root, defaults=manifest.get("defaults") or {})
        ids = [s.get("stage_id") for s in stages]
        if "finetune" not in ids:
            return StageResult(sid, "fail", f"unexpected stages: {ids}")
        return StageResult(sid, "ok", f"stages={ids}")

    return StageResult(sid, "fail", f"unknown stage type {stype!r}", exit_code=1)


def run_now_todos_smoke(
    repo_root: str | Path,
    *,
    bundle_path: str | Path = "configs/experiments/now_todos_smoke_bundle.json",
    stage_group: str = "cpu",
) -> tuple[dict[str, Any], int]:
    rr = Path(repo_root).expanduser().resolve()
    bundle = load_smoke_bundle(rr / bundle_path if not Path(bundle_path).is_absolute() else bundle_path)
    groups = bundle.get("stages") or {}
    if stage_group == "all":
        keys = ("verify", "cpu", "gpu")
    else:
        keys = (stage_group,)
    results: list[StageResult] = []
    for key in keys:
        for spec in groups.get(key) or []:
            if isinstance(spec, dict):
                results.append(run_stage(rr, spec))

    n_fail = sum(1 for r in results if r.status == "fail")
    n_skip = sum(1 for r in results if r.status == "skip")
    payload: dict[str, Any] = {
        "schema_version": "now_todos_smoke_run.v1",
        "stage_group": stage_group,
        "overall_status": "fail" if n_fail else ("ok" if not n_skip else "ok_with_skips"),
        "n_ok": sum(1 for r in results if r.status == "ok"),
        "n_skip": n_skip,
        "n_fail": n_fail,
        "stages": [r.as_dict() for r in results],
    }
    out_rel = str(bundle.get("report_out") or "reports/manuscript/now_todos_smoke.json")
    out_path = rr / out_rel
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload, (1 if n_fail else 0)
