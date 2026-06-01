"""Shared repro chain utilities (bundle load, argv runner, manifest steps, CLI formatting)."""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from harchoc.ml_env import repo_python_cmd


def load_json_bundle(path: str | Path, *, schema_version: str) -> dict[str, Any]:
    p = Path(path).expanduser().resolve()
    obj = json.loads(p.read_text(encoding="utf-8"))
    if obj.get("schema_version") != schema_version:
        raise ValueError(f"unsupported bundle schema: {obj.get('schema_version')!r} (expected {schema_version!r})")
    return obj


def format_repro_cmd(argv: list[str], *, mamba: bool) -> str:
    if mamba:
        return " ".join(repo_python_cmd(argv))
    return " ".join([sys.executable, *argv])


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def step_record(
    *,
    status: str,
    started_at: str | None = None,
    finished_at: str | None = None,
    message: str | None = None,
    artifacts: list[str] | None = None,
) -> dict[str, Any]:
    rec: dict[str, Any] = {"status": status}
    if started_at:
        rec["started_at"] = started_at
    if finished_at:
        rec["finished_at"] = finished_at
    if message:
        rec["message"] = message
    if artifacts:
        rec["artifacts"] = artifacts
    return rec


def overall_step_status(steps: dict[str, Any]) -> str:
    statuses = [str((steps.get(s) or {}).get("status") or "") for s in steps]
    if any(s == "failed" for s in statuses):
        return "failed"
    if all(s in ("ok", "skipped", "dry_run") for s in statuses):
        if any(s == "skipped" for s in statuses):
            return "partial"
        return "ok" if not any(s == "dry_run" for s in statuses) else "dry_run"
    return "partial"


def reproduce_command_block(
    *,
    ci_dry_run: list[str] | None = None,
    local: list[str] | None = None,
    extra: dict[str, list[str]] | None = None,
) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    if ci_dry_run:
        out["ci_safe_dry_run"] = list(ci_dry_run)
    if local:
        out["local"] = list(local)
    if extra:
        out.update(extra)
    return out


def run_argv_chain(
    steps: list[tuple[str, list[str]]],
    *,
    repo_root: str | Path,
    dry_run: bool = False,
    on_step: Callable[[str, list[str]], None] | None = None,
    run_argv: Callable[[list[str]], int] | None = None,
    mamba_for_step: Callable[[str], bool] | None = None,
    fail_label: str = "repro",
) -> int:
    """Run ordered (step_id, argv) pairs; argv is relative to repo_root (script path first)."""
    rr = Path(repo_root).expanduser().resolve()

    def _default_run(argv: list[str]) -> int:
        proc = subprocess.run([sys.executable, *argv], cwd=str(rr))
        return int(proc.returncode)

    runner = run_argv or _default_run
    mamba_fn = mamba_for_step or (lambda _sid: True)

    for step_id, argv in steps:
        if on_step is not None:
            on_step(step_id, argv)
        if dry_run:
            print(f"# {step_id}")
            print(format_repro_cmd(argv, mamba=mamba_fn(step_id)))
            continue
        rc = runner(argv)
        if rc != 0:
            print(
                f"{fail_label} step {step_id!r} failed with exit code {rc}",
                file=sys.stderr,
            )
            return rc
    return 0


def hsp_test_exports_present(
    repo_root: str | Path,
    hsp_artifacts: dict[str, Any],
    *,
    keys: tuple[str, ...] = ("gt_test", "preds_test"),
) -> bool:
    rr = Path(repo_root).expanduser().resolve()
    for key in keys:
        rel = str(hsp_artifacts.get(key) or "").strip()
        if not rel or not (rr / rel).is_file():
            return False
    return True
