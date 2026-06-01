"""Run integration gates and write reports/hsp/agent_batch_verify.json from real exit codes."""

from __future__ import annotations

import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))
from harchoc.script_entry import bootstrap_repo_imports

bootstrap_repo_imports()
_REPO = Path(__file__).resolve().parents[1]
_DEFAULT_OUT = _REPO / "reports/hsp/agent_batch_verify.json"
_MAMBA_ENV = "harchoc"


def _mamba_python(*script_args: str, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    cmd = ["mamba", "run", "-n", _MAMBA_ENV, "python", *script_args]
    run_env = {"PYTHONPATH": str(_REPO), **(env or {})}
    return subprocess.run(
        cmd,
        cwd=str(_REPO),
        capture_output=True,
        text=True,
        env={**dict(__import__("os").environ), **run_env},
    )


def _gpu_memory_note() -> str | None:
    try:
        proc = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return None
    if proc.returncode != 0 or not proc.stdout.strip():
        return None
    line = proc.stdout.strip().splitlines()[0]
    parts = [p.strip() for p in line.split(",")]
    if len(parts) != 2:
        return None
    used_mib, total_mib = int(float(parts[0])), int(float(parts[1]))
    if used_mib * 100 // max(total_mib, 1) >= 85:
        return f"GPU memory {used_mib}/{total_mib} MiB in use (strict_ml cuda_matmul may OOM until freed)."
    return None


def _run_unittest() -> dict[str, Any]:
    cmd = [
        "mamba",
        "run",
        "-n",
        _MAMBA_ENV,
        "scripts/run_tests.py",
        "-q",
    ]
    proc = subprocess.run(
        cmd,
        cwd=str(_REPO),
        capture_output=True,
        text=True,
        env={**dict(__import__("os").environ), "PYTHONPATH": str(_REPO), "HARCHOC_ALLOW_BASE_PYTHON": "1"},
    )
    tests_run = None
    m = re.search(r"^Ran (\d+) tests", proc.stdout + proc.stderr, re.MULTILINE)
    if m:
        tests_run = int(m.group(1))
    failures = len(re.findall(r"^FAILED ", proc.stdout + proc.stderr, re.MULTILINE))
    errors = len(re.findall(r"^ERROR:", proc.stdout + proc.stderr, re.MULTILINE))
    status = "ok" if proc.returncode == 0 else "failed"
    return {
        "command": "PYTHONPATH=. HARCHOC_ALLOW_BASE_PYTHON=1 mamba run -n harchoc python scripts/run_tests.py -q",
        "exit_code": proc.returncode,
        "tests_run": tests_run,
        "failures": failures,
        "errors": errors,
        "status": status,
    }


def main(argv: list[str] | None = None) -> int:
    import argparse

    p = argparse.ArgumentParser(description="Run agent batch verify gates and write JSON report.")
    p.add_argument("--out", type=Path, default=_DEFAULT_OUT)
    p.add_argument("--batch", default="dry_refactor_batches_1-4")
    args = p.parse_args(sys.argv[1:] if argv is None else argv)

    unittest = _run_unittest()

    strict_proc = _mamba_python(
        "scripts/strict_ml_smoke.py",
        env={"HARCHOC_STRICT_ML": "1"},
    )
    gpu_note = _gpu_memory_note() if strict_proc.returncode != 0 else None
    strict_smoke: dict[str, Any] = {
        "command": "HARCHOC_STRICT_ML=1 mamba run -n harchoc python scripts/strict_ml_smoke.py",
        "exit_code": strict_proc.returncode,
        "status": "ok" if strict_proc.returncode == 0 else "failed",
        "report": "reports/hsp/strict_ml_smoke.json",
    }
    if gpu_note:
        strict_smoke["gpu_note"] = gpu_note
    if strict_proc.returncode != 0:
        strict_smoke["stderr_tail"] = (strict_proc.stderr or "").strip()[-500:]

    repro_proc = _mamba_python("scripts/experiment.py", "repro", "--dry-run")
    finetune_proc = _mamba_python(
        "scripts/finetune.py",
        "--dry-run",
        "--out",
        "reports/transfer/finetune_agent_verify.json",
    )

    gates = {
        "unittest": unittest,
        "strict_ml_smoke": strict_smoke,
        "experiment_repro_dry_run": {
            "command": "mamba run -n harchoc python scripts/experiment.py repro --dry-run",
            "exit_code": repro_proc.returncode,
            "status": "ok" if repro_proc.returncode == 0 else "failed",
        },
        "finetune_dry_run": {
            "command": "mamba run -n harchoc python scripts/finetune.py --dry-run --out reports/transfer/finetune_agent_verify.json",
            "exit_code": finetune_proc.returncode,
            "status": "ok" if finetune_proc.returncode == 0 else "failed",
        },
    }
    overall_ok = all(g.get("status") == "ok" for g in gates.values())

    payload: dict[str, Any] = {
        "schema_version": "agent_batch_verify.v1",
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "batch": args.batch,
        "strict_ml": {
            "module": "harchoc/strict_ml.py",
            "capture_failure": True,
            "record_ml_failure": True,
        },
        **gates,
        "overall": "ok" if overall_ok else "failed",
    }
    if not overall_ok:
        failed = [k for k, g in gates.items() if g.get("status") != "ok"]
        payload["failed_gates"] = failed

    out_path = args.out if args.out.is_absolute() else (_REPO / args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {out_path}")
    print(f"overall: {payload['overall']}")
    return 0 if overall_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
