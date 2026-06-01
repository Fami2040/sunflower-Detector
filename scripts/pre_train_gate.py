"""
Pre-train gate: manifest + env reminders + unittest (and optional strict ML smoke).

Quick (CI-parity unittest, no GPU reports):

  python scripts/pre_train_gate.py --quick

Full (adds strict_ml_smoke when HARCHOC_STRICT_ML=1):

  HARCHOC_STRICT_ML=1 mamba run -n harchoc python scripts/pre_train_gate.py --full

Writes JSON only when --json-out is set (strict_ml_smoke uses a temp file otherwise).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import sys; from pathlib import Path; _r = Path(__file__).resolve().parent.parent; (str(_r) not in sys.path) and sys.path.insert(0, str(_r)); from harchoc.script_entry import bootstrap_repo_imports; bootstrap_repo_imports()

from harchoc.ml_env import default_mamba_env
from harchoc.strict_ml import strict_ml_enabled
from scripts._common_cli import cli_print, eprint, write_json

_REPO = Path(__file__).resolve().parents[1]
_MANIFEST = _REPO / "data" / "manifest.json"
_UNITTEST_CMD = (
    "PYTHONPATH=. HARCHOC_ALLOW_BASE_PYTHON=1 mamba run -n harchoc "
    "python -m unittest discover -s tests -q"
)


def _check_manifest() -> dict[str, Any]:
    ok = _MANIFEST.is_file()
    return {
        "status": "ok" if ok else "failed",
        "path": str(_MANIFEST.relative_to(_REPO)),
        "exists": ok,
    }


def _print_env_reminders() -> None:
    cli_print("Pre-train reminders:")
    cli_print("  export DATASET_ROOT=/path/to/extracted/dataset   # see data/manifest.json")
    cli_print("  export HARCHOC_EXPORT_DEVICE=cpu               # standalone eval.py / HSP chain")


def _run_unittest() -> dict[str, Any]:
    env_name = default_mamba_env()
    cmd = [
        "mamba",
        "run",
        "-n",
        env_name,
        "python",
        "-m",
        "unittest",
        "discover",
        "-s",
        "tests",
        "-q",
    ]
    proc = subprocess.run(
        cmd,
        cwd=str(_REPO),
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "PYTHONPATH": str(_REPO),
            "HARCHOC_ALLOW_BASE_PYTHON": "1",
        },
    )
    combined = proc.stdout + proc.stderr
    tests_run = None
    m = re.search(r"^Ran (\d+) tests", combined, re.MULTILINE)
    if m:
        tests_run = int(m.group(1))
    return {
        "command": _UNITTEST_CMD,
        "exit_code": proc.returncode,
        "tests_run": tests_run,
        "status": "ok" if proc.returncode == 0 else "failed",
        "output_tail": combined.strip()[-800:] if proc.returncode != 0 else None,
    }


def _run_strict_ml_smoke(*, embed_report: bool) -> dict[str, Any]:
    env_name = default_mamba_env()
    if not strict_ml_enabled():
        return {
            "status": "skipped",
            "reason": "HARCHOC_STRICT_ML not set",
            "command": f"HARCHOC_STRICT_ML=1 mamba run -n {env_name} python scripts/strict_ml_smoke.py",
        }

    with tempfile.TemporaryDirectory(prefix="pre_train_gate_") as td:
        smoke_out = Path(td) / "strict_ml_smoke.json"
        cmd = [
            "mamba",
            "run",
            "-n",
            env_name,
            "python",
            "scripts/strict_ml_smoke.py",
            "--out",
            str(smoke_out),
        ]
        proc = subprocess.run(
            cmd,
            cwd=str(_REPO),
            capture_output=True,
            text=True,
            env={**os.environ, "HARCHOC_STRICT_ML": "1", "PYTHONPATH": str(_REPO)},
        )
        result: dict[str, Any] = {
            "command": f"HARCHOC_STRICT_ML=1 mamba run -n {env_name} python scripts/strict_ml_smoke.py",
            "exit_code": proc.returncode,
            "status": "ok" if proc.returncode == 0 else "failed",
        }
        if proc.returncode != 0:
            result["stderr_tail"] = (proc.stderr or "").strip()[-500:]
        if embed_report and smoke_out.is_file():
            result["strict_ml_report"] = json.loads(smoke_out.read_text(encoding="utf-8"))
        return result


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Pre-train gate: manifest check, env reminders, unittest, optional strict ML smoke."
    )
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--quick",
        action="store_true",
        help="Run unittest discover -q (HARCHOC_ALLOW_BASE_PYTHON=1).",
    )
    mode.add_argument(
        "--full",
        action="store_true",
        help="Run --quick plus strict_ml_smoke when HARCHOC_STRICT_ML=1.",
    )
    p.add_argument(
        "--skip-strict-ml",
        action="store_true",
        help="With --full, do not run strict_ml_smoke even if HARCHOC_STRICT_ML=1.",
    )
    p.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="Write gate summary JSON (optional strict_ml payload path when --full).",
    )
    args = p.parse_args(sys.argv[1:] if argv is None else argv)

    manifest = _check_manifest()
    if manifest["status"] != "ok":
        cli_print(f"FAILED: missing {_MANIFEST}")
        return 1

    _print_env_reminders()

    report: dict[str, Any] = {
        "schema_version": "pre_train_gate.v1",
        "script": "pre_train_gate",
        "mode": "full" if args.full else "quick",
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "manifest": manifest,
        "reminders": {
            "DATASET_ROOT": "export DATASET_ROOT=/path/to/extracted/dataset",
            "HARCHOC_EXPORT_DEVICE": "export HARCHOC_EXPORT_DEVICE=cpu",
        },
    }

    unittest_result = _run_unittest()
    report["unittest"] = unittest_result
    if unittest_result["status"] != "ok":
        tail = unittest_result.get("output_tail")
        if tail:
            eprint(f"unittest output (tail):\n{tail}")
        if args.json_out:
            write_json(args.json_out, report)
            cli_print(f"Wrote {args.json_out}")
        return int(unittest_result["exit_code"]) or 1

    strict_result: dict[str, Any] | None = None
    if args.full and not args.skip_strict_ml:
        strict_result = _run_strict_ml_smoke(embed_report=args.json_out is not None)
        report["strict_ml_smoke"] = strict_result

    overall_ok = unittest_result["status"] == "ok" and (
        strict_result is None or strict_result.get("status") in ("ok", "skipped")
    )
    report["status"] = "ok" if overall_ok else "failed"

    if args.json_out:
        write_json(args.json_out, report)
        cli_print(f"Wrote {args.json_out}")

    if not overall_ok:
        failed: list[str] = []
        if unittest_result["status"] != "ok":
            failed.append("unittest")
        if strict_result and strict_result.get("status") == "failed":
            failed.append("strict_ml_smoke")
        cli_print(f"FAILED gates: {', '.join(failed)}")
        return 2

    cli_print("pre_train_gate: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
