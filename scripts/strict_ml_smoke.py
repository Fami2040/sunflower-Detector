"""
Strict ML integration smoke: torch + CUDA matmul + strict-mode import checks.

Run from repo root (agents / GPU dev):

  HARCHOC_STRICT_ML=1 mamba run -n harchoc python scripts/strict_ml_smoke.py

Fails loudly when CUDA or strict-mode surfaces are broken (no placeholder success).
"""

from __future__ import annotations

import argparse
import platform
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import sys; from pathlib import Path; _r = Path(__file__).resolve().parent.parent; (str(_r) not in sys.path) and sys.path.insert(0, str(_r)); from harchoc.script_entry import bootstrap_repo_imports; bootstrap_repo_imports()
from harchoc.gpu_probe import matmul_bench, torch_cuda_payload
from harchoc.strict_ml import capture_failure, require_torch, strict_ml_enabled
from scripts._common_cli import write_json

_REPO = Path(__file__).resolve().parents[1]
_DEFAULT_OUT = "reports/hsp/strict_ml_smoke.json"


def _check_cuda_matmul(*, n: int, iters: int) -> dict[str, Any]:
    torch_mod = require_torch()
    cuda_info = torch_cuda_payload(torch_mod)
    if not cuda_info.get("cuda_available"):
        msg = "CUDA is not available for the current PyTorch build/runtime."
        if cuda_info.get("device_error"):
            msg = f"{msg} ({cuda_info['device_error']})"
        raise RuntimeError(msg)
    if cuda_info.get("device_error"):
        raise RuntimeError(str(cuda_info["device_error"]))
    bench = matmul_bench(torch_mod, device="cuda", n=n, iters=iters)
    return {
        "status": "ok",
        "torch_version": cuda_info.get("torch_version"),
        "cuda": cuda_info,
        "bench": bench,
    }


def _check_gradcam_panel() -> dict[str, Any]:
    from harchoc.gradcam_panel import plan_gradcam_panel, select_panel_entries

    plan = plan_gradcam_panel(report_path=None, max_panels=4)
    picked = select_panel_entries([], max_panels=4)
    return {
        "status": "ok",
        "plan_status": plan.get("status"),
        "panel_entries": len(picked),
    }


def _check_dual_metric_report() -> dict[str, Any]:
    from harchoc.dual_metric_report import (
        build_dry_run_report,
        build_dual_metric_report,
        extract_detection_metrics,
        merge_dual_metric_from_paths,
    )

    _ = (build_dual_metric_report, merge_dual_metric_from_paths)
    dry = build_dry_run_report(out="reports/hsp/_strict_smoke_dual_metric.json", inputs={})
    metrics = extract_detection_metrics({"mAP50": 0.5, "mAP50_95": 0.3})
    return {
        "status": "ok",
        "dry_run_schema": dry.get("schema_version"),
        "extract_detection_keys": sorted(metrics.keys()),
    }


def _check_make_figures_dry_run() -> dict[str, Any]:
    from scripts.make_figures import main as make_figures_main

    with tempfile.TemporaryDirectory() as td:
        meta = Path(td) / "figures_run.json"
        rc = make_figures_main(["--dry-run", "--meta-out", str(meta)])
        if rc != 0:
            raise RuntimeError(f"make_figures --dry-run exited {rc}")
        if not meta.is_file():
            raise RuntimeError("make_figures --dry-run did not write meta JSON")
        return {"status": "ok", "exit_code": int(rc), "meta_out": str(meta)}


def _run_named_check(name: str, fn: Callable[[], dict[str, Any]]) -> dict[str, Any]:
    with capture_failure(name) as cap:
        result = fn()
    if cap.failed:
        return {
            "status": "failed",
            "error_type": cap.exc_type,
            "error": cap.exc_msg,
        }
    return result


def main(argv: list[str] | None = None) -> int:
    from harchoc.ml_env import reexec_script_in_mamba_env, should_reexec_in_mamba_for_torch

    if should_reexec_in_mamba_for_torch():
        reexec_script_in_mamba_env(argv)

    p = argparse.ArgumentParser(
        description="Strict ML smoke: CUDA matmul + gradcam/dual-metric/make_figures surfaces."
    )
    p.add_argument(
        "--out",
        default=_DEFAULT_OUT,
        help="Write JSON report (default: reports/hsp/strict_ml_smoke.json).",
    )
    p.add_argument("--n", type=int, default=1024, help="CUDA matmul size (NxN).")
    p.add_argument("--iters", type=int, default=10, help="CUDA matmul iterations.")
    p.add_argument(
        "--skip-make-figures",
        action="store_true",
        help="Skip make_figures --dry-run check.",
    )
    args = p.parse_args(sys.argv[1:] if argv is None else argv)

    checks: dict[str, Any] = {
        "cuda_matmul": _run_named_check("cuda_matmul", lambda: _check_cuda_matmul(n=int(args.n), iters=int(args.iters))),
        "gradcam_panel": _run_named_check("gradcam_panel", _check_gradcam_panel),
        "dual_metric_report": _run_named_check("dual_metric_report", _check_dual_metric_report),
    }
    if args.skip_make_figures:
        checks["make_figures_dry_run"] = {"status": "skipped"}
    else:
        checks["make_figures_dry_run"] = _run_named_check(
            "make_figures_dry_run", _check_make_figures_dry_run
        )

    cuda_ok = checks["cuda_matmul"].get("status") == "ok"
    cuda_available = bool(cuda_ok and (checks["cuda_matmul"].get("cuda") or {}).get("cuda_available"))
    torch_version = checks["cuda_matmul"].get("torch_version")
    all_ok = all(c.get("status") in ("ok", "skipped") for c in checks.values())

    report: dict[str, Any] = {
        "schema_version": "strict_ml_smoke.v1",
        "script": "strict_ml_smoke",
        "status": "ok" if all_ok else "failed",
        "strict_ml": strict_ml_enabled(),
        "cuda_available": cuda_available,
        "cuda_passed": cuda_ok,
        "torch_version": torch_version,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "checks": checks,
        "recorded_at": datetime.now(timezone.utc).isoformat(),
    }

    write_json(args.out, report)
    print(f"Wrote {args.out}")
    if not all_ok:
        failed = [k for k, v in checks.items() if v.get("status") not in ("ok", "skipped")]
        print(f"FAILED checks: {', '.join(failed)}", flush=True)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
