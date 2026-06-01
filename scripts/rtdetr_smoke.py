"""
RT-DETR 15-epoch smoke: GPU probe + optional train via mamba env ``harchoc``.

Never infer "no GPU" from base Python — this script re-execs under ``mamba run -n harchoc``.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import sys; from pathlib import Path; _r = Path(__file__).resolve().parent.parent; (str(_r) not in sys.path) and sys.path.insert(0, str(_r)); from harchoc.script_entry import bootstrap_repo_imports; bootstrap_repo_imports()
from harchoc.ml_env import (
    default_mamba_env,
    mamba_run_shell_command,
    probe_torch_via_mamba,
    reexec_script_in_mamba_env,
    should_reexec_in_mamba_for_torch,
)
from harchoc.post_train_eval import post_train_eval_skipped
from harchoc.strict_ml import capture_failure
from harchoc.train_config import load_train_config_json
from scripts._common_cli import write_json

_REPO = Path(__file__).resolve().parents[1]
_SMOKE_CONFIG = "configs/experiments/train_rtdetr_smoke_15ep.json"
_DEFAULT_OUT = "reports/hsp/rtdetr_smoke_15ep.json"


def _cuda_ok_from_gpu_payload(payload: dict[str, object], rc: int) -> bool:
    """Require check_gpu exit 0 and status ok — cuda_available alone is insufficient."""
    if rc != 0 or payload.get("status") != "ok":
        return False
    torch_info = payload.get("torch")
    if isinstance(torch_info, dict):
        return bool(torch_info.get("cuda_available"))
    return bool(payload.get("cuda_available"))


def _gpu_check_via_mamba(*, repo_root: Path) -> tuple[dict[str, object], int]:
    """Run check_gpu.py inside mamba; return parsed JSON + exit code."""
    env = default_mamba_env()
    out = repo_root / "reports" / "hsp" / "_rtdetr_smoke_gpu_check.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "mamba",
        "run",
        "-n",
        env,
        "python",
        str(repo_root / "scripts" / "check_gpu.py"),
        "--json-out",
        str(out),
    ]
    proc = subprocess.run(cmd, cwd=str(repo_root), check=False)
    rc = int(proc.returncode)
    if out.is_file():
        with capture_failure("read_gpu_check_json") as cap:
            payload = json.loads(out.read_text("utf-8"))
        if cap.failed:
            return {
                "status": "gpu_check_json_error",
                "error_type": cap.exc_type,
                "error": cap.exc_msg,
                "gpu_check_exit_code": rc,
                "mamba_env": env,
            }, 2
        if isinstance(payload, dict):
            return payload, rc
    probe = probe_torch_via_mamba()
    fallback_status = "missing_torch" if not probe.get("ok") else "gpu_check_missing_output"
    return {
        "status": fallback_status,
        "torch": probe,
        "mamba_env": env,
        "gpu_check_exit_code": rc,
    }, 2


def main(argv: list[str] | None = None) -> int:
    if should_reexec_in_mamba_for_torch():
        reexec_script_in_mamba_env(argv)

    p = argparse.ArgumentParser(description="RT-DETR 15-ep smoke probe and optional train.")
    p.add_argument(
        "--out",
        default=_DEFAULT_OUT,
        help="Write smoke metadata JSON (default: reports/hsp/rtdetr_smoke_15ep.json).",
    )
    p.add_argument(
        "--run-train",
        action="store_true",
        help="Run train.py smoke (15 epochs) in mamba env after GPU check passes.",
    )
    p.add_argument("--name", default="rtdetr_smoke_15ep", help="Ultralytics run name when --run-train.")
    args = p.parse_args(argv)

    repo_root = _REPO
    env = default_mamba_env()
    gpu_payload, gpu_rc = _gpu_check_via_mamba(repo_root=repo_root)
    cuda = _cuda_ok_from_gpu_payload(gpu_payload, gpu_rc)

    cmd_train = mamba_run_shell_command(
        "scripts/train.py",
        f"--name {args.name}",
        f"--config {_SMOKE_CONFIG}",
        env_name=env,
    )
    report: dict[str, object] = {
        "schema_version": "rtdetr_smoke_15ep.v1",
        "status": "gpu_ok_pending_train",
        "mamba_env": env,
        "config": _SMOKE_CONFIG,
        "policy": {
            "num_queries": 300,
            "documented_peak_gt_boxes_per_image": 1015,
            "accept_rtdetr_query_truncation": True,
        },
        "planned_train": {
            "epochs": 15,
            "imgsz": 1280,
            "batch": 1,
            "model": "rtdetr-l.pt",
        },
        "gpu_check": gpu_payload,
        "gpu_check_exit_code": gpu_rc,
        "cuda_available": cuda,
        "command_probe": mamba_run_shell_command(
            "scripts/check_gpu.py",
            "--json-out reports/hsp/gpu_check.json",
            env_name=env,
        ),
        "command_train": (
            f"export HARCHOC_MAX_EPOCHS=15 HARCHOC_MAX_IMGSZ=2048 DATASET_ROOT=... && {cmd_train}"
        ),
        "recorded_at": datetime.now(timezone.utc).isoformat(),
    }

    if not cuda:
        report["status"] = "gpu_unavailable"
        write_json(args.out, report)
        print(f"Wrote {args.out} (no CUDA in mamba env {env!r})")
        return 2

    if not args.run_train:
        report["status"] = "gpu_ok_pending_train"
        write_json(args.out, report)
        print(f"Wrote {args.out} (CUDA ok; pass --run-train to start 15-ep train)")
        return 0

    smoke_cfg_path = (repo_root / _SMOKE_CONFIG).resolve()
    merged_cfg = load_train_config_json(smoke_cfg_path, repo_root=repo_root)
    eval_section = merged_cfg.get("eval") if isinstance(merged_cfg.get("eval"), dict) else {}
    eval_skip = post_train_eval_skipped(cli_skip=False, eval_section=eval_section)
    report["post_train_eval_skipped"] = eval_skip

    train_cmd = [
        "mamba",
        "run",
        "-n",
        env,
        "python",
        str(repo_root / "scripts" / "train.py"),
        "--name",
        args.name,
        "--config",
        str(smoke_cfg_path),
    ]
    if eval_skip:
        train_cmd.append("--skip-eval")
    print("Running:", " ".join(train_cmd))
    proc = subprocess.run(train_cmd, cwd=str(repo_root), check=False)
    train_rc = int(proc.returncode)
    report["train_exit_code"] = train_rc
    if train_rc == 0:
        report["status"] = "train_complete"
    else:
        report["status"] = "train_failed"
        report["failure_phase"] = "train"
    write_json(args.out, report)
    print(f"Wrote {args.out}")
    return int(proc.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
