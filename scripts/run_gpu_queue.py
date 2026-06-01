"""Sequential GPU backlog queue — direct/CI CLI (canonical: ``./scripts/run_gpu_queue.sh``)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))
from harchoc.script_entry import bootstrap_repo_imports

bootstrap_repo_imports()

from harchoc.gpu_queue import DEFAULT_MIN_FREE_MIB, run_gpu_queue
from scripts._common_cli import add_dry_run_arg

_REPO = Path(__file__).resolve().parents[1]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Run gpu_queue manifest (dry-run or live).")
    p.add_argument(
        "--manifest",
        required=True,
        help="Path to gpu_queue_manifest.v1 JSON.",
    )
    add_dry_run_arg(p)
    p.add_argument(
        "--run",
        action="store_true",
        help="Execute jobs (default: dry-run prints stage commands).",
    )
    p.add_argument(
        "--resume",
        action="store_true",
        help="Resume from reports/gpu_queue/run_state.json.",
    )
    p.add_argument("--job", default=None, help="Run only this job id.")
    p.add_argument(
        "--state-path",
        default="reports/gpu_queue/run_state.json",
        help="Run state JSON path.",
    )
    p.add_argument(
        "--min-free-mib",
        type=int,
        default=DEFAULT_MIN_FREE_MIB,
        help="GPU wait threshold MiB free.",
    )
    args = p.parse_args(argv)
    dry_run = bool(args.dry_run) or not args.run
    return run_gpu_queue(
        args.manifest,
        repo_root=_REPO,
        dry_run=dry_run,
        resume=args.resume,
        job_filter=args.job,
        state_path=args.state_path,
        min_free_mib=args.min_free_mib,
    )


if __name__ == "__main__":
    raise SystemExit(main())
