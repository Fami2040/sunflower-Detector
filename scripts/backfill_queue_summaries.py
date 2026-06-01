"""Backfill HSP test summaries for queue jobs with train Done but no summary JSON."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))
from harchoc.script_entry import bootstrap_repo_imports

bootstrap_repo_imports()

from harchoc.backfill_queue_summaries import (
    GPU_QUEUE_SUMMARY_BACKFILLS,
    backfill_gpu_queue_summaries,
    backfill_queue_summary,
)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Eval-only HSP backfill for amp/sg/close sweeps (existing weights)."
    )
    p.add_argument("--job", action="append", default=[], help="Limit to job_id (repeatable).")
    p.add_argument("--force", action="store_true", help="Re-run even if summary exists.")
    args = p.parse_args(argv)

    repo = Path(__file__).resolve().parents[1]
    specs = GPU_QUEUE_SUMMARY_BACKFILLS
    if args.job:
        want = set(args.job)
        specs = tuple(s for s in specs if s.job_id in want)
        if not specs:
            raise SystemExit(f"no matching backfill specs for --job {args.job!r}")

    if len(specs) == len(GPU_QUEUE_SUMMARY_BACKFILLS):
        results = backfill_gpu_queue_summaries(
            repo_root=repo,
            skip_if_complete=not args.force,
        )
    else:
        results = [
            backfill_queue_summary(s, repo_root=repo, skip_if_complete=not args.force)
            for s in specs
        ]
    print(json.dumps(results, indent=2))
    failed = [r for r in results if r.get("status") not in ("complete", "skipped")]
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
