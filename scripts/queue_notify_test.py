"""Send a test GPU queue notification (uses configs/local/queue_notify.json — gitignored)."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

_r = Path(__file__).resolve().parent.parent
(str(_r) not in sys.path) and sys.path.insert(0, str(_r))
from harchoc.script_entry import bootstrap_repo_imports

bootstrap_repo_imports()

from harchoc.queue_notify import DEFAULT_NOTIFY_LOG, load_notify_config, notify_event


def main() -> int:
    ap = argparse.ArgumentParser(description="Test harchoc queue email notification")
    ap.add_argument("--dry-run", action="store_true", help="Log only; do not SMTP send")
    args = ap.parse_args()
    repo = Path(__file__).resolve().parents[1]
    cfg = load_notify_config(repo_root=repo)
    if cfg is None:
        print("No notify config — create configs/local/queue_notify.json or set HARCHOC_NOTIFY_EMAIL")
        return 1
    print("notify configured: yes (recipient not printed)")
    if args.dry_run:
        os.environ["HARCHOC_NOTIFY_DRY_RUN"] = "1"
    result = notify_event(
        repo_root=repo,
        event="test_ping",
        subject="[harchoc] queue notify test",
        body="Test notification from scripts/queue_notify_test.py",
        context={"job_id": "notify_test"},
    )
    print(json.dumps(result, indent=2))
    log_path = repo / DEFAULT_NOTIFY_LOG
    if log_path.is_file():
        last = log_path.read_text(encoding="utf-8").strip().splitlines()[-1]
        if cfg.email in last:
            print("ERROR: secret email leaked into notify log", file=sys.stderr)
            return 2
        print(f"log tail ok: {last[:120]}…")
    ok = result.get("delivered") or result.get("channel") == "dry_run"
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
