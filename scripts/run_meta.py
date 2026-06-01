from __future__ import annotations

import argparse
from pathlib import Path

import sys; from pathlib import Path; _r = Path(__file__).resolve().parent.parent; (str(_r) not in sys.path) and sys.path.insert(0, str(_r)); from harchoc.script_entry import bootstrap_repo_imports; bootstrap_repo_imports()
from harchoc.run_metadata import collect_run_metadata
from harchoc.schemas import with_schema_version
from scripts._common_cli import write_json


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Collect lightweight run metadata (CI-safe).")
    p.add_argument("--out", default="reports/run_meta.json", help="Where to write metadata JSON.")
    p.add_argument("--dry-run", action="store_true", help="Write minimal metadata only.")
    args = p.parse_args(argv)

    repo_root = Path(__file__).resolve().parents[1]
    manifest = repo_root / "data" / "manifest.json"

    meta = collect_run_metadata(repo_root=repo_root, dataset_manifest=manifest if manifest.exists() else None)
    payload = with_schema_version(
        {
        "status": "dry-run" if args.dry_run else "ok",
        "script": "run_meta",
        **meta,
        },
        schema_version="run_meta.v1",
    )
    out_path = write_json(args.out, payload)
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

