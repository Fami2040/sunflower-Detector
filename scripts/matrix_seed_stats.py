from __future__ import annotations

import argparse
from pathlib import Path

import sys

_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))
from harchoc.script_entry import bootstrap_repo_imports

bootstrap_repo_imports()
from harchoc.matrix_seed_stats import (
    MATRIX_SEED_STATS_V1,
    build_dry_run_matrix_seed_stats_v1,
    build_matrix_seed_stats_v1,
    parse_count_mae_json_args,
)
from scripts._common_cli import add_dry_run_arg, cli_print, read_json_dict, write_json


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Aggregate multi-seed mAP and count MAE from benchmark_matrix_train JSON."
    )
    add_dry_run_arg(p)
    p.add_argument(
        "--train-out",
        default="",
        help="benchmark_matrix_train.v1 JSON (required unless --dry-run with no preview).",
    )
    p.add_argument(
        "--out",
        default="",
        help="Output path (default: <train-out> with .seed_stats.json suffix).",
    )
    p.add_argument(
        "--count-mae-json",
        action="append",
        default=[],
        help="Per-run count MAE artifact: run_name=path/to/error_*_report.json or threshold_*_locked.json",
    )
    args = p.parse_args(argv)

    repo_root = Path(__file__).resolve().parents[1]
    count_mae_paths = parse_count_mae_json_args(list(args.count_mae_json or []))
    train_out = (args.train_out or "").strip()

    if args.dry_run:
        train_doc = None
        if train_out:
            train_path = Path(train_out)
            if not train_path.is_file():
                raise SystemExit(f"--train-out not found: {train_path}")
            train_doc = read_json_dict(train_path)
        out_path = (args.out or "").strip() or (
            str(Path(train_out).with_suffix(".seed_stats.json")) if train_out else "reports/matrix_seed_stats.json"
        )
        payload = build_dry_run_matrix_seed_stats_v1(
            out=out_path,
            source_train_out=train_out or None,
            train_doc=train_doc,
            count_mae_paths=count_mae_paths or None,
            repo_root=repo_root,
        )
        written = write_json(out_path, payload)
        cli_print(f"Wrote {written}")
        return 0

    if not train_out:
        raise SystemExit("--train-out is required (omit only with --dry-run).")
    train_path = Path(train_out)
    if not train_path.is_file():
        raise SystemExit(f"--train-out not found: {train_path}")

    train_doc = read_json_dict(train_path)
    out_path = (args.out or "").strip() or str(train_path.with_suffix(".seed_stats.json"))
    payload = build_matrix_seed_stats_v1(
        train_doc,
        source_train_out=train_path,
        count_mae_paths=count_mae_paths or None,
        repo_root=repo_root,
        schema_version=MATRIX_SEED_STATS_V1,
    )
    written = write_json(out_path, payload)
    cli_print(f"Wrote {written}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
