from __future__ import annotations

import argparse
import json
from pathlib import Path

import sys

_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))
from harchoc.script_entry import bootstrap_repo_imports

bootstrap_repo_imports()
from harchoc.cv_eval_core import aggregate_fold_metrics, kfold_assign, load_fold_metric
from harchoc.hsp_weights import HSP_DETECTION_WEIGHTS
from harchoc.schemas import with_schema_version
from harchoc.splits_io import read_split_list
from harchoc.script_scaffold import build_versioned_dry_run_payload, resolve_dataset_args
from scripts._common_cli import add_dataset_args, add_dry_run_arg, cli_print, require_existing_dir, write_json


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="K-fold split planning and fold-metric aggregation with bootstrap CIs.")
    add_dataset_args(p)
    add_dry_run_arg(p)
    p.add_argument("--weights", default=HSP_DETECTION_WEIGHTS, help="Path to model weights (metadata only in v1).")
    p.add_argument("--folds", type=int, default=5, help="Number of folds for split assignment.")
    p.add_argument("--seed", type=int, default=0, help="Random seed for fold assignment.")
    p.add_argument(
        "--splits-dir",
        default="data/splits",
        help="Directory under dataset root with train.txt for k-fold assignment.",
    )
    p.add_argument(
        "--fold-metrics",
        action="append",
        default=[],
        help="JSON file per fold with metrics (repeatable). When set, aggregates CIs instead of scaffold-only.",
    )
    p.add_argument("--out", default="reports/cv_eval/summary.json", help="Where to write aggregated metrics.")
    p.add_argument(
        "--write-fold-splits",
        default="",
        help="If set, write fold_N/{train,val,test}.txt under this directory for manual per-fold training.",
    )
    args = p.parse_args(argv)

    if args.dry_run:
        out_path = write_json(
            args.out,
            build_versioned_dry_run_payload(
                script="cv_eval",
                schema_version="cv_eval_run.v1",
                out=args.out,
                weights=args.weights,
                folds=args.folds,
                seed=args.seed,
            ),
        )
        cli_print(f"Wrote {out_path}")
        return 0

    spec = resolve_dataset_args(args)
    require_existing_dir(spec.root, what="Dataset root", hint="Export DATASET_ROOT=/path/to/extracted/dataset")
    root = Path(spec.root)
    train_txt = root / str(args.splits_dir) / "train.txt"
    train_list = read_split_list(train_txt, missing_ok=True)
    if not isinstance(train_list, list):
        train_list = []
    train_entries = [str(x) for x in train_list]

    fold_lists = kfold_assign(train_entries, folds=int(args.folds), seed=int(args.seed))
    folds_payload = [
        {"fold": i, "n_images": len(lst), "images": lst[:20], "truncated": len(lst) > 20}
        for i, lst in enumerate(fold_lists)
    ]

    fold_split_files: list[str] = []
    write_root = (args.write_fold_splits or "").strip()
    if write_root:
        wr = Path(write_root)
        wr.mkdir(parents=True, exist_ok=True)
        for i, lst in enumerate(fold_lists):
            fold_dir = wr / f"fold_{i}"
            fold_dir.mkdir(parents=True, exist_ok=True)
            p_train = fold_dir / "train.txt"
            p_train.write_text("\n".join(lst) + ("\n" if lst else ""), encoding="utf-8")
            fold_split_files.append(str(p_train))

    aggregation: dict[str, object] | None = None
    fold_paths = [Path(x) for x in args.fold_metrics]
    if fold_paths:
        docs = [load_fold_metric(p) for p in fold_paths]
        aggregation = aggregate_fold_metrics(docs)

    payload = with_schema_version(
        {
            "status": "ok",
            "script": "cv_eval",
            "weights": str(args.weights),
            "folds": int(args.folds),
            "seed": int(args.seed),
            "dataset_root": str(root),
            "train_split": str(train_txt),
            "n_train_images": len(train_entries),
            "fold_assignments": folds_payload,
            "fold_split_files": fold_split_files,
            "aggregation": aggregation,
            "notes": (
                "v1: k-fold lists + optional --fold-metrics JSON aggregation. "
                "Per-fold GPU train: use --write-fold-splits then train.py per fold list."
            ),
        },
        schema_version="cv_eval_run.v1",
    )
    out_path = write_json(args.out, payload)
    cli_print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
