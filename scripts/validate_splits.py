from __future__ import annotations

import argparse
from pathlib import Path

import sys

_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))
from harchoc.script_entry import bootstrap_repo_imports

bootstrap_repo_imports()
from harchoc.datasets import resolve_dataset
from harchoc.label_stats import peak_boxes_per_image
from harchoc.rtdetr_limits import (
    SUNFLOWER_DOCUMENTED_PEAK_GT_BOXES_PER_IMAGE,
    ULTRALYTICS_RTDETR_DEFAULT_NUM_QUERIES,
    rtdetr_query_cap_message,
)
from harchoc.split_leakage_audit import audit_split_leakage, load_group_csv, parse_group_key_spec, splits_from_split_dir
from harchoc.splits_io import read_split_list
from scripts._common_cli import add_dataset_args, add_dry_run_arg, eprint, require_conda_env, require_existing_dir, write_json


def _label_for_image(rel_img: Path) -> Path:
    parts = list(rel_img.parts)
    if not parts:
        return rel_img
    if parts[0] == "images":
        parts[0] = "labels"
    else:
        # Best effort: if user provided a path not under images/, keep it but still
        # force extension to .txt so the failure is obvious.
        pass
    return Path(*parts).with_suffix(".txt")


def main(argv: list[str] | None = None) -> int:
    require_conda_env()
    p = argparse.ArgumentParser(description="Validate split text files and label counterparts exist.")
    add_dataset_args(p)
    add_dry_run_arg(p)
    p.add_argument("--splits-dir", default="data/splits", help="Directory under dataset root with split txt files.")
    p.add_argument(
        "--require-test",
        action="store_true",
        help="Fail if test.txt is missing (otherwise, only validate it if present).",
    )
    p.add_argument(
        "--check-rtdetr-query-cap",
        action="store_true",
        help=(
            "After path checks, compare scanned peak GT boxes/image to RT-DETR num_queries "
            f"(default {ULTRALYTICS_RTDETR_DEFAULT_NUM_QUERIES}) and documented peak "
            f"({SUNFLOWER_DOCUMENTED_PEAK_GT_BOXES_PER_IMAGE})."
        ),
    )
    p.add_argument(
        "--num-queries",
        type=int,
        default=ULTRALYTICS_RTDETR_DEFAULT_NUM_QUERIES,
        help="RT-DETR decoder num_queries for --check-rtdetr-query-cap.",
    )
    p.add_argument(
        "--documented-peak-gt-boxes",
        type=int,
        default=SUNFLOWER_DOCUMENTED_PEAK_GT_BOXES_PER_IMAGE,
        help="Expected peak GT boxes/image on frozen sunflower splits (drift guard).",
    )
    p.add_argument(
        "--audit-leakage",
        action="store_true",
        help="Run stem / aug-sibling leakage audit and write JSON report.",
    )
    p.add_argument(
        "--audit-leakage-out",
        default="reports/split_leakage_audit.json",
        help="Output path for --audit-leakage JSON report.",
    )
    p.add_argument(
        "--group-key",
        default="",
        help="Optional group-key spec (stem, parent, prefix:N, csv:PATH) for group collision checks.",
    )
    args = p.parse_args(argv)

    if args.dry_run:
        return 0

    spec = resolve_dataset(manifest_path=args.manifest, default_dataset_name=args.default_dataset_name)
    require_existing_dir(spec.root, what="Dataset root", hint="Export DATASET_ROOT=/path/to/extracted/dataset")

    root = Path(spec.root)
    splits_dir = root / args.splits_dir

    split_files = {
        "train": splits_dir / "train.txt",
        "val": splits_dir / "val.txt",
        "test": splits_dir / "test.txt",
    }
    if args.require_test and not split_files["test"].exists():
        raise SystemExit(f"Missing required split file: {split_files['test']}")

    n_checked = 0
    n_missing_images = 0
    n_missing_labels = 0

    for split, txt_path in split_files.items():
        if split == "test" and not txt_path.exists() and not args.require_test:
            continue
        rel_imgs = read_split_list(txt_path, as_paths=True)
        for rel_img in rel_imgs:
            img_path = root / rel_img
            if not img_path.exists():
                n_missing_images += 1
                eprint(f"Missing image ({split}): {rel_img}")
                continue
            rel_lbl = _label_for_image(Path(rel_img) if isinstance(rel_img, str) else rel_img)
            lbl_path = root / rel_lbl
            if not lbl_path.exists():
                n_missing_labels += 1
                eprint(f"Missing label ({split}): {rel_lbl} (for {rel_img})")
                continue
            n_checked += 1

    if n_missing_images or n_missing_labels:
        eprint(
            f"Validation failed: checked={n_checked}, missing_images={n_missing_images}, missing_labels={n_missing_labels}"
        )
        return 2

    if args.audit_leakage:
        splits = splits_from_split_dir(splits_dir)
        group_key = (args.group_key or "").strip() or None
        csv_index: dict[str, str] | None = None
        if group_key:
            kind, arg = parse_group_key_spec(group_key)
            if kind == "csv":
                assert isinstance(arg, str)
                csv_index = load_group_csv(Path(arg), dataset_root=root)
        leakage = audit_split_leakage(
            splits,
            group_key_spec=group_key,
            csv_index=csv_index,
        )
        report = {
            "script": "validate_splits",
            "splits_dir": str(splits_dir.relative_to(root)),
            "leakage_audit": leakage,
        }
        out_path = write_json(args.audit_leakage_out, report)
        eprint(f"Wrote leakage audit: {out_path} (ok={leakage.get('ok')})")
        if not leakage.get("ok"):
            eprint("Leakage audit failed: cross-split stem/group collisions detected")
            return 3

    if args.check_rtdetr_query_cap:
        peak, peak_rel = peak_boxes_per_image(dataset_root=root, splits_dir=splits_dir)
        eprint(
            f"RT-DETR query cap: scanned_peak_gt_boxes_per_image={peak}"
            + (f" ({peak_rel})" if peak_rel else "")
        )
        if peak != int(args.documented_peak_gt_boxes):
            eprint(
                f"WARNING: scanned peak {peak} != documented peak {args.documented_peak_gt_boxes} "
                "(split/label drift?)"
            )
        num_queries = int(args.num_queries)
        if num_queries < peak:
            eprint("WARNING: " + rtdetr_query_cap_message(num_queries=num_queries, peak_gt=peak))
            return 1

    eprint(f"OK: checked={n_checked}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

