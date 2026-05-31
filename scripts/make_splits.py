from __future__ import annotations

import argparse
import random
from pathlib import Path

import sys; from pathlib import Path; _r = Path(__file__).resolve().parent.parent; (str(_r) not in sys.path) and sys.path.insert(0, str(_r)); from harchoc.script_entry import bootstrap_repo_imports; bootstrap_repo_imports()
from harchoc.datasets import resolve_dataset
from harchoc.split_leakage_audit import (
    assign_splits_by_group,
    load_group_csv,
    parse_group_key_spec,
    resolve_group_id,
)
from scripts._common_cli import add_dataset_args, add_dry_run_arg, eprint, require_existing_dir


_DEFAULT_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _norm_exts(exts: list[str] | None) -> set[str]:
    if not exts:
        return set(_DEFAULT_EXTS)
    out: set[str] = set()
    for e in exts:
        e = e.strip().lower()
        if not e:
            continue
        out.add(e if e.startswith(".") else f".{e}")
    return out or set(_DEFAULT_EXTS)


def _rel_to_root(root: Path, p: Path) -> str:
    return p.resolve().relative_to(root.resolve()).as_posix()


def _iter_images_in_dir(d: Path, exts: set[str]) -> list[Path]:
    if not d.exists():
        return []
    return [p for p in sorted(d.iterdir()) if p.is_file() and p.suffix.lower() in exts]


def _write_split_txt(out_path: Path, rel_paths: list[str]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(rel_paths) + ("\n" if rel_paths else ""), "utf-8")


def _make_from_folders(root: Path, exts: set[str]) -> dict[str, list[str]]:
    splits: dict[str, list[str]] = {}
    for split in ("train", "val", "test"):
        d = root / "images" / split
        imgs = _iter_images_in_dir(d, exts)
        splits[split] = [_rel_to_root(root, p) for p in imgs]
    return splits


def _make_random(
    *,
    root: Path,
    glob_pat: str,
    exts: set[str],
    seed: int,
    val_frac: float,
    test_frac: float,
    group_key: str | None = None,
) -> dict[str, list[str]]:
    if val_frac < 0 or test_frac < 0 or (val_frac + test_frac) >= 1.0:
        raise SystemExit("Invalid fractions: require val_frac>=0, test_frac>=0, and val_frac+test_frac < 1")

    candidates = [p for p in sorted(root.glob(glob_pat)) if p.is_file() and p.suffix.lower() in exts]
    if not candidates:
        raise SystemExit(f"No images matched glob {glob_pat!r} under {root}")

    rels = [_rel_to_root(root, p) for p in candidates]

    if group_key:
        kind, arg = parse_group_key_spec(group_key)
        csv_index: dict[str, str] | None = None
        if kind == "csv":
            assert isinstance(arg, str)
            csv_index = load_group_csv(Path(arg), dataset_root=root)

        def _group_for(rel: str) -> str:
            return resolve_group_id(rel, spec=group_key, csv_index=csv_index)

        return assign_splits_by_group(
            rels,
            group_for=_group_for,
            seed=seed,
            val_frac=val_frac,
            test_frac=test_frac,
        )

    rnd = random.Random(seed)
    idxs = list(range(len(candidates)))
    rnd.shuffle(idxs)
    shuffled = [candidates[i] for i in idxs]

    n = len(shuffled)
    n_val = int(n * val_frac)
    n_test = int(n * test_frac)
    n_train = n - n_val - n_test

    train = shuffled[:n_train]
    val = shuffled[n_train : n_train + n_val]
    test = shuffled[n_train + n_val :]

    return {
        "train": [_rel_to_root(root, p) for p in train],
        "val": [_rel_to_root(root, p) for p in val],
        "test": [_rel_to_root(root, p) for p in test],
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Materialize explicit train/val/test splits under data/splits/*.txt.")
    add_dataset_args(p)
    add_dry_run_arg(p)

    p.add_argument(
        "--mode",
        default="from-folders",
        choices=["from-folders", "random"],
        help="Split generation mode.",
    )
    p.add_argument(
        "--out-dir",
        default="data/splits",
        help="Directory under dataset root where split .txt files will be written.",
    )
    p.add_argument("--ext", action="append", default=None, help="Allowed image extension (repeatable).")

    p.add_argument(
        "--glob",
        default="images/**/*",
        help="(random mode) Glob under dataset root used to pick candidate images.",
    )
    p.add_argument("--seed", type=int, default=0, help="(random mode) RNG seed.")
    p.add_argument("--val-frac", type=float, default=0.1, help="(random mode) Fraction to assign to val.")
    p.add_argument("--test-frac", type=float, default=0.0, help="(random mode) Fraction to assign to test.")
    p.add_argument(
        "--group-key",
        default="",
        help=(
            "(random mode) Assign by group so members never span splits. "
            "Values: stem, parent, prefix:N, csv:PATH (or PATH.csv)."
        ),
    )

    args = p.parse_args(argv)

    if args.dry_run:
        return 0

    spec = resolve_dataset(manifest_path=args.manifest, default_dataset_name=args.default_dataset_name)
    require_existing_dir(spec.root, what="Dataset root", hint="Export DATASET_ROOT=/path/to/extracted/dataset")

    exts = _norm_exts(args.ext)
    root = Path(spec.root)
    out_dir = root / args.out_dir

    if args.mode == "from-folders":
        splits = _make_from_folders(root, exts)
    else:
        group_key = (args.group_key or "").strip() or None
        splits = _make_random(
            root=root,
            glob_pat=args.glob,
            exts=exts,
            seed=args.seed,
            val_frac=args.val_frac,
            test_frac=args.test_frac,
            group_key=group_key,
        )

    for split, rels in splits.items():
        _write_split_txt(out_dir / f"{split}.txt", rels)
        eprint(f"Wrote {out_dir / f'{split}.txt'} ({len(rels)} lines)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

