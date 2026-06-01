from __future__ import annotations

import argparse
from pathlib import Path

import sys; from pathlib import Path; _r = Path(__file__).resolve().parent.parent; (str(_r) not in sys.path) and sys.path.insert(0, str(_r)); from harchoc.script_entry import bootstrap_repo_imports; bootstrap_repo_imports()
from harchoc.datasets import resolve_dataset
from scripts._common_cli import add_dataset_args


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Print paths to sample images.")
    add_dataset_args(p)
    p.add_argument("--split", default="train", choices=["train", "val", "test"], help="Split to sample from.")
    p.add_argument("--n", type=int, default=5, help="How many images to print.")
    p.add_argument("--ext", action="append", default=None, help="Allowed extension (repeatable).")
    args = p.parse_args(argv)

    spec = resolve_dataset(
        manifest_path=args.manifest,
        default_dataset_name=args.default_dataset_name,
        dataset_name=args.dataset_name,
        dataset_root=args.dataset_root,
        yolo_data_yaml=args.yolo_data_yaml,
    )

    img_dir = Path(spec.root) / "images" / args.split
    if not img_dir.exists():
        raise SystemExit(f"Not found: {img_dir}")

    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"} if args.ext is None else {e if e.startswith(".") else f".{e}" for e in args.ext}
    imgs = [p for p in sorted(img_dir.iterdir()) if p.suffix.lower() in exts]
    for pth in imgs[: args.n]:
        print(pth)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

