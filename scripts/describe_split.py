from __future__ import annotations

import argparse
import os
from pathlib import Path
import statistics
import sys; from pathlib import Path; _r = Path(__file__).resolve().parent.parent; (str(_r) not in sys.path) and sys.path.insert(0, str(_r)); from harchoc.script_entry import bootstrap_repo_imports; bootstrap_repo_imports()
from harchoc.datasets import describe_dataset, resolve_dataset
from harchoc.yaml_minimal import parse_minimal_yaml_flat
from harchoc.run_metadata import collect_run_metadata
from harchoc.schemas import with_schema_version
from scripts._common_cli import add_dataset_args, add_dry_run_arg, cli_print, require_existing_dir, write_json


_IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _read_image_size(p: Path) -> tuple[int, int]:
    """Delegate to shared PIL reader (``harchoc.eval_export.read_image_size``)."""
    from harchoc.eval_export import read_image_size

    return read_image_size(p)


def _resolve_data_yaml(spec_root: Path, yolo_data_yaml: Path | None) -> Path | None:
    if yolo_data_yaml is not None and yolo_data_yaml.exists():
        return yolo_data_yaml
    p = (spec_root / "data.yaml").resolve()
    return p if p.exists() else None


def _split_paths_from_data_yaml(*, data_yaml: Path, split: str) -> tuple[Path | None, Path | None]:
    """
    Returns (base_dir, split_path_or_list_file) where:
    - base_dir is the directory used to resolve relative paths inside yaml
    - split_path_or_list_file may be a directory of images or a .txt list file
    """
    obj = parse_minimal_yaml_flat(data_yaml)
    split_val = obj.get(split)
    if not isinstance(split_val, str) or not split_val.strip():
        return None, None

    # Ultralytics supports "path:" as a base directory.
    base_dir = data_yaml.parent
    path_val = obj.get("path")
    if isinstance(path_val, str) and path_val.strip():
        pv = Path(path_val).expanduser()
        base_dir = (data_yaml.parent / pv).resolve() if not pv.is_absolute() else pv.resolve()

    sp = Path(split_val).expanduser()
    split_path = (base_dir / sp).resolve() if not sp.is_absolute() else sp.resolve()
    return base_dir, split_path


def _collect_images_from_dir(img_dir: Path) -> list[Path]:
    if not img_dir.exists():
        return []
    return [p for p in sorted(img_dir.iterdir()) if p.is_file() and p.suffix.lower() in _IMG_EXTS]


def _split_list_applies_to_dataset(split_file: Path, dataset_root: Path) -> bool:
    """True when at least one listed image path exists under dataset_root."""
    if not split_file.is_file():
        return False
    for ln in split_file.read_text("utf-8").splitlines():
        rel = ln.strip()
        if not rel or rel.startswith("#"):
            continue
        p = Path(rel)
        img = p if p.is_absolute() else (dataset_root / p)
        return img.is_file()
    return False


def _collect_images_from_list(list_file: Path, *, dataset_root: Path | None = None) -> list[Path]:
    if not list_file.exists():
        return []
    imgs: list[Path] = []
    for ln in list_file.read_text("utf-8").splitlines():
        s = ln.strip()
        if not s or s.startswith("#"):
            continue
        p = Path(s).expanduser()
        if p.is_absolute():
            imgs.append(p.resolve())
        elif dataset_root is not None:
            imgs.append((dataset_root / p).resolve())
        else:
            imgs.append((list_file.parent / p).resolve())
    return imgs


def _infer_label_path(*, dataset_root: Path, image_path: Path) -> Path:
    """
    Best-effort YOLO layout inference:
      <root>/images/<split>/<name>.<ext>  ->  <root>/labels/<split>/<name>.txt
    """
    try:
        rel = image_path.resolve().relative_to(dataset_root.resolve())
    except Exception:
        return image_path.with_suffix(".txt")
    parts = list(rel.parts)
    if parts and parts[0] == "images":
        return (dataset_root / "labels" / Path(*parts[1:])).with_suffix(".txt")
    return (dataset_root / "labels" / rel).with_suffix(".txt")


def _parse_yolo_label_file(p: Path) -> list[int]:
    """
    Returns list of class ids for all boxes in file.
    Missing/empty files -> [].
    """
    if not p.exists():
        return []
    out: list[int] = []
    for ln in p.read_text("utf-8").splitlines():
        s = ln.strip()
        if not s:
            continue
        toks = s.split()
        if not toks:
            continue
        try:
            cls = int(float(toks[0]))
        except Exception:
            continue
        out.append(cls)
    return out


def describe_split_stats(*, dataset_root: Path, split: str, split_files: list[Path] | None = None, yolo_data_yaml: Path | None = None) -> dict[str, object]:
    data_yaml = _resolve_data_yaml(dataset_root, yolo_data_yaml)

    images: list[Path]
    split_file_paths = [p.expanduser().resolve() for p in (split_files or []) if str(p).strip()]
    if split_file_paths:
        # If any looks like a list file, treat all as list files; otherwise treat as dirs.
        if any(p.suffix.lower() == ".txt" for p in split_file_paths):
            images = []
            for p in split_file_paths:
                images.extend(_collect_images_from_list(p, dataset_root=dataset_root))
        else:
            images = []
            for d in split_file_paths:
                images.extend(_collect_images_from_dir(d))
    else:
        # Standard layout fallback.
        images = _collect_images_from_dir(dataset_root / "images" / split)
        if not images and data_yaml is not None:
            _, sp = _split_paths_from_data_yaml(data_yaml=data_yaml, split=split)
            if sp is not None:
                images = (
                    _collect_images_from_list(sp, dataset_root=dataset_root)
                    if sp.suffix.lower() == ".txt"
                    else _collect_images_from_dir(sp)
                )

    widths: list[int] = []
    heights: list[int] = []
    file_sizes: list[int] = []
    boxes_per_image: list[int] = []
    class_counts: dict[str, int] = {}

    total_boxes = 0
    total_pixels = 0.0

    for img in images:
        w, h = _read_image_size(img)
        widths.append(w)
        heights.append(h)
        file_sizes.append(int(img.stat().st_size))
        total_pixels += float(w) * float(h)

        lbl = _infer_label_path(dataset_root=dataset_root, image_path=img)
        classes = _parse_yolo_label_file(lbl)
        n = len(classes)
        boxes_per_image.append(n)
        total_boxes += n
        for c in classes:
            k = str(c)
            class_counts[k] = class_counts.get(k, 0) + 1

    def _stats_int(xs: list[int]) -> dict[str, float | int | None]:
        if not xs:
            return {"n": 0, "min": None, "max": None, "mean": None, "median": None}
        return {
            "n": int(len(xs)),
            "min": int(min(xs)),
            "max": int(max(xs)),
            "mean": float(statistics.fmean(xs)),
            "median": float(statistics.median(xs)),
        }

    mp = (total_pixels / 1e6) if total_pixels > 0 else 0.0
    boxes_per_mp = (float(total_boxes) / mp) if mp > 0 else None

    return {
        "split": split,
        "source": {
            "dataset_root": str(dataset_root),
            "data_yaml": str(data_yaml) if data_yaml is not None else None,
            "split_files": [str(p) for p in split_file_paths] if split_file_paths else None,
        },
        "images": {
            "count": len(images),
            "width": _stats_int(widths),
            "height": _stats_int(heights),
            "file_size_bytes": _stats_int(file_sizes),
        },
        "labels": {
            "boxes_per_image": _stats_int(boxes_per_image),
            "class_counts": {k: int(v) for k, v in sorted(class_counts.items(), key=lambda kv: int(kv[0]))},
        },
        "density": {
            "total_boxes": int(total_boxes),
            "total_megapixels": float(mp),
            "boxes_per_megapixel": float(boxes_per_mp) if boxes_per_mp is not None else None,
        },
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Describe split statistics (scaffold).")
    add_dataset_args(p)
    add_dry_run_arg(p)
    p.add_argument("--split", default="test", choices=["train", "val", "test"], help="Split name to describe.")
    p.add_argument(
        "--split-file",
        action="append",
        default=[],
        help="Optional split source: a directory of images or a .txt listing image paths (repeatable).",
    )
    p.add_argument("--out", default="reports/split_stats.json", help="Where to write stats JSON.")
    args = p.parse_args(argv)

    if args.dry_run:
        repo_root = Path(__file__).resolve().parents[1]
        out_path = write_json(
            args.out,
            with_schema_version(
                {
                "status": "dry-run",
                "script": "describe_split",
                "meta": collect_run_metadata(
                    repo_root=repo_root,
                    dataset_manifest=Path(args.manifest),
                ),
                "split": args.split,
                "out": str(Path(args.out)),
                },
                schema_version="describe_split_run.v1",
            ),
        )
        cli_print(f"Wrote {out_path}")
        return 0

    spec = resolve_dataset(
        manifest_path=args.manifest,
        default_dataset_name=args.default_dataset_name,
        dataset_name=args.dataset_name,
        dataset_root=args.dataset_root,
        yolo_data_yaml=args.yolo_data_yaml,
    )
    require_existing_dir(spec.root, what="Dataset root", hint="Export DATASET_ROOT=/path/to/extracted/dataset")

    repo_root = Path(__file__).resolve().parents[1]
    split_files = [Path(x) for x in (args.split_file or [])]
    if not split_files:
        dataset_root = Path(spec.root)
        for candidate in (
            (dataset_root / "data" / "splits" / f"{args.split}.txt").resolve(),
            (repo_root / "data" / "splits" / f"{args.split}.txt").resolve(),
        ):
            if _split_list_applies_to_dataset(candidate, dataset_root):
                split_files = [candidate]
                break

    stats = describe_split_stats(
        dataset_root=Path(spec.root),
        split=str(args.split),
        split_files=split_files,
        yolo_data_yaml=spec.yolo_data_yaml,
    )
    payload = with_schema_version(
        {
        "status": "ok",
        "meta": collect_run_metadata(
            repo_root=repo_root,
            dataset_manifest=spec.manifest_path,
        ),
        "dataset": {"description": describe_dataset(spec)},
        **stats,
        },
        schema_version="describe_split_run.v1",
    )
    out_path = write_json(args.out, payload)
    if not os.getenv("CI"):
        cli_print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

