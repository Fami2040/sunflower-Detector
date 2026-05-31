from __future__ import annotations

from pathlib import Path

from harchoc.splits_io import read_split_list, resolve_split_entry


def count_yolo_boxes(label_path: Path) -> int:
    """Count non-empty YOLO label lines (one box per line). Missing file -> 0."""
    if not label_path.is_file():
        return 0
    n = 0
    for ln in label_path.read_text("utf-8", errors="replace").splitlines():
        s = ln.strip()
        if not s or s.startswith("#"):
            continue
        toks = s.split()
        if not toks:
            continue
        try:
            int(float(toks[0]))
        except (ValueError, TypeError):
            continue
        n += 1
    return n


def label_path_for_image(dataset_root: Path, rel_img: Path) -> Path:
    """Map ``images/.../name.ext`` -> ``labels/.../name.txt`` under dataset root."""
    parts = list(rel_img.parts)
    if parts and parts[0] == "images":
        return (dataset_root / "labels" / Path(*parts[1:])).with_suffix(".txt")
    return (dataset_root / "labels" / rel_img).with_suffix(".txt")


def peak_boxes_per_image(
    *,
    dataset_root: Path,
    splits_dir: Path,
    splits: tuple[str, ...] = ("train", "val", "test"),
) -> tuple[int, str | None]:
    """
    Return (peak_box_count, rel_image_path_with_peak) across split list files.
    """
    peak = 0
    peak_rel: str | None = None
    root = dataset_root.resolve()
    for split in splits:
        txt = splits_dir / f"{split}.txt"
        if not txt.is_file():
            continue
        rel_imgs = read_split_list(txt, as_paths=True)
        assert isinstance(rel_imgs, list)
        for rel_img in rel_imgs:
            rel = Path(rel_img)
            lbl = label_path_for_image(root, rel)
            if not lbl.is_file():
                img_path = resolve_split_entry(rel, dataset_root=root)
                lbl = label_path_for_image(root, Path(img_path.name))
                try:
                    rel_to_root = img_path.resolve().relative_to(root)
                    lbl = label_path_for_image(root, rel_to_root)
                except ValueError:
                    pass
            n = count_yolo_boxes(lbl)
            if n > peak:
                peak = n
                peak_rel = str(rel)
    return peak, peak_rel
