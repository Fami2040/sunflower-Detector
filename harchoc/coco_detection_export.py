from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from harchoc.data_yaml import labels_path_for_image
from harchoc.eval_export import iter_split_image_paths, load_gt_annotations, read_image_size
from harchoc.sunflower_dataset import CLASS_NAMES
from harchoc.tide_summary import SUNFLOWER_COCO_CATEGORIES, _xyxy_to_coco_xywh


def _repo_splits_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "data" / "splits"


def _gt_record_to_coco(
  *,
    image_id: int,
    file_name: str,
    img_path: Path,
    dataset_root: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    w, h = read_image_size(img_path)
    label_path = labels_path_for_image(dataset_root=dataset_root, image_path=img_path)
    anns_xyxy = load_gt_annotations(label_path=label_path, img_w=w, img_h=h)
    image = {"id": image_id, "file_name": file_name, "width": w, "height": h}
    coco_anns: list[dict[str, Any]] = []
    for ann in anns_xyxy:
        bbox = _xyxy_to_coco_xywh(ann["bbox"])
        coco_anns.append(
            {
                "id": len(coco_anns) + 1,
                "image_id": image_id,
                "category_id": int(ann["category_id"]),
                "bbox": bbox,
                "area": float(bbox[2] * bbox[3]),
                "iscrowd": 0,
            }
        )
    return image, coco_anns


def export_split_to_coco_json(
    *,
    split_file: Path,
    dataset_root: Path,
) -> dict[str, Any]:
    images: list[dict[str, Any]] = []
    annotations: list[dict[str, Any]] = []
    ann_id = 1
    for image_id, img_path, file_name in iter_split_image_paths(
        split_file, dataset_root=dataset_root
    ):
        if not img_path.is_file():
            continue
        image, anns = _gt_record_to_coco(
            image_id=image_id,
            file_name=Path(file_name).name,
            img_path=img_path,
            dataset_root=dataset_root,
        )
        images.append(image)
        for ann in anns:
            row = dict(ann)
            row["id"] = ann_id
            annotations.append(row)
            ann_id += 1
    return {
        "images": images,
        "annotations": annotations,
        "categories": list(SUNFLOWER_COCO_CATEGORIES),
    }


def materialize_coco_detection_tree(
    *,
    dataset_root: Path,
    out_root: Path,
    train_split: Path | None = None,
    val_split: Path | None = None,
) -> dict[str, str]:
    """
    Write COCO instances JSON + symlinked images for external DETR trainers.

    Returns relative paths: train_img_dir, train_ann, val_img_dir, val_ann.
    """
    train_split = train_split or (_repo_splits_dir() / "train.txt")
    val_split = val_split or (_repo_splits_dir() / "val.txt")
    layout = {
        "train_images": "images/train",
        "val_images": "images/val",
        "train_ann": "annotations/instances_train.json",
        "val_ann": "annotations/instances_val.json",
    }
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "annotations").mkdir(parents=True, exist_ok=True)

    for split_key, split_file, ann_key, img_sub in (
        ("train", train_split, "train_ann", "train_images"),
        ("val", val_split, "val_ann", "val_images"),
    ):
        coco = export_split_to_coco_json(split_file=split_file, dataset_root=dataset_root)
        ann_path = out_root / layout[ann_key]
        ann_path.write_text(json.dumps(coco, indent=2), encoding="utf-8")
        img_dir = out_root / layout[img_sub]
        img_dir.mkdir(parents=True, exist_ok=True)
        for _image_id, img_path, file_name in iter_split_image_paths(
            split_file, dataset_root=dataset_root
        ):
            if not img_path.is_file():
                continue
            dst = img_dir / Path(file_name).name
            if not dst.exists():
                os.symlink(img_path, dst)
    return layout


def class_names_for_export() -> list[str]:
    return list(CLASS_NAMES)
