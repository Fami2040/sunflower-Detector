from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

from harchoc.data_yaml import labels_path_for_image, read_class_names
from harchoc.splits_io import materialize_abs_split_list, read_split_list
from harchoc import strict_ml


def _repo_splits_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "data" / "splits"


def materialize_yolo_staging(
    *,
    dataset_root: Path,
    staging_root: Path,
    train_split: Path,
    val_split: Path,
) -> dict[str, str]:
    """
    Symlink images/labels into a YOLO layout for SuperGradients dataloaders.

    Returns relative subdirs for train/val image and label folders.
    """
    layout = {
        "train_images": "images/train",
        "train_labels": "labels/train",
        "val_images": "images/val",
        "val_labels": "labels/val",
    }
    for split_key, split_file in (("train", train_split), ("val", val_split)):
        img_sub = layout[f"{split_key}_images"]
        lbl_sub = layout[f"{split_key}_labels"]
        img_dir = staging_root / img_sub
        lbl_dir = staging_root / lbl_sub
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)
        abs_list = staging_root / f"{split_key}_abs_paths.txt"
        materialize_abs_split_list(
            split_source=split_file,
            dataset_root=dataset_root,
            out_path=abs_list,
        )
        for img_str in read_split_list(abs_list, missing_ok=True):
            img = Path(img_str)
            if not img.is_file():
                continue
            lbl = labels_path_for_image(dataset_root=dataset_root, image_path=img)
            dst_img = img_dir / img.name
            dst_lbl = lbl_dir / (img.stem + ".txt")
            if not dst_img.exists():
                os.symlink(img, dst_img)
            if lbl.is_file() and not dst_lbl.exists():
                os.symlink(lbl, dst_lbl)
    return layout


def train_bench_run(
    *,
    model_id: str,
    dataset_root: Path,
    runs_dir: Path,
    run_name: str,
    epochs: int,
    imgsz: int,
    batch: int,
    seed: int,
) -> dict[str, Any]:
    """
  Train YOLO-NAS via SuperGradients. Lazy-imports super_gradients; never downloads via matrix.
    """
    try:
        from super_gradients.training import Trainer
        from super_gradients.training.dataloaders.dataloaders import (
            coco_detection_yolo_format_train,
            coco_detection_yolo_format_val,
        )
        from super_gradients.training import models
    except ImportError as exc:
        return {
            "status": "skipped",
            "reason": f"missing_dependency:super_gradients ({exc})",
            "returncode": 1,
        }

    train_split = _repo_splits_dir() / "train.txt"
    val_split = _repo_splits_dir() / "val.txt"
    if not train_split.is_file() or not val_split.is_file():
        return {
            "status": "failed",
            "reason": "missing_repo_splits",
            "returncode": 1,
        }

    classes = read_class_names(dataset_root=dataset_root)
    run_dir = (runs_dir / run_name).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = run_dir / "sg_checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    staging_root = Path(tempfile.mkdtemp(prefix="sg_yolo_staging_"))
    try:
        layout = materialize_yolo_staging(
            dataset_root=dataset_root,
            staging_root=staging_root,
            train_split=train_split,
            val_split=val_split,
        )
        train_data = coco_detection_yolo_format_train(
            dataset_params={
                "data_dir": str(staging_root),
                "images_dir": layout["train_images"],
                "labels_dir": layout["train_labels"],
                "classes": classes,
            },
            dataloader_params={"batch_size": batch, "num_workers": 2},
        )
        val_data = coco_detection_yolo_format_val(
            dataset_params={
                "data_dir": str(staging_root),
                "images_dir": layout["val_images"],
                "labels_dir": layout["val_labels"],
                "classes": classes,
            },
            dataloader_params={"batch_size": batch, "num_workers": 2},
        )

        trainer = Trainer(experiment_name=run_name, ckpt_root_dir=str(ckpt_dir))
        model = models.get(model_id, num_classes=len(classes), pretrained_weights="coco")

        trainer.train(
            model=model,
            training_params={
                "max_epochs": int(epochs),
                "initial_lr": 1e-4,
                "lr_mode": "cosine",
                "cosine_final_lr_ratio": 0.1,
                "batch_size": batch,
                "ema": True,
                "mixed_precision": True,
                "seed": int(seed),
                "dataset": classes,
            },
            train_loader=train_data,
            valid_loader=val_data,
        )

        best_ckpt = ckpt_dir / run_name / "ckpt_best.pth"
        if not best_ckpt.is_file():
            candidates = sorted(ckpt_dir.rglob("ckpt_best.pth"))
            best_ckpt = candidates[0] if candidates else None
        weights_out = run_dir / "weights"
        weights_out.mkdir(parents=True, exist_ok=True)
        best_link = weights_out / "best.pth"
        if best_ckpt is not None and best_ckpt.is_file():
            if best_link.exists() or best_link.is_symlink():
                best_link.unlink()
            shutil.copy2(best_ckpt, best_link)
        else:
            return {
                "status": "failed",
                "reason": "no_checkpoint_produced",
                "returncode": 1,
                "run_dir": str(run_dir),
            }

        val_metrics: dict[str, Any] = {}
        val_metrics_error: str | None = None
        with strict_ml.capture_failure("supergradients_post_train_val_metrics") as cap:
            val_metrics = dict(trainer.test(model=model, test_loader=val_data, test_metrics_list=None) or {})
        if cap.failed:
            val_metrics_error = f"{cap.exc_type}: {cap.exc_msg}"

        ok_payload: dict[str, Any] = {
            "status": "ok",
            "returncode": 0,
            "run_dir": str(run_dir),
            "weights": str(best_link),
            "val_metrics": val_metrics,
        }
        if val_metrics_error is not None:
            ok_payload["val_metrics_error"] = val_metrics_error
        return ok_payload
    except Exception as exc:
        return {
            "status": "failed",
            "reason": str(exc),
            "exc_type": type(exc).__name__,
            "returncode": 1,
            "run_dir": str(run_dir),
        }
    finally:
        shutil.rmtree(staging_root, ignore_errors=True)
