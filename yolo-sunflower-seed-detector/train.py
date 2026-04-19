"""
Train a YOLO model for sunflower seed detection (e.g. fertilized vs unfertilized).

Requires: pip install ultralytics
Dataset: Ultralytics-style data.yaml (paths to train/val images and labels).
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train YOLO on sunflower seed dataset")
    p.add_argument(
        "--data",
        type=str,
        default=os.getenv("YOLO_DATA_YAML", "data/sunflower.yaml"),
        help="Path to data.yaml (train/val paths, class names, nc)",
    )
    p.add_argument(
        "--model",
        type=str,
        default=os.getenv("YOLO_MODEL", "yolov8n.pt"),
        help="Base weights, e.g. yolov8n.pt or yolov8s.pt",
    )
    p.add_argument("--epochs", type=int, default=int(os.getenv("YOLO_EPOCHS", "100")))
    p.add_argument("--imgsz", type=int, default=int(os.getenv("YOLO_IMGSZ", "640")))
    p.add_argument("--batch", type=int, default=int(os.getenv("YOLO_BATCH", "16")))
    p.add_argument(
        "--project",
        type=str,
        default=os.getenv("YOLO_PROJECT", "runs/sunflower"),
        help="Ultralytics project directory",
    )
    p.add_argument(
        "--name",
        type=str,
        default=os.getenv("YOLO_RUN_NAME", "train"),
        help="Run name under project/",
    )
    p.add_argument(
        "--device",
        type=str,
        default=os.getenv("YOLO_DEVICE", ""),
        help="cuda, cpu, or 0,1,... (empty = auto)",
    )
    p.add_argument(
        "--patience",
        type=int,
        default=int(os.getenv("YOLO_PATIENCE", "50")),
        help="Early stopping patience (epochs without improvement)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    data_path = Path(args.data)
    if not data_path.is_file():
        raise FileNotFoundError(
            f"data yaml not found: {data_path.resolve()}\n"
            "Create a Ultralytics data.yaml pointing to your train/val images and labels."
        )

    from ultralytics import YOLO

    model = YOLO(args.model)
    train_kw = dict(
        data=str(data_path.resolve()),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        project=args.project,
        name=args.name,
        patience=args.patience,
        exist_ok=True,
    )
    if args.device:
        train_kw["device"] = args.device

    model.train(**train_kw)
    print(f"Done. Weights under: {Path(args.project) / args.name}")


if __name__ == "__main__":
    main()
