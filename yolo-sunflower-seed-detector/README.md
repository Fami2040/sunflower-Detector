# yolo-sunflower-seed-detector

Train a YOLO model for sunflower seed / head detection.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate   # Windows
pip install -r requirements.txt
```

## Dataset

Prepare an Ultralytics `data.yaml` (paths to `train`/`val` images, `nc`, `names`). Example layout:

```
datasets/sunflower/
  images/train/ ...
  images/val/ ...
  labels/train/ ...
  labels/val/ ...
data/sunflower.yaml
```

## Train

**Kaggle dataset + fixed recipe** (`linaaabrahim/dataset1` via `kagglehub`):

```bash
python training.py
```

**CLI / local yaml** (generic):

```bash
python train.py --data data/sunflower.yaml --model yolov8n.pt --epochs 100 --imgsz 640 --batch 16
```

Environment overrides (optional): `YOLO_DATA_YAML`, `YOLO_MODEL`, `YOLO_EPOCHS`, `YOLO_IMGSZ`, `YOLO_BATCH`, `YOLO_DEVICE`, `YOLO_PROJECT`, `YOLO_RUN_NAME`, `YOLO_PATIENCE`.

## Create the GitHub repo

1. On GitHub: **New repository** → name `yolo-sunflower-seed-detector` (owner: your account).
2. Do not add README/license on GitHub (this folder already has them).
3. From this folder:

```bash
cd yolo-sunflower-seed-detector
git init
git add .
git commit -m "Initial commit: train.py and project skeleton"
git branch -M main
git remote add origin https://github.com/YOUR_USER/yolo-sunflower-seed-detector.git
git push -u origin main
```
