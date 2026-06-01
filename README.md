# 🌻 Sunflower Seed Counter 

A deep learning-powered system for automatic analysis of sunflower head images, designed to count **developed** and **aborted seeds**.

## HARCHOC

**HARCHOC** stands for **Helianthus annuus rapid classification heuristics for capitula**.
It is the practical workflow in this repository for fast sunflower head analysis, combining classification checks and dense seed detection.

![HARCHOC concept](HARCHOC1.png)

👉 **Try the system directly via Telegram:**  
https://t.me/sunflower_detector1_bot  

👉 **Annotated dataset (CVAT, 2500 images):**  
https://izba-memes.ru/share/y9xGFqCW  

---

## 🚀 Overview

This project provides an end-to-end pipeline for **digital phenotyping of sunflower heads**, combining:

- Deep learning–based seed detection (YOLO)  
- Image validation (sunflower vs non-sunflower)  
- Telegram bot interface for easy access  

The system is designed for research and breeding applications, enabling fast and objective seed counting.

---

## ✨ Features

- 🌱 **Seed Counting** – Detects and counts developed & aborted seeds  
- 🧠 **Image Validation** – Filters non-sunflower images automatically  
- ⚡ **Fast Processing** – Optimized with slicing (SAHI) for high-resolution images  
- 📱 **Telegram Bot Interface** – Easy to use  
- 📊 **Clean Output** – Returns structured count statistics (text-only)  

---

## 🧠 Model Details

**Detection Model (YOLO)**  
- Class 0 → **developed** seeds  
- Class 1 → **aborted** seeds  

**Dataset layout** — CVAT share lists **~2500** annotated images; the frozen modeling pool is **1093** head images with YOLO labels (~875 / 109 / 109 train/val/test). See [`data/README.md`](data/README.md). Install `cp data/data.yaml.example "$DATASET_ROOT/data.yaml"`.

**Classifier Model (YOLO Classification)**  
- Class 0 → Non-sunflower  
- Class 1 → Sunflower  



## ⚙️ Installation

```bash
git clone <your-repo-url>
cd <your-project>
pip install -r requirements.txt
```

### Optional: model zoo backends

Some scripts support optional ML backends, but **CI does not require them** (imports are lazy and dry-runs stay stdlib-only).

- **Ultralytics (YOLOv8/10/11, RT-DETR)**: install `ultralytics`
- **SuperGradients (YOLO-NAS)**: install `super-gradients` (import module `super_gradients`)

### Weights cache (no downloads)

This repo’s benchmark harness records deterministic weight resolution but **does not download weights**.

- **Default cache dir**: `data/weights/` (gitignored)
- **Override**: `WEIGHTS_CACHE_DIR=/path/to/cache`

See `docs/EXPERIMENTS.md` for the exact rules used by `scripts/benchmark_matrix.py`.

Verify the cache before a sweep (no downloads):

```bash
PYTHONPATH=. python scripts/check_weights_cache.py
```

### Validation-only budget caps (benchmark plan)

Export `HARCHOC_MAX_EPOCHS`, `HARCHOC_MAX_IMGSZ`, and `HARCHOC_MAX_BATCH` before sweeps so oversized configs fail during plan validation. See [`docs/training_budget.md`](docs/training_budget.md).

## Reproducible experiment entrypoints

These scripts are designed to be **CI-safe** via `--dry-run` and to use the repo’s dataset resolution conventions.

Generate a benchmark plan (no heavy imports):

```bash
export DATASET_ROOT=/path/to/dataset
export PYTHONPATH=.
python scripts/benchmark_matrix.py --dry-run --out reports/hsp/matrix_plan.json
```

Canonical scientific outputs live under **`reports/hsp/`** (and publication exports under `reports/manuscript/`). See [`reports/README.md`](reports/README.md).

Ops notes (disk, run naming `{model}_e{N}_s{seed}`, PR branch `pr/backlog-ci-dataset`): see `docs/EXPERIMENTS.md`. Budget caps: `docs/training_budget.md`.

Compute train/val/test drift proxies (stdlib):

```bash
DATASET_ROOT=/path/to/dataset \
python scripts/split_drift.py --out reports/hsp/split_drift_p0.json
```

The bot returns:

developed seeds
Total seeds

Non-sunflower images are automatically rejected.

📦 Project Structure
models/
  ├── best.pt
  ├── best2.pt
  └── classifier.pt

telegram_bot.py
requirements.txt
🧪 Dataset
Annotated using CVAT (~2500 images in the public share)
**1093** images in the frozen modeling pool with seed-level YOLO labels
See [`data/README.md`](data/README.md) for CVAT export vs modeling splits

📎 Download:
https://izba-memes.ru/share/y9xGFqCW

⚠️ Notes
GPU (CUDA) is recommended for faster inference
Ensure model files are placed in the models/ directory
Designed for research and prototyping purposes

This work was supported by the Ministry of Science and Higher Education of the Russian Federation (Federal Scientific and Technical Program for Development of Genetic Technologies for 2019-2030, Agreement No. 075-15-2025-528 dated May 29, 2025).

📄 License

This project is provided for research and educational use.
