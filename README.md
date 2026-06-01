# Sunflower Seed Counter

Deep learning pipeline for benchtop sunflower capitula: count **developed** and **aborted** seeds, with optional sunflower-vs-other gating and a Telegram interface.

## HARCHOC

**HARCHOC** (Helianthus annuus rapid classification heuristics for capitula) is the reproducible workflow in this repository: frozen splits, counting-first evaluation, and manuscript-grade exports under `reports/`.

![HARCHOC concept](HARCHOC1.png)

- **Telegram bot:** https://t.me/sunflower_detector1_bot  
- **Annotated dataset (CVAT):** https://izba-memes.ru/share/y9xGFqCW  

## Overview

- YOLO detection (classes 0 = developed, 1 = aborted)  
- Image validation (sunflower vs non-sunflower)  
- SAHI slicing for high-resolution trays  

Dataset layout, splits, and class names: [`data/README.md`](data/README.md). Scientific metrics and rebuttal text: [`reports/README.md`](reports/README.md).

## Installation

GPU work uses the `harchoc` mamba environment:

```bash
mamba env create -f envs/mamba.yml -y
mamba run -n harchoc python scripts/bootstrap_env.py
```

Optional backends (Ultralytics, SuperGradients) are documented in [`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md). CI uses lightweight imports without GPU deps.

Weights live in `models/` (gitignored `.pt` files). See [`models/README.md`](models/README.md).

## Experiments

Reproducible entrypoints, GPU queue, zoo matrix, and manuscript counting eval (`reports/hsp/`; internal label **HSP**):

[`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md)

Quick plan (no GPU):

```bash
export DATASET_ROOT=/path/to/dataset
PYTHONPATH=. python scripts/benchmark_matrix.py --dry-run --out reports/hsp/matrix_plan.json
```

## Project layout

| Path | Role |
|------|------|
| `scripts/` | Training, eval, `experiment.py` orchestration |
| `harchoc/` | Shared library |
| `configs/` | Experiment and bench configs |
| `data/splits/` | Frozen train/val/test lists |
| `reports/` | HSP JSON, manuscript markdown, validation |
| `models/` | `best2.pt`, `classifier.pt` (local) |

## Acknowledgement

This work was supported by the Ministry of Science and Higher Education of the Russian Federation (Federal Scientific and Technical Program for Development of Genetic Technologies for 2019–2030, Agreement No. 075-15-2025-528 dated May 29, 2025).

## License

Research and educational use.
