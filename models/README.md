# `models/` — frozen weights

Git-tracked **pointers only**; `.pt` files are local (see `.gitignore`).

| File | Role | Provenance |
|------|------|------------|
| **`best2.pt`** | Production YOLOv8m detector (developed / aborted) | **[Upstream `main`](https://github.com/Fami2040/sunflower-Detector)** — not trained in this fork. See [`docs/ORIGIN_MAIN_AND_DATASET.md`](../docs/ORIGIN_MAIN_AND_DATASET.md). |
| **`classifier.pt`** | Sunflower vs other gate (Telegram) | Upstream `main` |
| `best.pt` | Legacy / alternate (if present) | Upstream |

**HARCHOC eval:** headline test count MAE **61.3** @ locked conf **~0.15** — [`reports/hsp/dual_metric.json`](../reports/hsp/dual_metric.json).

Checksums: [`reports/hsp/baseline_models_manifest.json`](../reports/hsp/baseline_models_manifest.json).
