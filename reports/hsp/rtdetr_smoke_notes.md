# RT-DETR 15-ep smoke notes (2026-05-29)

## GPU check
- `reports/hsp/gpu_check.json` (via smoke JSON): **ok** (CUDA V100S-8Q, torch 2.6.0+cu124).

## Train attempts

1. **~12:02Z** — `rtdetr_smoke.py --run-train` failed at startup: Ultralytics `SyntaxError: 'num_queries' is not a valid YOLO argument` (policy field was forwarded). Fixed: `TRAIN_POLICY_ONLY_KEYS` in `harchoc/train_kwargs.py`.

2. **~12:03Z+** — Retry with `DATASET_ROOT` set; training under `runs/detect/runs/rtdetr_smoke_15ep/`.

## Status (refresh 2026-05-29 ~13:36Z UTC)
- **Process:** `train.py --name rtdetr_smoke_15ep` still running (`pgrep`); do **not** start a duplicate 15-ep run.
- **Progress:** `results.csv` has **6 / 15** epochs; `best.pt` / `last.pt` updated each epoch (~252 MB).
- **Wall time:** cumulative `time` column ~5059 s through epoch 6 (~14 min/epoch); full 15 ep likely **~3.5 h** total.

## Val loss `nan` (epochs 2–4)

| epoch | val/giou | val/cls | val/l1 | train losses | notes |
|------:|----------|---------|--------|--------------|-------|
| 1 | 3.23 | 0.035 | 1.44 | finite | baseline val ok |
| 2–4 | nan | nan | nan | finite | val metrics missing in CSV only |
| 5–6 | finite | finite | finite | finite | val path recovered |

**Assessment (smoke):** **Not blocking** for P0.4 harness pass. Training did not abort; train losses and LR schedule continued; epoch 5+ val columns are finite. Typical causes for intermittent DETR/RT-DETR val NaNs early in training include empty val matches, division in metric aggregation before boxes stabilize, or a skipped val batch—worth a one-line check in Ultralytics logs if it persists past epoch 10, but **acceptable for smoke** if run reaches `train_complete` with finite final-epoch val.

**Blocking would be:** OOM/kill, zero epochs written, or persistent NaN through final epoch with no `best.pt`.

## Post-train eval (2026-05-29 policy)

`train_rtdetr_smoke_15ep.json` sets `"eval": {"skip": true}` — smoke is **train-only**. Run HSP / test mAP with `HARCHOC_EXPORT_DEVICE=cpu` in a separate `eval.py` pass ([`docs/training_budget.md`](../../docs/training_budget.md) § Post-train eval).

## P0.4 backlog
- Prep gate: 15/15 epochs + weights under `runs/detect/runs/rtdetr_smoke_15ep/weights/`; JSON `train_complete` when `rtdetr_smoke.py --run-train` exits 0 with `eval.skip` (harness passes `--skip-eval`; `failure_phase` only on train failure).

## Commands
```bash
export DATASET_ROOT=/root/development/harchoc/sunflower-Detector/data/raw/extracted/dataset
export HARCHOC_MAX_EPOCHS=15 HARCHOC_MAX_IMGSZ=2048
mamba run -n harchoc python scripts/rtdetr_smoke.py --run-train
# monitor:
wc -l runs/detect/runs/rtdetr_smoke_15ep/results.csv
pgrep -af rtdetr_smoke_15ep
```
