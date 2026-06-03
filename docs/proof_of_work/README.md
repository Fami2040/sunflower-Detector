# Proof of work (GPU study, 2026-06)

Tracked artifacts for the post-zoo GPU queue and finetune pilot. Full logs and `reports/*` JSON remain local (gitignored); this folder holds reproducible summaries, log tails, and one tray finetune checkpoint.

## Production anchor

| Artifact | Test count MAE | mAP50 | Notes |
|----------|----------------:|------:|-------|
| `models/best2.pt` | **61.3** | **0.18** | Global deploy weights; SHA256 in `checkpoints/SHA256SUMS` |

## Zoo (close3, 100 ep, M-scale) — `reports/hsp/matrix_train.json`

| Model | Test MAE | mAP50 |
|-------|--------:|------:|
| yolov8m | 111.9 | 0.064 |
| yolo11m | 119.6 | 0.077 |
| yolo26m | 95.3 | 0.406 |
| yolov10m | 182.0 | 0.352 |

Higher mAP does not imply lower count MAE on this task.

## Finetune pilot (tray 350, stage1, base `best2`)

| Metric | Before | After |
|--------|-------:|------:|
| Tray 350 MAE | 162.0 | **7.0** |
| Canonical test MAE | 61.3 | **51.5** (gate **passed**, limit 67.4) |

- Queue summary: `queue_summaries/finetune_queue_finetune_weak_tray_2.json`
- Weights: `checkpoints/finetune_tray350_stage1_best.pt` (copy of `runs/transfer/finetune_350_s1/weights/best.pt`)
- Train log tails: `logs/finetune_weak_tray_350_tail.txt`, `logs/finetune_weak_tray_3a5-9_tail.txt`
- Post-zoo queue state: `gpu_queue_post_zoo_run_state.json` (8/8 jobs completed)

## Local-only (not in git)

- Full finetune logs: `reports/gpu_queue/logs/finetune_weak_tray_*/`
- Zoo run dirs: `runs/hsp_zoo/`, joined close3 retrains under `runs/`
