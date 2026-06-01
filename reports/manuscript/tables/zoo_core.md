# Model zoo (zoo_yolo_only (P0-5)) — test metrics @ locked conf

**Group:** `zoo_yolo_only` · **Snapshot:** 2026-06-01 · **Rows with MAE:** 3/4 complete; **YOLOv10m** 100-ep train in progress (not OOM-deferred). **Source:** `reports/hsp/matrix_train.json` + HSP error JSONs.

| Model ID | Run | Test MAE | mAP50 | mAP50-95 | Status |
|----------|-----|--------:|------:|---------:|--------|
| `yolov8m` | `yolov8m_e100_s0` | 111.9 | 0.064 | 0.029 | ok |
| `yolov10m` | `yolov10m_e100_s0` | — | — | — | **training** (draft) |
| `yolo11m` | `yolo11m_e100_s0` | 119.6 | 0.077 | 0.035 | ok |
| `yolo26m` | `yolo26m_e100_s0` | 95.3 | 0.406 | 0.167 | ok |

## Footnotes

1. Operating confidence fixed on **val** (`min_count_mae` on `data/splits/val.txt`) and applied unchanged on **test** (`data/splits/test.txt`, *n*=109).
2. Primary manuscript metrics: **test** count MAE and ranking mAP50 at locked conf; val metrics are threshold-selection transparency only.
3. **YOLOv10m:** 1-epoch VRAM probe peaked ~3.3 GiB on 8 GiB V100; prior “OOM deferral” was incorrect (row had not been trained). Refresh this table after `runs/hsp_zoo/yolov10m_e100_s0/weights/best.pt` exists (`manuscript-preflight`).
4. Anchor `best2.pt` test MAE **61.3** — not in this table (upstream production weights, same protocol).
