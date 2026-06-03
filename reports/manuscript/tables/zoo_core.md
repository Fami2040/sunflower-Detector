# Model zoo (zoo_yolo_only (P0-5)) — test metrics @ locked conf

**Group:** `zoo_yolo_only` · **Rows on disk:** 4/4 with test count MAE. **Train aggregate:** `reports/hsp/matrix_train.json` (train).

| Model ID | Run | Test MAE | mAP50 | mAP50-95 | Status |
|----------|-----|--------:|------:|---------:|--------|
| `yolov8m` | `yolov8m_e100_s0` | 111.9 | 0.064 | 0.029 | ok |
| `yolov10m` | `yolov10m_e100_s0` | 182.0 | 0.352 | 0.143 | ok |
| `yolo11m` | `yolo11m_e100_s0` | 119.6 | 0.077 | 0.035 | ok |
| `yolo26m` | `yolo26m_e100_s0` | 95.3 | 0.406 | 0.167 | ok |

## Footnotes

1. Operating confidence fixed on **val** (`min_count_mae` on `data/splits/val.txt`) and applied unchanged on **test** (`data/splits/test.txt`, *n*=109).
2. Primary manuscript metrics: **test** count MAE and ranking mAP50 at locked conf; val metrics are threshold-selection transparency only.
3. All four zoo rows have test count MAE from `reports/hsp/matrix_train.json` (P0-5 complete).
