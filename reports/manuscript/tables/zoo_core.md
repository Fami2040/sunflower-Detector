# Model zoo (zoo_yolo_only (P0-5)) — test metrics @ locked conf

**Group:** `zoo_yolo_only` · **Rows on disk:** 2/4 with test count MAE. **Train aggregate:** `reports/hsp/matrix_train.json` (train).

| Model ID | Run | Test MAE | mAP50 | mAP50-95 | Status |
|----------|-----|--------:|------:|---------:|--------|
| `yolov8m` | `yolov8m_e100_s0` | — | — | — | pending |
| `yolov10m` | `yolov10m_e100_s0` | — | — | — | pending |
| `yolo11m` | `yolo11m_e100_s0` | 119.6 | 0.077 | 0.035 | ok |
| `yolo26m` | `yolo26m_e100_s0` | 95.3 | 0.406 | 0.167 | ok |

## Footnotes

1. Operating confidence fixed on **val** (`min_count_mae` on `data/splits/val.txt`) and applied unchanged on **test** (`data/splits/test.txt`, *n*=109).
2. Primary manuscript metrics: **test** count MAE and ranking mAP50 at locked conf; val metrics are threshold-selection transparency only.
3. Empty MAE cells: train or HSP test eval not finished (P0-5 `zoo_matrix_train`); re-run `benchmark_matrix.py` with `--matrix-group zoo_yolo_only` after queue resume.
4. Partial aggregate (2/4) — table lists all `zoo_yolo_only` slots with graceful placeholders.
