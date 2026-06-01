# Headline metrics (production anchor)

**Model:** `models/best2.pt` · **Metric:** test count MAE @ val-locked conf · **Split:** `data/splits/test.txt`

| Split | Count MAE | 95% CI | mAP50 | mAP50-95 | Conf | Status |
|-------|----------:|--------|------:|---------:|-----:|--------|
| test | 61.3 | 51.3–71.3 (95%) | 0.180 | 0.060 | 0.15 | ok |
| val | 71.0 | 58.8–84.2 (95%) | — | — | 0.15 | ok |

## Footnotes

1. Operating confidence fixed on **val** (`min_count_mae` on `data/splits/val.txt`) and applied unchanged on **test** (`data/splits/test.txt`, *n*=109).
2. Primary manuscript metrics: **test** count MAE and ranking mAP50 at locked conf; val metrics are threshold-selection transparency only.
3. Source: [`reports/hsp/dual_metric.json`](reports/hsp/dual_metric.json) (`dual_metric_report.v1`).
