# Augmentation smoke — top ranked runs

**Primary metric:** test count MAE (15-ep smoke unless noted). **Reference best2:** 61.3 @ 100 ep. Showing top **10** ranked rows (equivalence duplicates excluded).

| Rank | ID | Run | MAE | Δ vs best2 | 95% CI | Key knobs |
|-----:|----|-----|----:|-----------:|--------|-----------|
| 1 | S1 | `aug_smoke_close3` | 68.9 | +7.6 | 60.1–78.5 (95%) | close_mosaic=3 only; mosaic active ep 0–11 |
| 2 | S9 | `aug_smoke_no_aug_yaml` | 73.2 | +12.0 | 62.1–84.2 (95%) | no aug_config; inline _BASELINE_DEFAULTS |
| 3 | S12 | `aug_smoke_amp_off` | 91.1 | +29.9 | 79.9–102.5 (95%) | S1 + amp=false |
| 4 | CLOSE10 | `aug_sweep_close10_15ep` | 96.7 | +35.4 | 85.8–108.3 (95%) | close_mosaic=2 scaled from 10 (15-ep sweep) |
| 5 | S10 | `aug_smoke_yolo11s` | 116.1 | +54.8 | 99.6–133.2 (95%) | train_smoke_rank_yolo11s_15ep.json + close3 aug |
| 6 | S5 | `aug_smoke_mosaic03` | 125.7 | +64.4 | 112.9–139.6 (95%) | mosaic=0.3, close_mosaic=3 |
| 7 | S8 | `aug_smoke_hsv_v045` | 125.9 | +64.6 | 112.8–139.9 (95%) | S3 + hsv_v=0.45 |
| 8 | S11 | `aug_smoke_musgd` | 136.3 | +75.0 | 123.7–150.0 (95%) | S10 + optimizer MuSGD, lr0=0.0001 |
| 9 | S4 | `aug_smoke_mosaic01` | 145.1 | +83.8 | 130.9–160.6 (95%) | mosaic=0.1, close_mosaic=3, translate=0.10 (≠ S1 close3 @ 0. |
| 10 | S3 | `aug_smoke_photometric` | 151.7 | +90.5 | 136.2–167.8 (95%) | mosaic=0, mixup=0, translate=0, scale=0, fliplr=0, hsv_s=0.4 |

## Footnotes

1. Operating confidence fixed on **val** (`min_count_mae` on `data/splits/val.txt`) and applied unchanged on **test** (`data/splits/test.txt`, *n*=109).
2. Full grid: [`reports/aug_smoke/leaderboard.md`](../../aug_smoke/leaderboard.md); index [`configs/experiments/aug_smoke_index.json`](../../../configs/experiments/aug_smoke_index.json).
