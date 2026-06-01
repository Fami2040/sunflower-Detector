# Aug smoke + sweep leaderboard

Generated: `2026-05-31T09:13:20Z` · primary metric: **test count MAE** (val-locked conf, test split, *n*=109).

## Reference (production)

| ID | run | epochs | test count MAE | 95% CI | notes |
|----|-----|--------|----------------|--------|-------|
| **best2** | `models/best2.pt` | 100 | **61.3** | 51.3–71.3 (95%) | production YOLOv8m 100 ep; HSP primary reference |

Source: `reports/hsp/fp_budget_sweep_test.json (locked conf 0.15, test split)`. No 15-ep smoke beats best2; best smoke/sweep is **+7.6 MAE** vs best2 (61.3).

## Rankings (15 ep)

| rank | smoke_id | run_name | MAE | 95% CI | key aug knobs | summary |
|------|----------|----------|-----|--------|---------------|---------|
| 1 | S1 | `aug_smoke_close3` | 68.9 | 60.1–78.5 (95%) | close_mosaic=3 only; mosaic active ep 0–11 | `reports/aug_smoke/s1_summary.json` |
| 2 | CLOSE25 *(sweep)* | `aug_sweep_close25_15ep` | 68.9 | 60.1–78.5 (95%) | close_mosaic=4 @ 15 ep (production 25; patience=11) | `reports/aug_smoke/sweep_close25_15ep_summary.json` |
| 3 | S9 | `aug_smoke_no_aug_yaml` | 73.2 | 62.1–84.2 (95%) | no aug_config; inline _BASELINE_DEFAULTS | `reports/aug_smoke/s9_summary.json` |
| 4 | S12 | `aug_smoke_amp_off` | 91.1 | 79.9–102.5 (95%) | S1 + amp=false | `reports/aug_smoke/s12_summary.json` |
| 5 | CLOSE10 *(sweep)* | `aug_sweep_close10_15ep` | 96.7 | 85.8–108.3 (95%) | close_mosaic=2 scaled from 10 (15-ep sweep) | `reports/aug_smoke/sweep_close10_15ep_summary.json` |
| 6 | S10 | `aug_smoke_yolo11s` | 116.1 | 99.6–133.2 (95%) | train_bench_yolo11s fields + epochs=15 | `reports/aug_smoke/s10_summary.json` |
| 7 | S5 | `aug_smoke_mosaic03` | 125.7 | 112.9–139.6 (95%) | mosaic=0.3, close_mosaic=3 | `reports/aug_smoke/s5_summary.json` |
| 8 | S8 | `aug_smoke_hsv_v045` | 125.9 | 112.8–139.9 (95%) | S3 + hsv_v=0.45 | `reports/aug_smoke/s8_summary.json` |
| 9 | S11 | `aug_smoke_musgd` | 136.3 | 123.7–150.0 (95%) | S10 + optimizer MuSGD, lr0=0.0001 | `reports/aug_smoke/s11_summary.json` |
| 10 | S4 | `aug_smoke_mosaic01` | 145.1 | 130.9–160.6 (95%) | mosaic=0.1, close_mosaic=3, translate=0.10 (≠ S1 close3 @ 0.05) | `reports/aug_smoke/s4_summary.json` |
| 11 | S2 | `aug_smoke_mosaic0` | 147.4 | 134.5–160.9 (95%) | mosaic=0, close_mosaic=0; translate=0.05, scale=0.15, fliplr=0.5 | `reports/aug_smoke/s2_summary.json` |
| 12 | S3 | `aug_smoke_photometric` | 151.7 | 136.2–167.8 (95%) | mosaic=0, mixup=0, translate=0, scale=0, fliplr=0, hsv_s=0.45, hsv_v=0.40, erasing=0.2 | `reports/aug_smoke/s3_summary.json` |

## Audit / duplicate class

Training-equivalent duplicates (canonical per cluster ranked above); not scored separately.

| smoke_id | run_name | MAE | 95% CI | key aug knobs | summary |
|----------|----------|-----|--------|---------------|---------|
| S0 | `aug_smoke_baseline` | 68.9 | 60.1–78.5 (95%) | production minimal; close_mosaic=15 (scales to 3 @ 15 ep) | `reports/aug_smoke/s0_summary.json` |
| S13 | `aug_smoke_patience5` | 68.9 | 60.1–78.5 (95%) | S1 + patience=5 only; same test MAE as S1 expected @ 15 ep when no early stop | `reports/aug_smoke/s13_summary.json` |
| S6 | `aug_smoke_erasing0` | 151.7 | 136.2–167.8 (95%) | S3 + erasing=0 | `reports/aug_smoke/s6_summary.json` |
| S7 | `aug_smoke_erasing03` | 151.7 | 136.2–167.8 (95%) | S3 + erasing=0.3 | `reports/aug_smoke/s7_summary.json` |

## Eval controls

Eval-only negative controls; excluded from ranked best smoke.

| smoke_id | run_name | MAE | 95% CI | key aug knobs | summary |
|----------|----------|-----|--------|---------------|---------|
| S14 | `aug_smoke_eval300` | 265.8 | 236.1–292.0 (95%) | S1 weights (aug_smoke_close3); eval --max-det 300 only — P0-1 truncation control, not aug ablation | `reports/aug_smoke/s14_summary.json` |

## Duplicate MAE clusters

Bit-identical test count MAE across runs — check `artifacts.preds_json.sha256` in summaries (follow-up **P1-AUG-DUP-MAE**; eval wiring is per-run weights, not shared preds).

- **68.9** ×4: **S0**, **S13**, **S1**, **CLOSE25** — `aug_smoke_baseline`, `aug_smoke_patience5`, `aug_smoke_close3`, `aug_sweep_close25_15ep` · preds `ad6f1621d8c2…` — identical test preds export (config equivalence or converged inference)
- **151.7** ×3: **S3**, **S6**, **S7** — `aug_smoke_photometric`, `aug_smoke_erasing0`, `aug_smoke_erasing03` · preds `41e79d287721…` — identical test preds export (config equivalence or converged inference)

## Regenerate

```bash
python scripts/experiment.py aug-leaderboard
```

Auto-refreshed after each aug smoke / sweep summary via `harchoc.aug_smoke_leaderboard.refresh_aug_smoke_leaderboard`.

Index: [`configs/experiments/aug_smoke_index.json`](../../configs/experiments/aug_smoke_index.json).
