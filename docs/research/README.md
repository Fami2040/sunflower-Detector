# Research literature notes (`docs/research/`)

Synthesis docs for augmentation, detectors, thresholds, domain shift, FP taxonomy, and explainability. They **do not** duplicate task status — see [`backlog.md`](../../backlog.md).

## Canonical artifact paths (cite these in papers)

All quantitative claims must use paths from [`reports/README.md`](../../reports/README.md):

| Layer | Directory | Examples |
|-------|-----------|----------|
| **HSP science** | [`reports/hsp/`](../../reports/hsp/README.md) | `dual_metric.json`, `threshold_*.json`, `gt_*.json`, `preds_*.json`, `error_*.json`, `split_drift_p0.json`, `matrix_train.json` |
| **Aug smokes** | `reports/aug_smoke/` | `leaderboard.md`, `comparative_analysis.json` |
| **Domain / transfer** | `reports/domains/`, `reports/transfer/` | `domain_eval.json`, `finetune.json` |
| **Publication** | [`reports/manuscript/`](../../reports/manuscript/README.md), [`reports/figures/`](../../reports/figures/README.md) | preflight manifest, tables, docx catalog |
| **Reviewer-2 CPU** | `reports/reviewer2_*.json`, [`reports/_llm/`](../../reports/_llm/) | [`_llm/index.md`](../../reports/_llm/index.md) |

**Avoid:** `reports/eval.json`, `reports/split_drift/report.json`, `reports/benchmarks/matrix_train.json` (manuscript), root `reports/weights_cache.json` — see stale-path table in [`reports/README.md`](../../reports/README.md).

**Ad-hoc script defaults** (`reports/thresholds/`, `reports/error_analysis/summary.json`) are for exploratory runs only; the HSP pipeline uses `reports/hsp/error_*.json`.

## Scans (action-oriented)

| Doc | Focus |
|-----|--------|
| [training_tech_scan_2026_augmentation.md](training_tech_scan_2026_augmentation.md) | `max_det`, mosaic, S0–S14 |
| [training_tech_scan_2026_detectors.md](training_tech_scan_2026_detectors.md) | RT-DETR query cap, zoo |
| [training_tech_scan_2026_eval_calibration.md](training_tech_scan_2026_eval_calibration.md) | HSP protocol, dual-metric, matrix |
| [augmentation_robustness_literature.md](augmentation_robustness_literature.md) | Counting-first aug review |
| [threshold_calibration_literature.md](threshold_calibration_literature.md) | Conf/NMS/calibration |
| [fp_taxonomy_literature.md](fp_taxonomy_literature.md) | TIDE buckets, FP taxonomy |
| [domain_shift_transfer_literature.md](domain_shift_transfer_literature.md) | Tray keys, finetune |
| [explainability_uncertainty_literature.md](explainability_uncertainty_literature.md) | Grad-CAM, calibration, fuzzy band |
| [arch_ema_bg_spike_literature.md](arch_ema_bg_spike_literature.md) | Background FP / EMA |

Manuscript-facing drafts: [`docs/manuscript/`](../manuscript/). Ops entrypoint: [`docs/EXPERIMENTS.md`](../EXPERIMENTS.md).
