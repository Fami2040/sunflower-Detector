# Reports (generated)

This directory stores **generated** experiment artifacts (JSON/CSV/MD summaries).

- Keep large or frequently-changing outputs untracked.
- Keep small “schema/contract” examples tracked only when helpful.

## Layout

| Path | Role |
|------|------|
| **`reports/hsp/`** | Canonical HSP baseline pipeline (eval exports, threshold sweeps, error analysis, dual-metric). |
| `reports/thresholds/` | Script default for ad-hoc sweeps — see [`thresholds/README.md`](thresholds/README.md). |
| `reports/_archive/` | Obsolete outputs (May 29 hygiene); see [`_archive/README.md`](_archive/README.md). |

## Canonical HSP path

**Manuscript and P0 gate metrics live under [`reports/hsp/`](hsp/README.md).** Start there for `dual_metric.json`, threshold sweeps, gt/preds exports, error analysis, split drift, and baseline model manifests.

One-page headline numbers: [`reports/hsp/p0_summary.md`](hsp/p0_summary.md).

## Archived (May 29 hygiene)

Obsolete artifacts were moved to [`reports/_archive/`](_archive/README.md) (legacy `eval/`, 800px bench YAMLs, zoo logs, duplicate GPU probes, etc.). Do not cite archived paths for P0 metrics.

## Do not use (stale / non-canonical)

If you see these paths outside `_archive/`, regenerate or use `reports/hsp/` instead:

| Path | Issue |
|------|--------|
| `reports/eval.json` | Early eval dump; archived |
| `reports/eval_hsp_yolov8n_img800.json` | **800px** model; archived |
| `reports/eval/` | Superseded by `reports/hsp/gt_*.json` / `preds_*.json` |
| `reports/eval_data.yaml`, `reports/eval_val_abs_paths.txt` | Root copies; canonical under `reports/hsp/` |

For the live HSP bundle layout, see [`reports/hsp/README.md`](hsp/README.md).
