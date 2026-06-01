# Error analysis outputs (redirect)

`error_analysis.py` writes two artifacts per run:

| Flag | Schema | HSP canonical (test) | HSP canonical (val) |
|------|--------|----------------------|---------------------|
| `--out` | `error_analysis_summary.v1` | [`reports/hsp/error_test.json`](../hsp/error_test.json) | [`reports/hsp/error_val.json`](../hsp/error_val.json) |
| `--report` | `error_analysis_report.v1` | [`reports/hsp/error_test_report.json`](../hsp/error_test_report.json) | [`reports/hsp/error_val_report.json`](../hsp/error_val_report.json) |

Script defaults (`reports/error_analysis/summary.json`, `report.json`) are for ad-hoc runs and CI `--light` mode only. The HSP pipeline uses **`reports/hsp/error_*.json`** (see `configs/experiments/error_analysis_test.json`, [`docs/RESEARCH_AND_OPS.md`](../../docs/RESEARCH_AND_OPS.md)).

### Key fields (`error_analysis_summary.v1` / `error_analysis_report.v1`)

| Field | Role |
|-------|------|
| `ambiguous_summary` | Aggregate ambiguous detection counts + conf band |
| `ambiguous_fp_crosstab` | Cross-tab **ambiguous** vs **not_ambiguous** per FP bucket (`background`, `localization`, `classification`, `dupe`, `tp`); `by_flag` for `low_conf_band` / `pred_pred_overlap` |
| `tide_bucket_summary` | TIDE-style bucket counts + delta-AP share proxy |
| `counting_metrics_excl_ambiguous_band` | Count MAE with low-conf band removed (**P1-UNCERT-FP**) |

Schema guard: `harchoc/error_analysis_schema.py`.

## Regenerate from existing HSP exports

Uses on-disk `gt_*.json` / `preds_*.json` only (no eval re-export). Locked confidence comes from the val sweep (`threshold_val.json` → `selected.row.conf_thr`) or any sweep JSON whose `locked.row.conf_thr` records the val-locked point (e.g. [`threshold_test_locked.json`](../hsp/threshold_test_locked.json)).

```bash
# Val summary + taxonomy report
mamba run -n harchoc python scripts/error_analysis.py \
  --gt-json reports/hsp/gt_val.json \
  --preds-json reports/hsp/preds_val.json \
  --locked-conf-from reports/hsp/threshold_val.json \
  --out reports/hsp/error_val.json \
  --report reports/hsp/error_val_report.json

# Test (config + FP crops for Grad-CAM)
mamba run -n harchoc python scripts/error_analysis.py \
  --config configs/experiments/error_analysis_test.json \
  --locked-conf-from reports/hsp/threshold_test_locked.json \
  --export-fp-crops --fp-crops-dir reports/error_analysis/fp_crops
```

`load_locked_conf` prefers `locked.row.conf_thr` when a locked block is present (test sweeps); val-only sweeps use `selected.row`. Either `threshold_val.json` or `threshold_test_locked.json` yields conf **~0.15** for HSP test error analysis.

Optional FP crops (test): [`reports/error_analysis/fp_crops/`](fp_crops/) — referenced from `error_test_report.json` → `fp_crops.results`.
