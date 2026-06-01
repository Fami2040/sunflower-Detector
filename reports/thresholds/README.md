# Threshold sweep outputs (redirect)

Script default `--out` is `reports/thresholds/sweep.json` (ad-hoc / light mode). For the **HSP baseline pipeline**, canonical sweep artifacts live under **`reports/hsp/`**:

| Split / role | JSON | CSV (optional) |
|--------------|------|----------------|
| Val (tuning) | [`reports/hsp/threshold_val.json`](../hsp/threshold_val.json) | `reports/hsp/threshold_val.csv` |
| Test (locked conf) | [`reports/hsp/threshold_test_locked.json`](../hsp/threshold_test_locked.json) | — |

Use `--locked-conf-from reports/hsp/threshold_val.json` on test. Specs: `configs/experiments/threshold_sweep_val.json`, `threshold_sweep_test_locked.json`. Commands: [`docs/research/threshold_calibration_literature.md`](../../docs/research/threshold_calibration_literature.md), [`docs/RESEARCH_AND_OPS.md`](../../docs/RESEARCH_AND_OPS.md).
