# Silent failure audit: `scripts/*.py`

> **Note (2026-05-29):** [`harchoc/strict_ml.py`](../../harchoc/strict_ml.py) was restored (see companion [`silent_failure_audit_harchoc.md`](silent_failure_audit_harchoc.md)). Helpers: `capture_failure`, `StrictWarnings`, `fail_or_warn`, `record_ml_failure`.

Grep: `grep -rn 'except Exception' scripts --include='*.py'` (re-audit 2026-05-29).

Companion: [`silent_failure_audit_harchoc.md`](silent_failure_audit_harchoc.md).

Before trusting [backlog model-improvement stack](../../backlog.md#model-improvement-stack-test-count-mae) experiments (threshold sweeps, matrix train, dual-metric), run verify gates: `reports/hsp/agent_batch_verify.json` workflow and `mamba run -n harchoc python scripts/strict_ml_smoke.py` (see [`harchoc/strict_ml.py`](../../harchoc/strict_ml.py)).

## Legend

| Risk | Meaning |
|------|---------|
| **high** | Swallows or masks failures on critical paths with no JSON / warning surface in default mode |
| **med** | Acceptable fallback but should log or surface in JSON |
| **low** | Import bootstrap, path relativization, per-line parse skip, or already fails loud (`SystemExit`) |

---

## Summary

| Metric | Count |
|--------|------:|
| Total `except Exception` sites | 43 |
| **HIGH remaining (default mode)** | **0** |
| med | 10 |
| low | 32 |
| **HIGH fixed (cumulative)** | 8 |

---

## Remaining HIGH (default mode)

None (2026-05-29): `benchmark_matrix._ultralytics_eval_one` records `results_dict_error` when parse fails.

---

## Fixes this pass

| File | Lines | Fix |
|------|-------|-----|
| `scripts/rtdetr_smoke.py` | 35–58, 105–107 | `capture_failure` for GPU JSON; fail-closed on probe | **fixed** |
| `scripts/check_weights_cache.py` | 109–122 | `capture_failure` for downloads; `error_type` in row | **fixed** |
| `scripts/benchmark_matrix.py` | 202–209 | `capture_failure` + `fail_or_warn` on eval JSON mAP parse | **fixed** |
| `scripts/benchmark_matrix.py` | 474–494 | `capture_failure` + `results_dict_error` on Ultralytics `results_dict` | **fixed** |
| `scripts/split_drift.py` | 170–174 | `capture_failure` + `fail_or_warn` on image size read | **fixed** |
| `scripts/threshold_sweep.py` | 428–440 | `capture_failure` + `fail_or_warn` on YOLO box parse | **fixed** |
| `scripts/train.py` | 164–175, 223–230 | `capture_failure` + `append_capture_warning` for version / weights probes | **fixed** |
| `scripts/eval.py` | 127–164 | `strict_warnings` on metrics / per-class parse | **fixed** |

---

## Audit table

| file | line | pattern | risk | recommended fix | status |
|------|-----:|---------|------|-----------------|--------|
| `scripts/benchmark_matrix.py` | 13 | `_path` bootstrap | low | Keep | — |
| `scripts/benchmark_matrix.py` | 202 | parse eval JSON mAP | med | `capture_failure` + `fail_or_warn` | **fixed** |
| `scripts/benchmark_matrix.py` | 474–494 | `results_dict` parse via `capture_failure` | med | bench-row `results_dict_error` (**DRY-MATRIX-RESULTS**) | **fixed** |
| `scripts/check_gpu.py` | 10 | `_path` bootstrap | low | Keep | — |
| `scripts/check_gpu.py` | 70 | matmul bench → `bench_failed` in JSON | low | Already surfaced | — |
| `scripts/check_gpu.py` | 164 | matmul bench → exit 2 | low | Already surfaced | — |
| `scripts/check_weights_cache.py` | 10 | `_path` bootstrap | low | Keep | — |
| `scripts/check_weights_cache.py` | 109 | download ultralytics weight | med | `capture_failure` + `error_type` | **fixed** |
| `scripts/cv_eval.py` | 9 | `_path` bootstrap | low | Keep | — |
| `scripts/dataset_from_manifest.py` | 7 | `_path` bootstrap | low | Keep | — |
| `scripts/describe_split.py` | 11 | `_path` bootstrap | low | Keep | — |
| `scripts/describe_split.py` | 160 | `relative_to` fallback label path | low | Keep | — |
| `scripts/describe_split.py` | 185 | skip bad YOLO class token | med | Count skipped lines in report | open |
| `scripts/error_analysis.py` | 10 | `_path` bootstrap | low | Keep | — |
| `scripts/error_analysis.py` | 45 | PIL import → reason string | low | Keep; narrow to `ImportError` optional | — |
| `scripts/error_analysis.py` | 455 | FP crop I/O | med | Already records per-crop reason | — |
| `scripts/eval.py` | 11 | `_path` bootstrap | low | Keep | — |
| `scripts/eval.py` | 69 | `relative_to` for val split | low | Keep | — |
| `scripts/eval.py` | 94 | train path fallback | med | Warn when train dir missing | open |
| `scripts/eval.py` | 139 | float mAP50 | med | `strict_warnings.warn` | **fixed** |
| `scripts/eval.py` | 145 | float mAP50-95 | med | `strict_warnings.warn` | **fixed** |
| `scripts/eval.py` | 161 | per-class maps | med | `strict_warnings.warn` | **fixed** |
| `scripts/eval.py` | 207 | skip bad label class id | med | Count in audit stats | open |
| `scripts/eval_domains.py` | 8 | `_path` bootstrap | low | Keep | — |
| `scripts/experiment.py` | 8 | `_path` bootstrap | low | Keep | — |
| `scripts/finetune.py` | 7 | `_path` bootstrap | low | Keep | — |
| `scripts/gpu_sanity.py` | — | **Removed** — use `scripts/check_gpu.py sanity` | — | — | — |
| `scripts/gpu_smoke_ultralytics.py` | — | **Removed** — use `scripts/check_gpu.py smoke-ultralytics` | — | — | — |
| `scripts/make_figures.py` | 8 | `_path` bootstrap | low | Keep | — |
| `scripts/make_splits.py` | 9 | `_path` bootstrap | low | Keep | — |
| `scripts/migrate_configs.py` | 11 | `_path` bootstrap | low | Keep | — |
| `scripts/migrate_configs.py` | 29 | YAML scalar parse fallback | low | Keep | — |
| `scripts/migrate_configs.py` | 243 | per-file migrate error in report | low | Already surfaced | — |
| `scripts/pick_sample_images.py` | 8 | `_path` bootstrap | low | Keep | — |
| `scripts/pipeline_request.py` | 14 | `_path` bootstrap | low | Keep | — |
| `scripts/rtdetr_smoke.py` | 18 | `_path` bootstrap | low | Keep | — |
| `scripts/rtdetr_smoke.py` | 35–58 | GPU check JSON / missing output | high | `capture_failure` + fail closed | **fixed** |
| `scripts/rtdetr_smoke.py` | 105–107 | CUDA from `cuda_available` only | high | Require `status==ok` and exit 0 | **fixed** |
| `scripts/run_meta.py` | 8 | `_path` bootstrap | low | Keep | — |
| `scripts/split_drift.py` | 9 | `_path` bootstrap | low | Keep | — |
| `scripts/split_drift.py` | 170 | skip unreadable image size | high | `capture_failure` + `fail_or_warn` | **fixed** |
| `scripts/strict_ml_smoke.py` | 22 | `_path` bootstrap | low | Keep | — |
| `scripts/threshold_sweep.py` | 12 | `_path` bootstrap | low | Keep | — |
| `scripts/threshold_sweep.py` | 420 | ultralytics import → SystemExit | low | Already surfaced | — |
| `scripts/threshold_sweep.py` | 428 | YOLO box parse | high | `capture_failure` + `fail_or_warn` | **fixed** |
| `scripts/train.py` | 14 | `_path` bootstrap | low | Keep | — |
| `scripts/train.py` | 93 | `relative_to` split path | low | Keep | — |
| `scripts/train.py` | 164–175 | torch / ultralytics version probe | high | `capture_failure` + warnings sink | **fixed** |
| `scripts/train.py` | 223 | resolve weights dir | high | `capture_failure` + `append_capture_warning` | **fixed** |
| `scripts/train.py` | 386 | ultralytics import → SystemExit | low | Already surfaced | — |
| `scripts/validate_splits.py` | 8 | `_path` bootstrap | low | Keep | — |
