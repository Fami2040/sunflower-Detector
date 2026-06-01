# Silent failure audit: `harchoc/*.py`

> **Note (2026-05-29):** [`harchoc/strict_ml.py`](../../harchoc/strict_ml.py) was restored to the full helper surface (`capture_failure`, `StrictWarnings`, `record_ml_failure`, `require_torch`, `require_cuda`, `fail_or_warn`, …). Opt-in strict runtime: `HARCHOC_STRICT_ML=1`.

Generated from `grep -n 'except Exception' harchoc --include='*.py'` (re-audit 2026-05-29; `rg` equivalent).

---

## Summary

| Metric | Count |
|--------|------:|
| Total `except Exception` sites | 18 |
| Excl. intentional `strict_ml.capture_failure` recorder | 17 |
| **HIGH remaining (default mode)** | **0** |
| MED | 5 |
| LOW | 12 |
| **HIGH fixed (this pass)** | 3 (gradcam panel paths) |

**Default mode** = `HARCHOC_STRICT_ML` unset. Failures must still appear in JSON / `strict_warnings` / `gradcam_errors`; strict mode additionally re-raises via `record_ml_failure` or `raise_if_strict`.

---

## Remaining HIGH (default mode)

_None._ Former HIGH gradcam paths now call `record_ml_failure` / `append_ml_error` and return `gradcam_errors` in the mosaic payload.

---

## Full table

| File | Line | Pattern | Risk | Recommended fix | Status |
|------|-----:|---------|------|-----------------|--------|
| `harchoc/strict_ml.py` | 58 | inside `capture_failure` | — | Intentional recorder (not a silent swallow) | — |
| `harchoc/gradcam_panel.py` | 137 | return `skipped` (matplotlib missing) | low | Keep skip status + reason | open |
| `harchoc/gradcam_panel.py` | 184 | panel CAM loop | med | `record_ml_failure` → `gradcam_errors` | **fixed** |
| `harchoc/gradcam_panel.py` | 188 | panel render loop | med | `record_ml_failure` → `gradcam_errors` | **fixed** |
| `harchoc/gradcam_panel.py` | 223 | return `False` (gradcam deps missing) | med | `append_ml_error` on deps import | **fixed** |
| `harchoc/gradcam_panel.py` | 322 | overlay fail → return `False` | med | `record_ml_failure` → `gradcam_errors` | **fixed** |
| `harchoc/gradcam_panel.py` | 329 | `pass` on `hook.remove` | low | Best-effort cleanup | — |
| `harchoc/gradcam_panel.py` | 334 | `pass` on `net.train` restore | low | Best-effort cleanup | — |
| `harchoc/eval_export.py` | 136 | `tolist` fallback `vals = row` | med | Narrow to `AttributeError`/`TypeError` | open |
| `harchoc/eval_export.py` | 163 | skip box on parse fail | med | `strict_warnings.warn("ultralytics_box_parse", …)` | **fixed** |
| `harchoc/eval_export.py` | — | PIL open via `capture_failure` + `strict_warnings` | med | `read_image_size` records `pil_image_open_failed` | **fixed** |
| `harchoc/supergradients_eval.py` | 95 | return failed dict (outer boundary) | med | Already includes `exc_type` | open |
| `harchoc/supergradients_train.py` | 174 | — | — | Post-train val via `capture_failure`; `val_metrics_error` on OK payload | **fixed** |
| `harchoc/supergradients_train.py` | 189 | return failed dict (outer boundary) | med | Already includes `exc_type` | open |
| `harchoc/gpu_probe.py` | 13 | return torch import error string | low | Keep for probe API | — |
| `harchoc/gpu_probe.py` | 53 | set `device_error` in JSON payload | low | Keep (explicit probe field) | — |
| `harchoc/gpu_probe.py` | 65 | — | — | `capture_failure` → `mem_get_info_error` | **fixed** |
| `harchoc/ml_env.py` | 92 | return `{ok: False, error: parse failed}` | low | Already surfaces error in payload | — |
| `harchoc/rtdetr_limits.py` | 24 | `SystemExit` on bad config int | low | Keep (fails loud) | — |
| `harchoc/rtdetr_limits.py` | 34 | `SystemExit` on bad env int | low | Keep (fails loud) | — |
| `harchoc/hpo_search.py` | 76 | `SystemExit` on bad HPO bounds | low | Keep (fails loud) | — |

**Removed from tree since prior audit** (no bare `except Exception`; use `capture_failure` or narrowed handlers): `threshold_protocol.py`, `model_zoo.py`, `label_stats.py`, `yaml_minimal.py`, `run_metadata.py`.

---

## Fixes this pass

| Area | Change |
|------|--------|
| **`capture_failure`** | Restored module; used in `gpu_probe`, `supergradients_train`, `eval_export.read_image_size`, `run_metadata`, etc. |
| **Grad-CAM errors** | `gradcam_panel.py` records structured `gradcam_errors`; strict mode re-raises via `record_ml_failure`. |
| **`eval_export` / `strict_warnings`** | Box parse and PIL open failures append to `strict_warnings` (included in eval JSON when caller passes `StrictWarnings`). |

---

## Usage

```bash
# Re-audit
grep -rn "except Exception" harchoc --include='*.py'

# Strict ML runtime
export HARCHOC_STRICT_ML=1
mamba run -n harchoc python scripts/check_gpu.py
```
