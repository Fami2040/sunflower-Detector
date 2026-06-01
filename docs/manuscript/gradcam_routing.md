# Grad-CAM routing (refactor audit closure)

**Status:** **Done** — custom implementation retained; canonical CLI routed.  
**Related:** [`architecture_recommendations.md`](architecture_recommendations.md) · [`refactor.md`](../../refactor.md) §3 · **P2-FIG-CAM** Done

---

## Canonical entry

Use **`experiment.py gradcam`** (not `make_figures.py` directly):

```bash
# After error_analysis --export-fp-crops on test split
mamba run -n harchoc python scripts/experiment.py gradcam --weights models/best2.pt
mamba run -n harchoc python scripts/experiment.py gradcam --dry-run
```

Delegates to `scripts/make_figures.py --figure fig_gradcam_panel` via `harchoc/experiment_argv.argv_for_gradcam`. See [`docs/EXPERIMENTS.md`](../EXPERIMENTS.md#unified-scriptsexperimentpy).

**Prerequisites:** `reports/hsp/error_test_report.json` with `fp_crops.results` (from `error_analysis.py --export-fp-crops`) and optional HSP weights for true overlays.

---

## Why custom YOLO graph (not pytorch-grad-cam)

| Concern | Repo choice |
|---------|-------------|
| Detection head / eval graph | Ultralytics YOLO needs **train-mode** forward on a **crop tensor** with backward through neck layer `model[-2]`; eval-mode graphs often drop grad paths (`gradcam_errors`, `status: partial`). |
| FP crop workflow | Overlays are bbox crops from `error_analysis` exports, not full-frame classifier CAM. |
| Strict ML surfacing | `harchoc/gradcam_panel.py` records `gradcam_errors`; `HARCHOC_STRICT_ML=1` re-raises via `record_ml_failure`. |
| CI / deps | No third-party CAM library in env or CI; implementation is torch + PIL only. |

Implementation: [`harchoc/gradcam_panel.py`](../../harchoc/gradcam_panel.py) (`_try_gradcam_overlay`, `_gradcam_scalar_target`).

---

## Deferred: pytorch-grad-cam

[pytorch-grad-cam](https://github.com/jacobgil/pytorch-grad-cam) is **not** adopted. YOLO/detection examples still require adapter work for our FP-crop + train-graph path; adding the dependency would not run in CI and does not improve the manuscript panel today.

Revisit only if we need multi-layer CAM ablations beyond the current FP mosaic — see [`refactor.md`](../../refactor.md) §7.
