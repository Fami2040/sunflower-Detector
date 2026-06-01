# Figures (`reports/figures/`)

Generated manuscript figures and `run.json` manifest (`figures_run.v1`).

## Journal style (MS-FIG-NORM)

`harchoc/figure_style.py` centralizes manuscript export defaults:

| Setting | Value |
|---------|--------|
| DPI | **300** (`savefig` + rcParams) |
| Fonts | Sans-serif: Arial → Helvetica → DejaVu Sans |
| Body text | 9 pt axes/labels; 8 pt ticks/legend |
| Panel labels | Bold **A**, **B**, **C**, … on each subplot or single-panel plot |

`scripts/make_figures.py` enables this by default (`--journal-style`; pass `--no-journal-style` for legacy 120–150 DPI output).

## Regenerate from HSP JSON

From the repo root. **CPU** (matplotlib + PIL only) for drift, PR curve, taxonomy, and ambiguous panels:

```bash
python scripts/make_figures.py \
  --out-dir reports/figures \
  --meta-out reports/figures/run.json \
  --split-drift-report reports/hsp/split_drift_p0.json \
  --threshold-csv reports/hsp/threshold_val.csv \
  --threshold-json reports/hsp/threshold_val.json \
  --error-report reports/hsp/error_test_report.json \
  --figure all
```

Omit `--error-report` to regenerate only `fig_split_drift` and `fig_pr_curve`.

Grad-CAM FP crop panel (crop mosaic without overlays is CPU; **torch/GPU** only when adding heatmaps):

```bash
mamba run -n harchoc python scripts/make_figures.py \
  --out-dir reports/figures \
  --meta-out reports/figures/run.json \
  --error-report reports/hsp/error_test_report.json \
  --figure fig_gradcam_panel
```

Optional `--weights path/to/best.pt` adds Grad-CAM heatmap overlays on each crop (default: taxonomy mosaic only).

## Outputs

| Figure | Path(s) | Source |
|--------|---------|--------|
| `fig_split_drift` | `split_drift/labels_class_dist_l1.png` (A), `labels_class_jsd_nats.png` (B), `width_ks_pvalues.png` (C) | `reports/hsp/split_drift_p0.json` |
| `fig_pr_curve` | `threshold/pr_f1_vs_conf.png` (panel A) | `reports/hsp/threshold_val.csv` (+ optional `threshold_val.json` for best-F1 marker) |
| `fig_gradcam_panel` | `fig_gradcam_panel.png` (A–L subpanels) | `reports/hsp/error_test_report.json` (`fp_crops.results` + `reports/error_analysis/fp_crops/`); optional `--weights` → GPU Grad-CAM |
| `fig_error_taxonomy` | `fig_error_taxonomy.png` | Same FP crops, diverse error types (**CPU**, no Grad-CAM) |
| `fig_ambiguous_panel` | `fig_ambiguous_panel.png` | Localization FP + low-conf band from `ambiguous_summary` (**CPU**) |
| `fig_concept` | `fig_concept.png`, `fig_concept.svg` | HSP pipeline: detection → locked conf count → optional deploy gate (**CPU**) |

Regenerate concept figure only:

```bash
python scripts/make_figures.py \
  --out-dir reports/figures \
  --meta-out reports/figures/run.json \
  --figure fig_concept
```

- **Grad-CAM**: pass `--weights models/best2.pt` (or HSP checkpoint); `gradcam_overlays` should match panel count. `status: partial` + `gradcam_errors` when overlays fail (e.g. eval-mode graph); set `HARCHOC_STRICT_ML=1` to surface failures.
- If matplotlib is missing, plots record `skipped` + reason in `rendered`.
