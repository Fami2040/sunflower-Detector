# Manuscript docx reproduction (HSP data, journal style)

Generated: 2026-06-01T11:42:54Z

Reproduces **quantitative** figures/tables aligned with `reports/plants-4336582.docx` from frozen HSP exports. Photos/setup figures (7–11) remain manual.

## Figures

| Docx | File | Status |
|------|------|--------|
| Figure 1 | `figures/figure_01_detection_example.png` | ok |
| Figure 2 | `figures/figure_02_training_curves.png` | ok |
| Figure 3 | `— (dataset spatial panels: not automated)` | manual |
| Figure 4 | `figures/figure_04_confusion_absolute.png` | ok |
| Figure 5 | `figures/figure_05_confusion_normalized.png` | ok |
| Figure 6 | `figures/figure_06_metrics_panels.png` | ok |
| Figure 7–11 | `manual photos / CVAT / architecture` | manual |

## Tables

| Docx | File |
|------|------|
| Table 1 | `tables/table_01_detection_metrics.md` |
| Table 2 | `tables/table_02_counting_summary.md` |
| Table 3 | `tables/table_03_error_bins.md` |

## Command

```bash
PYTHONPATH=. python scripts/experiment.py manuscript-docx-repro
```
