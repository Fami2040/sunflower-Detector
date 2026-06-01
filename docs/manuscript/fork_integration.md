# Fork integration vs upstream `main`

Agent-oriented map of what this repository adds relative to [Fami2040/sunflower-Detector](https://github.com/Fami2040/sunflower-Detector) `main`. Human submission text: [`reports/manuscript/`](../../reports/manuscript/).

## Upstream scope

- Telegram bot and sliced inference (`classifier.pt` gate + `best2.pt` detector)
- Shipped weights; training notebook/script with informal dataset references
- No frozen manuscript splits, counting-first protocol, or comparative experiment matrix

## Fork additions (HARCHOC)

| Module | Role | Manuscript use |
|--------|------|----------------|
| `data/splits/*.txt` + manifest | Frozen train/val/test | Methods — dataset |
| `scripts/eval.py` + HSP exports | GT/preds JSON, test mAP | Results Table 1 |
| `threshold_sweep.py` | Val-lock conf → test | Methods — operating point |
| `dual_metric.py` / `error_analysis.py` | Count MAE, TIDE buckets | Results, Discussion |
| `aug_smoke_index` + GPU queue | Literature-guided aug grid | Results — aug table |
| `benchmark_matrix` + `matrix_rows.v1` | Multi-detector zoo | Results — SOTA |
| `eval_domains.py` | Per-tray count MAE | Limitations / supplementary |
| `finetune.py` (planned GPU) | Tray holdout adaptation | Future work |
| `deploy_hsp_parity` | Deploy vs HSP conf check | Discussion — two-stage deploy |

## Evaluation vs production

```text
Manuscript metrics:  full-frame export @ conf 0.001 → val-lock → test count MAE
Production (bot):    classifier → SAHI slices → deploy filters (separate audit)
```

Figure 9 in the submitted docx should show the **production** path. All quantitative tables in this repo use the **HSP** path unless explicitly labeled deploy.

## Claims

**We claim:** reproducible counting-first evaluation on a versioned two-class seed dataset; systematic aug and detector comparisons on identical splits; transparent separation of ranking mAP and count MAE.

**We do not claim:** a new detection backbone; multi-site field validation; Telegram success rates without a deploy telemetry study.

## Cross-links

- Weights and study logic: [`ORIGIN_MAIN_AND_DATASET.md`](../ORIGIN_MAIN_AND_DATASET.md)
- Experiment–literature graph: [`study_lineage.md`](study_lineage.md)
- Reviewer gap map: [`reviewer_comments_backlog_gap.md`](reviewer_comments_backlog_gap.md)
