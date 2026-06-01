# Manuscript drafts (`docs/manuscript/`)

Repo-side **text and gap maps** for the sunflower HSP paper (LaTeX source is outside this repository). Quantitative numbers must match on-disk JSON under [`reports/hsp/`](../../reports/hsp/README.md), not prose here alone.

## Documents

| File | Role |
|------|------|
| [reviewer_comments_backlog_gap.md](reviewer_comments_backlog_gap.md) | Reviewer line → backlog ID → status |
| [val_test_map_gap.md](val_test_map_gap.md) | Val vs test mAP / count MAE narrative |
| [architecture_recommendations.md](architecture_recommendations.md) | Lit-backed architecture backlog |
| [literature_validated.md](literature_validated.md) | Citation registry (human index) |
| [literature_validated.json](literature_validated.json) | Machine-readable cites |
| [related_work_outline.md](related_work_outline.md) | §2 Related work outline |
| [gradcam_routing.md](gradcam_routing.md) | Grad-CAM entrypoint routing |
| [originality_contribution_peers.md](originality_contribution_peers.md) | MS-ORIG peer contrast |

## Generated publication artifacts (not in `docs/`)

Regenerated under [`reports/manuscript/`](../../reports/manuscript/README.md): preflight manifest, tables, docx catalog, backlog narrative. Figures: [`reports/figures/`](../../reports/figures/README.md). Reviewer CPU audits: `reports/reviewer2_*` ([`reviewer2_index.md`](../../reports/reviewer2_index.md)).

**Command:** `mamba run -n harchoc python scripts/experiment.py manuscript-preflight` — see [`docs/EXPERIMENTS.md` § Publication preflight](../EXPERIMENTS.md#publication-preflight-before-word-paste).
