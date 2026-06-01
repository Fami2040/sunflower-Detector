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
| [literature_doi_audit_2026-06-01.md](../../reports/manuscript/literature_doi_audit_2026-06-01.md) | DOI + `claim_fit` audit (2026-06-01) |
| [lit_audit/](../../reports/manuscript/lit_audit/README.md) | Per-paper validation reports |
| [related_work_outline.md](related_work_outline.md) | §2 Related work outline |
| [gradcam_routing.md](gradcam_routing.md) | Grad-CAM entrypoint routing |
| [originality_contribution_peers.md](originality_contribution_peers.md) | MS-ORIG peer contrast |

## Generated publication artifacts (not in `docs/`)

Regenerated under [`reports/manuscript/`](../../reports/manuscript/README.md): preflight manifest, tables, docx catalog, backlog narrative. **Git-tracked:** `reports/manuscript/**/*.md`, `reports/reviewer2.md` — see [`FRESHNESS.md`](../../reports/manuscript/FRESHNESS.md) before paste. Figures/JSON stay local under `reports/`.  
Integration map: [`fork_integration.md`](fork_integration.md), [`study_lineage.md`](study_lineage.md).  
LLM validation: [`reports/_llm/index.md`](../../reports/_llm/index.md).

**Command:** `mamba run -n harchoc python scripts/experiment.py manuscript-preflight` — see [`docs/EXPERIMENTS.md` § Publication preflight](../EXPERIMENTS.md#publication-preflight-before-word-paste).
