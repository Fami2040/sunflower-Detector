# Reports

Machine-readable experiment outputs (JSON/CSV) and publication markdown for the HSP pipeline. Paths here are citation sources for manuscript numbers.

- **Do not commit** run outputs (gitignored); tracked: this README, `reviewer2.md`, `manuscript/`, `_llm/`.
- **Regenerate:** [`docs/EXPERIMENTS.md`](../docs/EXPERIMENTS.md).

## Layers

| Layer | Path | Audience |
|-------|------|----------|
| **Human manuscript** | [`manuscript/`](manuscript/) | Abstract, rebuttal, methods/results prose for Word paste |
| **HSP metrics** | [`hsp/`](hsp/) | Canonical JSON + [`p0_summary.md`](hsp/p0_summary.md) headline card |
| **LLM validation** | [`_llm/`](_llm/) | Reproduce commands, audits, paste-check index |
| **Docx tables/figures** | [`manuscript/docx/`](manuscript/docx/) | Generated journal-style exports |

## Regeneration

| Goal | Command |
|------|---------|
| HSP + dual-metric | `experiment.py repro` |
| Reviewer validation JSON | `experiment.py reviewer2-repro` |
| Publication preflight | `experiment.py manuscript-preflight` |
| Docx-aligned tables/figures | `experiment.py manuscript-docx-repro` |

## Stale paths (do not cite)

| Avoid | Use |
|-------|-----|
| `reports/eval.json` | `reports/hsp/eval_*.json` |
| `reports/benchmarks/matrix_train.json` | `reports/hsp/matrix_train.json` |
| Flat `reviewer2_*.md` (removed) | `manuscript/` + `_llm/` |

Child catalogs: [`hsp/README.md`](hsp/README.md) (if present), [`manuscript/docx/README.md`](manuscript/docx/README.md).
