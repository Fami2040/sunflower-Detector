# Manuscript publication exports (`reports/manuscript/`)

**Freshness:** [`FRESHNESS.md`](FRESHNESS.md) — snapshot date and pending GPU rows before paste into Word.

## Submission source (markdown)

**Canonical text for journal submission and rebuttal** — not copy-paste from Word.

| File | Use |
|------|-----|
| [`abstract.md`](abstract.md) | IMRaD abstract |
| [`response_to_reviewers.md`](response_to_reviewers.md) | Point-by-point rebuttal |
| [`results_and_methods.md`](results_and_methods.md) | Methods, Results, comparative design |
| [`dataset.md`](dataset.md) | Dataset subsection (Methods) |

[`reports/plants-4336582.docx`](../plants-4336582.docx) is a **read-only reviewer snapshot** of the submitted draft. Use it for comment context and drift audits only ([`docx_vs_submission.md`](docx_vs_submission.md)).

LLM validation (reproduce, audits): [`../_llm/`](../_llm/).

## Generated artifacts (CPU)

After HSP metrics exist under [`reports/hsp/`](../hsp/README.md):

| Path | Role |
|------|------|
| [`preflight_manifest.json`](preflight_manifest.json) | Step status (`reviewer2_repro` → figures → tables → docx → aug → narrative) |
| [`tables/`](tables/) | `headline_metrics.md`, `aug_smoke_top_n.md`, `zoo_core.md` |
| [`docx/`](docx/README.md) | Quantitative figures 1–2, 4–6 and Tables 1–3 (`manuscript-docx-repro`) |
| [`narrative_from_backlog.md`](narrative_from_backlog.md) | Structured export from backlog |

```bash
mamba run -n harchoc python scripts/experiment.py manuscript-preflight
PYTHONPATH=. HARCHOC_ALLOW_BASE_PYTHON=1 python scripts/experiment.py reviewer2-paste-check
```

Docs: [`docs/EXPERIMENTS.md` § Publication preflight](../../docs/EXPERIMENTS.md#publication-preflight-before-word-paste).
