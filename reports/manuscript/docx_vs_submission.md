# Submitted docx vs repository submission (drift audit)

The file `reports/plants-4336582.docx` is a **frozen reviewer snapshot**, not the build target.
Canonical submission text: `reports/manuscript/*.md`.

Docx on disk: **True**

## Known metric drift (expected until journal resubmission from markdown)

| Claim in docx | Repo value | Notes |
|---------------|------------|-------|
| test mAP50 0.793 (docx abstract) | docx 0.793 → repo 0.18 | submission uses reports/manuscript/abstract.md; technical detail in reports/_llm/map50_investigation.md |
| mean relative counting error 13.2% (docx) | docx 13.2 → repo 12.04 | use full test n=109 stats from reports/manuscript/abstract.md |
| 80% images relative error <2% (docx) | docx 80.0 → repo 13.8 | separate n=50 blinded audit from full test n=109 |
| Telegram 96.4% success / 15–30s latency (docx) | docx 96.4% → repo None | export deploy logs before citing; not in repo JSON |
| test count MAE 61.3 (repo headline) | docx not found in docx extract → repo 61.3 | headline MAE in reports/manuscript/abstract.md Results |

## Submission source of truth

- Abstract / rebuttal / methods: `reports/manuscript/`
- Tables / quantitative figures: `reports/manuscript/docx/`
- Regenerate parity: `python scripts/experiment.py reviewer2-paste-check`

