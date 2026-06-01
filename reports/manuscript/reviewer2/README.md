# Reviewer 2 artifacts

| Kind | Path | Tracked |
|------|------|---------|
| Verbatim comments | [`reports/reviewer2.md`](../../reviewer2.md) | Yes |
| Formal rebuttal | [`response_to_reviewers.md`](../response_to_reviewers.md) | Yes |
| Co-author draft | [`reviewer2_rebuttal_for_coauthor.md`](../reviewer2_rebuttal_for_coauthor.md) | Yes |
| CPU repro chain | `experiment.py reviewer2-repro` | [`reviewer2_repro.json`](../../../configs/experiments/reviewer2_repro.json) |

Regenerate counting / mAP50 / confusion / paste-check JSON:

```bash
mamba run -n harchoc python scripts/experiment.py reviewer2-repro
PYTHONPATH=. HARCHOC_ALLOW_BASE_PYTHON=1 python scripts/experiment.py reviewer2-repro --dry-run
```

Canonical status: [`docs/manuscript/status.md`](../../../docs/manuscript/status.md).
