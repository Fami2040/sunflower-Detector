# Manuscript narrative (from backlog)

*Generated from backlog · 2026-06-03*

## Methods status

| Metric | Value |
|--------|-------|
| Headline metric | Test count MAE @ val-locked conf |
| best2 | 61.3 MAE @ conf ~0.15 — [`ORIGIN_MAIN_AND_DATASET.md`](docs/ORIGIN_MAIN_AND_DATASET.md) |
| Dataset | CVAT 1093 — [`data/manifest.json`](data/manifest.json) (`y9xGFqCW`) |
| Aug 100 ep confirm | 64.1 — did not beat best2 |
| Next GPU job | Post-zoo → [`gpu_queue_post_zoo.json`](configs/experiments/gpu_queue_post_zoo.json) (base: `models/best2.pt`, see [`fine |

## Results available

- Headline test count MAE and locked conf: see anchor **best2** and `reports/hsp/dual_metric.json`, `reports/hsp/p0_summary.md`.
- Error / TIDE / confusion: `reports/hsp/error_test_report.json`, `eval.py --confusion-matrix-only`.
- Domain trays: `reports/domains/domain_eval.json`.
- Aug closed: `reports/aug_smoke/leaderboard.md` (production `robustness_minimal`).
- Manuscript drafts (repo): `docs/manuscript/reviewer_comments_backlog_gap.md` (MS-* Done sections).

## Limitations / open

- **Open Next/Blocked IDs:** none
- Archive holds **0** ticket tokens; cross-link sample: n/a.

## Repro commands

**reviewer2 repro:**
```bash
mamba run -n harchoc python scripts/experiment.py reviewer2-repro
```
```bash
PYTHONPATH=. HARCHOC_ALLOW_BASE_PYTHON=1 python scripts/experiment.py reviewer2-repro --dry-run
```
```bash
mamba run -n harchoc python scripts/experiment.py repro --stage post-zoo
```

**manuscript hsp:**
```bash
mamba run -n harchoc python scripts/experiment.py repro
```
```bash
PYTHONPATH=. HARCHOC_ALLOW_BASE_PYTHON=1 python scripts/experiment.py repro --dry-run
```

**manuscript full:**
```bash
mamba run -n harchoc python scripts/experiment.py repro --stage full
```
```bash
PYTHONPATH=. HARCHOC_ALLOW_BASE_PYTHON=1 python scripts/experiment.py repro --stage full --dry-run
```

**manuscript preflight:**
```bash
mamba run -n harchoc python scripts/experiment.py manuscript-preflight
```
```bash
PYTHONPATH=. HARCHOC_ALLOW_BASE_PYTHON=1 python scripts/experiment.py manuscript-preflight --dry-run
```
```bash
mamba run -n harchoc python scripts/experiment.py repro --stage preflight
```

**aug compare:**
```bash
mamba run -n harchoc python scripts/experiment.py aug-compare
```
```bash
PYTHONPATH=. HARCHOC_ALLOW_BASE_PYTHON=1 python scripts/experiment.py aug-compare --dry-run
```

**backlog narrative:**
```bash
mamba run -n harchoc python scripts/experiment.py backlog-narrative
```
```bash
PYTHONPATH=. HARCHOC_ALLOW_BASE_PYTHON=1 python scripts/experiment.py backlog-narrative --dry-run
```

**figures repro:**
```bash
mamba run -n harchoc python scripts/experiment.py figures-repro
```
```bash
PYTHONPATH=. HARCHOC_ALLOW_BASE_PYTHON=1 python scripts/experiment.py figures-repro --dry-run
```

**tables repro:**
```bash
mamba run -n harchoc python scripts/experiment.py tables-repro
```
```bash
PYTHONPATH=. HARCHOC_ALLOW_BASE_PYTHON=1 python scripts/experiment.py tables-repro --dry-run
```

**Gap index:** [reports/_llm/programmatic_gaps.md](reports/_llm/programmatic_gaps.md) · [docs/manuscript/reviewer_comments_backlog_gap.md](docs/manuscript/reviewer_comments_backlog_gap.md)
