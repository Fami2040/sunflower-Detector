# Manuscript narrative (from backlog)

*Generated from backlog · 2026-06-01*

## Methods status

| Metric | Value |
|--------|-------|
| Headline metric | Test count MAE at val-locked conf (not val mAP alone) |
| best2 | 61.3 MAE @ conf ~0.15 — upstream [`main`](https://github.com/Fami2040/sunflower-Detector) — [`ORIGIN_MAIN_AND_DATASET.md |
| Dataset | CVAT ~2500 — [`data/manifest.json`](data/manifest.json) (`y9xGFqCW`) |
| Aug HARCHOC retrain (100 ep) | 64.1 — did not beat best2 — [`aug_confirm_winner_100ep_summary.json`](reports/aug_smoke/aug_confirm_winner_100ep_summary |
| Error mix @ locked conf | FN ~53% ΔAP share; among FPs loc ~58%, bg ~35%, cls ~3% — [`error_test_report.json`](reports/hsp/error_test_report.json) |
| Next GPU job | `zoo_matrix_p0_5` in [`gpu_queue_full.json`](configs/experiments/gpu_queue_full.json) |

**Model stack (reference):**

| Step | Focus | Status |
|------|-------|--------|
| 1 | Train/export parity (`max_det=3000`) | Done |
| 2 | Full YOLO @ 1280, 100 ep | CLI [`train_yolov8m_baseline.json`](configs/experiments/train_yolov8m_baseline.json) |
| 3 | Val lock → test; FP budget | Done |
| 4 | Aug ablations | Closed |
| 5 | Model zoo + count columns | Next |
| 6 | Tray / domain | Next |
| 7 | RT-DETR query cap | P1-RTDETR-COUNT-REFRESH |
| 8 | Count levers (finetune, hard-neg, dupe post, head-ROI eval) | P1/P2 — see [Now](#now) |

**Active queue (Now):**

| ID | Status | Blocker → archive? | Next (open refs) |
|----|--------|-------------------|------------------|
| P0-5 | Next | — | — |
| P1-ZOO-PARITY | Next | — | — |
| P1-ZOO-PROV | Partial | — | — |
| P2-RTDETR-V2 | Partial | — | — |
| P2-DEIM-EVAL | Ready | — | — |
| DATA-ACQ-GEN | Next | — | — |
| P1-FINETUNE-TRAY | Partial | — | — |
| P1-RTDETR-COUNT-REFRESH | Next | — | — |
| P1-RTDETR-Q | Blocked | GPU | COUNT-REFRESH |
| P1-CV-TRAIN | Next | GPU | — |
| P2-HEAD-ROI-EVAL | Next | — | — |
| P2-RTDETR-IDX | Next | — | — |
| MS-SOTA | Blocked | — | — |
| MS-GEN | Partial | — | — |
| MS-ABS | Next | — | — |
| MS-EXPLAIN | Partial | — | — |
| LIT-VALIDATE | Next | — | — |

## Results available

- Headline test count MAE and locked conf: see anchor **best2** and `reports/hsp/dual_metric.json`, `reports/hsp/p0_summary.md`.
- Error / TIDE / confusion: `reports/hsp/error_test_report.json`, `eval.py --confusion-matrix-only`.
- Domain trays: `reports/domains/domain_eval.json`.
- Aug closed: `reports/aug_smoke/leaderboard.md` (production `robustness_minimal`).
- Manuscript drafts (repo): `docs/manuscript/reviewer_comments_backlog_gap.md` (MS-* Done sections).

## Limitations / open

- **Blocked:** P1-RTDETR-Q, MS-SOTA
- **Partial:** P1-ZOO-PROV, P2-RTDETR-V2, P1-FINETUNE-TRAY, MS-GEN, MS-EXPLAIN
- **Open Next/Blocked IDs:** P0-5, P1-ZOO-PARITY, DATA-ACQ-GEN, P1-RTDETR-COUNT-REFRESH, P1-CV-TRAIN, P2-HEAD-ROI-EVAL, P2-RTDETR-IDX, MS-ABS, LIT-VALIDATE, P1-RTDETR-Q, MS-SOTA
- Archive holds **50** ticket tokens; cross-link sample: P1-RTDETR-Q, P1-CV-TRAIN.

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

**Gap index:** [reports/reviewer2_programmatic_gaps.md](reports/reviewer2_programmatic_gaps.md) · [docs/manuscript/reviewer_comments_backlog_gap.md](docs/manuscript/reviewer_comments_backlog_gap.md)
