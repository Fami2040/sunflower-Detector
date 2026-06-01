# Backlog (local, not committed)

Task tracker for HARCHOC sunflower seed counting. **Updated:** 2026-06-01 · **Branch:** `pr/backlog-ci-dataset`

| Anchor | Value |
|--------|--------|
| Headline metric | Test **count MAE** @ val-locked conf |
| **best2** | **61.3** MAE @ conf ~0.15 — [`ORIGIN_MAIN_AND_DATASET.md`](docs/ORIGIN_MAIN_AND_DATASET.md) |
| Dataset | CVAT ~2500 — [`data/manifest.json`](data/manifest.json) (`y9xGFqCW`) |
| Aug 100 ep confirm | **64.1** — did not beat best2 |
| **Next GPU job** | `zoo_matrix_p0_5` — [`gpu_queue_full.json`](configs/experiments/gpu_queue_full.json) |

**Doc map:** [EXPERIMENTS](docs/EXPERIMENTS.md) · [reports/manuscript](reports/manuscript/) · [p0_summary](reports/hsp/p0_summary.md) · [reviewer2](reports/reviewer2.md) · [backlog_archive.md](backlog_archive.md)

**Convention:** Active work in [Now](#now) only. Completed items: [backlog_archive.md](backlog_archive.md).

---

## Dataset

Classes **0 = developed**, **1 = aborted**. Splits [`data/splits/*.txt`](data/splits/) (875 / 109 / 109). Val = threshold selection; **test** = manuscript metrics. Weights: `models/best2.pt`. GPU: `mamba run -n harchoc python …`

---

## Model stack

| Step | Status |
|------|--------|
| HSP protocol + threshold lock | Done |
| Aug ablations | **Closed** — anchor **61.3** |
| Zoo + count columns | **P0-5** Next |
| Tray / domain / finetune | Partial / queued |

---

## Now

| ID | Pri | Status | Next |
|----|-----|--------|------|
| **P0-5** | P0 | **Next** | `zoo_yolo_only` 4×100 ep → [`matrix_train.json`](reports/hsp/matrix_train.json) |
| **P1-ZOO-PARITY** | P1 | Next | Count MAE columns after P0-5 |
| **DATA-ACQ-GEN** | P1 | Next | Weak-tray acquisition — [archive § Data](backlog_archive.md) |
| **P1-FINETUNE-TRAY** | P1 | Partial | GPU stages — [FINETUNE_WEAK_TRAYS](docs/FINETUNE_WEAK_TRAYS.md) |
| **P1-RTDETR-COUNT-REFRESH** | P1 | Next | Refresh RT-DETR count MAE or document skip |
| **MS-SOTA** | P1 | Blocked | P0-5 complete |
| **MS-GEN** | P1 | Partial | Tray spread narrative |
| **MS-ABS** | P2 | Next | Use [`reports/manuscript/abstract.md`](reports/manuscript/abstract.md) |
| **LIT-VALIDATE** | P2 | **Partial Done** | 11/11 DOI audit 2026-06-01 — [`literature_doi_audit_2026-06-01.md`](reports/manuscript/literature_doi_audit_2026-06-01.md); cite only from [`literature_validated.json`](docs/manuscript/literature_validated.json) |

**GPU order:** `kill_stray_gpu_jobs.sh` → **P0-5** → `gpu_queue_post_zoo.json` (domain audit, finetune). Details: [EXPERIMENTS § GPU queue](docs/EXPERIMENTS.md#gpu-sequential-queue).

---

## Aug (closed)

Anchor **61.3**; best smoke **68.9**; 100-ep confirm **64.1**; mosaic-off rejected **147.4** (S2; LWCD reports mosaic disabled for **corn**, not evidence for sunflower — [`lit_audit/lwcd_yolo2025.md`](reports/manuscript/lit_audit/lwcd_yolo2025.md)). Leaderboard: [`reports/aug_smoke/leaderboard.md`](reports/aug_smoke/leaderboard.md). Full table: [archive § Aug](backlog_archive.md).

---

## Reporting

| Output | Path |
|--------|------|
| Headline | [`p0_summary.md`](reports/hsp/p0_summary.md) |
| Manuscript prose | [`reports/manuscript/`](reports/manuscript/) |
| HSP JSON | [`reports/hsp/`](reports/hsp/) |
| Reviewer validation | [`reports/_llm/`](reports/_llm/) |
| Gap map (agents) | [`docs/manuscript/reviewer_comments_backlog_gap.md`](docs/manuscript/reviewer_comments_backlog_gap.md) |
| Literature DOI audit | [`literature_doi_audit_2026-06-01.md`](reports/manuscript/literature_doi_audit_2026-06-01.md) · [`lit_audit/`](reports/manuscript/lit_audit/README.md) |

```bash
mamba run -n harchoc python scripts/experiment.py repro --stage full
PYTHONPATH=. HARCHOC_ALLOW_BASE_PYTHON=1 python scripts/experiment.py manuscript-preflight --dry-run
```

---

## Permanent

Extend `experiment.py` / `harchoc/*` before new `scripts/*.py` — [.cursor/rules/extend-before-add-script.mdc](.cursor/rules/extend-before-add-script.mdc).
