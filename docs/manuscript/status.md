# Manuscript and GPU status (canonical)

**Updated:** 2026-06-01 · **Branch:** `pr/backlog-ci-dataset`  
**Reviewer theme map:** [`reviewer_comments_backlog_gap.md`](reviewer_comments_backlog_gap.md) (IDs only — no duplicate integration table)  
**Task IDs:** [`backlog.md`](../../backlog.md#now)

---

## Headline numbers (safe to cite)

| Metric | Value | Evidence |
|--------|------:|----------|
| Test count MAE | **61.3** | `reports/hsp/dual_metric.json` |
| Test mAP50 | **0.18** | `reports/hsp/eval_test_map.json` |
| Locked conf | **~0.15** | `reports/hsp/threshold_val.json` |
| Dataset | **1093** images | `data/manifest.json` (`sunflower-cvat-1093`, share `y9xGFqCW`) |

Do **not** cite pooled test mAP **0.79** without reconciling to HSP exports — see [`val_test_map_gap.md`](val_test_map_gap.md).

---

## Freshness (re-check after GPU jobs)

| Track | Status | Notes |
|-------|--------|-------|
| HSP + threshold lock | **Stable** | Anchor **61.3** |
| Aug S0–S14 + 100 ep | **Closed** | Best smoke **68.9**; confirm **64.1**; S2 mosaic-off **147.4** rejected |
| Zoo `zoo_yolo_only` | **3/4 + v10m** | v8m **111.9**, 11m **119.6**, 26m **95.3**; v10m when `runs/hsp_zoo/yolov10m_e100_s0/weights/best.pt` exists |
| RT-DETR @ 1280 (Ultralytics) | **Blocked (8 GiB)** | 1-ep VRAM probe OOM |
| External DETR (DEIM / RT-DETRv2) | **Partial** | D-FINE trained; DEIM/rtdetrv2 port issues |
| Weak-tray finetune | **Queued** | After P0-5 — [`gpu_queue_post_zoo.json`](../../configs/experiments/gpu_queue_post_zoo.json) |
| Literature DOI audit | **Done** | [`literature_validated.json`](literature_validated.json); [`literature_doi_audit_2026-06-01.md`](../../reports/manuscript/literature_doi_audit_2026-06-01.md) |

**Regenerate after zoo / finetune:**

```bash
mamba run -n harchoc python scripts/experiment.py tables-repro
mamba run -n harchoc python scripts/experiment.py manuscript-preflight
```

---

## Now (active work)

| ID | Pri | Status | Next |
|----|-----|--------|------|
| **P0-5** | P0 | In progress | Finish v10m; repair [`matrix_train.json`](../../reports/hsp/matrix_train.json) when idle |
| **P1-ZOO-PARITY** | P1 | Next | `tables-repro` / MS-SOTA after P0-5 |
| **DATA-ACQ-GEN** | P1 | Next | +50 year cohort — [`data/cohorts/README.md`](../../data/cohorts/README.md) |
| **P1-FINETUNE-TRAY** | P1 | Queued | `gpu_queue_post_zoo.json` (gated on P0-5) |
| **MS-SOTA** | P1 | Partial | 3/4 zoo MAE; v10m footnote |
| **MS-GEN** | P1 | CPU | [`domain_count_mae.json`](../../reports/domains/domain_count_mae.json) |
| **MS-ABS** | P2 | Next | [`abstract.md`](../../reports/manuscript/abstract.md) |

**GPU order:** finish **P0-5** → `./scripts/run_gpu_queue.sh --manifest configs/experiments/gpu_queue_post_zoo.json --run --resume`

### P0-5 repair (when GPU idle)

```bash
mamba run -n harchoc python scripts/benchmark_matrix.py --group zoo_yolo_only --no-dry-run \
  --runs-dir runs/hsp_zoo --train-out reports/hsp/matrix_train.json
```

### Post-zoo wiring test (1 ep)

```bash
./scripts/run_gpu_queue.sh --manifest configs/experiments/gpu_queue_post_zoo_smoke.json --run --job finetune_tray_smoke_1ep
```

---

## Runnable verification (`now-todos-smoke`)

```bash
PYTHONPATH=. HARCHOC_ALLOW_BASE_PYTHON=1 python scripts/experiment.py now-todos-smoke --stage cpu
PYTHONPATH=. HARCHOC_ALLOW_BASE_PYTHON=1 python scripts/experiment.py now-todos-smoke --stage verify
HARCHOC_RUN_GPU_SMOKE=1 mamba run -n harchoc python scripts/experiment.py now-todos-smoke --stage gpu
```

Report: [`reports/manuscript/now_todos_smoke.json`](../../reports/manuscript/now_todos_smoke.json) (gitignored). Bundle: [`now_todos_smoke_bundle.json`](../../configs/experiments/now_todos_smoke_bundle.json).

| Stage | What runs |
|-------|-----------|
| verify | Config paths; matrix gate (blocked until 4/4 zoo rows) |
| cpu | Live `tables-repro`, `weak_tray_plan_smoke.json`, reviewer2 dry-run, queue dry-run |
| gpu | 1-ep finetune smoke (optional) |

---

## Stub vs live artifacts

| Path | Status | Combat |
|------|--------|--------|
| `preflight_manifest.json` | STUB | `manuscript-preflight` (live) |
| `tables/tables_manifest.json` | LIVE after smoke / tables-repro | Re-run when zoo completes |
| `tables/zoo_core.md` | PARTIAL until v10m | `tables-repro` |
| `matrix_train.json` | PARTIAL during train | `benchmark_matrix.py` when idle |
| `weak_tray_plan_smoke.json` | LIVE smoke | Does not overwrite production plan |
| `transfer/finetune.json` | STALE dry-run | Use post_zoo queue or archive |
| `now_todos_smoke.json` | LIVE harness output | `now-todos-smoke` |

**Do not cite:** v10m zoo row until weights exist; finetune tray outcomes; stale preflight.

---

## Closed

- HSP anchor **61.3** / aug grid closed / literature audit **11/11**
- Smoke harness + [`gpu_queue_post_zoo_smoke.json`](../../configs/experiments/gpu_queue_post_zoo_smoke.json)

---

## Acquisition (+50 images, other year)

Do not merge into canonical `data/splits/test.txt` without new SHA + Methods sentence. See [`data/cohorts/README.md`](../../data/cohorts/README.md) and [`cohort_zeroshot_eval_smoke.json`](../../configs/experiments/cohort_zeroshot_eval_smoke.json).
