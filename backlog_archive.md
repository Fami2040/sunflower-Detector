# Backlog archive

Historical detail moved from `backlog.md` (2026-06-01). Active tasks: [`backlog.md`](backlog.md).

---

## Aug (closed)

**No further aug GPU** unless schedule audit explicitly reopens. Program: literature-guided aug tactics vs **anchor `best2.pt`**, **same** frozen test split + locked conf as zoo/SOTA work ([`ORIGIN_MAIN_AND_DATASET.md`](docs/ORIGIN_MAIN_AND_DATASET.md)).

| Outcome | MAE / note |
|---------|------------|
| **Anchor `best2`** (legacy weights, fair HSP re-measure) | **61.3** on `data/splits/test.txt` |
| Best 15-ep smoke (literature aug) | **68.9** — did not beat anchor |
| 100-ep confirm (full train, minimal aug) | **64.1** — did not beat anchor |
| Best 15 ep smoke | S1 **68.9** |
| mosaic=0 (S2) | **147.4** — rejected |
| close10 @ 15 ep | **96.7** — rejected |
| close25 @ 15 ep | **68.9** ≡ S1 (audit-only) |

Rankings + CIs: [`reports/aug_smoke/leaderboard.md`](reports/aug_smoke/leaderboard.md). Phase A job summaries: `sweep_close10_15ep_summary.json`, `sweep_close25_15ep_summary.json`. Cancelled/deferred: **P1-AUG-CLOSE-100EP**, **P1-AUG-SCHEDULE-SMOKE** (see archive).

---

## Data acquisition

**ID:** **DATA-ACQ-GEN** · supports **MS-GEN**, domain eval, finetune loop.

**Problem:** Val≫test mAP and tray spread reflect **session / tray / lighting** shift; aug is saturated — need labeled coverage of weak cells.

**Axes** (metadata via [`domain_tags.example.csv`](data/domain_tags.example.csv)): `tray_key`, `variety`, `maturity`, `lighting`, `site`, `capture_date`.

**Priorities:** (1) Audit weak trays — [`domain_count_mae.json`](reports/domains/domain_count_mae.json), [`catalog.json`](reports/domains/catalog.json). (2) Balance: ≥20 img/tray in train or explicit holdout; ≥3 lighting conditions; no tray >25% of train without stratified val. (3) LOFO: hold out 2–3 `tray_key`s; eval via [`eval_domains.py`](scripts/eval_domains.py). (4) Fixed protocol: `imgsz=1280`, classes 0/1, `validate_splits.py` after ingest. (5) Train new data with **`robustness_minimal`** unchanged.

**Study arms:** A coverage → ↓ worst-tray MAE; B LOFO → holdout within +10% global MAE; C finetune (**P1-FINETUNE-TRAY**) on holdout tray — playbook [FINETUNE_WEAK_TRAYS](docs/FINETUNE_WEAK_TRAYS.md).

```bash
export DATASET_ROOT="$(pwd)/data/raw/extracted/dataset"
python scripts/validate_splits.py --require-test
python scripts/eval_domains.py --import-domain-tags data/domain_tags.csv \
  --catalog reports/domains/catalog.json --out reports/domains/domain_eval.json
python scripts/eval_domains.py --write-domain-splits --domains-dir data/domains
mamba run -n harchoc python scripts/eval_domains.py --merge-tray-count-mae \
  --device cpu --locked-conf-from reports/hsp/threshold_val.json \
  --out reports/domains/domain_eval.json
python scripts/experiment.py domain-tray-audit --out reports/domains/weak_tray_plan.json
mamba run -n harchoc python scripts/finetune.py --stage 1 --tray-key <TRAY> --train-mode tray_adapt \
  --dataset-root "$DATASET_ROOT" --out reports/transfer/finetune_<TRAY>_s1.json
mamba run -n harchoc python scripts/split_drift.py --extended --out reports/hsp/split_drift_extended.json
```

---

## GPU runbook

```bash
export DATASET_ROOT="$(pwd)/data/raw/extracted/dataset"
export HARCHOC_EXPORT_DEVICE=cpu

python scripts/pre_train_gate.py --quick
HARCHOC_STRICT_ML=1 mamba run -n harchoc python scripts/pre_train_gate.py --full
mamba run -n harchoc python scripts/check_weights_cache.py --sync-repos-manifest
mamba run -n harchoc python scripts/check_weights_cache.py --download --strict --out reports/hsp/weights_cache.json
mamba run -n harchoc python scripts/validate_splits.py --require-test
mamba run -n harchoc python scripts/check_gpu.py --json-out reports/hsp/gpu_check.json
```

**Queue (one GPU, sequential):** [`./scripts/run_gpu_queue.sh`](scripts/run_gpu_queue.sh) — all subprocesses via `mamba run -n harchoc python …`. Stop strays before a fresh run: `./scripts/kill_stray_gpu_jobs.sh`. Do not spawn parallel `train.py` or ad-hoc orchestrators.

| Phase | Manifest / job | Status |
|-------|----------------|--------|
| Aug confirm | [`gpu_queue_aug_confirm.json`](configs/experiments/gpu_queue_aug_confirm.json) | Done — **64.1** MAE |
| Aug smokes S0–S14 + Tier 2 | [`gpu_queue_aug_pending.json`](configs/experiments/gpu_queue_aug_pending.json) | Done 2026-05-30 |
| RT-DETR refresh | [`gpu_queue_full.json`](configs/experiments/gpu_queue_full.json) | `skip_if` on 8 GiB — **P1-RTDETR-COUNT-REFRESH** |
| **Zoo (P0-5)** | `zoo_matrix_p0_5` in [`gpu_queue_full.json`](configs/experiments/gpu_queue_full.json) (`matrix_group: zoo_yolo_only`) | **Next** (~**480** min, 4 YOLO rows) |
| **Post-zoo** | [`gpu_queue_post_zoo.json`](configs/experiments/gpu_queue_post_zoo.json) | **Queued** — repro/preflight → domain audit → finetune weak trays (after P0-5) |
| CV 5-fold / `zoo_scale` / full `zoo_core` DETR | full manifest tail or >8 GiB | Defer on 8 GiB |

**Pipeline (confirm → full):**

```bash
./scripts/run_gpu_queue.sh pipeline-dry-run   # preview
./scripts/run_gpu_queue.sh pipeline-run       # nohup: reports/gpu_queue/nohup.log
./scripts/run_gpu_queue.sh resume             # or: --job zoo_matrix_p0_5
```

State: `reports/gpu_queue/run_state.json`, logs under `reports/gpu_queue/logs/{job_id}/`. Details: [EXPERIMENTS § GPU queue](docs/EXPERIMENTS.md#gpu-sequential-queue).

**Zoo groups** ([`zoo_comparison_design.md`](docs/zoo_comparison_design.md)): on **8 GiB**, P0-5 = **`zoo_yolo_only`** (4 Ultralytics M-scale rows; queue job `zoo_matrix_p0_5`). Full **`zoo_core`** (10 rows incl. external DETR + RT-DETR) and **`zoo_scale`** / **`sota_2026`** → **>8 GiB** or after a YOLO family wins. Bench: [`configs/bench/`](configs/bench/), rows [`matrix_rows.v1.json`](configs/zoo/matrix_rows.v1.json).

```bash
mamba run -n harchoc python scripts/benchmark_matrix.py --group zoo_yolo_only --dry-run \
  --out reports/hsp/matrix_plan.json
mamba run -n harchoc python scripts/benchmark_matrix.py --group zoo_yolo_only --no-dry-run \
  --runs-dir runs/hsp_zoo --train-out reports/hsp/matrix_train.json
```

**Notify (optional):** `configs/local/queue_notify.json` from [`queue_notify.example.json`](configs/local/queue_notify.example.json); test: `python scripts/queue_notify_test.py --dry-run`.

**Post-zoo → Word (one chain, `harchoc/repro_chain.py`):**

```bash
mamba run -n harchoc python scripts/experiment.py repro --stage full          # HSP + preflight
mamba run -n harchoc python scripts/experiment.py manuscript-preflight        # after exports exist
PYTHONPATH=. HARCHOC_ALLOW_BASE_PYTHON=1 python scripts/experiment.py manuscript-preflight --dry-run   # plan only
PYTHONPATH=. HARCHOC_ALLOW_BASE_PYTHON=1 python scripts/experiment.py manuscript-preflight           # live on reports/hsp/*.json (CPU, no GPU train)
export DATASET_ROOT="$(pwd)/data/raw/extracted/dataset"
PYTHONPATH=. python scripts/experiment.py manuscript-docx-repro   # docx Fig 1,2,4–6 + Tables 1–3 → reports/manuscript/docx/
```

Stages: `post-zoo` = reviewer2 only; `preflight` = reviewer2 → figures → tables → **docx** → aug → narrative. Manifest: [`preflight_manifest.json`](reports/manuscript/preflight_manifest.json). Layout: [`reports/manuscript/README.md`](reports/manuscript/README.md) · index [`reports/README.md`](reports/README.md).

---

## Reporting quick ref

**Index:** [`reports/README.md`](reports/README.md) (scientific layers — avoid citing stale root paths).

| Output | Path |
|--------|------|
| Headline | [`p0_summary.md`](reports/hsp/p0_summary.md) |
| HSP core (cite for science) | [`reports/hsp/README.md`](reports/hsp/README.md) — `dual_metric.json`, `threshold_*.json`, `gt_*.json`, `preds_*.json`, `error_*.json` |
| Thresholds / errors / TIDE | `reports/hsp/threshold_*.json`, `error_*.json`, `tide_bucket_summary*.json` |
| Confusion 3×3 | `eval.py --confusion-matrix-only` → `reports/hsp/*_confusion.json` |
| Zoo matrix train | [`matrix_train.json`](reports/hsp/matrix_train.json) (not `reports/benchmarks/matrix_train.json`) |
| Domain | [`domain_eval.json`](reports/domains/domain_eval.json) |
| Aug | [`reports/aug_smoke/leaderboard.md`](reports/aug_smoke/leaderboard.md) |
| Figures | [`reports/figures/manifest.json`](reports/figures/manifest.json) |
| Reviewer-2 | [`reports/manuscript/`](reports/manuscript/), [`reports/_llm/`](reports/_llm/), `reviewer2_*.json` |
| Gap map | [`reviewer_comments_backlog_gap.md`](docs/manuscript/reviewer_comments_backlog_gap.md) |
| Preflight / tables / docx / narrative | [`manuscript/README.md`](reports/manuscript/README.md), [`preflight_manifest.json`](reports/manuscript/preflight_manifest.json) |

**CI:** `PYTHONPATH=. HARCHOC_ALLOW_BASE_PYTHON=1 python -m unittest discover -s tests`

---

## Archive

*Completed work — do not duplicate in [Now](#now).*

### P0 · aug · zoo prep

| ID | Evidence |
|----|----------|
| P0-0–P0-4 | split drift, max_det/S14, baselines, HSP chain ~61.3, RT-DETR smoke |
| P1-AUG (S0–S14), Phase A close10/25, 100ep confirm | [`leaderboard.md`](reports/aug_smoke/leaderboard.md), [`aug_smoke_index.json`](configs/experiments/aug_smoke_index.json) |
| P1-AUG-DUP-MAE, CLOSE-SCALE, AMP/SG HSP eval (diagnostic 204.2) | equivalence_classes; diagnostic summaries in `reports/aug_smoke/`, `reports/hsp/` |
| P1-ZOO-READY, P2-ZOO-EXPAND, P2-DEIM-REPOS | [`weights_cache.json`](reports/hsp/weights_cache.json), [`detector_sources.v1.json`](configs/external/detector_sources.v1.json) |
| P2-SAHI-MATRIX, P2-FIG-CONCEPT, ARCH-MOSAIC0-AB, P2-AUG-RANK-REPORT | scaffolds + figures in `reports/figures/` |

### Manuscript & science (Done)

MS-ORIG, MS-LIT, MS-SPLIT-MAPNARR, MS-VAL-MAPDOWN, MS-ASYM-NARR, MS-FP-LOC-NARR, MS-DOMAIN-ADAPT, MS-DEPLOY-2STG, MS-FUZZY-BOUND, MS-MANUAL-*, MS-FIG-NORM, MS-REPRO, MS-VAL-MAP-CAVEAT — see [gap map](docs/manuscript/reviewer_comments_backlog_gap.md).

### Eval · threshold · figures

P1-FP-BUDGET, P1-TIDE/TIDECV, P2-COUNT-SEL, P1-DOMAIN-EVAL, P1-FINETUNE-TRAY (tray_adapt splits + audit), P2-DET-CONFUSION, P2-FIG*, R-SCI-1/2, P1-SPLIT-DRIFT-RICH, P2-ASYM-SEED, P1-RTDETR-MAXDET — artifacts under `reports/hsp/`, `reports/figures/`, `reports/domains/`.

### GPU probes · queue jobs (historical)

VRAM/autobatch/AMP smokes; aug Tier-2 jobs (amp/sg/close sweeps) Done; RT-DETR skipped on 8 GiB. Job table: `reports/gpu_queue/jobs/*.json`.

### DRY / agents

All items in [refactor.md](refactor.md) §1–§6 Done. Agent batches (prep gate, zoo harness, domain tags, fp_budget, figures): see refactor + git history.

<details>
<summary>Infrastructure scaffolds (historical)</summary>

CI, splits, train/eval metadata, analysis scripts (`threshold_sweep`, `error_analysis`, `make_figures`, …). Superseded detail: git history, [refactor.md](refactor.md).

</details>

---

## Permanent

Extend `experiment.py`, `benchmark_matrix.py`, `eval.py`, `train.py`, `harchoc/*` before new top-level scripts — [.cursor/rules/extend-before-add-script.mdc](.cursor/rules/extend-before-add-script.mdc).
