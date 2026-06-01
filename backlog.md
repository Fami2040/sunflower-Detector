# Backlog (local, not committed)

Task tracker for HARCHOC sunflower seed counting. **Updated:** 2026-06-01 · **Branch:** `pr/backlog-ci-dataset`

| Anchor | Value |
|--------|--------|
| Headline metric | Test **count MAE** at val-locked conf (not val mAP alone) |
| **best2** | **61.3** MAE @ conf ~0.15 — [`dual_metric.json`](reports/hsp/dual_metric.json) |
| Aug winner (100 ep) | **64.1** on [`robustness_minimal.yaml`](configs/aug/robustness_minimal.yaml) — [`aug_confirm_winner_100ep_summary.json`](reports/aug_smoke/aug_confirm_winner_100ep_summary.json) |
| Error mix @ locked conf | FN ~53% ΔAP share; among FPs loc ~58%, bg ~35%, cls ~3% — [`error_test_report.json`](reports/hsp/error_test_report.json) |
| **Next GPU job** | `zoo_matrix_p0_5` in [`gpu_queue_full.json`](configs/experiments/gpu_queue_full.json) |

**Doc map:** [RESEARCH_AND_OPS](docs/RESEARCH_AND_OPS.md) · [EXPERIMENTS](docs/EXPERIMENTS.md) · [FINETUNE_WEAK_TRAYS](docs/FINETUNE_WEAK_TRAYS.md) · [zoo_comparison_design](docs/zoo_comparison_design.md) · [HSP_BASELINE_MODELS](docs/HSP_BASELINE_MODELS.md) · [reports/hsp/README](reports/hsp/README.md) · [p0_summary](reports/hsp/p0_summary.md) · [reviewer gap](docs/manuscript/reviewer_comments_backlog_gap.md) · [refactor.md](refactor.md)

**Convention:** Only **Next** / **Partial** / **Blocked** appear in [Now](#now). Everything finished lives in [Archive](#archive).

---

## Dataset

- Classes: **0 = developed**, **1 = aborted** ([`data/README.md`](data/README.md), `harchoc/sunflower_dataset.py`).
- Layout: `DATASET_ROOT` → `images/{train,val}/`, `labels/{train,val}/`; splits [`data/splits/*.txt`](data/splits/) (875 / 109 / 109).
- Early stop on `val.txt`; manuscript metrics on **`test.txt`** via `eval.py`.
- Weights: `models/best2.pt` (`harchoc.hsp_weights.HSP_DETECTION_WEIGHTS`); gate `models/classifier.pt`.
- GPU: `mamba run -n harchoc python …` — [RESEARCH_AND_OPS §4](docs/RESEARCH_AND_OPS.md#4-operations).

---

## Model stack (reference)

| Step | Focus | Status |
|------|--------|--------|
| 1 | Train/export parity (`max_det=3000`) | Done |
| 2 | Full YOLO @ 1280, 100 ep | CLI [`train_yolov8m_baseline.json`](configs/experiments/train_yolov8m_baseline.json) |
| 3 | Val lock → test; FP budget | Done |
| 4 | Aug ablations | **Closed** — keep `robustness_minimal` @ 100 ep; details [Aug (closed)](#aug-closed) |
| 5 | Model zoo + count columns | **P0-5** Next — partial: no count winner vs best2 yet |
| 6 | Tray / domain | **MS-GEN** Partial · **DATA-ACQ-GEN** Next |
| 7 | RT-DETR query cap | **P1-RTDETR-COUNT-REFRESH** |
| 8 | Count levers (finetune, hard-neg, dupe post, head-ROI eval) | P1/P2 — see [Now](#now) |

**Defer:** new backbone/head ROI training until eval-only mask shows ↓bg FP without ↑FN — [architecture_recommendations](docs/manuscript/architecture_recommendations.md).

---

## Now

### Science · training · eval

| ID | Pri | Status | Blocker | Next |
|----|-----|--------|---------|------|
| **P0-5** | P0 | **Next** | — | `zoo_core` 10×100 ep → [`matrix_train.json`](reports/hsp/matrix_train.json); weights cache Done; queue job `zoo_matrix_p0_5`; per-row `test_count_mae` + fast 3×3 via `eval.py --confusion-matrix-only` |
| **P1-ZOO-PARITY** | P1 | Next | P0-5 | Matrix columns: count MAE + accuracy–efficiency; gate on MAE not mAP alone |
| **P1-ZOO-PROV** | P1 | Partial | P0-5 | Provenance in matrix `weights` + [`detector_sources.v1.json`](configs/external/detector_sources.v1.json) |
| **P2-RTDETR-V2** | P2 | Partial | P0-5 | External DETR rows trained/eval via matrix; prep: `check_weights_cache --download` |
| **P2-DEIM-EVAL** | P2 | Ready | P0-5 train | HSP eval wired; run after matrix train |
| **DATA-ACQ-GEN** | P1 | Next | Weak-tray audit | Acquisition + LOFO — [§ Data acquisition](#data-acquisition) |
| **P1-FINETUNE-TRAY** | P1 | **Partial** | Domain splits on disk | `tray_adapt` train lists + staged `finetune.py` — [FINETUNE_WEAK_TRAYS](docs/FINETUNE_WEAK_TRAYS.md); GPU tray runs open |
| **P1-RTDETR-COUNT-REFRESH** | P1 | Next | — | Fresh RT-DETR test count MAE or document stale summaries (`gpu_queue_full` `skip_if`) |
| **P1-RTDETR-Q** | P1 | Blocked | GPU / CUDA @ `model.to` | Use COUNT-REFRESH path; nq≥1024 smoke |
| **P1-CV-TRAIN** | P1 | Next | GPU | 5-fold training (routing Done — archive) |
| **P2-HEAD-ROI-EVAL** | P2 | Next | — | Eval-only ROI mask on preds before any head detector |
| **P2-RTDETR-IDX** | P2 | Next | — | `eval_idx` / latency sweeps |

**Order:** P0-5 → DATA-ACQ-GEN / finetune / inference levers — [RESEARCH_AND_OPS §3](docs/RESEARCH_AND_OPS.md#3-prioritized-roadmap).

### Manuscript (repo draft vs LaTeX)

| ID | Pri | Status | Blocker | Next |
|----|-----|--------|---------|------|
| **MS-SOTA** | P1 | Blocked | P0-5 | SOTA table after `matrix_train.json` — [gap §3](docs/manuscript/reviewer_comments_backlog_gap.md) |
| **MS-GEN** | P1 | Partial | DATA-ACQ-GEN | §4 draft Done; per-tray numbers + acquisition — [gap §4](docs/manuscript/reviewer_comments_backlog_gap.md#4-manuscript-draft--generalization--cross-tray-counting-discussion) |
| **MS-ABS** | P2 | Next | — | Abstract IMRaD — [gap §1](docs/manuscript/reviewer_comments_backlog_gap.md) |
| **MS-EXPLAIN** | P2 | Partial | — | Ren 2025 cite; `fig_concept` Done — archive |
| **R2-NARRATIVE** | P2 | Done | — | `experiment.py backlog-narrative` → `reports/manuscript/narrative_from_backlog.md` + `backlog_narrative.json` |
| **MS-PREFLIGHT** | P1 | Done | — | `experiment.py manuscript-preflight` / `repro --stage preflight` → [`preflight_manifest.json`](reports/manuscript/preflight_manifest.json) |
| **LIT-VALIDATE** | P2 | Next | — | [`literature_validated.json`](docs/manuscript/literature_validated.json) |

*Other MS-* narrative items: repo draft **Done** — [Manuscript archive](#manuscript--science-done).

### Blockers & dedup (read before re-queueing GPU)

| Issue | Unblock / action |
|-------|------------------|
| **MS-SOTA** | Finish **P0-5** |
| **P1-RTDETR-Q** | Avoid parallel GPU; use full-queue `skip_if` |
| Do **not** re-train aug recipe duplicates | [`equivalence_classes`](configs/experiments/aug_smoke_index.json) in [`aug_smoke_index.json`](configs/experiments/aug_smoke_index.json) — S0≡S1≡S13≡CLOSE25; S3≡S6≡S7; AMP≡SG diagnostic only |

---

## Aug (closed)

**No further aug GPU** unless schedule audit explicitly reopens. Production recipe: [`robustness_minimal.yaml`](configs/aug/robustness_minimal.yaml) (`mosaic=0.1`, `close_mosaic=15`, `mixup=0`).

| Outcome | MAE / note |
|---------|------------|
| 100 ep confirm | **64.1** vs best2 **61.3** |
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
mamba run -n harchoc python scripts/check_gpu.py --json-out reports/gpu_check.json
```

**Queue (one GPU, sequential):** [`./scripts/run_gpu_queue.sh`](scripts/run_gpu_queue.sh) — all subprocesses via `mamba run -n harchoc python …`. Stop strays before a fresh run: `./scripts/kill_stray_gpu_jobs.sh`. Do not spawn parallel `train.py` or ad-hoc orchestrators.

| Phase | Manifest / job | Status |
|-------|----------------|--------|
| Aug confirm | [`gpu_queue_aug_confirm.json`](configs/experiments/gpu_queue_aug_confirm.json) | Done — **64.1** MAE |
| Aug smokes S0–S14 + Tier 2 | [`gpu_queue_aug_pending.json`](configs/experiments/gpu_queue_aug_pending.json) | Done 2026-05-30 |
| RT-DETR refresh | [`gpu_queue_full.json`](configs/experiments/gpu_queue_full.json) | `skip_if` on 8 GiB — **P1-RTDETR-COUNT-REFRESH** |
| **Zoo** | `zoo_matrix_p0_5` in full manifest | **Next** (~2000 min) |
| CV 5-fold / `zoo_scale` | full manifest tail | Defer |

**Pipeline (confirm → full):**

```bash
./scripts/run_gpu_queue.sh pipeline-dry-run   # preview
./scripts/run_gpu_queue.sh pipeline-run       # nohup: reports/gpu_queue/nohup.log
./scripts/run_gpu_queue.sh resume             # or: --job zoo_matrix_p0_5
```

State: `reports/gpu_queue/run_state.json`, logs under `reports/gpu_queue/logs/{job_id}/`. Details: [EXPERIMENTS § GPU queue](docs/EXPERIMENTS.md#gpu-sequential-queue).

**Zoo groups** ([`zoo_comparison_design.md`](docs/zoo_comparison_design.md)): **`zoo_core`** (10 rows, default) → optional **`zoo_scale`** / **`sota_2026`** after a family wins. Bench: [`configs/bench/`](configs/bench/), rows [`matrix_rows.v1.json`](configs/zoo/matrix_rows.v1.json).

```bash
mamba run -n harchoc python scripts/benchmark_matrix.py --group zoo_core --dry-run \
  --out reports/hsp/matrix_plan.json
mamba run -n harchoc python scripts/benchmark_matrix.py --group zoo_core --no-dry-run \
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

Stages: `post-zoo` = reviewer2 only; `preflight` = figures + tables + aug + narrative + reviewer2. Manifest: [`preflight_manifest.json`](reports/manuscript/preflight_manifest.json).

---

## Reporting quick ref

| Output | Path |
|--------|------|
| Headline | [`p0_summary.md`](reports/hsp/p0_summary.md) |
| Thresholds / errors / TIDE | `reports/hsp/threshold_*.json`, `error_*.json`, `tide_bucket_summary*.json` |
| Confusion 3×3 | `eval.py --confusion-matrix-only` |
| Domain | [`domain_eval.json`](reports/domains/domain_eval.json) |
| Aug | [`reports/aug_smoke/leaderboard.md`](reports/aug_smoke/leaderboard.md) |
| Gap map | [`reviewer_comments_backlog_gap.md`](docs/manuscript/reviewer_comments_backlog_gap.md) |
| Preflight / tables / narrative | [`preflight_manifest.json`](reports/manuscript/preflight_manifest.json), `reports/manuscript/tables/`, [`narrative_from_backlog.md`](reports/manuscript/narrative_from_backlog.md) |

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
