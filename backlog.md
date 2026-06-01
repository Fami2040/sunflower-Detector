# Backlog (local, not committed)

Single task tracker for HARCHOC / sunflower seed-counting. **Validated:** 2026-05-31 · **Branch:** `pr/backlog-ci-dataset`

| | |
|---|---|
| **Priority** | P0 (credibility) · P1 · P2 |
| **Status (active)** | **Next** / **Partial** / **Blocked** only below |
| **Archive** | [Done & tested](#done--tested-archive) at bottom |

**Doc map:** [RESEARCH_AND_OPS](docs/RESEARCH_AND_OPS.md) · [EXPERIMENTS](docs/EXPERIMENTS.md) · [zoo_comparison_design](docs/zoo_comparison_design.md) · [HSP_BASELINE_MODELS](docs/HSP_BASELINE_MODELS.md) · [reports/hsp/README](reports/hsp/README.md) · [p0_summary](reports/hsp/p0_summary.md) · [refactor.md](refactor.md)

---

## Dataset (sunflower-cvat-2500)

Class **0 = developed**, **1 = aborted** only (`harchoc/sunflower_dataset.py`, [`data/README.md`](data/README.md)).

- **Layout:** `DATASET_ROOT` → `images/{train,val}/`, `labels/{train,val}/`; splits `data/splits/{train,val,test}.txt` (875 / 109 / 109).
- **Metrics:** early stop on `val.txt`; manuscript on **`test.txt`** via `eval.py`.
- **Weights:** `models/best2.pt` (HSP/matrix); `harchoc.hsp_weights.HSP_DETECTION_WEIGHTS`; override `DETECTION_MODEL`. Gate: `models/classifier.pt`.

**GPU / env:** `mamba run -n harchoc python …` — [RESEARCH_AND_OPS §4](docs/RESEARCH_AND_OPS.md#4-operations).

---

## Model improvement stack (test count MAE)

**Success metric:** test **count MAE** at val-locked conf — not val mAP alone.

| Step | What | Active IDs |
|------|------|------------|
| 1 | Train/export parity | **Done** (`max_det=3000`; S14 eval-only **265.8** @ max_det 300 confirms cap — [`s14_summary.json`](reports/aug_smoke/s14_summary.json)) |
| 2 | Full YOLO @ 1280, 100 ep | CLI [`train_yolov8m_baseline.json`](configs/experiments/train_yolov8m_baseline.json) |
| 3 | Val lock → test; FP budget | **Done** |
| 4 | Aug ablations | **Done** — S0–S14 + 100-ep confirm (**64.1** vs **best2 61.3**); Phase A close sweeps **closed** (close10 **96.7** rejected; close25 ≡ S1 **68.9** audit-only) — keep `robustness_minimal` @ 100 ep — [§ Aug close sweeps (closed)](#aug-close-sweeps-closed) |
| 5 | Model zoo + count columns | **P0-5** **Ready** (`zoo_core_8gb` 8×100 ep on 8 GiB path); full matrix **26 rows** — [zoo design](docs/zoo_comparison_design.md) |
| 6 | Tray / domain | **MS-GEN** Partial; **DATA-ACQ-GEN** Next — [§ Data acquisition](#data-acquisition-for-generalization-study-design) |
| 7 | RT-DETR query cap | **P1-RTDETR-COUNT-REFRESH** (queue skipped prior summaries) |

**Defer (low MAE ROI):** [architecture_recommendations § Defer](docs/manuscript/architecture_recommendations.md).

---

## Active queue

### Manuscript (repo draft vs LaTeX paste)

| ID | Pri | Status | Task | Blocker | Next |
|----|-----|--------|------|---------|------|
| MS-ABS | P2 | Next | Abstract IMRaD | — | [gap §1](docs/manuscript/reviewer_comments_backlog_gap.md) |
| MS-SOTA | P1 | Blocked | SOTA table (YOLO + RT-DETR + NAS) | **P0-5** | [gap §3](docs/manuscript/reviewer_comments_backlog_gap.md) |
| MS-GEN | P1 | Partial | Generalization + per-tray metrics | **DATA-ACQ-GEN** tray coverage | §4 repo draft Done — [gap §4](docs/manuscript/reviewer_comments_backlog_gap.md#4-manuscript-draft--generalization--cross-tray-counting-discussion); acquisition plan [§ Data acquisition](#data-acquisition-for-generalization-study-design) |
| MS-EXPLAIN | P2 | Partial | Ren 2025 cite in manuscript prose | — | `fig_concept` Done — [archive § Figures](#figures--explainability-done) |
| LIT-VALIDATE | P2 | Next | Maintain [`literature_validated.json`](docs/manuscript/literature_validated.json) | — | ongoing when adding cites |

*All other MS-* narrative rows **Done** (repo draft) — see [Manuscript archive](#manuscript--science-narrative-done).*

### Science · eval · training

| ID | Pri | Status | Task | Blocker | Next |
|----|-----|--------|------|---------|------|
| P0-5 | P0 | **Next** | **`zoo_core`** 10×100 ep → `matrix_train.json` | — | Weights **20/20** Ultralytics + **4/4** external cached ([`weights_cache.json`](reports/hsp/weights_cache.json)); queue job `zoo_matrix_p0_5` (~2000 min). Tier 1–2 aug jobs skip on dry-run — run `./scripts/run_gpu_queue.sh resume` or `--job zoo_matrix_p0_5`. Optional **`zoo_scale`** / **`sota_2026`** deferred — [zoo_comparison_design](docs/zoo_comparison_design.md) |
| P2-DEIM-EVAL | P2 | **Ready** | HSP test eval for external DETR checkpoints | **P0-5** train | Wired in `benchmark_matrix.py` + `harchoc/external_detector_eval.py` (export + `error_analysis` @ locked conf); needs GPU matrix train for artifacts |
| P2-DEIM-REPOS | P2 | Done | Upstream repos via `harchoc.bench_assets` / `check_weights_cache.py` | — | Canonical [`detector_sources.v1.json`](configs/external/detector_sources.v1.json); clones under `external/` on `--download`; generated [`external_repos.v1.json`](configs/external/external_repos.v1.json) via `--sync-repos-manifest`; pins in [`weights_manifest.json`](data/weights/weights_manifest.json) |
| ARCH-MOSAIC0-AB | P1 | **Done (15 ep)** | S2 vs S0 mosaic-off | — | 15-ep: **S1 68.9** vs **S2 147.4** — mosaic-off **rejected**; **no 100-ep S2** (low ROI). Confirm @ 100 ep **64.1** on `robustness_minimal` — [archive § Aug smokes](#aug-smokes--15-ep-sweeps-gpu-queue-2026-05-2930) |
| DATA-ACQ-GEN | P1 | Next | Field/lab capture plan for tray & lighting diversity | Weak-tray audit | Stratified acquisition + LOFO eval protocol — [§ Data acquisition](#data-acquisition-for-generalization-study-design) |
| P1-RTDETR-COUNT-REFRESH | P1 | Next | RT-DETR test count MAE (640/1280/nq1024) | — | RT-DETR jobs stay in [`gpu_queue_full.json`](configs/experiments/gpu_queue_full.json) for **`skip_if` conditional re-run only** (stale summaries / job transcripts); re-run or document dates |
| P1-RTDETR-Q | P1 | **Blocked** | `num_queries` ≥1024 15-ep smoke (live GPU) | contention | prior notes: CUDA driver @ `model.to` — see **P1-RTDETR-COUNT-REFRESH** |
| P2-AUG-RANK-REPORT | P2 | Done | Aug smoke + sweep leaderboard artifact | P1-AUG archive | [`reports/aug_smoke/leaderboard.md`](reports/aug_smoke/leaderboard.md) from index + CIs |
| P1-VRAM-RTDETR | P1 | Done | rtdetr-l batch probe @ 1280 | — | queue skipped — [archive § GPU probes](#gpu-probes-tested) |
| P1-ZOO-PARITY | P1 | Next | Matrix accuracy–efficiency columns | **P0-5** | |
| P1-ZOO-PROV | P1 | Partial | Provenance per matrix row | **P0-5** | `detector_sources.v1.json` + matrix `weights` block for external; Ultralytics manifest in [`weights_manifest.json`](data/weights/weights_manifest.json) |
| P1-CV-TRAIN | P1 | Next | Per-fold GPU training | GPU | routing Done — [archive § Scaffolding](#scaffolding-code-done) |
| P2-FIG-CONCEPT | P2 | Done | Pipeline diagram `fig_concept` (redesigned) | — | [archive § Figures](#figures--explainability-done) |
| P2-RTDETR-IDX | P2 | Next | `eval_idx` / latency sweeps | — | |
| P2-RTDETR-V2 | P2 | Partial | RT-DETRv2 / D-FINE / DEIM in matrix | **P0-5** train | Bench + [`detector_sources.v1.json`](configs/external/detector_sources.v1.json) + **train** + **HSP eval** wired; prep via `bench_assets` (`check_weights_cache --download`) |
| P2-SAHI-MATRIX | P2 | Done | SAHI matrix eval protocol scaffold | — | dry-run plan JSON; GPU eval TBD |

**Credibility order:** P0-4 Done → P1 aug **Done** → **P0-5** zoo → threshold / domain (test MAE). [RESEARCH_AND_OPS §3](docs/RESEARCH_AND_OPS.md#3-prioritized-roadmap).

### Blockers (zoo plan execution)

| Blocker | Affects | Unblock |
|---------|---------|---------|
| ~~**GPU queue idle / not started**~~ | ~~Tier 1–2, **P0-5**~~ | **Unblocked** — confirm + Tier 2 complete; **`zoo_matrix_p0_5`** is next runnable job in [`gpu_queue_full.json`](configs/experiments/gpu_queue_full.json) |
| ~~**P2-DEIM-EVAL**~~ | ~~HSP test MAE for 4 external DETR rows post-train~~ | **Ready** — wired in matrix; run after **P0-5** train |
| **P1-RTDETR-Q** | Live nq1024 15-ep smoke | GPU contention / CUDA @ `model.to` — use **P1-RTDETR-COUNT-REFRESH** `skip_if` path in full queue |
| **MS-SOTA** | Manuscript SOTA table | **P0-5** `matrix_train.json` |
| **15-ep aug gap vs best2** (+7.6 MAE) | Replacing `models/best2.pt` on 15-ep alone | **Resolved for recipe choice** — 100-ep confirm **64.1** vs **61.3**; zoo / finetune dominate further MAE gains |
| ~~**P1-AUG-CLOSE-SCALE** (config)~~ | ~~Misleading `close_mosaic=15` warnings on 15-ep YAMLs~~ | **Done** — runtime scale + explicit smoke YAMLs; close10/close25 sweeps wired |

Dedup audit: do not re-train **S0/S13/CLOSE25** (preds `ad6f1621…`, canonical **S1**), **S6/S7** (preds `41e79d28…`, canonical **S3**), or re-queue **`aug_sweep_15_close25`** (preds dedup vs complete smokes). **AMP≡SG** @ 15 ep diagnostic (preds `e4853607…`, MAE **204.2** — not ranked). See [dedup_root_cause.md](reports/aug_smoke/dedup_root_cause.md), [`equivalence_classes`](configs/experiments/aug_smoke_index.json).

### GPU execution tiers (post aug_pending)

One GPU, sequential — [manifest map](docs/EXPERIMENTS.md#gpu-queue-manifest-map). Rankings: [`leaderboard.md`](reports/aug_smoke/leaderboard.md).

| Tier | Jobs / backlog | Manifest |
|------|----------------|----------|
| **Done** | S0–S14 smokes; **P1-AUG-100EP-WINNER**; Tier 2 amp/sg HSP eval + close10/close25 Phase A | [`gpu_queue_aug_pending.json`](configs/experiments/gpu_queue_aug_pending.json) (2026-05-30); confirm + Tier 2 in [`gpu_queue_full.json`](configs/experiments/gpu_queue_full.json) (summaries verified 2026-05-31) |
| **Tier 1** | **P1-RTDETR-COUNT-REFRESH** (rtdetr ×3 + vram probe — `skip` on 8 GiB) | [`gpu_queue_full.json`](configs/experiments/gpu_queue_full.json) — conditional `skip_if` only |
| **Next** | **`zoo_core`** 10×100 ep → `matrix_train.json` | [`gpu_queue_full.json`](configs/experiments/gpu_queue_full.json) job `zoo_matrix_p0_5` (~2000 min) |
| **Defer** | **P1-CV-TRAIN** (5-fold), **`zoo_scale`** / **`sota_2026`** | tail of full queue / separate manifest |

**P0-5 order:** prep gate + weights cache **Done** → aug confirm + Tier 2 **Done** → **`zoo_matrix_p0_5`** (`--group zoo_core`, external rows get chained HSP eval) → optional **`zoo_scale`**. RT-DETR refresh remains conditional skip on 8 GiB. Design: [zoo_comparison_design](docs/zoo_comparison_design.md).

### Follow-ups from aug queue analytics (2026-05-30)

| ID | Why |
|----|-----|
| ~~**P1-AUG-CLOSE Phase A**~~ | **Done** — close10 **96.7** (rejected); close25 **68.9** ≡ S1 (audit-only); decision: keep `close_mosaic=15` @ 100 ep — [§ Aug close sweeps (closed)](#aug-close-sweeps-closed). |
| ~~**P1-AMP-HSP-EVAL** / **P1-SG-HSP-EVAL**~~ | **Done (diagnostic)** — both **204.2** MAE, identical preds `e4853607…`; no production impact. |
| **P0-5** | **Next GPU** — `./scripts/run_gpu_queue.sh resume` or `--job zoo_matrix_p0_5`. |
| **DATA-ACQ-GEN** | Tray/lighting acquisition for generalization — [§ Data acquisition](#data-acquisition-for-generalization-study-design). |
| **P1-RTDETR-COUNT-REFRESH** | RT-DETR jobs skipped on 8 GiB; manuscript/SOTA needs fresh test count MAE or explicit stale-date caveat. |
| **P2-AUG-RANK-REPORT** | Leaderboard: [`reports/aug_smoke/leaderboard.md`](reports/aug_smoke/leaderboard.md) (regenerate via `experiment.py aug-leaderboard`). |

---

## Aug close sweeps (closed)

**Status:** Phase A **Done** (2026-05-31). **No further aug GPU** before zoo unless schedule audit (Phase C) explicitly reopens.

**Production baseline (frozen):** [`robustness_minimal.yaml`](configs/aug/robustness_minimal.yaml) — `mosaic=0.1`, `close_mosaic=15` @ 100 ep, `mixup=0`. Confirm: **64.1** MAE ([`aug_confirm_winner_100ep_summary.json`](reports/aug_smoke/aug_confirm_winner_100ep_summary.json)) vs **best2 61.3**.

### Closed (do not re-run)

| Axis | Evidence | ID |
|------|----------|-----|
| mosaic=0 (LWCD-style) | S2 **147.4** @ 15 ep | ARCH-MOSAIC0-AB **Done** |
| mosaic≥0.3 | S5 **125.7** | P1-AUG-MOSAIC archive |
| photometric-only / erasing sweep | S3≡S6≡S7 **151.7** | P1-AUG-DUP-MAE |
| close15 @ 15 ep | ≡ S0/S1 **68.9** | recipe dedup-skipped |
| close25 @ 15 ep | ≡ S1 **68.9** (preds `ad6f1621…`) | P1-AUG-DUP-MAE + preds dedup on `aug_sweep_15` |
| close10 @ 15 ep | **96.7** — worse than S1 | rejected; distinct preds |
| S0/S1/S13/CLOSE25, S3/S6/S7 re-trains | preds SHA equivalent | [`equivalence_classes`](configs/experiments/aug_smoke_index.json) |
| 100-ep close_mosaic {10, 25} | No Phase A winner; close10 failed gate | **P1-AUG-CLOSE-100EP cancelled** |
| patience schedule smoke @ 100 ep | Optional; not scheduled | **P1-AUG-SCHEDULE-SMOKE deferred** — run only after confirm `results.csv` audit |

### Phase A — **Done**

| Job | Result | Summary |
|-----|--------|---------|
| `aug_sweep_15_close10` | **96.7** MAE — rejected (worse than S1 **68.9**) | [`sweep_close10_15ep_summary.json`](reports/aug_smoke/sweep_close10_15ep_summary.json) |
| `aug_sweep_15_close25` | **68.9** MAE — ≡ S1 (audit-only; preds dedup skips re-queue) | [`sweep_close25_15ep_summary.json`](reports/aug_smoke/sweep_close25_15ep_summary.json) |

**Decision:** Neither arm beats S1. Keep `close_mosaic=15` @ 100 ep. Rankings: [`leaderboard.md`](reports/aug_smoke/leaderboard.md).

Re-run only if summaries deleted:

```bash
./scripts/run_gpu_queue.sh close-phase-dry-run   # close25 skips: preds duplicate of S0
GPU_QUEUE_MANIFEST=configs/experiments/gpu_queue_full.json ./scripts/run_gpu_queue.sh resume
```

### Phase B / C — cancelled / deferred

| ID | Status | Rationale |
|----|--------|-----------|
| **P1-AUG-CLOSE-100EP** | **Cancelled** | close10 failed gate (**96.7** > **72.3**); close25 duplicate outcome — no 15-ep winner |
| **P1-AUG-SCHEDULE-SMOKE** | **Deferred** | Parse `runs/aug_confirm_winner_100ep/results.csv` first; enable [`gpu_queue_aug_close_100ep.json`](configs/experiments/gpu_queue_aug_close_100ep.json) only if tail never fired |

**Explicitly out of scope:** AMP/SG 15-ep smokes (diagnostic only — **Done**, amp≡sg preds), RT-DETR aug, Albumentations pipeline, copy-paste (no masks).

*Historical plan text (pre-Phase-A):* [`docs/research/augmentation_robustness_literature.md`](docs/research/augmentation_robustness_literature.md), [`training_tech_scan_2026_augmentation.md`](docs/research/training_tech_scan_2026_augmentation.md). Promotion gate was 15-ep MAE ≤ **72.3** (S1 + 5%).

---

## Data acquisition for generalization (study design)

**ID:** **DATA-ACQ-GEN** · supports **MS-GEN**, **P1-DOMAIN-EVAL**, **P1-FINETUNE-LOOP**. Literature: [`domain_shift_transfer_literature.md`](docs/research/domain_shift_transfer_literature.md), [`augmentation_robustness_literature.md`](docs/research/augmentation_robustness_literature.md) §3 (real session diversity > synthetic aug).

**Problem:** Val≫test **detection mAP** and tray-level count spread are driven by **session / tray / lighting** shift, not fixable by more mosaic. Aug is **saturated** at `robustness_minimal`; generalization gains need **labeled coverage** of under-represented conditions.

### Domain axes (metadata)

Join every new capture to [`data/domain_tags.example.csv`](data/domain_tags.example.csv) schema via `eval_domains.py --import-domain-tags`:

| Field | Examples | Why |
|-------|----------|-----|
| `tray_key` | `349-10-2`, `3a2-2` | Primary LOFO / holdout unit ([`domain_tags.py`](harchoc/domain_tags.py)) |
| `variety` | PV545, Other | Biological appearance shift |
| `maturity` | early, ripe | Size / contrast of developed vs aborted |
| `lighting` | benchtop, LED, field, flash | Dominant benchtop shift axis (photometric DR insufficient alone — S8 failed) |
| `site` | lab-a, field-a | Session / operator / camera drift |
| `capture_date` | ISO date | Optional temporal drift |

### Collection priorities

1. **Audit weak trays first:** [`domain_count_mae.json`](reports/domains/domain_count_mae.json) + [`catalog.json`](reports/domains/catalog.json) — target trays with high test MAE or low train representation.
2. **Minimum balance:** ≥ **20 images / tray_key** in train *or* explicit holdout list; ≥ **3 lighting conditions** across train; no single tray > **25%** of train images without stratified val slot.
3. **Holdout protocol (LOFO):** Reserve **2–3 tray_keys** entirely from train/val; evaluate only on `data/domains/test_{tray_key}.txt` ([`eval_domains.py --write-domain-splits`](scripts/eval_domains.py)). Report count MAE per tray in manuscript (**MS-GEN**).
4. **Fixed protocol:** Same `imgsz=1280`, top-down pose, YOLO label format; class **0=developed**, **1=aborted** only. Re-run `validate_splits.py` after ingest.
5. **Aug policy on new data:** Train with **`robustness_minimal.yaml`** unchanged; do **not** increase mosaic/mixup to compensate for missing trays.

### Study arms (no new model code)

| Arm | Design | Success metric |
|-----|--------|----------------|
| **A — Coverage** | Add N sessions from weak lighting × tray cells | ↓ worst-tray count MAE in `domain_count_mae.json` |
| **B — LOFO** | Train without holdout trays; eval per-tray splits | Holdout tray MAE within **+10%** of global test MAE |
| **C — Finetune** | **P1-FINETUNE-LOOP** stage-1/2 on 1 holdout tray (K-shot) | Δ MAE vs zero-shot on same tray |

### Ingest checklist

```bash
export DATASET_ROOT="$(pwd)/data/raw/extracted/dataset"
# 1. Drop images/labels under DATASET_ROOT; update data/manifest.json if root moves
# 2. Append stems to data/splits/{train,val,test}.txt OR regenerate with stratification script
python scripts/validate_splits.py --require-test
python scripts/eval_domains.py --import-domain-tags data/domain_tags.csv \
  --catalog reports/domains/catalog.json --out reports/domains/domain_eval.json
python scripts/eval_domains.py --write-domain-splits --domains-dir data/domains
mamba run -n harchoc python scripts/split_drift.py --extended --out reports/hsp/split_drift_extended.json
```

**Manuscript hook:** Cross-tray generalization paragraph (**MS-GEN** §4) cites LOFO table + acquisition axes; aug paragraph cites **64.1** confirm and closed mosaic-off ablation.

---

## Runbook (GPU)

```bash
export DATASET_ROOT="$(pwd)/data/raw/extracted/dataset"
export HARCHOC_EXPORT_DEVICE=cpu   # eval / HSP chain

python scripts/pre_train_gate.py --quick
HARCHOC_STRICT_ML=1 mamba run -n harchoc python scripts/pre_train_gate.py --full

mamba run -n harchoc pip install gdown   # once — DEIM Google Drive checkpoints
mamba run -n harchoc python scripts/check_weights_cache.py --sync-repos-manifest
mamba run -n harchoc python scripts/check_weights_cache.py --download --strict --out reports/hsp/weights_cache.json
mamba run -n harchoc python scripts/validate_splits.py --require-test
mamba run -n harchoc python scripts/check_gpu.py --json-out reports/gpu_check.json
mamba run -n harchoc python scripts/rtdetr_smoke.py --run-train   # P0-4 Done
mamba run -n harchoc python scripts/experiment.py repro --dry-run
mamba run -n harchoc python scripts/benchmark_matrix.py --group zoo_core --dry-run \
  --out reports/hsp/matrix_plan.json   # 10-row plan
mamba run -n harchoc python scripts/benchmark_matrix.py --group zoo_core --no-dry-run \
  --runs-dir runs/hsp_zoo --train-out reports/hsp/matrix_train.json   # P0-5 (Ultralytics + NAS + external DETR; HSP eval chained)
```

**Sequential GPU queue** (one job per GPU): [`gpu_queue_aug_pending.json`](configs/experiments/gpu_queue_aug_pending.json) **complete** (2026-05-30 15:08 UTC). **Canonical ops:** [`./scripts/run_gpu_queue.sh`](scripts/run_gpu_queue.sh). All train/eval subprocesses run as **`mamba run -n harchoc python …`** (env override: `HARCHOC_MAMBA_ENV`, binary override: `MAMBA_BIN`).

**Do not** spawn ad-hoc `train.py`, `/tmp/*orchestrator*.sh`, or parallel Cursor GPU subagents — they race the queue. Before a fresh `run`, stop strays (not while `pipeline-run` / `run_gpu_queue.py` owns the GPU):

```bash
./scripts/kill_stray_gpu_jobs.sh
```

### GPU pipeline (live, nohup + mamba)

**Whole backlog in one detached session:** confirm (100 ep) → full manifest (RT-DETR → Tier 2 → **P0-5** zoo → CV tail). Implemented in [`run_gpu_queue.sh`](scripts/run_gpu_queue.sh) (`pipeline-run` | `pipeline-resume` | `pipeline-dry-run`).

```bash
export DATASET_ROOT="$(pwd)/data/raw/extracted/dataset"
export HARCHOC_EXPORT_DEVICE=cpu

# optional: preview both manifests (no GPU)
./scripts/run_gpu_queue.sh pipeline-dry-run

# live — setsid+nohup wrapper; each phase invokes mamba run -n harchoc python scripts/run_gpu_queue.py
./scripts/run_gpu_queue.sh pipeline-run
# → pid in reports/gpu_queue/nohup.pid
# → logs:  reports/gpu_queue/nohup.log
# → state: reports/gpu_queue/run_state.json  (manifest switches confirm → full when phase 1 completes)
# → per-job: reports/gpu_queue/logs/{job_id}/*.log

tail -f reports/gpu_queue/nohup.log
cat reports/gpu_queue/run_state.json
watch -n 5 nvidia-smi
```

**Phase order inside `pipeline-run`:**

1. **`gpu_queue_aug_confirm.json`** — job `aug_confirm_winner_100ep` (100 ep `robustness_minimal`, train + HSP eval). Skipped when summary already verified; waits if the same job is already in flight (does not restart).
2. **`gpu_queue_full.json`** — preflight → vram/RT-DETR → amp/sg eval + close sweeps → `zoo_matrix_p0_5` → `cv_fold_train` (with manifest `skip` / `skip_if` as documented in tier table).

Override manifest paths: `GPU_QUEUE_MANIFEST_CONFIRM`, `GPU_QUEUE_MANIFEST_FULL`. After a failure mid-pipeline: `./scripts/run_gpu_queue.sh pipeline-resume`.

**Single-manifest nohup** (one JSON only — same mamba wrapper):

```bash
export DATASET_ROOT="$(pwd)/data/raw/extracted/dataset"
./scripts/run_gpu_queue.sh dry-run          # → dry_run_plan.txt
GPU_QUEUE_MANIFEST=configs/experiments/gpu_queue_full.json ./scripts/run_gpu_queue.sh run
./scripts/run_gpu_queue.sh resume           # resume current manifest in run_state.json
```

State/logs: `reports/gpu_queue/run_state.json`, `reports/gpu_queue/logs/{job_id}/{stage}.log`. See [EXPERIMENTS § GPU queue](docs/EXPERIMENTS.md#gpu-sequential-queue).

**Email notify (optional):** copy [`configs/local/queue_notify.example.json`](configs/local/queue_notify.example.json) → `configs/local/queue_notify.json` (gitignored) with your address + SMTP app password. Fires on each queue job complete/fail, each zoo matrix row, and manifest finish. Audit trail: `reports/gpu_queue/notify_log.jsonl` (no recipient stored). Test: `python scripts/queue_notify_test.py --dry-run`. Disable: `HARCHOC_NOTIFY_DISABLE=1`.

**Queue snapshot (2026-05-31):** S0–S14 + Phase A sweeps **complete** in [`aug_smoke_index.json`](configs/experiments/aug_smoke_index.json). **Next GPU job:** `zoo_matrix_p0_5` in [`gpu_queue_full.json`](configs/experiments/gpu_queue_full.json) — prior Tier 1–2 jobs skip on dry-run. Rankings: [`leaderboard.md`](reports/aug_smoke/leaderboard.md).

```bash
# Start zoo (skips completed aug jobs):
export DATASET_ROOT="$(pwd)/data/raw/extracted/dataset"
./scripts/run_gpu_queue.sh resume
# or: ./scripts/run_gpu_queue.sh --job zoo_matrix_p0_5
```

### P1-AUG-100EP-WINNER (confirm manifest) — **Done**

100-ep **production** [`robustness_minimal.yaml`](configs/aug/robustness_minimal.yaml) confirm vs **best2** (~61.3 MAE). Result: **64.1** test count MAE ([`aug_confirm_winner_100ep_summary.json`](reports/aug_smoke/aug_confirm_winner_100ep_summary.json)).

```bash
# Re-run only if summary deleted (skip gate will skip when verified):
GPU_QUEUE_MANIFEST=configs/experiments/gpu_queue_aug_confirm.json ./scripts/run_gpu_queue.sh run
```

Next GPU: **P0-5** `zoo_matrix_p0_5` — see [§ GPU execution tiers](#gpu-execution-tiers-post-aug_pending).

### Zoo matrix (2026-05-30)

| Group | Rows | Trainable now | Notes |
|-------|------|---------------|--------|
| **`zoo_core`** | 10 | Ultralytics (6) + external (4) | `yolov8/10/11/26m`, `rtdetr nq1024/x`, DEIM stack — queue default |
| **`sota_deim`** / **`zoo_detr_stack`** | 4 each | **Yes** (after `check_weights_cache --download`) | Same four external rows; filter aliases — [`matrix_rows.v1.json`](configs/zoo/matrix_rows.v1.json) |
| **`sota_2026`** | ~22 | Yes (Ultralytics) | Full hub set incl. YOLO26 + scale ladders |
| **`zoo_scale`** | 14 | Yes | n/s/l/x/b per family (excludes core `*m` picks) — run only after a family wins `zoo_core` |

Design / pruning: [`docs/zoo_comparison_design.md`](docs/zoo_comparison_design.md). Bench configs: [`configs/bench/`](configs/bench/).

Full HSP chain: [EXPERIMENTS § Threshold sweep](docs/EXPERIMENTS.md#threshold-sweep--error-analysis-real-preds). Budget caps: [training_budget.md](docs/training_budget.md).

---

## Reporting quick ref

| Output | Path |
|--------|------|
| Headline metrics | [`p0_summary.md`](reports/hsp/p0_summary.md) |
| Thresholds / errors / TIDE | `reports/hsp/threshold_*.json`, `error_*.json`, `tide_bucket_summary*.json` |
| Domain eval | [`domain_eval.json`](reports/domains/domain_eval.json) |
| Aug smokes | [`reports/aug_smoke/`](reports/aug_smoke/) |
| Gap map | [`reviewer_comments_backlog_gap.md`](docs/manuscript/reviewer_comments_backlog_gap.md) |

**CI:** `PYTHONPATH=. HARCHOC_ALLOW_BASE_PYTHON=1 python -m unittest discover -s tests` (462 tests)

---

## Done & tested (archive)

*Moved here 2026-05-29 — active queue above is Next/Partial/Blocked only.*

### P0 credibility

| ID | Evidence |
|----|----------|
| P0-0 | [`split_drift_p0.json`](reports/hsp/split_drift_p0.json) ok |
| P0-1 | `max_det: 3000`; S14 eval-only @ max_det 300 — test MAE **265.8** ([`s14_summary.json`](reports/aug_smoke/s14_summary.json); prior [`s14_maxdet_truncation.json`](reports/hsp/s14_maxdet_truncation.json)) |
| P1-AUG (smokes S0–S14) | **15/15 complete** — [`gpu_queue_aug_pending`](configs/experiments/gpu_queue_aug_pending.json) finished 2026-05-30 15:08 UTC — [§ Aug smokes](#aug-smokes--15-ep-sweeps-gpu-queue-2026-05-2930) |
| P1-AUG-DUP-MAE | **S0≡S1≡S13≡CLOSE25** (rank **S1**; preds `ad6f1621…`) and **S3≡S6≡S7** (rank **S3**; preds `41e79d28…`) — **do not re-train** — [`dedup_root_cause.md`](reports/aug_smoke/dedup_root_cause.md), [`equivalence_classes`](configs/experiments/aug_smoke_index.json), [`leaderboard.md`](reports/aug_smoke/leaderboard.md) |
| P1-AUG-CLOSE (Phase A) | close10 **96.7** rejected; close25 ≡ S1 **68.9** — keep `close_mosaic=15` @ 100 ep — [`sweep_close10_15ep_summary.json`](reports/aug_smoke/sweep_close10_15ep_summary.json), [`sweep_close25_15ep_summary.json`](reports/aug_smoke/sweep_close25_15ep_summary.json) |
| P1-AUG-100EP-WINNER | **64.1** test MAE @ 100 ep `robustness_minimal` — [`aug_confirm_winner_100ep_summary.json`](reports/aug_smoke/aug_confirm_winner_100ep_summary.json) |
| P1-AUG-CLOSE-SCALE | `apply_close_mosaic_epoch_scale` + smoke YAMLs — [archive wiring] |
| P1-AMP-HSP-EVAL | **204.2** MAE — diagnostic; preds ≡ SG — [`amp_on_smoke_15ep_summary.json`](reports/hsp/amp_on_smoke_15ep_summary.json) |
| P1-SG-HSP-EVAL | **204.2** MAE — diagnostic; preds ≡ AMP — [`sg_yolo_nas_s_smoke_15ep_summary.json`](reports/aug_smoke/sg_yolo_nas_s_smoke_15ep_summary.json) |
| P1-AUG-CLOSE-100EP | **Cancelled** — no Phase A winner |
| P1-AUG-SCHEDULE-SMOKE | **Deferred** — schedule audit on confirm run first |
| P1-AUG-MOSAIC (15 ep) | Queue sweeps **removed** (≡ **S2** mosaic0, **S4** mosaic01, **S5** mosaic03 smokes); historical sweep summaries in [§ Aug smokes](#aug-smokes--15-ep-sweeps-gpu-queue-2026-05-2930) |
| P1-AMP-SMOKE (15 ep train) | `amp_smoke_15ep_on` train Done — HSP eval **Done** **204.2** MAE (diagnostic; preds ≡ SG) |
| P1-SG (15 ep train) | `sg_yolo_nas_s_smoke` train Done — HSP eval **Done** **204.2** MAE (diagnostic; preds ≡ AMP) |
| P0-2 | [`baseline_models_manifest.json`](reports/hsp/baseline_models_manifest.json) |
| P0-3 | HSP chain on `best2.pt`: test MAE ~61.3 @ conf ~0.15 |
| P0-4 | RT-DETR 15-ep smoke harness — [`rtdetr_smoke_notes.md`](reports/hsp/rtdetr_smoke_notes.md) |
| MS-VAL-MAP-CAVEAT | [`val_test_map_gap.md`](docs/manuscript/val_test_map_gap.md) |

### Manuscript & science narrative (Done)

| ID | Doc |
|----|-----|
| MS-ORIG | [`originality_contribution_peers.md`](docs/manuscript/originality_contribution_peers.md), gap §2 |
| MS-LIT | [`related_work_outline.md`](docs/manuscript/related_work_outline.md), gap §6 |
| MS-SPLIT-MAPNARR, MS-VAL-MAPDOWN | [`val_test_map_gap.md`](docs/manuscript/val_test_map_gap.md) §5 |
| MS-ASYM-NARR | gap §10, [`asymmetric_seed_policy.json`](configs/eval/asymmetric_seed_policy.json) |
| MS-FP-LOC-NARR | gap §11, [`fig_error_taxonomy.png`](reports/figures/fig_error_taxonomy.png) |
| MS-DOMAIN-ADAPT | gap §12, [`domain_eval.json`](reports/domains/domain_eval.json) |
| MS-DEPLOY-2STG | gap §14, **R-SCI-2** |
| MS-FUZZY-BOUND | gap §15, [`fig_ambiguous_panel.png`](reports/figures/fig_ambiguous_panel.png) |
| MS-MANUAL-BASE, MS-MANUAL-N50 | gap §8, §17 |
| MS-FIG-NORM | `harchoc/figure_style.py`, `--journal-style` — see also [Figures archive](#figures--explainability-done) |
| MS-REPRO | `experiment.py repro`, [`manuscript_repro_bundle.json`](configs/experiments/manuscript_repro_bundle.json) |

### Eval · domain · deploy (Done)

| ID | Evidence |
|----|----------|
| R-SCI-1 / SCI-MAP-CPU | [`eval_test_map.json`](reports/hsp/eval_test_map.json), `map-cpu` |
| R-SCI-2 | [`deploy_hsp_parity.json`](reports/hsp/deploy_hsp_parity.json) |
| P1-DOMAIN-EVAL | `domain_eval.v1` + count MAE per tray — [`domain_count_mae.json`](reports/domains/domain_count_mae.json) |
| P1-FINETUNE-LOOP | splits + `--stage 1|2` scaffold; full 25+25 ep = GPU follow-up |
| P1-UNCERT-FP | `ambiguous_fp_crosstab` in error reports |
| P1-SPLIT-DRIFT-RICH | `--extended` on `split_drift.py` |
| P1-TIDECV | `tidecv_compare.v1` sidecar |
| P1-TIDE | count-share proxy + FP crops — [`tide_bucket_summary*.json`](reports/hsp/tide_bucket_summary.json); optional `--tidecv` |
| P2-COUNT-SEL | count-first threshold selector in `threshold_sweep` / `dual_metric.json` |
| P2-ASYM-SEED | [`asymmetric_seed_policy.json`](configs/eval/asymmetric_seed_policy.json) |
| P1-RTDETR-MAXDET / META | `validate_rtdetr_infer_max_det`; matrix metadata |
| P1-ZOO-READY | Prep gate via `harchoc.bench_assets` / [`weights_cache.json`](reports/hsp/weights_cache.json) (20 Ultralytics + 4 external); matrix **26** rows — [`matrix_rows.v1.json`](configs/zoo/matrix_rows.v1.json), validate: `benchmark_matrix.py --validate-zoo` |
| P2-ZOO-EXPAND | YOLO26 + YOLO10/11 scales + `rtdetr-x` + external DETR registry + `zoo_core` groups — 2026-05-30 |
| ARCH-EMA-BG-SPIKE | cite-only; impl cancelled — [arch_ema_bg_spike](docs/research/arch_ema_bg_spike_literature.md) |

### FP budget & threshold (Done)

| ID | Evidence |
|----|----------|
| P1-FP-BUDGET (val) | `experiment.py fp-budget-sweep`, `threshold_sweep --fp-budget-sweep-out` → [`fp_budget_sweep.json`](reports/hsp/fp_budget_sweep.json) (109 val images, 19 sweep rows) |
| P1-FP-BUDGET (test) | `experiment.py --config fp_budget_sweep_test.json fp-budget-sweep` → [`fp_budget_sweep_test.json`](reports/hsp/fp_budget_sweep_test.json) + [`fp_budget_sweep_test.md`](reports/hsp/fp_budget_sweep_test.md) (test MAE **61.3** @ locked conf **0.15**; F1-max +18.1 MAE) |

### Figures & explainability (Done)

| ID | Evidence |
|----|----------|
| P2-FIG-CAM | `experiment.py gradcam` → [`gradcam_routing.md`](docs/manuscript/gradcam_routing.md) |
| P2-FIG | [`fig_error_taxonomy.png`](reports/figures/fig_error_taxonomy.png), [`fig_ambiguous_panel.png`](reports/figures/fig_ambiguous_panel.png), [`fig_concept.png`](reports/figures/fig_concept.png) |
| P2-FIG-CONCEPT | [`fig_concept.png`](reports/figures/fig_concept.png), [`fig_concept.svg`](reports/figures/fig_concept.svg) via `make_figures.py --figure fig_concept` |

### Scaffolding (code Done)

| ID | Evidence |
|----|----------|
| P2-SEED-MAE | `matrix_seed_stats.v1` columns in `benchmark_matrix.py` (needs **P0-5** matrix rows) |
| P1-CV-TRAIN (routing) | `experiment.py cv-eval` → `argv_for_cv_eval` (**DRY-SPRAWL-CV**) |
| P1-DOMAIN-TAGS | `eval_domains.py --import-domain-tags` → `catalog.json` + `domain_eval.json`; schema [`data/domain_tags.example.csv`](data/domain_tags.example.csv) |
| P1-RTDETR-Q (config) | `configs/models/rtdetr-l_nq1024.yaml`, [`train_rtdetr_queries_smoke_15ep.json`](configs/experiments/train_rtdetr_queries_smoke_15ep.json) |

### GPU probes tested

| ID | Result | Report |
|----|--------|--------|
| P1-VRAM | yolov8m 1-ep @ 1280 batch=1 → **7416 MiB** peak | [`train_batch_probe.json`](reports/hsp/train_batch_probe.json) |
| P1-AMP-SMOKE | AMP **on** OK; AMP **off** OOM on 8 GiB V100 | [`amp_smoke_1ep.json`](reports/hsp/amp_smoke_1ep.json) |
| P1-AUTOBATCH | dry-run OK; live `batch=-1` → OOM @ ~60% util target | [`autobatch_probe.json`](reports/hsp/autobatch_probe.json) |
| P1-RTDETR-Q | nq=1024; ep1 ok; retries **failed** (CUDA driver error @ model.to) after GPU contention | [`rtdetr_queries_smoke_notes.json`](reports/hsp/rtdetr_queries_smoke_notes.json) |
| P1-RTDETR-AMP | matrix bench `amp` / `grad_clip` flags wired | refactor **DRY** / bench configs |

**Open GPU follow-ups:** **P0-5** `zoo_core_8gb` only. RT-DETR jobs skipped on 8 GiB. **DATA-ACQ-GEN** parallel (no GPU).

### Aug smokes + 15-ep sweeps (GPU queue 2026-05-29/30)

HSP eval: test split, conf locked from [`threshold_val.json`](reports/hsp/threshold_val.json) (~0.15). **Queue complete:** [`gpu_queue_aug_pending`](configs/experiments/gpu_queue_aug_pending.json) finished **2026-05-30 15:08 UTC** — all **S0–S14** in [`aug_smoke_index.json`](configs/experiments/aug_smoke_index.json).

**Reference:** **best2** **61.3** MAE @ 100 ep ([`dual_metric.json`](reports/hsp/dual_metric.json)). Best 15-ep smoke **S1 close3 68.9** (+7.6 vs best2).

**15-ep rankings (distinct recipes):** S1 **68.9** → S9 **73.2** → S12 **91.1** → S10 **116.1** → S5 **125.7** → S8 **125.9** → S11 **136.3** → S4 **145.1** → S2 **147.4**. Full table + CIs: [`reports/aug_smoke/leaderboard.md`](reports/aug_smoke/leaderboard.md).

**Eval control:** **S14** eval-only @ max_det 300 → **265.8** MAE ([`s14_summary.json`](reports/aug_smoke/s14_summary.json)) — validates `max_det=3000` export cap (P0-1).

**Audit (do not re-rank):** S0≡S1≡S13≡CLOSE25 @ 68.9; S3≡S6≡S7 @ 151.7 — [`dedup_root_cause.md`](reports/aug_smoke/dedup_root_cause.md), [`equivalence_classes`](configs/experiments/aug_smoke_index.json).

**Conclusions (15 ep + Phase A):** (1) **close_mosaic=3 (S1)** wins; **S9** near winner (**73.2**). (2) **close10 96.7** — shorter tail rejected. (3) **close25 ≡ S1** — schedule-equivalent @ 15 ep (audit-only). (4) **100-ep confirm** **64.1** on `robustness_minimal` (~**+2.8** vs **best2 61.3**). (5) **Aug closed** — next MAE gains from **P0-5** zoo / **DATA-ACQ-GEN**.

**Sweeps (Tier 2):** [`sweeps_15ep`](configs/experiments/aug_smoke_index.json) — close10 + close25 **complete**; close15 recipe-skipped (≡ S0/S1).

**Follow-ups:** **P0-5** zoo, **DATA-ACQ-GEN** — [§ Data acquisition](#data-acquisition-for-generalization-study-design). Phase B/C **cancelled/deferred** — [§ Aug close sweeps (closed)](#aug-close-sweeps-closed).

### GPU queue jobs (train-only → eval-only follow-up)

| Job | Status | Notes |
|-----|--------|-------|
| `amp_smoke_15ep_on` | train Done | `skip_eval`; weights ready — skipped on re-run via `weights_run_name` |
| `amp_smoke_15ep_on_hsp_eval` | **Done** | **204.2** MAE — diagnostic; preds ≡ SG — [`amp_on_smoke_15ep_summary.json`](reports/hsp/amp_on_smoke_15ep_summary.json) |
| `sg_yolo_nas_s_smoke` | train Done | weights `runs/sg_yolo_nas_s_smoke_15ep/` — skipped on re-run |
| `sg_yolo_nas_s_hsp_eval` | **Done** | **204.2** MAE — diagnostic; preds ≡ AMP — [`sg_yolo_nas_s_smoke_15ep_summary.json`](reports/aug_smoke/sg_yolo_nas_s_smoke_15ep_summary.json) |
| `aug_sweep_15_close10` | **Done** | **96.7** MAE — rejected |
| `aug_sweep_15_close25` | **Done** | **68.9** MAE — ≡ S1; preds dedup skips re-queue |
| RT-DETR ×3 + vram_probe | skipped | Jobs stay in [`gpu_queue_full.json`](configs/experiments/gpu_queue_full.json) for **`skip_if` conditional re-run only** — prior `reports/gpu_queue/jobs/*.json` — **P1-RTDETR-COUNT-REFRESH** |

### DRY refactor (all Done)

See [refactor.md](refactor.md) §1–§6: `script_entry`, `deploy_filters`, `json_io`, `experiment_argv`, `finetune`→`train`, `platt` isotonic, TIDE adapter, eval `--locked-conf-from`, cv-eval routing, legacy train shim, bot locked-conf env.

### Agent batches & plumbing (2026-05-27/29)

Prep gate, zoo harness, 2026 research scans, micro-smoke tier, `hsp_weights.py`, domain tags CLI, fp_budget sweep, figure/Grad-CAM routing. Details: [refactor.md](refactor.md).

<details>
<summary>Historical reference (infrastructure scaffolds)</summary>

Infrastructure & CI; dataset/splits; train/eval/metadata; model zoo; analysis scripts (`threshold_sweep`, `error_analysis`, `make_figures`, …). Superseded detail lives in git history and [refactor.md](refactor.md).

</details>

---

## Permanent (always-on)

**Fight sprawl + DRY** — extend `experiment.py`, `benchmark_matrix.py`, `eval.py`, `train.py`, `harchoc/*` before new scripts ([extend-before-add-script](.cursor/rules/extend-before-add-script.mdc)).
