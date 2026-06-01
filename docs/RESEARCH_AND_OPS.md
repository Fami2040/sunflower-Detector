# HARCHOC research & operations (consolidated)

Single entry point for **how we train, evaluate, and improve** sunflower seed detection (`sunflower-cvat-2500`). Deep dives stay in [`research/`](research/) and [`EXPERIMENTS.md`](EXPERIMENTS.md); this doc layers **ops → science → backlog**.

**Validated:** 2026-05-29 (consolidated from ten doc-validation passes + three 2026 tech scans).

---

## How to use this document

| Layer | Read when you need… |
|-------|---------------------|
| [§1 Executive summary](#1-executive-summary) | What to do this week |
| [§2 Document map](#2-document-map) | Where the full literature lives |
| [§3 Prioritized roadmap](#3-prioritized-roadmap) | P0 / P1 / P2 with status |
| [§ Manuscript reviewer response (MS-*)](#manuscript-reviewer-response-ms) | Gap map, MS-* backlog, lit hooks |
| [§4 Operations](#4-operations) | Mamba, budgets, bootstrap, matrix |
| [§5 Data & splits](#5-data--splits) | Layout, classes, leakage, drift |
| [§6 Training recipe](#6-training-recipe) | Defaults @ 1280, schedule, VRAM |
| [§7 Augmentation](#7-augmentation) | Recipe + S0–S14 smokes |
| [§8 Model zoo & DETR](#8-model-zoo--detr) | YOLO matrix + RT-DETR policy |
| [§9 Eval & HSP protocol](#9-eval--hsp-protocol) | Val tune → lock → test |
| [§10 Error / FP / explainability](#10-error--fp--explainability) | TIDE-style analysis, crops, XAI |
| [§11 Domain shift](#11-domain-shift--transfer) | Tray keys, catalog, finetune |
| [§12 Env & libraries](#12-environment--optional-libraries) | What to add to `harchoc` env |
| [§13 Refactors & code](#13-refactors--small-code-improvements) | DRY wins, not new scripts |
| [§14 Do not chase](#14-do-not-chase) | Hype filter |

---

## 1. Executive summary

**Task:** dense **developed** (0) / **aborted** (1) seed boxes on benchtop head images @ **imgsz=1280**; success = **test count MAE**, not val mAP alone. Ordered improvement stack: [backlog § Model improvement stack](../backlog.md#model-improvement-stack-test-count-mae).

**Strengths:** Frozen splits, val→lock→test threshold protocol, bench/matrix harness, conservative aug, bootstrap + budget caps, mamba env wiring.

**Credibility gate (2026-05-29):** P0 steps 0–3 and S14 control are **Done** on `models/best2.pt` — see [p0_summary](../reports/hsp/p0_summary.md) and [s14_maxdet_truncation.json](../reports/hsp/s14_maxdet_truncation.json).

**Still open ([stack steps 3–5](../backlog.md#model-improvement-stack-test-count-mae)):**

| Stack step | Backlog (status in [work queue](../backlog.md#work-queue-p0--p2)) |
|------------|----------------------------------------------------------------------|
| **5** zoo gate | **P0-4** RT-DETR 15-ep smoke (**Partial**; [`rtdetr_smoke_notes`](../reports/hsp/rtdetr_smoke_notes.md)) → **P0-5** full matrix (**Blocked** on P0-4) |
| **3** threshold / trust | **P1-FP-BUDGET** (count-first `--select min_count_mae`), **MS-FUZZY-BOUND**, **P1-UNCERT-FP** |
| **4** aug | **P1-AUG** S0–S14 (test MAE primary), **ARCH-MOSAIC0-AB** |
| **5** zoo follow-on | **P1-ZOO-PARITY**, **P2-SEED-MAE** (after P0-5) |

**Parallel manuscript metric:** **R-SCI-1** test mAP + `dual_metric` detection row (`HARCHOC_EXPORT_DEVICE=cpu`; [training_budget](training_budget.md)).

**Done (manuscript infra):** **MS-REPRO** (`experiment.py repro`), **MS-VAL-MAP-CAVEAT** ([val_test_map_gap](manuscript/val_test_map_gap.md)), validated literature registry, strict ML smoke.

---

## 2. Document map

### Operations & entrypoints

| Doc | Role |
|-----|------|
| [EXPERIMENTS.md](EXPERIMENTS.md) | Live CLI, configs, matrix, threshold workflow |
| [../reports/README.md](../reports/README.md) | Canonical `./reports` layout (HSP science vs manuscript vs archive) |
| [research/README.md](research/README.md) | Literature scans + anti-sprawl path table for citations |
| [HSP_BASELINE_MODELS.md](HSP_BASELINE_MODELS.md) | `best2.pt` / `classifier.pt`, deploy vs HSP eval |
| [training_budget.md](training_budget.md) | `HARCHOC_MAX_*`, RT-DETR smoke, export device |
| [../data/README.md](../data/README.md) | Dataset tree, `data.yaml`, splits |
| [../backlog.md](../backlog.md) | Single task tracker (not duplicated here) |
| [manuscript/reviewer_comments_backlog_gap.md](manuscript/reviewer_comments_backlog_gap.md) | Reviewer comment → backlog / **MISSING** map (**MS-***) |
| [manuscript/related_work_outline.md](manuscript/related_work_outline.md) | §2 Related Work outline + cite table (**MS-LIT** Done) |

### 2026 tech scans (action-oriented)

Index + canonical `reports/` paths: [`docs/research/README.md`](research/README.md).

| Doc | Focus |
|-----|--------|
| [training_tech_scan_2026_augmentation.md](research/training_tech_scan_2026_augmentation.md) | `max_det`, mosaic/`close_mosaic`, **S0–S14** commands |
| [training_tech_scan_2026_detectors.md](research/training_tech_scan_2026_detectors.md) | RT-DETR query cap, zoo gaps, SAHI |
| [training_tech_scan_2026_eval_calibration.md](research/training_tech_scan_2026_eval_calibration.md) | HSP protocol, counting metrics, matrix gaps |

### Literature (reference)

| Doc | Focus |
|-----|--------|
| [augmentation_robustness_literature.md](research/augmentation_robustness_literature.md) | Counting-first aug review |
| [threshold_calibration_literature.md](research/threshold_calibration_literature.md) | Conf/NMS/calibration, agri counting |
| [fp_taxonomy_literature.md](research/fp_taxonomy_literature.md) | TIDE buckets, Layer B review |
| [arch_ema_bg_spike_literature.md](research/arch_ema_bg_spike_literature.md) | GrainNet EMA / background FP — implement vs cancel (**ARCH-EMA-BG-SPIKE** Done) |
| [explainability_uncertainty_literature.md](research/explainability_uncertainty_literature.md) | Ambiguous band, calibration, Grad-CAM |
| [domain_shift_transfer_literature.md](research/domain_shift_transfer_literature.md) | Session/tray shift, finetune |

---

## 3. Prioritized roadmap

**Task status (single source of truth):** [backlog.md § Work queue](../backlog.md#work-queue-p0--p2) — one P0→P2 table with ID, status, blockers, and doc links.

**Model improvement stack → backlog** ([narrative + defer list](../backlog.md#model-improvement-stack-test-count-mae)):

| Step | What | Backlog IDs |
|------|------|-------------|
| 1 | Train/export parity (`max_det=3000`, frozen splits, HSP) | P0-0…P0-3 **Done**, S14 |
| 2 | Full YOLO recipe @ 1280 (`train_yolov8m_baseline.json`, 100 ep) | [EXPERIMENTS](EXPERIMENTS.md) |
| 3 | Val operating point → lock → test; count-first selection | **P1-FP-BUDGET**, **MS-FUZZY-BOUND**, **P1-UNCERT-FP** |
| 4 | Aug ablations (primary: test MAE) | **P1-AUG** S0–S14, **ARCH-MOSAIC0-AB** |
| 5 | Model zoo + count columns | **P0-4** → **P0-5**, **P1-ZOO-PARITY**, **P2-SEED-MAE** |
| 6 | Tray / domain shift | **P1-FINETUNE-LOOP**, **P1-DOMAIN-EVAL** |
| 7 | RT-DETR capacity ablation (zoo row only) | **P1-RTDETR-Q** |

This section maps backlog IDs to research scans (rationale only; no duplicate status column).

| Backlog ID | Theme | Primary doc |
|------------|--------|-------------|
| P0-4, P0-5 | Credibility gate → zoo matrix | [detectors scan](research/training_tech_scan_2026_detectors.md), [EXPERIMENTS § matrix](EXPERIMENTS.md#model-zoo-benchmark-matrix) |
| P0-0 … P0-3, S14 | Done — drift, max_det, HSP on `best2.pt` | [eval scan](research/training_tech_scan_2026_eval_calibration.md), [aug scan](research/training_tech_scan_2026_augmentation.md) |
| P1-AUG* | Aug S0–S14 + sweeps | [aug scan](research/training_tech_scan_2026_augmentation.md) §5 |
| P1-RTDETR-* | Query cap, imgsz, AMP | [detectors scan](research/training_tech_scan_2026_detectors.md) |
| P1-FP-BUDGET, P2-COUNT-SEL | Count-first thresholds | [eval scan](research/training_tech_scan_2026_eval_calibration.md), [threshold_calibration_literature](research/threshold_calibration_literature.md) |
| P1-TIDE*, P2-FIG* | Error taxonomy / figures | [fp_taxonomy_literature](research/fp_taxonomy_literature.md), [explainability_literature](research/explainability_uncertainty_literature.md) |
| R-SCI-1, R-SCI-2 | mAP + deploy vs manuscript | [HSP_BASELINE_MODELS](HSP_BASELINE_MODELS.md), [p0_summary](../reports/hsp/p0_summary.md) |

---

## Manuscript reviewer response (MS-*)

**Gap map (comment → backlog / MISSING):** [`docs/manuscript/reviewer_comments_backlog_gap.md`](manuscript/reviewer_comments_backlog_gap.md). **Tasks:** [`backlog.md`](../backlog.md#work-queue-p0--p2) rows prefixed **MS-*** plus **ARCH-*** / **LIT-*** / **SCI-*** — status and blockers live there only (not duplicated here).

**Validated registry:** [`literature_validated.json`](manuscript/literature_validated.json) · [`architecture_recommendations.md`](manuscript/architecture_recommendations.md)

### Literature hooks (reviewer-requested cites)

| Registry ID | Theme | Repo synthesis |
|-------------|--------|----------------|
| `yang2024_oct_tl` | Transfer / domain adaptation | [domain_shift_transfer_literature.md](research/domain_shift_transfer_literature.md) §10 |
| `ren2025_scripta_interp` | Interpretability (trust) | [explainability_uncertainty_literature.md](research/explainability_uncertainty_literature.md) + **P2-FIG-CAM** Done |
| `alshehri2025_uav` | Two-stage deploy analogy | [HSP_BASELINE_MODELS.md](HSP_BASELINE_MODELS.md) |
| `yao2025_hfuzzy` | Graded trust on detections (§270 analogy; not 3rd class) | [explainability_uncertainty_literature.md](research/explainability_uncertainty_literature.md) · **MS-FUZZY-BOUND** |

**Related deep dives:** error taxonomy and localization-vs-cls framing — [fp_taxonomy_literature.md](research/fp_taxonomy_literature.md) (pairs with explainability § on ambiguous/overlap FPs).

### How repo artifacts address reviewer themes

- **SOTA breadth (not YOLOv8-only):** `benchmark_matrix.py` / P0-5 zoo trains seven Ultralytics slots + YOLO-NAS with shared bench JSON @ 1280; `reports/hsp/matrix_train.json` and [detectors scan](research/training_tech_scan_2026_detectors.md) document RT-DETR query-cap policy alongside YOLO rows.
- **Counting credibility vs mAP alone:** `experiment.py dual-metric` → [`dual_metric.json`](../reports/hsp/dual_metric.json) merges detection mAP with val-selected / test-locked counting MAE, FP/img, and calibration fields from the HSP chain ([§9](#9-eval--hsp-protocol), [p0_summary](../reports/hsp/p0_summary.md)).
- **Explainability (Ren 2025):** `make_figures.py --figure fig_gradcam_panel` + `harchoc/gradcam_panel.py` on FP crops from `error_analysis.py --export-fp-crops`; literature workflow in [explainability_uncertainty_literature.md](research/explainability_uncertainty_literature.md).
- **Generalization / val≫test (Yang 2024):** `split_drift.py` → [`split_drift_p0.json`](../reports/hsp/split_drift_p0.json) (KS/L1/JSD gate); tray/session slices via `eval_domains.py` — see [domain_shift_transfer_literature.md](research/domain_shift_transfer_literature.md). Val mAP caveats: [`docs/manuscript/val_test_map_gap.md`](manuscript/val_test_map_gap.md) when present.
- **Threshold / boundary discipline (Yao 2025, FP taxonomy):** val `threshold_sweep.py` → test `--locked-conf-from` only ([`threshold_test_locked.json`](../reports/hsp/threshold_test_locked.json), `harchoc/threshold_protocol.py`); Layer-A TIDE-style buckets + optional FP crops align with [fp_taxonomy_literature.md](research/fp_taxonomy_literature.md) for localization-dominant errors at locked conf.
- **Two-stage deploy analogy (Alshehri 2025):** production path `classifier.pt` gate + `best2.pt` SAHI detection vs single-stage HSP `eval.py` ([HSP_BASELINE_MODELS.md](HSP_BASELINE_MODELS.md), [reviewer gap §14](manuscript/reviewer_comments_backlog_gap.md#14-manuscript-draft--two-stage-deploy-discussion); **MS-DEPLOY-2STG** Done, **R-SCI-2** [`deploy_hsp_parity.json`](../reports/hsp/deploy_hsp_parity.json)).

---

## 4. Operations

### Mamba env (non-negotiable)

```bash
export DATASET_ROOT="$(pwd)/data/raw/extracted/dataset"
mamba run -n harchoc python scripts/<script>.py ...
```

- Helpers: `harchoc/ml_env.py`, `scripts/check_gpu.py`, `scripts/rtdetr_smoke.py` (re-exec if base Python lacks torch).
- CI: `HARCHOC_ALLOW_BASE_PYTHON=1` for unittest only.

Details: [EXPERIMENTS.md](EXPERIMENTS.md) § GPU environment, [training_budget.md](training_budget.md).

### Budget caps

| Variable | Default | Purpose |
|----------|---------|---------|
| `HARCHOC_MAX_EPOCHS` | 500 | Cap runaway epochs |
| `HARCHOC_MAX_IMGSZ` | 2048 | Cap resolution |
| `HARCHOC_MAX_BATCH` | 16 | Cap batch in bench JSON |

Short smokes: `export HARCHOC_MAX_EPOCHS=15 HARCHOC_MAX_IMGSZ=2048`.

### Bootstrap checklist

```bash
python scripts/bootstrap_env.py --env harchoc --verify-manifests
python scripts/bootstrap_env.py --env harchoc --with-super-gradients   # YOLO-NAS only
python scripts/bootstrap_env.py --env harchoc --with-external-detr      # DEIM / D-FINE / RT-DETRv2
mamba run -n harchoc python scripts/check_gpu.py --json-out reports/hsp/gpu_check.json
mamba run -n harchoc python scripts/validate_splits.py --require-test
```

### Matrix train (after P0)

```bash
mamba run -n harchoc python scripts/check_weights_cache.py --download --strict --out reports/hsp/weights_cache.json
mamba run -n harchoc python scripts/benchmark_matrix.py --out reports/hsp/matrix_plan.json
mamba run -n harchoc python scripts/benchmark_matrix.py --no-dry-run \
  --runs-dir runs/hsp_zoo \
  --train-out reports/hsp/matrix_train.json \
  --out reports/hsp/matrix_plan.json
```

Run naming: `{model}_e{N}_s{seed}`. Do not use deprecated `reports/_archive/hsp/bench_800/` (use `configs/bench` @ 1280).

---

## 5. Data & splits

- **Classes:** 0 = **developed**, 1 = **aborted** (`harchoc/sunflower_dataset.py`).
- **Labels:** YOLO `class_id cx cy w h` (normalized); pair `images/…` ↔ `labels/…` same stem.
- **Splits:** `data/splits/{train,val,test}.txt` — **875 / 109 / 109**; CVAT folder names ≠ modeling boundary.
- **Ultralytics val** = early stop; **test** = manuscript metrics (`eval.py`).
- **Done:** group-wise splits, leakage audit (stem/group; pHash stub), domain catalog (`eval_domains.py`).
- **Next:** split drift richer proxies (P1); asymmetric seed eval policy **Done** ([`configs/eval/asymmetric_seed_policy.json`](../configs/eval/asymmetric_seed_policy.json)).

Refs: [data/README.md](../data/README.md), [domain_shift_transfer_literature.md](research/domain_shift_transfer_literature.md).

---

## 6. Training recipe

### Committed defaults (`train_bench_base.json` + `robustness_minimal.yaml`)

| Knob | Value | Rationale |
|------|-------|-----------|
| `imgsz` | 1280 | Small seeds on tray |
| `epochs` / `patience` | 100 / 50 | Standard zoo |
| `optimizer` | AdamW, `lr0=2e-4` | Legacy sunflower recipe |
| `max_det` (train) | 3000 | Dense counting |
| `mosaic` | 0.1 | Low vs YOLO26 default ~1.0 |
| `mixup` / `cutmix` | 0 | Count integrity |
| `close_mosaic` | 15 | Late calibration tail — **actionable:** sweep {10, 15, 25} vs early-stop interaction ([#18013](https://github.com/ultralytics/ultralytics/issues/18013); **P1-AUG-CLOSE**) |
| `conf` / `iou` | 0.05 / 0.3 | Export/sweep alignment |

### `max_det` parity (step 1 — Done on bench base)

`train_bench_base.json` sets **train and eval** `max_det=3000` (`"eval": {"max_det": 3000}`). Legacy smokes or one-off configs at **300** truncate dense preds → count FN; aug scan **S14** is the deliberate negative control ([`s14_maxdet_truncation.json`](../reports/hsp/s14_maxdet_truncation.json)).

### VRAM & batch

- Probed: `yolov8n→batch=2`, `yolov8m→batch=1` on ~7.5 GiB.
- **Next:** per-model 1-ep ladder for all zoo slots.

### Schedule guard (Done)

`epochs - patience >= close_mosaic` in `harchoc/train_config.py` (bench config CI). On **15-ep** smokes, `close_mosaic=15` disables mosaic from epoch 0 — use scaled `close_mosaic` (e.g. 3) per [aug scan](research/training_tech_scan_2026_augmentation.md).

---

## 7. Augmentation

**Stack step 4** ([backlog stack](../backlog.md#model-improvement-stack-test-count-mae)): aug ablations with **test count MAE** primary — **P1-AUG** S0–S14, **ARCH-MOSAIC0-AB**.

**Verdict (2026 scans):** We are **not** under-augmented vs Ultralytics defaults; val≫test is more likely **eval caps + split drift + schedule** than missing mosaic.

### Keep

- Low mosaic (0.1), no mixup, mild geometry, photometric HSV/erasing.
- `aug_config` → `configs/aug/robustness_minimal.yaml` on all `train_bench_*.json`.

### Run next (GPU): S0–S14 @ 15 epochs

Primary metric: **test count MAE** via `error_analysis.py` with locked conf from val sweep.

| Smoke | Intent |
|-------|--------|
| S0 | Baseline (documents 15-ep + `close_mosaic` issue) |
| S1–S3 | Mosaic-off / photometric-only |
| S4–S6 | `close_mosaic` scaled for 15 ep |
| S14 | Eval @ `max_det=300` **negative control** |

Full commands: [training_tech_scan_2026_augmentation.md](research/training_tech_scan_2026_augmentation.md) §5 (all `mamba run -n harchoc`).

### Defer

- YOLO26-style mosaic≈1, mixup>0, copy-paste (needs masks).
- MuSGD smoke until YOLO11 baseline justifies optimizer change.

Deep review: [augmentation_robustness_literature.md](research/augmentation_robustness_literature.md).

---

## 8. Model zoo & DETR

### Ultralytics matrix (7) + YOLO-NAS (SG)

- Configs: `configs/bench/*.yaml` + `configs/experiments/train_bench_*.json`.
- Weights: `data/weights/` + `check_weights_cache.py --download --strict`.
- **Done:** train/eval chain, seed mAP via `--aggregate-seeds`, `max_det` not clobbered at train time.

### RT-DETR policy (Done)

| Item | Value |
|------|--------|
| Model | `rtdetr-l.pt` (Ultralytics v1) |
| `num_queries` | 300 (committed) |
| Peak GT / image | 1015 |
| Policy | `accept_rtdetr_query_truncation: true` |
| Guard | `train.py`, `bench_config`, `validate_splits --check-rtdetr-query-cap` |

Raising queries requires **custom `RTDETRDecoder` YAML** + full retrain ([#20688](https://github.com/ultralytics/ultralytics/issues/20688)) — **Next** ablation, not kwargs alone.

### Detector stack gaps (research only until P0 clear)

- RT-DETRv2/v4, D-FINE, DEIM not in zoo.
- Bench configs now use **`max_det: 3000`**; S14 documents truncation at 300 ([`s14_maxdet_truncation.json`](../reports/hsp/s14_maxdet_truncation.json)).
- SAHI only on deploy (`run_infer_once.py`, bot), not matrix eval.

Details: [training_tech_scan_2026_detectors.md](research/training_tech_scan_2026_detectors.md).

---

## 9. Eval & HSP protocol

**Stack step 3** ([backlog stack](../backlog.md#model-improvement-stack-test-count-mae)): tune on **val** → lock conf → report on **test**; select operating point with **`--select min_count_mae`** (**P1-FP-BUDGET**). Graded trust on detections (**MS-FUZZY-BOUND**: locked conf + low-conf score band + `ambiguous_summary`; not a third class — §10).

**Baseline weights:** [HSP_BASELINE_MODELS.md](HSP_BASELINE_MODELS.md) (`best2.pt` vs `classifier.pt`, deploy SAHI vs manuscript export). Checksums: [baseline_models_manifest](../reports/hsp/baseline_models_manifest.json). Spec: `configs/experiments/eval_hsp_baseline.json`.

**Headline metrics (frozen):** [reports/hsp/p0_summary.md](../reports/hsp/p0_summary.md). **Canonical artifacts:** [reports/hsp/README.md](../reports/hsp/README.md).

### Pipeline (implemented; regen via configs)

```text
eval.py (val/test export @ conf 0.001, max_det 3000)
  → threshold_sweep.py (val: --iou-grid, optional isotonic)
  → threshold_sweep.py (test: --locked-conf-from threshold_val.json)
  → error_analysis.py (val + test)
  → experiment.py dual-metric
  → make_figures.py
```

**Guards:** No test tuning without `--locked-conf-from` (`harchoc/threshold_protocol.py`). `load_locked_conf` reads `locked.row` on test-locked sweeps ([`threshold_lock.py`](../harchoc/threshold_lock.py)).

**Copy-paste commands:** [EXPERIMENTS.md § Threshold sweep + error analysis](EXPERIMENTS.md#threshold-sweep--error-analysis-real-preds). GPU runbook: [backlog § Runbook](../backlog.md#runbook-gpu).

### Gaps vs 2026 counting practice

| Gap | Recommendation |
|-----|----------------|
| MS Methods sentence for count-first selection | **P1-FP-BUDGET** Partial — config + `dual_metric` done; manuscript text open |
| Matrix seeds: mAP only | Extend `matrix_seed_stats` for MAE (P2) |
| No TIDE ΔAP | Optional `tidecv`; Layer A buckets already in `error_analysis.py` |

Details: [training_tech_scan_2026_eval_calibration.md](research/training_tech_scan_2026_eval_calibration.md), [threshold_calibration_literature.md](research/threshold_calibration_literature.md).

---

## 10. Error / FP / explainability

### MS-FUZZY-BOUND (reviewer §270)

**Graded trust on detections**, not a third YOLO class or relabel protocol: val-**locked conf** gate, a **low-conf score band** on exported boxes, **`ambiguous_summary`** in `error_analysis.py`, and FP taxonomy (background / localization ≫ cls). Yao 2025 cites fuzzy **regression** as an analogy for graded trust only ([explainability_literature](research/explainability_uncertainty_literature.md)). Open: **P1-UNCERT-FP** cross-tab `ambiguous_detections` × FP buckets; **P2-FIG** `fig_ambiguous_panel`.

### Implemented (Layer A)

- TIDE-style buckets: background / localization / classification / dupe (`error_analysis.py`).
- Count MAE / rRMSE + bootstrap CI; area strata + conf×taxonomy grid (`error_taxonomy.py`).
- `--export-fp-crops` for reviewer panels; `--locked-conf-from` alignment.

### Open

- Real test exports for manuscript.
- TIDE **delta-AP** (not just bucket counts).
- Layer B: `review_category` on crop manifest.
- `fig_ambiguous_panel` / taxonomy PNGs (`make_figures.py` partial).

### Dense tray notes

- **Dupe** and overlap dominate; **Cls** = developed↔aborted confusion.
- Interpret **Miss** only after **max_det** parity (P0).

Refs: [fp_taxonomy_literature.md](research/fp_taxonomy_literature.md), [explainability_uncertainty_literature.md](research/explainability_uncertainty_literature.md).

---

## 11. Domain shift & transfer

| Capability | Status |
|------------|--------|
| Tray keys from stems (`349-10-2`, …) | **Done** (`harchoc/domain_tags.py`) |
| Domain catalog + optional split lists | **Done** (`eval_domains.py`) |
| Per-domain mAP / count | **Scaffold** (use domain lists + `eval.py`) |
| `finetune.py` train loop | **Scaffold** |

**Before domain science:** P0 `max_det` parity; then per-tray metrics on frozen test.

Ref: [domain_shift_transfer_literature.md](research/domain_shift_transfer_literature.md).

---

## 12. Environment & optional libraries

### Already in workflow

- `ultralytics`, `torch` (via `bootstrap_env.py`)
- `scipy` (split drift KS, stats CI)
- `matplotlib` (optional drift plots)
- `super-gradients` (optional; YOLO-NAS)

### Consider adding (P2, when feature lands)

| Package | Use | When |
|---------|-----|------|
| `tidecv` | Official TIDE ΔAP cross-check | After real preds exported |
| `imagehash` + `Pillow` | Perceptual-hash leakage audit | If enabling pHash in `split_leakage_audit` |
| `sahi` | Matrix/deploy parity eval | P2 deploy-parity group; already on bot path |
| `wandb` or `mlflow` | Artifact tracking | Log existing JSON + `run_metadata` paths only |

**Do not add** to default env without a wired script: DEIM/D-FINE repos, RT-DETRv4 training stack (separate venv if experimented).

---

## 13. Refactors & small code improvements

Prefer extending existing entrypoints ([`.cursor/rules/extend-before-add-script.mdc`](../.cursor/rules/extend-before-add-script.mdc)). **Canonical DRY / sprawl audit:** [`refactor.md`](../refactor.md) (backlog **DRY-*** rows in [backlog.md](../backlog.md)).

| Item | Effort | Impact |
|------|--------|--------|
| `train_bench_base.json`: `"eval": {"max_det": 3000}` | Low | **P0** counting fairness |
| `dual_metric_report`: prefer `locked.counting_metrics` | Low | **Done** — `resolve_counting_metrics()` |
| `matrix_seed_stats`: ingest error-analysis MAE | Medium | Multi-seed counting table |
| `threshold_protocol`: `--select min_count_mae` | Medium | Val tuning aligned to MAE |
| `train.py`: optional runtime `close_mosaic` guard | Low | Catch bad smoke configs |
| `finetune.py`: implement loop (not new script) | Medium | Domain shift P2 |
| `eval_domains.py`: wire per-domain eval | Medium | Tray-level tables |
| Bench YAML: matrix metadata for RT-DETR (query cap note) | Low | Provenance |

**No new** `scripts/threshold_calibration.py` (forbidden in backlog).

---

## 14. Do not chase

Aligned with [backlog stack defer](../backlog.md#model-improvement-stack-test-count-mae) and [architecture_recommendations § Defer](manuscript/architecture_recommendations.md) (low test-MAE ROI):

- Default YOLO26 mosaic≈1 / high mixup for counting.
- RT-DETRv4 / huge ViT teachers before query cap + eval `max_det` parity fixed.
- Lowering `num_queries` for speed (opposite of dense trays).
- SAHI-as-default **training** metric before full-frame baseline stable.
- **Third detect class** or fuzzy detect head for “uncertainty” / “ambiguous” seeds — use 2-class YOLO + score band + `ambiguous_summary` (**MS-FUZZY-BOUND**).
- COCO AP leaderboard without test count MAE.
- Copy-paste online aug without mask labels.
- Replacing greedy matching with Hungarian for YOLO counting path.
- Objects365 mega-pretrain pivot before plumbing fixes.

---

## 15. Maintenance

- When a research doc changes, update its **Validated** footer; bump this doc’s roadmap table if P0/P1 shifts.
- **Single task tracker:** [backlog.md](../backlog.md) — do not duplicate open tasks here long-term.
- **CI:** `PYTHONPATH=. HARCHOC_ALLOW_BASE_PYTHON=1 HARCHOC_QUIET=1 python -m unittest discover -s tests` (203+ tests).

*Consolidated 2026-05-29 from validated `docs/research/*`, `EXPERIMENTS.md`, `training_budget.md`, and `backlog.md`.*
