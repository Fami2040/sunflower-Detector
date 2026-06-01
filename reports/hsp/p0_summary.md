# HSP P0 summary (from locked artifacts)

Snapshot from canonical JSON under `reports/hsp/`. Regenerate after re-running the val-tune → test-lock chain.

**Sources:** `dual_metric.json`, `threshold_val.json`, `threshold_test_locked.json`, `split_drift_p0.json`, `eval_val.json`, `eval_test.json`, `s14_maxdet_truncation.json` (2026-05-29 branch `pr/backlog-ci-dataset`).

---

## Operating point

| Split | Locked conf | F1 | Precision | Recall | Count MAE | n images |
|-------|-------------|-----|-----------|--------|-----------|----------|
| Val | **0.15** | 0.642 | 0.630 | 0.654 | **71.0** | 109 |
| Test | **0.15** | 0.610 | 0.629 | 0.592 | **61.3** | 109 |

### Methods (draft — threshold / operating point)

We chose the deployment confidence on the **validation** split only by minimizing per-image **count MAE** over a confidence grid (`--select min_count_mae`; `configs/experiments/threshold_sweep_val.json`), using the same category-aware greedy matcher as counting evaluation (match IoU **0.3**; preds exported at conf **0.001**, NMS IoU **0.3**, `max_det` **3000**). The selected confidence (**0.15** on current `models/best2.pt` artifacts) was **locked** and applied **unchanged** on the held-out **test** split (`threshold_sweep.py --locked-conf-from`; `threshold_test_locked.json`); test rows report the locked operating point only—no re-tuning or IoU/calibration search on test.

At the locked point, false positives remain large on dense trays (**~217 FP/image** on val, **~194** on test; `dual_metric.json` → `operating_point.*_row`), which motivates count-first selection over F1-max alone (**P1-FP-BUDGET**; [eval calibration scan §4](../../docs/research/training_tech_scan_2026_eval_calibration.md#p1--counting-first-tuning-no-new-script)). FP taxonomy and error analysis show **background and localization errors dominate class confusion**; the high FP rate is therefore treated as a class-agnostic operating-point budget, not solved by adding a detect class.

For reviewer **§270 / boundary seeds** (**MS-FUZZY-BOUND**), we do **not** introduce a third YOLO class or relabel protocol. Instead we report **graded trust on two-class detections**: the locked-confidence gate, a post-hoc **low-confidence score band** on exported boxes (default upper bound = locked conf + **0.15**), `ambiguous_summary` and `ambiguous_fp_crosstab` in `error_analysis.py`, and optional **count MAE excluding the band** (**P1-UNCERT-FP** Done). Yao et al. (2025) hierarchical fuzzy **regression** is cited only as an analogy for graded output trust ([explainability literature](../../docs/research/explainability_uncertainty_literature.md); registry `yao2025_hfuzzy`). Qualitative panel: [`fig_ambiguous_panel.png`](../figures/fig_ambiguous_panel.png) (**P2-FIG** Done).

- Conf selected on val (`threshold_val.json`, **`min_count_mae`**) and applied on test via `--locked-conf-from`.
- Test counting MAE 95% CI (bootstrap): **51.3 – 71.3** — confirmed in `threshold_test_locked.json` → `locked.counting_metrics.mae_ci` (point 61.27; low 51.30; high 71.29; n=109); matches `dual_metric.json` test row.
- Val rRMSE @ operating point: **0.172**; test rRMSE: **0.149** (`dual_metric.json`).
- FP per image @ locked conf (**category-aware** match IoU 0.3): val **217.0**, test **193.6** (`dual_metric.json` → `operating_point.val_selected_row` / `test_locked_row`; same rows in sweep `locked` / selected rows).

---

## Calibration (val sweep)

From `threshold_val.json` (isotonic run on val exports):

| Metric | Value |
|--------|-------|
| Mode | **isotonic** (`calibrator`: `isotonic_pava`) |
| ECE (10 bins) | **0.082** (`calibration_metrics.ece`) |
| Score pairs | 184 125 (`calibration.n_pairs`) |

Post-hoc calibration is for sweep/counting diagnostics on val; locked conf **0.15** is unchanged on test.

---

## Split drift (P0 gate)

| Check | Status |
|-------|--------|
| Report `status` | **ok** |
| Acceptance `status` | **ok** (all pairs: train_vs_val, val_vs_test, train_vs_test) |
| Leakage audit | **ok** |

`val_vs_test` KS p-values (width / height / boxes-per-image) all > 0.42; no resplit required before threshold tuning on current frozen splits.

---

## Asymmetric seed distribution (reviewer §96–98)

Policy: [`configs/eval/asymmetric_seed_policy.json`](../../configs/eval/asymmetric_seed_policy.json) (`asymmetric_seed_policy.v1`). Class counts from [`split_drift_p0.json`](split_drift_p0.json).

| Split | Developed | Aborted | n images |
|-------|----------:|--------:|---------:|
| train | **55.1%** | **44.9%** | 875 |
| val | **52.7%** | **47.3%** | 109 |
| test | **55.4%** | **44.6%** | 109 |

Developed seeds outnumber aborted by ~10 percentage points (~**55% / ~45%** of boxes on test). This reflects biological head composition, not deliberate class balancing. Class mix is stable train→test (`train_vs_test` `class_dist_l1` **0.006**).

**Eval policy (Methods / Results draft):** Headline metrics use the frozen **test** split only; **val** selects and locks the global confidence threshold (conf **0.15**) with no test re-tuning. Per-class developed/aborted counts and `cls_confusion` in FP breakdowns come from test `error_analysis` at the locked point; primary claim remains **total** count MAE (**61.3**). Manuscript draft prose: [reviewer gap §10](../../docs/manuscript/reviewer_comments_backlog_gap.md#10-manuscript-draft--asymmetric-seed-distribution-methods--results) (**MS-ASYM-NARR** Done).

---

## Originality vs crop seed-detection peers (reviewer theme #2)

**MS-ORIG** repo draft Done. Introduction bullets + peer cite table: [`originality_contribution_peers.md`](../../docs/manuscript/originality_contribution_peers.md); gap mirror [§2](../../docs/manuscript/reviewer_comments_backlog_gap.md#2-manuscript-draft--originality-vs-crop-seed-detection-introduction). Registry peers: `grainnet2025`, `lwcd_yolo2025` ([`literature_validated.json`](../../docs/manuscript/literature_validated.json)); dense-counting context: [`fp_taxonomy_literature.md`](../../docs/research/fp_taxonomy_literature.md).

**Positioning (one line):** Benchtop **sunflower** developed/aborted counting at ~500 instances/image with **validation-locked test count MAE** and **TIDE-style** FP taxonomy—vs GrainNet/LWCD kernel counters and GWHD field heads; no public sunflower-seed peer benchmark at this density.

---

## Manual counting baseline (reviewer §41–43)

**MS-MANUAL-BASE** repo draft; no timed human study in repo. Manuscript prose + comparison table: [reviewer gap §8](../../docs/manuscript/reviewer_comments_backlog_gap.md#8-manuscript-draft--manual-counting-baseline-methods--results).

**Human protocol (Methods draft):** Single expert per tray image; separate **developed** / **aborted** tallies on zoomable full-res view; **10% blind recount**; budget **~1 person-hour per tray** (planning range **45–75 min**, up to **90 min** with recount). Distinct from [**MS-MANUAL-N50**](#manual-validation-n50-reviewer-theme-17) (*n*=50 density-stratified test audit).

| Modality | Time / image | Count MAE | Throughput (trays/h) |
|----------|-------------:|----------:|---------------------:|
| Expert manual | **45–75 min** *(placeholder)* | **40–120** *(placeholder)* | **0.8–1.3** *(placeholder)* |
| YOLOv8m @ locked conf **0.15** | **15–45 s** *(placeholder)* | **61.3** (CI **51.3–71.3**) | **80–240** *(placeholder)* |

Model MAE from [`dual_metric.json`](dual_metric.json) / [`threshold_test_locked.json`](threshold_test_locked.json); mean GT ≈ **554** seeds/image (~**11%** rMAE-scale error). Replace manual and wall-clock placeholders after a timed study; cite `grainnet2025` for peer reporting style only until then.

> On held-out test trays (*n*=109), model count MAE was **61.3** seeds/image (95% CI **51.3–71.3**) at ~15–45 s/image (GPU placeholder); expert manual throughput is budgeted at ~45–75 min/image (MAE **40–120** placeholder pending timed study).

---

## Manual validation *n*=50 (reviewer theme #17)

**MS-MANUAL-N50** repo draft; no human recount Results in repo yet. Manuscript prose + sampling table: [reviewer gap §17](../../docs/manuscript/reviewer_comments_backlog_gap.md#17-manuscript-draft--manual-validation-n50-methods).

**Methods (draft):** **n=50** images sampled **only** from held-out **test** (*N*=109); excluded from train/val and from threshold locking. **Density-stratified** quotas on GT boxes/image (tertile cuts **492** / **651**; pool **38 / 35 / 36**); sample **17 / 16 / 17**. Where possible, ≤**1** image per **`tray_key`** per stratum (**41** trays in test). Two independent readers + adjudicator; annotation/count QA only—not **MS-MANUAL-BASE** timed throughput.

| Stratum | GT boxes/image | Pool *n* | Sample *n* |
|---------|----------------|--------:|-----------:|
| Low (≤492) | 169–492 (mean 364) | 38 | **17** |
| Mid (493–651) | 494–651 (mean 580) | 35 | **16** |
| High (≥652) | 653–830 (mean 729) | 36 | **17** |
| **Total** | full test mean **554** | **109** | **50** |

Density stats from [`split_drift_p0.json`](split_drift_p0.json) test `labels.boxes_per_image`. Document PRNG seed + image list in supplementary material when recounts are completed.

> Manual validation comprised n=50 test images sampled with density-stratified quotas (tertiles on ground-truth boxes per image; 17/16/17) from the held-out test split only, with at most one image per tray key when possible; independent readers verified counts without influencing the locked operating point.

---

## FP localization vs classification (reviewer §142–145)

Artifacts: [`tide_bucket_summary.json`](tide_bucket_summary.json) (`localization_dominates_classification`: **true**); [`error_test_report.json`](error_test_report.json) → `fp_breakdown`, `tide_bucket_summary`; figure [`fig_error_taxonomy.png`](../figures/fig_error_taxonomy.png) (**P2-FIG** Done).

| TIDE-proxy ΔAP share (test) | Share |
|-----------------------------|------:|
| Miss (FN) | **52.9%** |
| Loc | **27.5%** |
| Bkg | **16.7%** |
| Cls | **2.9%** |

Loc+Bkg vs Cls mass ratio: **15.1×** (`loc_plus_bkg_over_cls_ratio`). At locked conf **0.15**, **27 509** FPs split **62%** localization / **38%** background; **1 825** matched boxes show developed↔aborted confusion (~**6.6%** of **27 477** TPs).

**Methods / Results draft:** TIDE-style error bucketing on test exports; qualitative panel from FP crop manifest. Manuscript prose: [reviewer gap §11](../../docs/manuscript/reviewer_comments_backlog_gap.md#11-manuscript-draft--fp-localization-vs-classification-methods--results) (**MS-FP-LOC-NARR** Done, repo draft; LaTeX paste = user action). Pipeline: [EXPERIMENTS § TIDE buckets](../../docs/EXPERIMENTS.md#tide-bucket-dap--fp-crop-manifest-p1-tide).

> False positives at the locked operating point were overwhelmingly background and localization errors (62% and 38% of 27 509 FPs); developed/aborted confusion affected only 1 825 matched boxes (~15× less mass than Loc+Bkg in TIDE-proxy buckets). Representative error types: `fig_error_taxonomy.png`.

---

## Fuzzy boundary / graded trust (reviewer §270)

Literature: [`yao2025_hfuzzy`](../../docs/manuscript/literature_validated.json) · [explainability § Reviewer cites](../../docs/research/explainability_uncertainty_literature.md#reviewer-cites-ren-2025-yao-2025). Artifacts: [`error_test_report.json`](error_test_report.json) → `ambiguous_summary`, `ambiguous_fp_crosstab`; figure [`fig_ambiguous_panel.png`](../figures/fig_ambiguous_panel.png) (**P2-FIG** Done).

| Metric (test @ locked conf **0.15**) | Value |
|--------------------------------------|------:|
| Ambiguous band | **(0.15, 0.30]** (locked conf + **0.15**) |
| Predictions in band | **19 668** / **56 811** (~**35%**) |
| Ambiguous among FP buckets | **13 050** (~**47%** of **27 509** FPs) |
| Count MAE (headline) | **61.3** |
| Count MAE excl. band (diagnostic) | **214.0** |

**Methods (draft):** Hard developed/aborted labels unchanged; graded trust via val-locked conf gate + post-hoc low-conf score band + `ambiguous_detections` cross-tabbed against TIDE-style FP buckets (`error_analysis.py`; **P1-UNCERT-FP** Done). Not a third detect class or relabel protocol.

**Discussion draft:** Yao et al. (2025) hierarchical fuzzy **regression** (tabular data; registry `yao2025_hfuzzy`) cited **only as analogy** for reporting gradual output trust on two-class detections. Most ambiguous-box mass at the operating point is background or localization error, not developed↔aborted confusion (§11). Manuscript prose: [reviewer gap §15](../../docs/manuscript/reviewer_comments_backlog_gap.md#15-manuscript-draft--fuzzy--hierarchical-boundary-seeds-discussion) (**MS-FUZZY-BOUND** Done, repo draft; LaTeX paste = user action).

> Rather than a third detect class for boundary-ambiguous seeds, we report graded trust on two-class outputs—a val-locked confidence gate plus a low-confidence score band—by analogy to Yao et al. (2025) hierarchical fuzzy regression; most ambiguous-box mass at the operating point was background or localization error, not developed↔aborted confusion (`fig_ambiguous_panel.png`).

---

## Domain adaptation plan (reviewer §306–308 / §360)

Literature: [`yang2024_oct_tl`](../../docs/manuscript/literature_validated.json) · [domain shift synthesis §10](../../docs/research/domain_shift_transfer_literature.md#10-reviewer-cite-yang-et-al-2024-transfer-learning). Per-tray eval: [`domain_eval.json`](../domains/domain_eval.json) (`domain_eval.v1`; **41/52** trays ok on CPU, 11 train-only / no test split). Finetune roadmap: [`finetune_tray_stage1.json`](../../configs/experiments/finetune_tray_stage1.json) / [`finetune_tray_stage2.json`](../../configs/experiments/finetune_tray_stage2.json) via `finetune.py --stage 1|2` (**P1-FINETUNE-LOOP** Done).

| Artifact | Role |
|----------|------|
| `reports/domains/catalog.json` | 52 `tray_key` groups; split membership |
| `data/domains/test_{tray_key}.txt` | Per-tray eval lists (`--write-domain-splits`) |
| `domain_eval.json` | Per-tray mAP50 on test slices; tray spread **~0.18–0.53** vs pooled test ≈0.79 |
| Stage 1 / 2 configs | 25+25 ep; frozen backbone → full unfreeze (`finetune_stage{1,2}.yaml`) |

**Discussion draft:** Cross-tray ranking mAP varies across session slices, supporting staged transfer fine-tuning (Yang 2024 **analogy** for source→target adaptation; TFA/Gandhi freeze schedule) rather than retraining from scratch. Field / multi-site generalization remains future work (**MS-GEN**). Manuscript prose: [reviewer gap §12](../../docs/manuscript/reviewer_comments_backlog_gap.md#12-manuscript-draft--domain-adaptation-plan-discussion) (**MS-DOMAIN-ADAPT** Done, repo draft; LaTeX paste = user action). Full GPU stage1→stage2 + before/after tray count MAE not yet run.

> Cross-tray ranking mAP varied substantially across held-out session slices (supplementary domain catalog), motivating staged transfer fine-tuning on new trays—by analogy to Yang et al. (2024) and TFA/Gandhi YOLO freeze schedules—rather than training from scratch; field and multi-site validation remain future work.

---

## S14 · `max_det=300` negative control

Artifact: [`s14_maxdet_truncation.json`](s14_maxdet_truncation.json) (`s14_maxdet_control.v1`). Same `models/best2.pt`, test split, locked conf **0.15** (val); only export `max_det` differs.

| `max_det` | Locked count MAE | Locked recall | Export cap hit |
|-----------|------------------|---------------|----------------|
| **3000** (P0) | **61.3** | 0.592 | 6/109 images @ 3000 preds |
| **300** (S14) | **261.7** (+200.4) | 0.365 | **109/109** @ 300 preds |

Mean GT ≈ **554** boxes/image; **94.5%** of test images have GT > 300. Truncation at `max_det=300` is material for counting-first metrics—use **3000** for smokes S0–S13 and manuscript eval; S14 documents the old `eval.max_det: 300` failure mode.

---

## Detection mAP

**Primary metric for the manuscript is test count MAE** (table above), not val or test ranking mAP.

**Val vs test mAP (reviewer):** Peak **training-time** val mAP50 (≈0.97; early stop on `data/splits/val.txt`) is **not** generalization. Full evidence + split narrative (**MS-SPLIT-MAPNARR** Done): [val_test_map_gap.md §5](../../docs/manuscript/val_test_map_gap.md#5-manuscript-draft--val-map-vs-test--results--22). `dual_metric.json` → `metric_roles` / per-row `split_role_label`.

### Manuscript draft — paste into §2.2 (**MS-VAL-MAPDOWN** Done)

**Repo draft complete;** external LaTeX paste is a user action. Canonical block: [val_test_map_gap.md §5 — Paste into §2.2](../../docs/manuscript/val_test_map_gap.md#paste-into-22-latex).

> Peak validation mAP50 during training (≈0.97; early stop on validation) informed checkpoint selection only—not field performance. Test ranking mAP50 ≈0.79 under the training convention; frozen splits pass distributional checks ([`split_drift_p0.json`](split_drift_p0.json); **MS-SPLIT-MAPNARR**). Primary metric: **test count MAE 61.3** (95% CI **51.3–71.3**, *n*=109) at conf **0.15** locked on val; val MAE **71.0** for threshold transparency only. Detection mAP is supplementary.

`eval_val.json` / `eval_test.json` are **export-only** (`mAP50` null there). Test ranking mAP lives in [`eval_test_map.json`](eval_test_map.json) (merged into `dual_metric.json` test row; val detection row empty until optional `eval_val_map.json`).

**Merge wiring (2026-05-29):** `experiment.py dual-metric` / `harchoc/dual_metric_report.py` pull mAP from a sibling **`eval_test_map.json`** (or `--eval-test-map`) when the primary eval is export-only. Re-run dual-metric after creating the map file.

To populate test mAP (omit `--export-only`; GPU ~minutes @ `imgsz=1280`; needs ~free 8 GiB VRAM):

```bash
mamba run -n harchoc python scripts/eval.py \
  --weights models/best2.pt \
  --split-file data/splits/test.txt \
  --imgsz 1280 --max-det 3000 \
  --out reports/hsp/eval_test_map.json
mamba run -n harchoc python scripts/experiment.py dual-metric \
  --eval-val reports/hsp/eval_val.json --eval-test reports/hsp/eval_test.json \
  --sweep reports/hsp/threshold_val.json --sweep-test reports/hsp/threshold_test_locked.json \
  --error-val reports/hsp/error_val.json --error-test reports/hsp/error_test.json \
  --out reports/hsp/dual_metric.json
```

**2026-05-29:** map eval was **not** written — `eval.py` hit **CUDA OOM** (8 GiB GPU, ~6.2 GiB already in use by another process). Counting/locked metrics are unaffected.

Do not run map eval during routine P0 regen unless you need fresh mAP.

---


---

## Cross-tray generalization (reviewer #4 / MS-GEN)

Artifacts: [`domain_eval.json`](../domains/domain_eval.json) (`domain_eval.v1`), [`domain_count_mae.json`](../domains/domain_count_mae.json) (`domain_count_mae.v1`), [`catalog.json`](../domains/catalog.json) (`domain_metadata_tags` scaffold — variety/maturity/lighting/site **TBD**). **41/52** trays have held-out test slices; count MAE uses the **same** val-locked conf as pooled test (**0.15**; `threshold_val.json`), exports @ conf **0.001**, match IoU **0.3**.

| Metric (per-tray test slices, *n*=41) | Value |
|----------------------------------------|------:|
| Count MAE mean | **67.9** |
| Count MAE median | **49.5** |
| Count MAE range | **17.0 – 243.5** |
| mAP50 range | **0.005 – 0.74** |
| mAP50 median | **0.42** |
| Pooled test count MAE (reference) | **61.3** |
| Pooled test mAP50 (reference) | **≈0.79** |

**Discussion draft:** Held-out tray slices show substantial spread in both ranking mAP and count MAE at the val-locked confidence; multi-site, field-light, and variety/maturity claims remain outside the present benchtop corpus (**P1-DOMAIN-TAGS** tags TBD in catalog). Manuscript prose: [reviewer gap §4](../../docs/manuscript/reviewer_comments_backlog_gap.md#4-manuscript-draft--generalization--cross-tray-counting-discussion) (**MS-GEN** repo draft **Done**; LaTeX paste = user action).

> Held-out tray slices showed substantial spread in both ranking mAP and count MAE at the val-locked confidence (supplementary `domain_eval.v1`), while multi-site and field-light validation remain outside the present benchtop dataset.

Regenerate: `DATASET_ROOT=data/raw/extracted/dataset mamba run -n harchoc python scripts/eval_domains.py --merge-tray-count-mae --out reports/domains/domain_eval.json --device cpu`.

## Two-stage deploy (reviewer line 415)

Literature: [`alshehri2025_uav`](../../docs/manuscript/literature_validated.json) · parity: [`deploy_hsp_parity.json`](deploy_hsp_parity.json) (`deploy_hsp_parity.v1`; **R-SCI-2** Done). Wiring: [HSP_BASELINE_MODELS § Two-stage](../../docs/HSP_BASELINE_MODELS.md#two-stage-deploy-analogy-alshehri-2025).

| Stage | Production deploy | Manuscript HSP eval |
|-------|-------------------|---------------------|
| 1 — content gate | `classifier.pt` (sunflower vs other; conf ≥ **0.5**) | Not used (all split images are heads) |
| 2 — seed detection | `best2.pt` + SAHI (slice **500**, overlap **0.35**) | `best2.pt` full-frame @ `imgsz` **1280** |
| Counting conf | Post-filter **0.06** / **0.04** per class | Val-locked **0.15** (`threshold_val.json`) |

Default deploy per-class thresholds are **~0.09–0.11** below locked HSP conf (`deploy_hsp_parity.json` → `comparison.deploy_vs_hsp_locked_delta_*`). Optional `HARCHOC_LOCKED_CONF*` / `experiment.py deploy-parity --locked-conf-from` aligns bot filters for parity checks.

**Discussion draft:** Alshehri et al. (2025) cite is a **conceptual** two-stage robustness analogy (UAV action recognition, not seed boxes); production uses gate→detect, manuscript uses single-stage full-frame counting. Manuscript prose: [reviewer gap §14](../../docs/manuscript/reviewer_comments_backlog_gap.md#14-manuscript-draft--two-stage-deploy-discussion) (**MS-DEPLOY-2STG** Done, repo draft; LaTeX paste = user action).

> Field deployment applies a two-stage sunflower gate plus SAHI seed detection (Alshehri et al., 2025, by analogy), whereas reported counting metrics use full-frame evaluation on benchtop heads at a validation-locked confidence without the deploy classifier.

---

## Deploy vs manuscript (threshold summary)

**Manuscript / P0:** full-frame Ultralytics export @ `conf=0.001`, NMS IoU **0.3**, `imgsz` **1280**, locked operating conf **0.15** (val-tuned).

**Deploy (Telegram / SAHI):** two-stage path above; sliced inference — not the locked **0.15** point without `HARCHOC_LOCKED_CONF*`.

Details: [docs/HSP_BASELINE_MODELS.md](../../docs/HSP_BASELINE_MODELS.md) (deploy vs `eval.py` paths).

---

## Weights & exports

- Weights: `models/best2.pt` (sha256 in `threshold_test_locked.json` meta).
- Match IoU for sweeps: **0.3** (category-aware).
- Export hyperparams: conf **0.001**, NMS IoU **0.3**, `max_det` **3000**, `imgsz` **1280** (see [README.md](README.md)).

---

## Reproducibility bundle

Index: [`configs/experiments/manuscript_repro_bundle.json`](../../configs/experiments/manuscript_repro_bundle.json). One-command regen: `mamba run -n harchoc python scripts/experiment.py repro` ([EXPERIMENTS § bundle](../../docs/EXPERIMENTS.md#manuscript-reproducibility-bundle)).

| Kind | Path |
|------|------|
| Train recipe | `configs/experiments/train_yolov8m_baseline.json` |
| Eval / exports | `configs/experiments/eval_hsp_baseline.json` → `eval_*.json`, `gt_*.json`, `preds_*.json` |
| Thresholds | `threshold_sweep_val.json`, `threshold_sweep_test_locked.json` → `threshold_*.json` |
| Errors + table | `error_analysis_*.json` → `error_*.json`; `dual_metric.json` |
| Split SHA256 | `data/splits/{train,val,test}.txt` (in bundle `repo_splits.files`) |

---

## Not in P0 headline metrics

- `rtdetr_smoke_15ep.json`: GPU probe **ok**; 15-ep train **`train_in_progress`** (6/15 ep in `results.csv` as of 2026-05-29); val **nan** ep 2–4 only—see `rtdetr_smoke_notes.md` (smoke-acceptable).
- Obsolete paths live under `reports/_archive/`; do not cite for P0 (see [reports/README.md](../README.md)).
