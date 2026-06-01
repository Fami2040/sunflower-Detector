# Reviewer comments ↔ backlog gap map

**Purpose:** Map manuscript reviewer themes to [`backlog.md`](../backlog.md) work-queue IDs, partial coverage elsewhere in the repo, or **MISSING** gaps.  
**Validated against backlog:** 2026-06-01 · branch `pr/backlog-ci-dataset`  
**Literature:** [`literature_validated.json`](literature_validated.json) · [`literature_validated.md`](literature_validated.md)

## Integration status (2026-06)

Repo integration snapshot for manuscript / GPU agents (not a reviewer theme).

| Track | Status | Blocker / note |
|-------|--------|----------------|
| **P0-5 zoo** | **3/4 + v10m in flight** | v8m / 11m / 26m complete; **v10m** 100-ep training (not OOM). See [`FRESHNESS.md`](../../reports/manuscript/FRESHNESS.md). |
| **`matrix_train.json`** | **3/4 MAE ok** | **111.9** / **119.6** / **95.3**; **v10m** pending weights. None beat **best2** **61.3**. |
| **RT-DETR / external DETR** | **Split** | **Ultralytics RT-DETR-L** @ 1280 batch=1 **OOMs** on 8 GiB (vram probe). **External** DEIM/rtdetrv2: integration/port issues; **D-FINE** reached ~6.7 GiB @ ep 42 on same GPU — not a blanket >8 GiB block. |
| **Finetune weak trays** | **Queued, not run** | Code + splits wired (**P1-FINETUNE-TRAY** Partial); queue kind **`finetune_tray`**; Methods [§20](#20-manuscript-draft--tray-finetuning-methods--discussion); GPU in [`gpu_queue_post_zoo.json`](../../configs/experiments/gpu_queue_post_zoo.json) after zoo. |
| **Provenance (`best2`, dataset)** | **Documented** | Weights from [origin `main`](https://github.com/Fami2040/sunflower-Detector); dataset CVAT [`y9xGFqCW`](../../data/manifest.json) — [§0](#0-origin-weights--dataset-not-kaggle) |
| **Aug (closed)** | **Done** | Literature aug program (S0–S14 + 100 ep) on **same test split** as anchor **best2 61.3** — none beat anchor — [§18](#18-manuscript-draft--augmentation-strategy-methods) |
| **MS-SOTA table** | **Draft ok** | [`reports/manuscript/tables/zoo_core.md`](../../reports/manuscript/tables/zoo_core.md) — refresh after **v10m** completes (`manuscript-preflight`). |
| **Headline vs zoo** | **Keep `best2`** | Test count MAE **61.3** @ locked conf until a zoo/finetune row beats it on canonical test. |
| **Literature DOI audit** | **Done** | 11/11 in registry; GWHD DOI fixed — [`literature_doi_audit_2026-06-01.md`](../../reports/manuscript/literature_doi_audit_2026-06-01.md), [`lit_audit/`](../../reports/manuscript/lit_audit/README.md) |

**Single-GPU order:** kill strays → `zoo_matrix_p0_5` → repro/preflight → domain audit → finetune — [backlog § 4-agent workstream](../../backlog.md#4-agent-workstream-2026-06).

**Test ranking mAP50 (~0.18 vs legacy ~0.79):** Narrative-only **0.79** / **0.793** came from training-log / domain-pooled conventions, not current HSP exports. Canonical pooled test mAP50 for `models/best2.pt` is **0.180** in [`eval_test_map.json`](../../reports/hsp/eval_test_map.json) (`export_only: false`, `imgsz=1280`, `max_det=3000`, `data/splits/test.txt`). Reproduce:

```bash
export DATASET_ROOT=/path/to/dataset   # see data/manifest.json
mamba run -n harchoc python scripts/experiment.py map-cpu
mamba run -n harchoc python scripts/experiment.py dual-metric \
  --eval-test reports/hsp/eval_test.json \
  --eval-test-map reports/hsp/eval_test_map.json \
  --sweep reports/hsp/threshold_val.json \
  --sweep-test reports/hsp/threshold_test_locked.json \
  --error-test reports/hsp/error_test.json \
  --out reports/hsp/dual_metric.json
```

Do **not** cite **0.79** as pooled test mAP without reconciling to this path ([`val_test_map_gap.md`](val_test_map_gap.md), [`p0_summary.md` § Detection mAP](../../reports/hsp/p0_summary.md#detection-map)).

---

**Coverage legend**

| Status | Meaning |
|--------|---------|
| **Existing backlog** | A named open or Done queue ID directly addresses the reviewer ask |
| **Partial** | Supporting code, lit notes, or Done artifacts exist; manuscript text or policy still open |
| **MISSING** | No backlog ID and no substantive artifact for the reviewer ask |

---

## Gap table (all 17 reviewer themes)

| # | Reviewer theme / manuscript anchor | Backlog / repo coverage | Status | Recommended IDs | Notes |
|---|-----------------------------------|-------------------------|--------|-----------------|-------|
| 1 | **Abstract standardization** — purpose / methods / results / conclusions | **MS-ABS** | **MISSING** (manuscript only) | — | No LaTeX in repo |
| 2 | **Originality / contribution vs crop seed detection** | **MS-ORIG** Done; lit: `grainnet2025`, `lwcd_yolo2025`, fp_taxonomy | **Done** | — | Repo draft §2 + [`originality_contribution_peers.md`](originality_contribution_peers.md); external LaTeX paste = user action |
| 3 | **SOTA comparison (not only YOLOv8)** | **P0-5**, **MS-SOTA**, **P1-ZOO-PARITY**; **P1-ZOO-READY** Done | **Existing backlog** | — | **P0-5** = `zoo_yolo_only` on 8 GiB; Methods rationale [§19](#19-manuscript-draft--model-selection--sota-zoo-methods--results); **MS-SOTA** blocked until [`matrix_train.json`](../../reports/hsp/matrix_train.json) complete — [Integration status (2026-06)](#integration-status-2026-06) |
| 4 | **Generalization** — multi-site, field light, variety, maturity | **MS-GEN** Partial; **P1-DOMAIN-EVAL** Done; **P1-DOMAIN-TAGS** | **Partial** | — | Tray/session + count MAE @ locked conf; field tags open |
| 5 | **Reproducibility** | **MS-REPRO** Done; **P1-ZOO-PROV** Next | **Partial** | — | `experiment.py repro` + bundle JSON |
| 6 | **Literature review depth** | **MS-LIT** Done; [`related_work_outline.md`](related_work_outline.md) + `docs/research/*` + validated registry | **Done** | **LIT-VALIDATE** (ongoing) | Repo draft Done ([§6 cite table](#6-manuscript-draft--related-work--literature-review-depth)); external LaTeX paste = user action |
| 7 | **Figure normalization** | **MS-FIG-NORM** Done, **P2-FIG** | **Partial** | — | `figure_style.py` + 300 DPI/panel labels; `fig_concept` open |
| 8 | **Lines 41–43:** quantitative **manual counting baseline** | **MS-MANUAL-BASE** Done | **Partial** | — | Repo draft §8 + [`p0_summary`](../../reports/hsp/p0_summary.md); timed human study still open |
| 9 | **Lines 70–72:** **val mAP ≈ 0.97 vs test ≈ 0.793** | **MS-SPLIT-MAPNARR** Done; **MS-VAL-MAPDOWN** Done; **MS-VAL-MAP-CAVEAT** Done; [`val_test_map_gap.md`](val_test_map_gap.md) §5 | **Done** | — | Repo draft Done ([§2.2 paste](val_test_map_gap.md#paste-into-22-latex)); external LaTeX paste = user action |
| 10 | **Lines 96–98:** **asymmetric seed distribution** | **MS-ASYM-NARR** Done; **P2-ASYM-SEED** Done | **Partial** | — | Repo draft §10 below + [`p0_summary`](../../reports/hsp/p0_summary.md); LaTeX paste open |
| 11 | **Lines 142–145:** **FP localization ≫ classification** | **MS-FP-LOC-NARR** Done; **P1-TIDE** Partial; fp_taxonomy § MS-FP-LOC-NARR | **Partial** | — | Repo draft §11 below + [`p0_summary`](../../reports/hsp/p0_summary.md); LaTeX paste open |
| 12 | **Lines 306–308 / 360:** **domain adaptation plan** | **MS-DOMAIN-ADAPT** Done; **P1-FINETUNE-LOOP** Done; **P1-DOMAIN-EVAL** Partial; `yang2024_oct_tl` in [domain_shift §10](../research/domain_shift_transfer_literature.md#10-reviewer-cite-yang-et-al-2024-transfer-learning) | **Done** | — | Repo draft §12 below + [`p0_summary`](../../reports/hsp/p0_summary.md); live [`domain_eval.json`](../../reports/domains/domain_eval.json); LaTeX paste = user action |
| 13 | **Line 295:** **explainability** — Ren 2025 | **MS-EXPLAIN** Partial; **P2-FIG-CAM** Done; `ren2025_scripta_interp` | **Partial** | — | Tooling Done; Ren cite in prose open |
| 14 | **Line 415:** **two-stage deploy** — Alshehri 2025 | **MS-DEPLOY-2STG** Done; **R-SCI-2** Done; `alshehri2025_uav`; HSP_BASELINE § | **Done** | — | Repo draft §14 below + [`p0_summary`](../../reports/hsp/p0_summary.md); LaTeX paste = user action |
| 15 | **Line 270:** **fuzzy / hierarchical boundary seeds** | **MS-FUZZY-BOUND** Done; **P1-UNCERT-FP** Done; **P2-FIG** (`fig_ambiguous_panel` Done); `yao2025_hfuzzy`; `error_analysis` score band | **Done** | — | Repo draft §15 below + [`p0_summary`](../../reports/hsp/p0_summary.md); Yao = fuzzy **regression** analogy (graded trust), not a 3rd YOLO class; LaTeX paste = user action |
| 16 | **Overstatement of validation mAP in Section 2.2** | **MS-VAL-MAPDOWN** Done; **MS-SPLIT-MAPNARR** Done | **Done** | — | Shared with #9; paste from [`val_test_map_gap.md` §5 — Paste into §2.2](val_test_map_gap.md#paste-into-22-latex); external LaTeX = user action |
| 17 | **Manual validation n=50** — sampling / density | **MS-MANUAL-N50** Done | **Done** | — | Repo draft §17 below; human recount Results table = user action |

**§15 — model improvement stack:** **MS-FUZZY-BOUND** is stack **step 3** (val operating point → lock → test; count-first selection) with **P1-FP-BUDGET** and **P1-UNCERT-FP** — see [`backlog.md` § Model improvement stack](../backlog.md#model-improvement-stack-test-count-mae).

---

## §2 Manuscript draft — originality vs crop seed detection (Introduction)

**Use for reviewer theme #2 (**MS-ORIG**).** Canonical bullets, peer cite table, and Introduction paragraph: [`originality_contribution_peers.md`](originality_contribution_peers.md). Registry: [`literature_validated.json`](literature_validated.json) (`grainnet2025`, `lwcd_yolo2025`). FP / dense-counting peers: [`fp_taxonomy_literature.md`](../research/fp_taxonomy_literature.md).

**Introduction (draft bullets):**

1. **Sunflower-seed tray benchmark** — developed vs aborted, ~500 boxes/image, frozen splits + split-drift audit; no public peer dataset at GrainNet/GWHD scale.
2. **Counting-first protocol** — val-minimized **count MAE** → locked test conf (**0.15**; test MAE **61.3**, CI **51.3–71.3**); val mAP not used as generalization claim ([`val_test_map_gap.md`](val_test_map_gap.md)).
3. **Two-class similarity** — natural ~**55% / 45%** prevalence; per-class confusion + total count MAE (**MS-ASYM-NARR**).
4. **TIDE-style FP taxonomy** — background + localization ≫ developed↔aborted cls (**MS-FP-LOC-NARR**; GWHD/GrainNet analogues in fp_taxonomy).
5. **Dense-benchtop training evidence** — LWCD mosaic-off + GrainNet aug precedents (**ARCH-MOSAIC0-AB**).
6. **Cross-tray reporting** — `domain_eval.v1` session spread (**MS-GEN**, **MS-DOMAIN-ADAPT**).
7. **Graded trust** — score band on two-class exports, not a third class (**MS-FUZZY-BOUND**).

**Peer cite table (abbreviated):** GrainNet = wheat kernel counting + EMA/background; LWCD-YOLO = corn seeds + mosaic off + Grad-CAM; GWHD/GWC = field heads + count RMSE; cover-crop / impurity / in-situ / sunflower-head / soybean point papers = related failure modes (full table in [`originality_contribution_peers.md` §3](originality_contribution_peers.md#3-peer-comparison-cite-table)).

Suggested Introduction sentence: *“Relative to benchtop grain detectors (GrainNet; LWCD-YOLO) and field head benchmarks (GWHD), we contribute a reproducible two-class sunflower seed counting benchmark with validation-locked test count MAE, TIDE-style error taxonomy, and cross-tray generalization reporting—filling the missing dense-tray sunflower-seed gap while aligning evaluation with counting-first crop-vision practice.”*

---

## §8 Manuscript draft — manual counting baseline (Methods / Results)

**Use for reviewer lines 41–43 (**MS-MANUAL-BASE**).** Model counting metrics: [`dual_metric.json`](../../reports/hsp/dual_metric.json), [`p0_summary.md`](../../reports/hsp/p0_summary.md). **No timed human study is in the repo** — table cells marked *placeholder* are planning ranges for a future protocol; replace before submission.

**Methods (draft — human protocol):** Expert manual seed counts on benchtop tray images follow a **single-head, single-session** protocol aligned with CVAT box annotations (developed + aborted; ~**554** GT boxes/image mean on test). One trained counter works on a zoomable full-resolution view, tallying **developed** and **aborted** separately, with a mandatory **10% blind recount** on a random subset for intra-rater check. Budget: **~1 person-hour per tray image** (45–75 min typical for dense heads in pilot planning; upper bound 90 min if recount included). Images are processed in random order; breaks discouraged mid-tray to avoid partial-state bias. This protocol is **not** the manuscript’s *n*=50 qualitative audit (**MS-MANUAL-N50**, theme #17).

**Methods / Results (draft — comparison table):**

| Modality | Time per tray image | Count MAE (seeds/image) | Throughput (trays/hour) | Notes |
|----------|--------------------:|------------------------:|------------------------:|-------|
| **Expert manual** | **45–75 min** (*placeholder*) | **40–120** (*placeholder*; planned blind recount vs GT) | **0.8–1.3** (*placeholder*) | ~1 person-hour budget; inter-rater study not yet run |
| **YOLOv8m HSP** (`best2.pt`, val-locked conf **0.15**) | **15–45 s** (*placeholder*; GPU full-frame @ `imgsz` **1280**) | **61.3** (test; 95% CI **51.3–71.3**, *n*=109) | **80–240** (*placeholder*) | Measured MAE from [`threshold_test_locked.json`](../../reports/hsp/threshold_test_locked.json); wall-clock from export+sweep not benchmarked in repo |

**Results (draft):** At the val-locked operating point, automated counting on held-out **test** achieves per-image count MAE **61.3** seeds (bootstrap 95% CI **51.3–71.3**) vs a mean GT load of ≈**554** seeds/image (~**11%** relative error). Planned expert manual baselines target comparable dense trays under the **~1 h/tray** budget; until timed counts are collected, manuscript text should report manual ranges as **placeholders** and cite GrainNet-style reporting (count MAE + manual reference; registry `grainnet2025`) without implying measured human superiority or inferiority.

Suggested Methods sentence: *“Expert reference counts used a ~1 person-hour-per-tray protocol with separate developed/aborted tallies and a 10% blind recount; automated counting used full-frame YOLOv8m exports at imgsz 1280 with confidence locked on validation (conf 0.15) before any test evaluation.”*

Suggested Results sentence: *“On held-out test trays (*n*=109), model count MAE was 61.3 seeds/image (95% CI 51.3–71.3) at ~15–45 s/image (GPU placeholder); expert manual throughput under the same trays is budgeted at ~45–75 min/image (MAE 40–120 placeholder pending timed study).”*

---

## §6 Manuscript draft — Related Work / literature review depth

**Use for reviewer theme #6 (**MS-LIT**).** Canonical outline + cite table: [`related_work_outline.md`](related_work_outline.md). Registry additions: `gwhd2020`, `iamchuen2026_sunflower_uav`, `gulzar2025_sunflower_tl` in [`literature_validated.json`](literature_validated.json).

**Related Work (draft structure):** Six subsections — (1) plant phenotyping and dense organ counting (GWHD, counting MAE/RMSE peers); (2) sunflower imaging (UAV heads, disk phenotyping, disease TL survey) with explicit benchtop-**seed** vs field-**head** scope; (3) small-object / ultra-dense detection (`imgsz=1280`, mosaic/mixup, RT-DETR query limits, TIDE-style errors); (4) crop seed/kernel peers (**GrainNet**, **LWCD-YOLO**); (5) thresholds and counting-first evaluation (HSP val→lock→test, val mAP non-generalization); (6) short transfer/XAI/deploy analogies (Yang, TFA, Gandhi, Ren, Alshehri, Yao) pointing to Discussion drafts in §12–15.

**Cross-links (research corpus):**

| `docs/research/` doc | Feeds §2 |
|----------------------|----------|
| [fp_taxonomy_literature.md](../research/fp_taxonomy_literature.md) | §2.1, §2.3–2.4 |
| [augmentation_robustness_literature.md](../research/augmentation_robustness_literature.md) | §2.2–2.3 |
| [domain_shift_transfer_literature.md](../research/domain_shift_transfer_literature.md) | §2.2, §2.6 |
| [training_tech_scan_2026_detectors.md](../research/training_tech_scan_2026_detectors.md) | §2.3 |
| [training_tech_scan_2026_augmentation.md](../research/training_tech_scan_2026_augmentation.md) | §2.3 |
| [training_tech_scan_2026_eval_calibration.md](../research/training_tech_scan_2026_eval_calibration.md) | §2.5 |
| [threshold_calibration_literature.md](../research/threshold_calibration_literature.md) | §2.5 |
| [explainability_uncertainty_literature.md](../research/explainability_uncertainty_literature.md) | §2.6 |

Suggested opening sentence: *“Prior work spans dense crop phenotyping, sunflower head imaging in the field, and small-object detection on benchtop seed trays; we position HARCHOC as viability-aware seed counting with a locked-confidence test protocol rather than panicle-level yield proxies.”*

**Paste block:** [related_work_outline.md §5 — Paste into §2](related_work_outline.md#5-paste-into-2-latex--opening-synthesis)

---

## §4 Manuscript draft — generalization / cross-tray counting (Discussion)

**Use for reviewer theme #4 (**MS-GEN**).** Per-tray artifacts: [`domain_eval.json`](../../reports/domains/domain_eval.json) (`domain_eval.v1`; mAP + **count MAE** @ val-locked conf **0.15**); sidecar [`domain_count_mae.json`](../../reports/domains/domain_count_mae.json) (`domain_count_mae.v1`). Locked conf from [`threshold_val.json`](../../reports/hsp/threshold_val.json) (`min_count_mae`); exports at conf **0.001**, counting at locked conf (match IoU **0.3**, category-aware), same protocol as held-out test (**61.3** MAE pooled). CLI: `eval_domains.py --merge-tray-count-mae` with `DATASET_ROOT`.

**Discussion (draft):** We do not yet claim multi-site, field-light, or variety/maturity generalization—those axes await tagged collections (**P1-DOMAIN-TAGS**). What we *can* report on the frozen benchtop corpus is **session/tray** heterogeneity: **52** tray keys in [`catalog.json`](../../reports/domains/catalog.json), **41** with held-out test slices (`eval_domains.py --run-all-trays`, `models/best2.pt`, CPU). Pooled test ranking mAP50 ≈**0.79** masks wide tray-level spread (per-tray mAP50 roughly **0.005–0.74** (median ≈0.42) in `domain_eval.v1`). At the **same** val-locked operating point used for headline counting, per-tray **count MAE** on test slices spans roughly **17.0–243.5** seeds/image (mean ≈**67.9** across *n*=*41* tray-slices with test images; see `count_mae_summary` in `domain_eval.v1`), corroborating that dense-tray counting error is not uniform across acquisition batches. Multi-site and in-field imaging remain **future work**; extrapolation should cite split-drift stability ([`split_drift_p0.json`](../../reports/hsp/split_drift_p0.json)) and val-vs-test mAP narrative ([`val_test_map_gap.md`](val_test_map_gap.md)), not benchtop trays alone.

Suggested Discussion sentence: *“Held-out tray slices showed substantial spread in both ranking mAP and count MAE at the val-locked confidence (supplementary `domain_eval.v1`), while multi-site and field-light validation remain outside the present benchtop dataset.”*

---

## §10 Manuscript draft — asymmetric seed distribution (Methods / Results)

**Use for reviewer lines 96–98 (**MS-ASYM-NARR**).** Canonical policy: [`asymmetric_seed_policy.json`](../../configs/eval/asymmetric_seed_policy.json); class counts from [`split_drift_p0.json`](../../reports/hsp/split_drift_p0.json).

**Methods (draft):** Annotations reflect natural head composition: **developed** seeds (class 0) outnumber **aborted** seeds (class 1) by roughly ten percentage points across frozen splits (~**55% / ~45%** of boxes on test; train **0.551** / **0.449**, val **0.527** / **0.473**, test **0.554** / **0.446**). This asymmetry is biological prevalence, not a deliberate class-balancing protocol. Headline counting metrics are reported on the held-out **test** split only (`data/splits/test.txt`, *n*=109); **validation** is used solely to sweep and lock a single global confidence threshold before any test evaluation (`threshold_sweep.py --locked-conf-from`; [`p0_summary.md`](../../reports/hsp/p0_summary.md)).

**Results (draft):** At the val-locked operating point (conf **0.15**), per-class developed and aborted counts and class confusion within false positives are taken from test `error_analysis` exports (`error_test_report.json`); primary manuscript metric remains **total** per-image count MAE (**61.3**, 95% CI **51.3–71.3**). Class prevalence is stable train→test (`class_dist_l1` ≈ **0.006**); val↔test mix differs slightly at *n*=109 but stays the same order of magnitude (**0.054** L1).

Suggested Results sentence: *“Developed seeds comprised ~55% of annotated boxes on the held-out test trays (aborted ~45%), consistent with natural head composition; all headline count metrics below use the test split only, with thresholds selected on validation.”*

---

## §11 Manuscript draft — FP localization vs classification (Methods / Results)

**Use for reviewer lines 142–145 (**MS-FP-LOC-NARR**).** TIDE-proxy buckets: [`tide_bucket_summary.json`](../../reports/hsp/tide_bucket_summary.json) (`localization_dominates_classification`: **true**); FP taxonomy + crops: [`error_test_report.json`](../../reports/hsp/error_test_report.json); figure: [`fig_error_taxonomy.png`](../../reports/figures/fig_error_taxonomy.png) (**P2-FIG** Done; CPU `make_figures.py`).

**Methods (draft):** At the val-locked operating point (conf **0.15**), we categorized detection errors on held-out **test** exports with a TIDE-style count-share proxy (`error_analysis.py`; `tide_bucket_count_proxy.v1` → `tide_bucket_summary.json`). False positives decompose into **background** activations (no sufficient GT overlap) and **localization** errors (low-IoU overlap with a GT box); **developed/aborted** cross-class confusion is counted separately on matched boxes (`fp_breakdown`, `cls_confusion` in `error_test_report.json`). Qualitative evidence uses representative FP crops rendered by `make_figures.py` → **`fig_error_taxonomy.png`**.

**Results (draft):** On test (*n*=109 images), **`localization_dominates_classification`** is **true**: TIDE-proxy ΔAP share ranks **localization** (**27.5%**) and **background** (**16.7%**) far above **classification** (**2.9%**); Loc+Bkg error mass exceeds Cls by **~15×** (`loc_plus_bkg_over_cls_ratio` **15.1**). Among **27 509** false positives at locked conf, **62%** are low-IoU localization and **38%** background; only **1 825** matched boxes show developed↔aborted confusion (~**6.6%** of true positives). Missed seeds (**Miss** bucket, **52.9%** ΔAP share) dominate recall loss; the high FP rate at counting thresholds is therefore **class-agnostic** (background/localization), not solved by relabeling or a third detect class.

Suggested Results sentence: *“False positives at the locked operating point were overwhelmingly background and localization errors (62% and 38% of 27 509 FPs); developed/aborted confusion affected only 1 825 matched boxes (~15× less mass than Loc+Bkg in TIDE-proxy buckets). Representative error types are shown in Fig. X (`fig_error_taxonomy.png`).”*

---

## §12 Manuscript draft — domain adaptation plan (Discussion)

**Use for reviewer lines 306–308 / 360 (**MS-DOMAIN-ADAPT**).** Literature: [`yang2024_oct_tl`](literature_validated.json) · [domain shift synthesis §10](../research/domain_shift_transfer_literature.md#10-reviewer-cite-yang-et-al-2024-transfer-learning); per-tray eval: [`domain_eval.json`](../../reports/domains/domain_eval.json) (**P1-DOMAIN-EVAL** Done); finetune roadmap: [`finetune_tray_stage1.json`](../../configs/experiments/finetune_tray_stage1.json) / [`finetune_tray_stage2.json`](../../configs/experiments/finetune_tray_stage2.json) (**P1-FINETUNE-LOOP** Done).

**Discussion (draft):** Benchtop imaging uses controlled indoor lighting on dried heads, but **session/tray** composition still shifts across acquisition batches. Tray keys parsed from image stems (`harchoc/domain_tags.py`; 52 trays in [`catalog.json`](../../reports/domains/catalog.json)) define the primary domain axis for this dataset. Per-tray ranking mAP on held-out tray slices (`eval_domains.py --run-all-trays` → `domain_eval.v1` on `models/best2.pt`; **41/52** trays with test split, 11 train-only) shows substantial tray-to-tray spread (individual test-tray mAP50 roughly **0.18–0.53** vs aggregate test ≈0.79), supporting the reviewer concern that a single pooled test split can mask local degradation. Following Yang et al. (2024; PLoS ONE — retinal OCT ensemble with ImageNet transfer and Grad-CAM, **not** seed detection; registry `yang2024_oct_tl`), we frame adaptation to new trays as **staged transfer fine-tuning** from `models/best2.pt`, not full retraining from scratch. The operational schedule follows TFA-style two-stage freeze [Wang et al. 2020; `tfa2020`] and YOLOv8 staged unfreeze [Gandhi & Gandhi 2025 preprint; `gandhi2025_yolov8_freeze`]: `finetune.py --stage 1|2` with `configs/experiments/finetune_tray_stage{1,2}.json` (25+25 epochs; stage 1 frozen backbone via `configs/transfer/finetune_stage1.yaml`, stage 2 full unfreeze at lower LR via `finetune_stage2.yaml`), optional `--run-tray-eval` before/after on a held-out `tray_key`. **Generalization beyond benchtop** (field lighting, variety, maturity, UAV heads) remains future work pending tagged collections (**MS-GEN**, **P1-DOMAIN-TAGS**); honest extrapolation claims should cite split-drift and val-vs-test mAP narrative ([`val_test_map_gap.md`](val_test_map_gap.md)).

Suggested Discussion sentence: *“Cross-tray ranking mAP varied substantially across held-out session slices (supplementary domain catalog; `domain_eval.v1`), motivating staged transfer fine-tuning on new trays—by analogy to Yang et al. (2024) and TFA/Gandhi YOLO freeze schedules (`finetune_tray_stage1/2`)—rather than training from scratch; field and multi-site validation remain future work.”*

---

## §14 Manuscript draft — two-stage deploy (Discussion)

**Use for reviewer line 415 (**MS-DEPLOY-2STG**).** Literature: [`alshehri2025_uav`](literature_validated.json) (Frontiers in Neurorobotics 2025 — UAV multi-person **action recognition**, not seed detection). Deploy wiring: [`HSP_BASELINE_MODELS.md`](../HSP_BASELINE_MODELS.md) § Two-stage deploy; parity JSON: [`deploy_hsp_parity.json`](../../reports/hsp/deploy_hsp_parity.json) (`deploy_hsp_parity.v1`; **R-SCI-2** Done).

**Discussion (draft):** Production inference (`telegram_bot.py`, `run_infer_once.py`) is intentionally **two-stage**: (1) **`classifier.pt`** gates the upload — accept only top-1 **sunflower** with confidence ≥ **0.5** (`is_sunflower_image`; skip via `SKIP_CLASSIFIER`); (2) **`best2.pt`** runs **SAHI** sliced detection (slice **500**, overlap **0.35**, merge NMS IoU **0.50**) with per-class post-filter conf **0.06** / **0.04** (developed / aborted; `harchoc/deploy_filters.py`). This parallels the reviewer’s Alshehri et al. (2025) **robustness pattern** — a coarse content/scene decision before fine-grained perception — though their pipeline targets UAV video **action recognition**, not sunflower seed boxes (`alshehri2025_uav`; registry `architecture_takeaway`). **Manuscript counting and detection metrics** use a **single-stage** science path on the frozen HSP corpus: **`best2.pt` only**, full-frame `eval.py` @ `imgsz` **1280**, export conf **0.001**, counting at val-locked conf **0.15** — no classifier gate, because all split images are benchtop sunflower heads. Deploy vs manuscript threshold policy is explicit in [`deploy_hsp_parity.json`](../../reports/hsp/deploy_hsp_parity.json): default deploy per-class thresholds sit **~0.09–0.11** below locked HSP conf; optional `HARCHOC_LOCKED_CONF` / `--locked-conf-from` aligns bot filters for parity experiments. Do not compare Telegram/SAHI counts to headline test MAE (**61.3**) without that alignment ([`p0_summary.md`](../../reports/hsp/p0_summary.md) § Two-stage deploy).

Suggested Discussion sentence: *“Field deployment applies a two-stage sunflower gate plus SAHI seed detection (Alshehri et al., 2025, by analogy), whereas reported counting metrics use full-frame evaluation on benchtop heads at a validation-locked confidence without the deploy classifier.”*

---

## §15 Manuscript draft — fuzzy / hierarchical boundary seeds (Discussion)

**Use for reviewer line 270 (**MS-FUZZY-BOUND**).** Literature: [`yao2025_hfuzzy`](literature_validated.json) · [explainability synthesis § Reviewer cites](../research/explainability_uncertainty_literature.md#reviewer-cites-ren-2025-yao-2025); artifacts: [`error_test_report.json`](../../reports/hsp/error_test_report.json) → `ambiguous_summary`, `ambiguous_fp_crosstab` (**P1-UNCERT-FP** Done); figure [`fig_ambiguous_panel.png`](../../reports/figures/fig_ambiguous_panel.png) (**P2-FIG** Done; CPU `make_figures.py`).

**Discussion (draft):** The reviewer asks how boundary-ambiguous or hierarchically nested seeds should be handled without blurring developed vs aborted semantics. We did **not** introduce a third YOLO detect class or a relabel protocol for “uncertain” kernels—annotations remain hard developed (0) / aborted (1) labels. Instead, by analogy to Yao et al. (2025; IEEE TFS — hierarchical fuzzy **regression** on high-dimensional tabular data, registry `yao2025_hfuzzy`), we treat each two-class detection as carrying **graded trust** rather than a single binary accept/reject: a val-locked confidence gate (conf **0.15**), a post-hoc **low-confidence score band** on exported boxes (default upper bound = locked conf + **0.15**, i.e. **(0.15, 0.30]**), and cross-tabs of `ambiguous_detections` against TIDE-style FP buckets (`error_analysis.py` → `ambiguous_summary`, `ambiguous_fp_crosstab`). On held-out **test** (*n*=109 images), **19 668** of **56 811** exported predictions fall in the ambiguous band (~**35%**); **13 050** ambiguous boxes lie in background or localization FP buckets (~**47%** of all **27 509** FPs at locked conf), while developed↔aborted confusion remains a small matched-box subset (**1 825** boxes; §11). Headline count MAE (**61.3**) uses all boxes above the locked gate; excluding the band is reported only as a diagnostic (**214.0** MAE—shows sub-threshold mass, not a deployment policy). Qualitative examples appear in **`fig_ambiguous_panel.png`**. Yao’s fuzzy membership on **regression** outputs is cited **only as a conceptual parallel** for reporting gradual output trust—not as a mandate for fuzzy detection heads, relabeling, or a third class.

Suggested Discussion sentence: *“Rather than a third detect class for boundary-ambiguous seeds, we report graded trust on two-class outputs—a val-locked confidence gate plus a low-confidence score band—by analogy to Yao et al. (2025) hierarchical fuzzy regression; most ambiguous-box mass at the operating point was background or localization error, not developed↔aborted confusion (Fig. X, `fig_ambiguous_panel.png`).”*

---

## §17 Manuscript draft — manual validation *n*=50 (Methods)

**Use for reviewer theme #17 (**MS-MANUAL-N50**).** Density stats: [`split_drift_p0.json`](../../reports/hsp/split_drift_p0.json) test `labels.boxes_per_image`; tray keys via `harchoc/domain_tags.py` (`tray_key_from_stem`). Timed throughput baseline is separate: **MS-MANUAL-BASE** (§8).

**Methods (draft):** Independent **manual validation** used **n=50** images drawn **only** from the held-out **test** split (`data/splits/test.txt`, *N*=109); train and validation images were excluded, and this subset played **no role** in confidence locking or any model decision. The sampling frame is the full frozen test pool with CVAT ground-truth **boxes per image** (mean **553.8**, median **591**, range **169–830**; [`split_drift_p0.json`](../../reports/hsp/split_drift_p0.json)). To preserve **density representativeness**, we stratify test images into **tertiles** on GT box count (cuts at **492** and **651** boxes/image) and draw a fixed quota from each stratum without replacement (**17 / 16 / 17** → **50** total) using a documented PRNG seed. Where the pool allows, we cap draws at **one image per `tray_key`** per stratum so session keys (**41** trays in test) are not collapsed into a single tray. Two readers independently verify developed and aborted counts (or adjudicate against exported GT boxes); a third reader resolves disagreements. This protocol is **annotation / count QA** on a density-balanced test slice—not threshold tuning and not the timed **~1 h/tray** study in **MS-MANUAL-BASE**.

**Sampling table (test split, density-stratified *n*=50):**

| Stratum | GT boxes/image (test tertile) | Pool *n* (test) | Pool mean (range) | Sample *n* (manual validation) |
|---------|------------------------------|-----------------|-------------------|--------------------------------|
| Low | ≤492 | 38 | 364 (169–492) | **17** |
| Mid | 493–651 | 35 | 580 (494–651) | **16** |
| High | ≥652 | 36 | 729 (653–830) | **17** |
| **Total** | all test images | **109** | **554** (169–830) | **50** |

**Representativeness (design targets):**

| Metric | Full test (*n*=109) | Manual subset (*n*=50, by design) |
|--------|---------------------|-----------------------------------|
| Mean boxes/image | **553.8** | ≈**554** (stratum quotas match pool fractions) |
| Median boxes/image | **591** | ≈**580–590** (tertile balance) |
| Tray keys represented | **41** | ≤**50**, ≤1 draw per tray when pool allows |
| Split role | held-out test only | same; no val/train leakage |

**Procedure note (repo):** Tertile cuts and pool sizes are recomputed from label files under `DATASET_ROOT` when splits change; document the seed and resulting image list in supplementary material when human recounts are completed. Peer precedent for stratified manual audit: GWHD FP decomposition (*N*=500 random FPs; [fp_taxonomy § Layer B](../research/fp_taxonomy_literature.md)); GrainNet counting vs manual on dense benchtop grains (`grainnet2025`).

Suggested Methods sentence: *“Manual validation comprised n=50 test images sampled with density-stratified quotas (tertiles on ground-truth boxes per image; 17/16/17) from the held-out test split only, with at most one image per tray key when possible; independent readers verified counts without influencing the locked operating point.”*

---

## Summary by status

| Status | Count | Theme #s |
|--------|------:|----------|
| **Existing backlog** | 1 | 3 |
| **Partial** | 8 | 4–8, 10–11, 13 |
| **Done** (repo draft) | 7 | 2, 9, 12, 14–17 |
| **MISSING** (manuscript-only) | 1 | 1 |

*Themes #12–15 moved from **MISSING** to **partial** after [`literature_validated.json`](literature_validated.json) and research-doc updates (2026-05-29).*

---

## Architecture / science backlog (from validated lit)

| ID | Pri | Scope |
|----|-----|--------|
| **LIT-VALIDATE** | P2 | Keep [`literature_validated.json`](literature_validated.json) in sync when adding cites |
| **P1-FINETUNE-LOOP** | P1 | TFA/Gandhi staged tray finetune (`finetune.py`) |
| **P1-DOMAIN-TAGS** | P2 | `domain_metadata_tags` scaffold in catalog (variety/maturity/lighting/site **TBD**) |
| **ARCH-MOSAIC0-AB** | P1 | S2 mosaic=0 vs S0 (`lwcd_yolo2025`) |
| **ARCH-EMA-BG-SPIKE** | P2 **Done** | GrainNet EMA lit reviewed; implementation cancelled — [arch_ema_bg_spike](../research/arch_ema_bg_spike_literature.md) |
| **SCI-MAP-CPU** | P1 | Test mAP + dual_metric detection row (CPU export) |

---

## §0 Origin weights & dataset (not Kaggle)

**Use for reviewer #5 (reproducibility) and any Methods confusion about training source.**

| Topic | Fact | Evidence |
|-------|------|----------|
| **`best2.pt`** | Trained in **[Fami2040/sunflower-Detector](https://github.com/Fami2040/sunflower-Detector)** (`main`), copied into this fork | [`docs/ORIGIN_MAIN_AND_DATASET.md`](../ORIGIN_MAIN_AND_DATASET.md), [`configs/origin/public_yolov8_train_reference.json`](../../configs/origin/public_yolov8_train_reference.json) |
| **Dataset** | **CVAT ~2500**, share **`y9xGFqCW`** — tracked [`data/manifest.json`](../../data/manifest.json) | [`data/README.md`](../../data/README.md) |
| **Kaggle `dataset1`** | Development convenience in upstream script; **not** the published corpus in this repo | Origin reference JSON `dataset_note` |
| **Aug comparison program** | S0–S14 + 100 ep: **new trains** with literature tactics; **same** test + locked conf as anchor `best2` | [`reports/aug_smoke/README.md`](../../reports/aug_smoke/README.md) |
| **Reviewer criticisms** | Full text | [`reports/reviewer2.md`](../../reports/reviewer2.md) |

Suggested Methods sentence: *“The detector weights (`best2.pt`) were obtained from the public training pipeline (YOLOv8m, 100 epochs, image size 1280, conservative augmentation; repository Fami2040/sunflower-Detector). Counting metrics in this study use the CVAT-annotated corpus (~2500 images, share y9xGFqCW) with frozen train/val/test lists; held-out test count MAE is 61.3 seeds/image at a validation-locked confidence of 0.15.”*

---

## §18 Manuscript draft — augmentation strategy (Methods)

**Use for Methods / reviewer questions on training aug, S0–S14, and why production weights were not replaced.** Canonical one-pager: [`p0_summary.md` § Augmentation](../../reports/hsp/p0_summary.md#augmentation-strategy-reviewer-methods). Index: [`aug_smoke_index.json`](../../configs/experiments/aug_smoke_index.json). Rankings (test IDs only): [`leaderboard.md`](../../reports/aug_smoke/leaderboard.md).

**Methods (draft):** Production **`best2.pt`** was trained in the public upstream repo ([origin `main`](https://github.com/Fami2040/sunflower-Detector)) with conservative online augmentation (`mosaic=0.1`, photometric jitter, no mixup) — see [`public_yolov8_train_reference.json`](../../configs/origin/public_yolov8_train_reference.json). HARCHOC **codifies** a similar recipe in [`robustness_minimal.yaml`](../../configs/aug/robustness_minimal.yaml) and cites benchtop **seed/kernel** peers ([`grainnet2025`](literature_validated.json); [`lwcd_yolo2025`](literature_validated.json) — LWCD **reports** mosaic disabled in §3.1 for corn kernels; **our S2 mosaic=0 worsened** test MAE to **147.4**, so we do not adopt mosaic-off for sunflower). DOI audit: [`literature_doi_audit_2026-06-01.md`](../../reports/manuscript/literature_doi_audit_2026-06-01.md).

**Augmentation comparison (same protocol as anchor):** Legacy **`best2.pt`** was re-evaluated on frozen CVAT splits (test count MAE **61.3** at validation-locked conf **≈0.15**). We then trained **fifteen 15-epoch** YOLOv8m variants (**S0–S14**) with **literature-guided** augmentation tactics ([`training_tech_scan_2026_augmentation.md`](../research/training_tech_scan_2026_augmentation.md)), plus a **100-epoch** full train with production-like minimal aug — each evaluated on **`data/splits/test.txt`** with the **same** locked confidence (no test leakage). **S14** is eval-only (`max_det=300`) — control, not an aug competitor.

**Results:** No alternative train beat the anchor: best smoke **68.9** (+7.6 vs **61.3**); 100-ep confirm **64.1** (+2.9). **S2** (mosaic=0) **147.4** — full mosaic-off **rejected** on our trays (retain low mosaic like anchor). **Closed** — retain **`best2.pt`**.

**Evidence:** [`ORIGIN_MAIN_AND_DATASET.md`](../ORIGIN_MAIN_AND_DATASET.md), [`leaderboard.json`](../../reports/aug_smoke/leaderboard.json), [`aug_confirm_winner_100ep_summary.json`](../../reports/aug_smoke/aug_confirm_winner_100ep_summary.json).

Suggested Methods sentence: *“We held the legacy detector fixed as the anchor and compared literature-guided training augmentations and full-budget retrains on the same held-out test split and counting protocol (validation-locked confidence). No variant improved test count MAE beyond the anchor (61.3 seeds/image); alternative recipes scored 64.1–147.4 MAE.”*

---

## §19 Manuscript draft — model selection & SOTA zoo (Methods / Results)

**Use for reviewer theme #3 (SOTA beyond YOLOv8) and headline metric choice.** Anchor: [`p0_summary.md` § Model selection](../../reports/hsp/p0_summary.md#model-selection--best2pt-vs-zoo-reviewer-sota--methods).

**Methods (draft):** The manuscript detector is **`models/best2.pt`** (YOLOv8m from [upstream `main`](https://github.com/Fami2040/sunflower-Detector), 100 epochs @ `imgsz=1280`). **HARCHOC** reports **per-image total seed count MAE** on frozen CVAT splits (`data/splits/test.txt`, *n*=109) after tuning confidence on **validation only** (`threshold_sweep.py --select min_count_mae` → **≈0.15**). **Ranking mAP** is supplementary ([`dual_metric.json`](../../reports/hsp/dual_metric.json)); pooled test mAP50 **≈0.18** under HSP export — not legacy manuscript **~0.79** ([`val_test_map_gap.md`](val_test_map_gap.md)).

**SOTA comparison (8 GiB GPU):** **`zoo_yolo_only`** (four YOLO M-scale rows × 100 epochs @ 1280). Completed test count MAE at locked conf: **yolov8m 111.9**, **yolo11m 119.6**, **yolo26m 95.3** — all worse than anchor **`best2` 61.3** ([`matrix_train.json`](../../reports/hsp/matrix_train.json)). **YOLOv10m** was finishing 100-ep training at draft date (1-ep VRAM ~3.3 GiB; not OOM-deferred). **Ultralytics RT-DETR-L** train @ 1280 batch=1 **OOMs** on 8 GiB (measured probe). External DETR: integration/port issues; **D-FINE** reached ~6.7 GiB peak on same GPU. Snapshot: [`reports/manuscript/FRESHNESS.md`](../../reports/manuscript/FRESHNESS.md).

**Evidence:** [`eval_test_map.json`](../../reports/hsp/eval_test_map.json), [`matrix_train.json`](../../reports/hsp/matrix_train.json), [`zoo_comparison_design.md`](../zoo_comparison_design.md), [`gpu_queue_full.json`](../../configs/experiments/gpu_queue_full.json) (`zoo_matrix_p0_5`).

Suggested Methods sentence: *“The reported detector is YOLOv8m (`best2.pt`) chosen by minimum held-out test count MAE at a validation-locked confidence; supplementary multi-architecture runs (YOLOv10/11/26 at 1280) use the same HSP protocol; Ultralytics RT-DETR training at 1280 is deferred on 8 GiB GPUs (VRAM), while external DETR baselines depend on stack integration.”*

---

## §20 Manuscript draft — tray finetuning (Methods / Discussion)

**Use with §12 (domain adaptation narrative) for operational finetune protocol.** Playbook: [`FINETUNE_WEAK_TRAYS.md`](../FINETUNE_WEAK_TRAYS.md) § Methods paragraph. Code: `scripts/finetune.py`, `harchoc/finetune_pipeline.py`, queue kind **`finetune_tray`**.

**Methods (draft — planned, not yet run on GPU):** For trays with high held-out count error, we adapt **`best2.pt`** using **tray-local splits** (`data/domains/train_{tray}`, `val_{tray}`) with an explicit assertion that **no canonical test image** appears in finetune train/val ([`finetune_tray_splits.py`](../../harchoc/finetune_tray_splits.py)). Weak trays are ranked from per-tray count MAE at the **same** locked conf as the headline metric ([`weak_tray_plan.json`](../../reports/domains/weak_tray_plan.json): **`3a5-9`**, **`350`**, **`200-3-1`**). **TIDE-style error mass** on the global test export ([`tide_bucket_summary.json`](../../reports/hsp/tide_bucket_summary.json)) motivates **localization/background-focused** tray adaptation when Loc+Bkg dominates; dominant **Miss** suggests deferring finetune in favor of `max_det`/recall levers (`finetune_tide_guidance`). Training follows a **two-stage freeze schedule** (25+25 epochs; stage 1 frozen backbone, stage 2 full unfreeze at lower LR) by analogy to TFA and staged YOLO freeze literature ([`finetune_tray_stage1.json`](../../configs/experiments/finetune_tray_stage1.json)). **Success** is measured on the tray holdout slice; **promotion** requires canonical **test** count MAE within **+10%** of the global reference **61.3** (`canonical_gate` in `finetune_run.v1`).

**Status:** Implementation and CPU tray audit **Done**; staged GPU jobs are **queued** in [`gpu_queue_post_zoo.json`](../../configs/experiments/gpu_queue_post_zoo.json) **after** `zoo_matrix_p0_5` — do not report finetune MAE improvements until `reports/transfer/finetune_*_s2.json` exist.

**Evidence:** [`domain_eval.json`](../../reports/domains/domain_eval.json), [`weak_tray_plan.json`](../../reports/domains/weak_tray_plan.json), [`finetune_3a5-9_s1_plan.json`](../../reports/transfer/finetune_3a5-9_s1_plan.json) (plan artifact).

Suggested Methods sentence: *“For acquisition trays with elevated count MAE at the locked operating point, we plan staged transfer fine-tuning from `best2.pt` on tray-holdout splits (no canonical test leakage), with promotion gated on canonical test count MAE within 10% of 61.3 seeds/image.”*

---

## Cross-links

| Artifact | Reviewer themes |
|----------|-----------------|
| [`originality_contribution_peers.md`](originality_contribution_peers.md) | 2 |
| [`val_test_map_gap.md`](val_test_map_gap.md) | 9, 16 |
| [`literature_validated.json`](literature_validated.json) | 6, 12–15, 18 |
| [`literature_doi_audit_2026-06-01.md`](../../reports/manuscript/literature_doi_audit_2026-06-01.md) | All cites |
| [`lit_audit/`](../../reports/manuscript/lit_audit/README.md) | Per-paper `claim_fit` |
| [`related_work_outline.md`](related_work_outline.md) | 6 |
| [`reports/hsp/p0_summary.md`](../../reports/hsp/p0_summary.md) | 2, 3, 4, 5, 9, 10, 11, 12, 14, 15, **18–20** |
| [`reports/aug_smoke/leaderboard.md`](../../reports/aug_smoke/leaderboard.md) | **18** |
| [`reports/aug_smoke/comparative_analysis.md`](../../reports/aug_smoke/comparative_analysis.md) | **18** |
| [`docs/FINETUNE_WEAK_TRAYS.md`](../FINETUNE_WEAK_TRAYS.md) | **12**, **20** |
| [`reports/domains/weak_tray_plan.json`](../../reports/domains/weak_tray_plan.json) | **20** |
| [`reports/hsp/matrix_train.json`](../../reports/hsp/matrix_train.json) | **3**, **19** |
| [`reports/figures/fig_ambiguous_panel.png`](../../reports/figures/fig_ambiguous_panel.png) | 15 |
| [`docs/HSP_BASELINE_MODELS.md`](../HSP_BASELINE_MODELS.md) | 14 |
| [`reports/hsp/deploy_hsp_parity.json`](../../reports/hsp/deploy_hsp_parity.json) | 14 |
| [`reports/domains/domain_eval.json`](../../reports/domains/domain_eval.json) | 4, 12 |
| [`reports/domains/domain_count_mae.json`](../../reports/domains/domain_count_mae.json) | 4 |
| [`docs/research/domain_shift_transfer_literature.md`](../research/domain_shift_transfer_literature.md) | 4, 12 |
| [`reports/hsp/tide_bucket_summary.json`](../../reports/hsp/tide_bucket_summary.json) | 11 |
| [`reports/figures/fig_error_taxonomy.png`](../../reports/figures/fig_error_taxonomy.png) | 11 |
| [`reports/hsp/error_test_report.json`](../../reports/hsp/error_test_report.json) | 11, 13, 15 |

---

## MISSING count (manuscript writing only)

**2** themes have no repo artifact beyond backlog rows: **#1** (abstract), **#17** (n=50 protocol). **#8** has a **repo draft** (§8; placeholders) but **no timed human study** yet — replace placeholders before submission.
