# Explainability & uncertainty for seed/kernel detection (P1 research)

Literature synthesis for **harchoc** P1 (*explainability*, *ambiguous seeds*) on **HSP benchtop sunflower-seed** counting (`imgsz=1280`, ~500 boxes/head, frozen `data/splits/test.txt`). Scope: existing YOLO checkpoints + exported preds through `error_analysis.py` / `threshold_sweep.py`—no architecture swap (P2PNet, MC-dropout retrain) in P1.

**Companions:** [`threshold_calibration_literature.md`](threshold_calibration_literature.md), [`training_tech_scan_2026_eval_calibration.md`](training_tech_scan_2026_eval_calibration.md), [`fp_taxonomy_literature.md`](fp_taxonomy_literature.md). **Out of scope for P1:** full PDQ/LIME-at-full-frame, 3-seed ensembles, video tracking, `telegram_bot.py` SAHI paths. **UAV sunflower-head papers** are cited only as threshold/crowd analogues—not the primary deployment setting (see § Applicability).

---

## Applicability to benchtop sunflower-seed counting (HSP)

| Dimension | HSP (this repo) | Closest literature peers | Use in P1 |
|-----------|-----------------|--------------------------|-----------|
| **Imaging** | Fixed-camera head photos on tray; single scene per image | Corn/wheat **benchtop kernels** (LWCD-YOLO, GrainNet), cover-crop seed mixes | Grad-CAM, overlap/duplicate FPs, ambiguity flags |
| **Task metric** | **Count MAE / rRMSE** at locked conf (manuscript) | GWHD counting RMSE; SoyCountNet MAE; Zu et al. mobile seed counting | Tie uncertainty proxies to **counting**, not mAP-only |
| **Density** | Hundreds of instances per image; adhesion common | P2PNet / SoyCountNet overlap penalties; Kumari centroid clustering | `pred_pred_overlap` ambiguity; dupe taxonomy |
| **Classes** | developed (0) / aborted (1) | Multi-class impurity papers (Bagherpour 2025) | `cls_confusion` + conf×taxonomy grid already in `error_analysis.py` |
| **Field / UAV** | Not primary capture mode | Iamchuen et al. 2026 sunflower **heads**; MTL-PlotCounter seedlings | Conf+IoU grid precedent for `threshold_sweep.py`; do **not** block P1 on LOFO or tiled mosaics |

**Reviewer-facing framing:** Benchtop seeds share **crowd-counting failure modes** (overlap, duplicate boxes, boundary ambiguity) with field seed papers, but differ in **GSD, motion blur, and domain shift**—session/tray metadata (`eval_domains.py`) is the relevant shift axis, not plot-level UAV blocks.

**Implemented hooks (2026-05-29):**

- `error_analysis.py`: TIDE-style dual-IoU FP buckets, `ambiguous_detections` (`low_conf_band`, `pred_pred_overlap`), area strata, conf×taxonomy grid, counting MAE + bootstrap CI; `--locked-conf-from` aligns conf with val sweep.
- `threshold_sweep.py`: val tune → `--locked-conf-from` on test; `--calibrate isotonic|platt`; `--calibration-metrics` (ECE / reliability bins).
- `make_figures.py`: `fig_gradcam_panel` scaffold (`harchoc/gradcam_panel.py`); `fig_ambiguous_panel` still **todo**.

---

## 1. Explainability: Grad-CAM, attention, LIME in agricultural small-object settings

### Grad-CAM (dominant post-hoc choice)

**Gradient-weighted Class Activation Mapping (Grad-CAM)** backpropagates class-specific gradients into late convolutional feature maps to produce a coarse spatial heatmap of “where the model looked.” It is the most common XAI tool in recent agri-detection papers because it needs no architecture change and pairs naturally with CNN/YOLO backbones.

| Domain | Finding relevant to harchoc |
|--------|----------------------------|
| **Wheat spikes (GWHD)** | Grad-CAM used *quantitatively* per YOLOv5 detection-layer scale to score which head attends to labeled spikes; drove architectural refinement (remove weak large-scale head, add micro-scale). |
| **Corn kernels (LWCD-YOLO)** | Grad-CAM on YOLOv11n vs LWCD-YOLO; partial kernel attention linked to missed/incorrect detections; full-kernel coverage accompanied higher mAP ([Sun et al., 2025](https://doi.org/10.3390/agriculture15181968)). |
| **Tomato maturity (TAA-YOLOv8)** | Grad-CAM on attention-enhanced YOLOv8 showed fruit-centered activations as “stakeholder trust” evidence ([Rahim et al., 2026](https://doi.org/10.3390/agriculture16101130)). |
| **Agri OD review** | Names Grad-CAM + SHAP as standard; notes **small-object detection** and **real-time XAI on edge** remain open gaps ([Khan et al., 2025](https://doi.org/10.3390/agriculture15131351)). |

**Practical hook for harchoc:** Run Grad-CAM on backbone/neck layers for a *fixed panel* of images: TP, localization-FP, duplicate/overlap-FP, FN-adjacent regions. Overlay on crops from `error_analysis.py --export-fp-crops`; render via `make_figures.py --error-report … --figure fig_gradcam_panel` (optional torch overlay in `harchoc/gradcam_panel.py`). Library: [pytorch-grad-cam](https://github.com/jacobgil/pytorch-grad-cam) (YOLO/detection examples).

### Attention mechanisms (in-model interpretability)

Attention modules (CBAM, custom channel–spatial blocks, BiFPN weighting) are often justified with Grad-CAM “after” plots. They improve small-target recall but are **not** explanations by themselves—they change the model. For reviewers, the credible story is: *architectural change → metric gain → Grad-CAM shows tighter focus on seed bodies vs background/edge.*

### LIME / SHAP / detection-specific XAI

| Method | Ag / small-object fit | Caveat |
|--------|----------------------|--------|
| **LIME** | Used in ag mainly for **classification** (disease, stress); superpixel masks are coarse for 10–30 px seeds. | Adapted to OD via surrogates (e.g. SODEx); vanilla LIME/RISE emphasize **classification**, not localization ([Andres et al. / Martinez-Seras et al., 2024](https://doi.org/10.1016/j.rineng.2024.103498)). |
| **SHAP** | Cited in ag reviews for pixel attribution; popular for tabular + image classifiers. | Expensive; fragile for dense many-instance scenes. |
| **D-RISE / D-MFPP** | Black-box OD explainers with similarity over class, IoU, and objectness ([ref. 6](#5-references-12-sources)). | Heavier than Grad-CAM; industrial robotics focus in ref. 6, not benchtop seed counting at scale. |

**Recommendation:** Prioritize **Grad-CAM** (+ optional **Score-CAM** if gradients are noisy on YOLO heads). Treat LIME/SHAP as supplementary figures on **single-seed crops** or **image-level “seed present?”** ablations, not full-frame dense detection.

---

## 2. Uncertainty: epistemic vs aleatoric; MC dropout & ensembles for detection

### Definitions (detection context)

| Type | Meaning | Typical cause in seed trays |
|------|---------|------------------------------|
| **Aleatoric** | Irreducible noise in data | Motion blur, glare, touching/occluded kernels, label ambiguity at boundaries |
| **Epistemic** | Model ignorance | Rare poses, tray/session shift, sparse similar examples in train |

Object detectors usually emit a **scalar confidence** (classification score × objectness). That score is **not** a calibrated uncertainty measure—it mixes both types and is often overconfident ([Hall et al., WACV 2020](https://arxiv.org/abs/1811.10800); [WACV PDF](https://openaccess.thecvf.com/content_WACV_2020/papers/Hall_Probabilistic_Object_Detection_Definition_and_Evaluation_WACV_2020_paper.pdf)).

### MC dropout (MC-Drop)

Keep dropout active at inference; run \(T\) forward passes; aggregate boxes/labels. **Gal & Ghahramani** cast dropout training as approximate Bayesian inference and use MC samples at test time as an **epistemic** (model) uncertainty proxy ([arXiv:1506.02142](https://arxiv.org/abs/1506.02142)); they do not separate aleatoric uncertainty in that framework.

- **Stochastic-YOLO** (YOLOv3 + MC-Drop in each detection head): converts \(T\) box samples into probabilistic boxes (corner covariances for spatial quality, class×objectness scores for label quality) and evaluates with **PDQ**; uses feature-map **caching** so only layers after the first dropout block are re-sampled ([arXiv:2009.02967](https://arxiv.org/abs/2009.02967)).
- **Practical epistemic proxies** (without PDQ): spread of box corners or class scores across MC/TTA passes; disagreement across a small ensemble ([ref. 8–10](#5-references-12-sources)).
- **Cost:** still \(T\) stochastic head passes (cached, not always full-network); Ultralytics YOLO does not enable dropout at inference by default—requires training-time dropout injection or approximations.

### Deep ensembles

Train \(K\) models (different seeds); disagreement → epistemic proxy. Strong baseline in OD uncertainty surveys; highest cost (training + storage). For harchoc, a **minimal 3-seed ensemble** on the best config may suffice for a methods subsection without full zoo retrain.

### Probabilistic object detection (research-grade)

**Probabilistic bounding boxes (PBoxes)** + **PDQ** metric jointly score spatial and label distributions ([arXiv:1811.10800](https://arxiv.org/abs/1811.10800), [challenge overview](https://nikosuenderhauf.github.io/roboticvisionchallenges/object-detection.html)). Valuable conceptually (overlap ↔ spatial uncertainty) but **heavy** to implement on custom seed data vs extending existing YOLO confidence + disagreement heuristics.

### Lightweight proxies (good enough for P1)

1. **Low confidence** near operating threshold — default band in `error_analysis.py` is \([τ, τ+0.15]\) or override via locked conf from `threshold_sweep.py` (`--locked-conf-from`).
2. **High IoU between predictions** (`pred_pred_overlap`, default IoU ≥ 0.5) — duplicate boxes on one kernel → “ambiguous / crowd-like.”
3. **TTA spread** (flip/scale variants; count or box jitter) without retraining — **not yet wired**.
4. **Train–test / session eval** (`eval_domains.py`) as epistemic shift detector.
5. **Post-hoc calibration on val** — `threshold_sweep.py --calibrate isotonic|platt` + `--calibration-metrics` (ECE); apply locked conf to test; report alongside uncalibrated `eval.py` mAP ([eval scan](training_tech_scan_2026_eval_calibration.md) §1.3).

---

## 3. Ambiguous seeds, overlap, and crowd-counting overlap

Dense seed trays share failure modes with **crowd counting** and **overlapping instance detection**:

| Problem | Manifestation in seed counting | Literature pattern |
|---------|-------------------------------|-------------------|
| **Overlap / adhesion** | One box covers two kernels, or two boxes on one kernel | Cover-crop mix detection stresses overlap/thickness ([Kumari et al., 2025](https://doi.org/10.3390/agriengineering4040059)); overlapping rice seeds via contour pre-label + Faster R-CNN ([JFPE 13787](https://doi.org/10.1111/jfpe.13787)) |
| **Duplicate predictions** | Count inflation | P2PNet: point proposals + **one-to-one Hungarian matching** (training) instead of NMS-style suppression in dense crowds ([ICCV 2021](https://arxiv.org/abs/2107.12746)); SoyCountNet adds **nearest-neighbor + overlap penalties** on top of P2PNet ([Frontiers 2026](https://doi.org/10.3389/fpls.2026.1743104)) |
| **Missed small/ambiguous seeds** | Under-count on smooth or low-contrast seeds | Zu et al.: DL under-counted “visually ambiguous” species on mobile bench backgrounds ([Front. Plant Sci. 2025](https://doi.org/10.3389/fpls.2025.1659781)) |
| **NMS suppression** | Lost detections in clusters | Detection-based crowd methods report NMS fails on tiny, crowded heads; P2PNet uses point proposals + assignment instead ([ICCV 2021 P2PNet](https://arxiv.org/abs/2107.12746)) |

**Conceptual bridge for reviewers:** Frame “ambiguous seed” instances as *(i)* **label ambiguity** (annotator disagreement on touching boundaries), *(ii)* **prediction ambiguity** (low conf or high ensemble/TTA variance), *(iii)* **structural ambiguity** (high local density). Report all three in a small **qualitative panel** plus aggregate rates on test (`ambiguous_summary` in `error_analysis_summary.v1`).

---

## 4. What to implement first (low effort, high reviewer value)

Aligned with existing scaffolds: `error_analysis.py`, `threshold_sweep.py`, `make_figures.py` (`fig_error_taxonomy`, `fig_gradcam_panel`).

| Priority | Deliverable | Effort | Reviewer value | Repo status (2026-05-29) |
|----------|-------------|--------|----------------|--------------------------|
| **A** | **Ambiguity flags on real preds:** low `conf`, pairwise pred IoU > 0.5, band from locked sweep | Low | Quantified “uncertain / ambiguous seeds” on test | **Done** in `error_analysis.py`; needs **real** HSP exports |
| **B** | **Grad-CAM panel** (12–24 images): TP, localization-FP, duplicate-FP, FN | Low–med | Visual trust; seed-body focus | **Partial** — `gradcam_panel.py` + `make_figures.py`; needs `--export-fp-crops` + weights |
| **C** | **Manuscript figures** `fig_error_taxonomy` + `fig_ambiguous_panel` | Low | Closes qualitative gap | Taxonomy **todo**; ambiguous panel **todo** |
| **D** | **TTA disagreement** (3–5 augments): std of count or box centers per image | Med | Epistemic proxy without retrain | **Not started** |
| **E** | **3-model seed ensemble** disagreement on test subset | Med–high | Stronger uncertainty story | **Not started** |
| **F** | MC-Dropout / P2PNet branch | High | Post-P1 research | **Out of scope** P1 |

**Do not block P1 on:** full PDQ pipeline, LIME at full resolution, point-only re-labeling, or ensemble retraining.

---

## 5. References (12 sources)

| # | Citation | URL |
|---|----------|-----|
| 1 | Selvaraju et al., *Grad-CAM*, ICCV 2017 | https://arxiv.org/abs/1610.02391 |
| 2 | Wang et al., wheat spike detection + interpretive Grad-CAM per scale, *Plant Methods* 2023 | https://doi.org/10.1186/s13007-023-01020-2 |
| 3 | Sun et al., LWCD-YOLO corn kernel detection + Grad-CAM, *Agriculture* 2025 | https://doi.org/10.3390/agriculture15181968 |
| 4 | Rahim et al., TAA-YOLOv8 tomato maturity + Grad-CAM, *Agriculture* 2026 | https://doi.org/10.3390/agriculture16101130 |
| 5 | Khan et al., agri object detection review (XAI, small objects), *Agriculture* 2025 | https://doi.org/10.3390/agriculture15131351 |
| 6 | Andres, Martinez-Seras et al., black-box OD explainability (D-RISE, D-MFPP, D-Deletion), *Results Eng.* 2024 | https://doi.org/10.1016/j.rineng.2024.103498 · [arXiv:2411.00818](https://arxiv.org/abs/2411.00818) |
| 7 | pytorch-grad-cam (object-detection tutorials: Faster R-CNN, YOLOv5) | https://github.com/jacobgil/pytorch-grad-cam |
| 8 | Gal & Ghahramani, MC Dropout, ICML 2016 | https://arxiv.org/abs/1506.02142 |
| 9 | Azevedo et al., Stochastic-YOLO / MC-Drop, arXiv 2020 | https://arxiv.org/abs/2009.02967 |
| 10 | Hall et al., Probabilistic Object Detection & PDQ, WACV 2020 | https://arxiv.org/abs/1811.10800 |
| 11 | Song et al., P2PNet crowd counting & localization, ICCV 2021 | https://arxiv.org/abs/2107.12746 |
| 12 | Liu et al., SoyCountNet (P2PNet + overlap penalties, field soybean seeds), *Front. Plant Sci.* 2026 | https://doi.org/10.3389/fpls.2026.1743104 |

**Additional context (overlap / counting):** Kumari et al., cover-crop seed-mix detection ([DOI 10.3390/agriengineering4040059](https://doi.org/10.3390/agriengineering4040059)); Zu et al., automated seed counting / ambiguous-species under-count ([DOI 10.3389/fpls.2025.1659781](https://doi.org/10.3389/fpls.2025.1659781)); overlapping rice seeds ([JFPE 13787](https://doi.org/10.1111/jfpe.13787)).

---

## 6. Prioritized implementation roadmap (3 tiers)

### Tier 1 — Ship for P1 / manuscript (days, no retrain)

**Goal:** Quantify and visualize ambiguity + explanations on **real test predictions** (HSP protocol: [`EXPERIMENTS.md`](../EXPERIMENTS.md), [`reports/hsp/`](../../reports/hsp/README.md)).

1. Run val export → `threshold_sweep.py` (optional `--calibrate isotonic`) → test with `--locked-conf-from` → `error_analysis.py` with same lock; cross-tab `ambiguous_summary` × `error_taxonomy` / `conf_taxonomy_grid`.
2. `error_analysis.py --export-fp-crops` → `make_figures.py --error-report … --figure fig_gradcam_panel` (weights optional for true Grad-CAM overlay).
3. Add `fig_ambiguous_panel` renderer (select rows from `ambiguous_detections` in report JSON) — **only remaining Tier-1 figure gap**.
4. Methods paragraph: % ambiguous on test, breakdown by FP bucket; cite calibration/ECE from sweep JSON if `--calibration-metrics` used on val.

**Exit criteria:** Reproducible `error_analysis_summary.v1` + sweep JSON on frozen test preds; Grad-CAM mosaic + ambiguous panel PNGs.

### Tier 2 — Strengthen uncertainty claims (1–2 weeks, optional light retrain)

1. **TTA disagreement module** post-`eval.py`: fixed aug set, per-image count std and box jitter stats; flag high-variance images.
2. **Calibration narrative:** compare Platt vs isotonic on val ECE; lock conf; note dual-metric table still uses `error_analysis` counting (eval scan gap: calibrated scores not merged into `dual_metric_report.v1`).
3. **Small ensemble (K=3)** same architecture, seeds {0,1,2}; disagreement IoU/label on matched boxes for epistemic subset analysis (not full zoo).
4. Optional: **Score-CAM** side-by-side if Grad-CAM highlights tray stripe/edge background.

**Exit criteria:** Uncertainty subsection with epistemic *proxy* correlated with known hard cases (`data/eval_sets/asymmetric.txt`, integration test images).

### Tier 3 — Research extensions (post-P1 / separate PR)

1. **Point-supervision pilot:** center-point labels on subset; compare duplicate rate vs YOLO+NMS (P2PNet / SoyCountNet overlap-penalty loss)—only if annotation budget allows.
2. **MC-Dropout or probabilistic heads** in Ultralytics training loop (Stochastic-YOLO-style); compare to ensemble on corner std across MC samples; optionally **PDQ** if probabilistic boxes are exported.
3. **Active review queue:** export ambiguous crops for human re-label → feed back to `finetune.py` / asymmetric eval set.

---

## 7. Mapping to harchoc backlog & code

| Backlog item | Tier 1 anchor | Code (2026-05-29) |
|--------------|---------------|-------------------|
| Classifier explainability | Grad-CAM panel on exported FP crops | `harchoc/gradcam_panel.py`, `make_figures.py --figure fig_gradcam_panel` |
| Uncertainty / ambiguous seeds | Ambiguity rules + taxonomy cross-tab | `error_analysis.py` → `ambiguous_summary`, `conf_taxonomy_grid` |
| FP taxonomy + crops | `--export-fp-crops` + taxonomy buckets | TIDE-style `fp_breakdown`; see [`fp_taxonomy_literature.md`](fp_taxonomy_literature.md) |
| Threshold / calibration | Val sweep + lock + ECE | `threshold_sweep.py --calibrate`, `--calibration-metrics` |
| Manuscript figures | `fig_error_taxonomy`, explain/ambiguous panels | Grad-CAM **partial**; taxonomy + ambiguous PNG **todo** |
| Integration test images | Tag ambiguous cases | `tests/assets`, `data/eval_sets/asymmetric.txt` |

**Dependencies:** Real GT/pred export on test ([`data/examples/README.md`](../../data/examples/README.md)); baseline `best.pt`; val `threshold_sweep` JSON for `--locked-conf-from` on test analysis.

---

## Reviewer cites (Ren 2025, Yao 2025)

| Registry ID | DOI | HARCHOC use |
|-------------|-----|-------------|
| `ren2025_scripta_interp` | [10.1016/j.scriptamat.2024.116350](https://doi.org/10.1016/j.scriptamat.2024.116350) | Interpretability **framing** for breeding trust — pair with **Done** `fig_gradcam_panel` (**MS-EXPLAIN**) |
| `yao2025_hfuzzy` | [10.1109/TFUZZ.2025.3549791](https://doi.org/10.1109/TFUZZ.2025.3549791) | **Methods analogy** — graded trust on detections: locked conf, `ambiguous_summary`, FP taxonomy (large FP volume is class-agnostic); `fig_ambiguous_panel` + **P1-UNCERT-FP** (**MS-FUZZY-BOUND**) |

Ren et al. (Scripta Materialia 2025) harmonize physics and DL for **material property** prediction — not vision. Yao et al. (IEEE TFS 2025) optimize fuzzy topology for **high-dimensional regression** — cite as related work on gradual membership, not as a mandate for a third YOLO class or relabeling protocol.

---

*Validated 2026-05-29. Primary URLs/DOIs for refs. 1–12 and additional context rechecked (HTTP 200 / resolvable DOI). Code cross-check: `scripts/error_analysis.py`, `scripts/threshold_sweep.py`, `scripts/make_figures.py`, `harchoc/gradcam_panel.py`, [`training_tech_scan_2026_eval_calibration.md`](training_tech_scan_2026_eval_calibration.md).*

**Validated literature registry:** [`docs/manuscript/literature_validated.md`](../manuscript/literature_validated.md) · **Related Work (MS-LIT):** [`related_work_outline.md`](../manuscript/related_work_outline.md)
