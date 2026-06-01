# Literature audit: GrainNet 2025 (`grainnet2025`)

**Paper:** Wang, X. et al. *GrainNet: efficient detection and counting of wheat grains based on an improved YOLOv7 modeling.* Plant Methods **21**, 44 (2025).  
**DOI:** [10.1186/s13007-025-01363-y](https://doi.org/10.1186/s13007-025-01363-y)  
**Fetched:** 2026-06-01 (BioMed Central HTML via WebFetch; DOI redirect timed out)  
**Scope:** Wheat **kernel/grain counting** claims in this repo vs what the paper reports. Architecture (EMA, ASF-GD) is cited only where it affects counting interpretation.

---

## 1. What the paper actually measures for counting

| Item | Paper (verified) |
|------|------------------|
| **Task** | Benchtop **wheat grains** scattered on a platform; detection boxes → **per-image predicted count** vs **manual count** (LabelImg boxes; “GT” in figures). |
| **Counting metrics** | **MAE**, **RMSE**, **R²** (Eqs. 11–13); also **“counting accuracy”** (%) in Table 6 / abstract. |
| **Evaluation split for headline counting** | **Validation set** only: *n* = number of images in the **verification set** (910 images after offline augmentation; 8:2 split from 4,552 augmented images). **No separate held-out test split** for counting MAE/R² in the main counting section. |
| **Grains per image (manual)** | Table 5: per-image totals **10–300** (wide CV across three variety labels G/Y/B). |
| **How count is obtained** | Model prediction count = detections on each image (standard detection-counting pipeline); compared to manual grain number per image. |
| **Dataset scale** | **1,198** original images → offline aug → **4,552** images (rotation, flip, translation, brightness, **cutout**, **mixup**, **mosaic**); train **3,642** / val **910**. |
| **Adhesion / density (image regions)** | Low **2–20**, medium **21–50**, high **>50** grains per localized area (not tray-scale ~500). |
| **Training** | YOLOv7-derived **GrainNet** (P-conv lightweight blocks, **ASF-GD** neck, **EMA** in head); 640×640, 200 epochs, batch 16, SGD. |

---

## 2. Headline counting numbers (paper)

**Primary comparison (Table 6 / abstract / conclusions — five detectors on validation counting):**

| Model | R² | MAE | Notes |
|-------|-----|-----|--------|
| **GrainNet** | **0.93** | **5.97** | Best among Faster R-CNN, YOLOv5/7/8, GrainNet |
| — | — | — | **Counting accuracy 94.47%** (Table 6 / abstract) |
| — | — | — | Conclusions also give **RMSE 23.15** for GrainNet |

**Subset / condition studies (not interchangeable with Table 6):**

- **Table 7:** MAE **&lt; 1** at imaging heights 10–15 cm (high adhesion); MAE rises at **20 cm**; MAE **&lt; 4** at low/medium adhesion, worse at **high** adhesion (fixed 20 cm height).
- **Fig. 10 / §3.4.1:** **200** random field images (2024 Jiaozuo); per-variety R² **0.94, 0.93, 0.93** (GrainNet vs YOLOv7).
- **ANOVA block (§Discussion):** Repeated training (*n*=5 runs per model) reports GrainNet **R² 0.921**, **MAE 5.57**, **RMSE 20.53** — **different experimental frame** than Table 6 headline; do not mix without labeling.

**Detection metrics (same paper, not counting):** P **97.32%**, R **92.05%**, mAP@0.5 **93.15%**, F1 **0.946**, **29.10 FPS** — reported alongside counting, not replaced by it.

---

## 3. Repo claims vs paper (counting-focused)

| Repo claim / usage | Source (examples) | Paper | Verdict |
|--------------------|-------------------|-------|---------|
| Dense **kernel counting** peer on benchtop grains | `results_and_methods.md`, `originality_contribution_peers.md` | Wheat grains, adhesion scenarios, 10–300 grains/image | **OK** — cite as **wheat**, benchtop, dense/touching kernels; density band ≠ sunflower ~500 boxes/image |
| Reports **count MAE** and **R²** vs manual | Registry, fp_taxonomy, MS-ORIG | MAE **5.97**, R² **0.93** on **validation** images | **OK** — add **split** (val, not locked test) and crop |
| **Heavy offline aug** (cutout, mixup, mosaic) | `augmentation_robustness_literature.md`, registry | 1,198 → 4,552; aug list in §2 | **OK** |
| **YOLOv7 + attention** shorthand | Table row in `results_and_methods.md` | YOLOv7 + **ASF-GD** + **EMA** + P-conv lightweight | **Understated** — “attention only” omits ASF-GD; safe text: “YOLOv7-based GrainNet (ASF-GD, EMA)” |
| **Count MAE + rRMSE** as peer reporting norm | `literature_validated.json` architecture_takeaway | Paper defines **RMSE**, not **rRMSE** | **Partial** — cite GrainNet **MAE + RMSE + R²**; rRMSE is **harchoc/GWHD convention**, not GrainNet’s headline |
| **Counting-first** vs mAP alone | `study_lineage.md` | Strong counting section **and** prominent mAP/P/R/F1 | **Partial** — peer is counting-aware, not counting-only protocol like val-lock → **test** |
| **Val-locked conf → test MAE** parity | `originality_contribution_peers.md` delta row | No `min_count_mae` threshold lock; counting on **val** at trained detector settings | **OK as contrast** — do not imply GrainNet used our HSP lock |
| **Single-class-style counting** | `originality_contribution_peers.md` | Three variety classes (G/Y/B) for **detection**; counting aggregates **total grains/image** | **Mostly OK** — counting is total-count; detection is multi-class by variety |
| **Manual comparison / MS-MANUAL-BASE precedent** | `reviewer_comments_backlog_gap.md` | Manual counts are **label reference** for R²/MAE; no timed human counting study vs model | **Partial** — precedent for **metric pairing**, not for **~1 h/tray human baseline** |
| Headline MAE **61.3** vs GrainNet **5.97** | Implicit bench marks | Different crop, splits, grains/image (~554 vs 10–300) | **Not comparable** — never rank or imply superiority from MAE magnitude |
| **Counting accuracy** 94.47% | Rare in repo | Table 6 / abstract | **Omitted in repo** — optional if defining same formula as paper |
| EMA reduces **background** FPs (counting narrative) | `arch_ema_bg_spike_literature.md` | Stated qualitatively + confusion matrices; **no** TIDE-style FP buckets | **OK cite-only** — do not claim GrainNet quantifies same taxonomy as harchoc |

---

## 4. Numbers safe to cite in manuscript (counting)

Use with split and crop explicit:

- GrainNet vs manual on **validation** images: **MAE 5.97**, **R² 0.93**, **counting accuracy 94.47%**, **RMSE 23.15** (conclusions).
- Per-image manual load **10–300** grains; augmented corpus **4,552** images (**910** validation).
- Comparison frame: five detectors (Faster R-CNN, YOLOv5/7/8, GrainNet) on the **same validation counting task**.

Avoid without qualification:

- Equating GrainNet validation MAE to harchoc **test** MAE **61.3**.
- Calling GrainNet **rRMSE**-primary (paper uses RMSE).
- Citing ANOVA **MAE 5.57** as the same result as Table 6 **5.97**.
- Implying GrainNet reports a **timed manual counting baseline** arm.

---

## 5. Suggested Related Work / Methods phrasing (counting)

> Following benchtop wheat grain studies that score detectors by per-image **count MAE**, **RMSE**, and **R²** against manual totals on a validation split (GrainNet: MAE 5.97, R² 0.93 on 910 augmented validation images with 10–300 grains per image), we report held-out **test** count MAE at a confidence chosen on validation only.

---

## 6. Validation status

| Field | Value |
|-------|--------|
| Registry ID | `grainnet2025` |
| DOI resolves | Yes (BioMed Central; direct DOI fetch timed out) |
| Counting metrics in paper | **Confirmed** (MAE, RMSE, R², counting accuracy %) |
| Repo counting claims | **Mostly aligned**; fix **rRMSE attribution**, **val vs test**, **manual baseline** wording, **YOLOv7+attention** shorthand |
| Re-audit trigger | Paper correction, new version, or repo manuscript cites new GrainNet numbers |
