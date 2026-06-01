# Literature audit: `lwcd_yolo2025`

**Registry id:** `lwcd_yolo2025`  
**DOI:** [10.3390/agriculture15181968](https://doi.org/10.3390/agriculture15181968)  
**Title:** LWCD-YOLO: A Lightweight Corn Seed Kernel Fast Detection Algorithm Based on YOLOv11n  
**Authors:** Chen et al. (*Agriculture* 2025, 15(18), 1968)  
**Audit date:** 2026-06-01  
**Source fetched:** MDPI HTML — [https://www.mdpi.com/2077-0472/15/18/1968](https://www.mdpi.com/2077-0472/15/18/1968) (DOI landing `https://www.mdpi.com/10.3390/agriculture15181968` returned 404; journal article URL resolves)

---

## Claim under audit

**Repo claim:** Benchtop **corn seed kernel** detection; **Ultralytics-style mosaic augmentation disabled** during YOLO training — cited as precedent for HARCHOC **ARCH-MOSAIC0-AB** (smoke arm **S2**, `mosaic=0`).

---

## Verdict

| Item | Status |
|------|--------|
| Task = dense benchtop corn seeds (kernels) | **Confirmed** (§2.1, 244 images, ~28k kernels, tray platform) |
| Mosaic disabled in training | **Confirmed** (§2.3, single explicit sentence) |
| Mosaic-off ablation (on vs off) in paper | **Not reported** — global training policy only |
| Equivalence to HARCHOC `mosaic=0` / `close_mosaic=0` | **Plausible** (YOLOv11n + “default” other hparams); paper does not name YAML keys |

**Overall:** The **mosaic-off** citation is **valid** for manuscript related work and as an external precedent; it does **not** by itself justify adopting mosaic-off on sunflower trays (HARCHOC S2 smoke worsened count MAE vs anchor).

---

## Primary evidence (mosaic)

**Section 2.3 — Experimental environment and parameters** (training setup for all experiments):

> Mosaic data augmentation was disabled to ensure environmental consistency, and pre-trained weights were not loaded.

**Rationale in paper:** “environmental consistency” across comparisons (architecture / loss ablations), not counting MAE or missed-kernel rate.

**Scope:** Stated once; no other occurrences of “mosaic”, “mixup”, or `close_mosaic` in the full-text HTML fetch.

**Training stack (same paragraph):** YOLOv11n; images 640×640; SGD (momentum 0.937, lr 0.02 → 0.00001); 1000 epochs max; batch 32; early stop on test mAP plateau (100 epochs); **other parameters retain their default values** (Ultralytics defaults implied).

---

## Related augmentation (not mosaic)

**Section 2.1.3 — Data preprocessing** describes **offline** training-set expansion before YOLO training:

- Random rotation, scaling, flipping, contrast, cropping  
- **10×** augmentation per training image (161 → 1610 images; 18,659 → 186,590 boxes)  
- **No augmentation on the test set**

This is **separate** from online mosaic: the paper applies heavy **dataset-level** geometric jitter **and** disables **mosaic** in the detector trainer.

---

## Task and dataset (context)

| Aspect | Paper |
|--------|--------|
| Object | Commercial **corn seed kernels** (six varieties, benchtop) |
| Imaging | Industrial camera, LED tray platform; 4448×3000 capture, annotated minimum rectangles |
| Density | Training: 161 images / 18,659 boxes (~116 boxes/image); test: 83 / 9,319 (~112/image) |
| Split | Variety holdout: LP206 + CS5 test; four varieties train |
| Environments | Fixed layout + random spread with adhesion/occlusion (both in test) |
| Model | LWCD-YOLO (YOLOv11n + PConv/EMA backbone, MSFFM neck, WIoU) |
| Metrics | Box mAP (P, R, mAP@0.5, mAP@0.5:0.95); **no count MAE** |

---

## Gaps and caveats

1. **No mosaic on/off table** — disabling mosaic is a fixed protocol choice, not an ablated axis.  
2. **Wording** — “disabled”, not `mosaic=0`; `close_mosaic` not discussed.  
3. **Metric mismatch** — HARCHOC optimizes **tray count MAE**; LWCD-YOLO reports detection mAP only.  
4. **Crop / species** — maize kernels vs sunflower **developed/aborted** embryos; precedent is **regime** (dense benchtop seeds), not transferable hyperparameters.  
5. **HARCHOC empirical check** — 15-ep smoke **S2** (`mosaic=0`) gave test count MAE **147.4** vs anchor **61.3** ([`results_and_methods.md`](../results_and_methods.md)); LWCD precedent supports **running** the ablation, not **accepting** mosaic-off for our headline model.

---

## HARCHOC mapping

| Backlog / arm | Use of this audit |
|---------------|-------------------|
| **ARCH-MOSAIC0-AB** / **S2** | External **mosaic-off** precedent for dense benchtop seeds; cite in Methods/Related Work |
| **MS-ORIG** | Peer: lightweight YOLO on kernels + Grad-CAM (§3, elsewhere in paper) |
| **P1-AUG** | Literature scan anchor; decision rule remains **test count MAE**, not peer mAP |

**Safe cite wording:** Sun et al. (2025) train YOLOv11n on dense benchtop corn-kernel images with **mosaic augmentation turned off** for consistent experimental conditions, alongside separate offline geometric expansion of the training set.

---

## References

- Sun et al., LWCD-YOLO, *Agriculture* **2025**, 15(18), 1968. https://doi.org/10.3390/agriculture15181968  
- Internal registry: [`docs/manuscript/literature_validated.json`](../../../docs/manuscript/literature_validated.json) (`lwcd_yolo2025`)
