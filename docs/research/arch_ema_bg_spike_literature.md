# ARCH-EMA-BG-SPIKE: GrainNet EMA and background-FP ablation

Research note for backlog **ARCH-EMA-BG-SPIKE** (P2). Evaluates whether to implement **Efficient Multi-scale Attention (EMA)** from GrainNet (Wang et al., Plant Methods 2025) to reduce **background** false positives in harchoc dense sunflower-seed counting.

**Registry:** `grainnet2025` in [`literature_validated.json`](../manuscript/literature_validated.json). **FP taxonomy:** [`fp_taxonomy_literature.md`](fp_taxonomy_literature.md). **Verdict (2026-05-29):** **Cancel custom EMA spike** — cite GrainNet in Related Work / MS-ORIG; pursue lower-cost levers first (threshold, mosaic, hard negatives, deploy filter).

---

## 1. GrainNet 2025 — what EMA does

| Item | Detail |
|------|--------|
| **Paper** | Wang et al., *GrainNet: efficient detection and counting of wheat grains based on an improved YOLOv7 modeling*, Plant Methods 2025. [DOI:10.1186/s13007-025-01363-y](https://doi.org/10.1186/s13007-025-01363-y) |
| **Task** | Benchtop wheat kernels; six backgrounds, three adhesion levels; counting MAE / RMSE / R² vs manual |
| **Base** | YOLOv7 + lightweight P-conv blocks + **ASF-GD** neck (SSFF + gather-distribute + CPAM) |
| **EMA role** | **Efficient Multi-scale Attention** [Ouyang et al., ICASSP 2023] inserted in the **detection head** (not Ultralytics-default neck blocks) |
| **Stated benefit** | “Reduces background noise interference” — multi-scale channel + spatial reweighting (1×1 and 3×3 branches, grouped feature maps, Fig. 6) |
| **Incremental ablation** | On YOLOv7 + lightweight + ASF-GD (exp. 3): adding EMA raised **mAP@0.5** by **+0.61 pp** (92.54 → 93.15%) and **F1** by +1.2 pp; full GrainNet reports P=97.32%, R=92.05%, mAP@0.5=93.15% vs Faster R-CNN / YOLOv5/7/8 |
| **Attention sweep** | Table 12 / Fig. 13: EMA vs SE / ECA / CBAM / SimAM on YOLOv7; authors report EMA wins on parameter/FLOP budget; standalone EMA-on-YOLOv7 cited at mAP@0.5 **89.22**, mAP@0.5:0.95 **56.18** (wheat counting task) |
| **Confusion evidence** | Fig. 13 confusion matrices per attention mechanism — qualitative support for background-FP reduction, not TIDE-style bucket counts |

**Takeaway:** EMA is a **custom head module** bundled with a **YOLOv7-specific** neck redesign, not a training-hyperparameter or aug knob.

---

## 2. Related 2025–2026 evidence (background vs precision)

| Source | EMA use | FP / precision lesson |
|--------|---------|------------------------|
| **DAER-YOLO** (J. Imaging 2025) | C3k2-**EMA** in neck | **Solo EMA:** mAP@50 ↑ but **Precision ↓** vs baseline — “limitations of relying on a single attention mechanism”; full **C3k2-iEMA** needed to rebound precision while cutting FPs |
| **Fire detection + Mamba** (PMC 2025) | EMA after neck upsample | Background-heavy scenes; paired with Wise-IoU — architecture + loss, not EMA alone |
| **EMAF-Net** (Sensors 2026) | EMHA (EfficientNet + MHSA), not GrainNet EMA | Rural road agri scenes — different mechanism |

**Caution:** Attention modules can **increase recall at the cost of precision** (more background activations) unless paired with fusion/loss changes — opposite of harchoc’s need if background FPs are already high at locked conf.

---

## 3. harchoc alignment

### Observed error mix (test, locked conf ~0.15)

From [`error_test_report.json`](../../reports/hsp/error_test_report.json) `fp_breakdown`:

| Bucket | Count | Share of FP buckets |
|--------|------:|--------------------:|
| **localization** | 17,107 | ~58% |
| **background** | 10,402 | ~35% |
| **classification** | 1,825 | ~6% |
| **dupe** | 0 | — |

**MS-FP-LOC-NARR** narrative: **localization + background ≫ class confusion** — GrainNet’s EMA motivation (tray texture, hull chips, glare) is **directionally relevant** for ~35% of FP buckets, but **does not address** the larger localization / partial-box bucket without separate losses or post-filters.

### Stack fit

| Factor | Implication |
|--------|-------------|
| **Ultralytics YOLOv8m @ 1280** | No built-in EMA; requires custom `yaml` + registered modules or forked train loop |
| **Improvement stack step 3–4** | Threshold lock + mosaic/aug ablations are **config-only** and higher ROI |
| **Deploy path** | `harchoc/deploy_filters.py` + locked conf already gate FPs at inference |
| **Peer parity** | MS-ORIG / MS-SOTA cite GrainNet for **count MAE + kernel-counting** positioning, not mandatory architecture match |

---

## 4. Implement vs cancel

### Decision: **Cancel implementation spike** (research **Done**)

| Criterion | Implement EMA | Cancel (chosen) |
|-----------|---------------|-----------------|
| **Engineering** | Custom YOLO graph surgery, CI import weight, matrix row parity | Zero graph change |
| **Isolated ablation** | Confounded with GrainNet ASF-GD / YOLOv7 | N/A |
| **Expected ROI** | Uncertain on YOLOv8; DAER-YOLO shows precision can worsen | Document cite + alternative levers |
| **Manuscript** | Optional “explored and deferred” sentence | “GrainNet uses EMA for background noise; we address Bkg/Loc via TIDE taxonomy, threshold lock, and aug” |
| **GPU budget** | Full retrain + error_analysis per zoo member | Deferred |

**No config stub:** Unlike **ARCH-MOSAIC0-AB** (`mosaic=0` in `configs/aug/*.yaml`), EMA is **not** expressible in `train_*.json` or `configs/aug/` without a non-Ultralytics model definition — a stub would imply a runnable experiment path that does not exist.

### Recommended alternatives (ranked)

1. **P1-FP-BUDGET / count-first threshold** — reduce background FPs at operating point without retrain.
2. **ARCH-MOSAIC0-AB (S2)** — LWCD-style mosaic-off; benchtop consistency precedent.
3. **Layer B hard negatives** — tray texture / debris crops from `--export-fp-crops` (fp_taxonomy § Layer B).
4. **Deploy filter tuning** — `DeployFilterConfig` after locked conf (production path).
5. **Long-term** — only if a future **custom neck experiment** track is opened (with ASF-GD-level scope), revisit EMA as one block in a **combined** ablation (cf. DAER-YOLO iEMA), not a standalone spike.

---

## 5. Manuscript one-liner (optional)

> Dense kernel-counting peers suppress tray background via attention (GrainNet EMA in the detection head; Wang et al., 2025). Our test error taxonomy shows background and localization false positives dominate over class confusion at the val-locked operating point; we prioritize threshold calibration, augmentation consistency, and TIDE-style reporting rather than retrofitting YOLOv8 with GrainNet’s YOLOv7-specific EMA module.

---

## 6. Backlog / doc cross-links

| Artifact | Update |
|----------|--------|
| **ARCH-EMA-BG-SPIKE** | **Done** — ablation note (this file); implementation cancelled |
| **MS-ORIG / MS-SOTA** | Cite `grainnet2025` for counting MAE peer; EMA as Related Work only |
| **fp_taxonomy_literature.md** | §7 cross-link |
| **architecture_recommendations.md** | Moved to “Defer / do not chase” |

---

*Validated 2026-05-29 against Plant Methods HTML landing page and harchoc `error_test_report.json`. No GPU training performed.*
