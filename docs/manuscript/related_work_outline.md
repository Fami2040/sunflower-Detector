# Related Work outline (§2) — repo draft

**Backlog:** **MS-LIT** Done (repo draft 2026-05-29) · **Reviewer theme:** [gap §6](reviewer_comments_backlog_gap.md#6-literature-review-depth)  
**Registry:** [`literature_validated.json`](literature_validated.json) · [`literature_validated.md`](literature_validated.md)  
**External LaTeX:** paste subsections below into manuscript §2 — source `.tex` is outside this repo.

---

## 1. Purpose and scope

Deepen §2 **Related Work** from the internal research corpus (`docs/research/*.md`), not ad-hoc web search. Three axes the reviewer asked for:

| Axis | What to cover | Primary synthesis docs |
|------|---------------|-------------------------|
| **Plant phenotyping** | Dense organ counting, overlap/duplicate FPs, counting metrics (MAE/RMSE), benchmark rigor | [fp_taxonomy](../research/fp_taxonomy_literature.md), [threshold_calibration](../research/threshold_calibration_literature.md), [eval scan](../research/training_tech_scan_2026_eval_calibration.md) |
| **Sunflower** | Head UAV detection, disk geometry, disease/TL surveys; position benchtop **seed** task vs field **head** work | [domain_shift](../research/domain_shift_transfer_literature.md) §83–87, [augmentation](../research/augmentation_robustness_literature.md) §1.3, [explainability](../research/explainability_uncertainty_literature.md) § Applicability |
| **Small-object / dense-scene** | ~500 boxes/image, `imgsz=1280`, mosaic/mixup policy, RT-DETR query caps, TIDE-style errors | [detectors scan](../research/training_tech_scan_2026_detectors.md), [aug scan](../research/training_tech_scan_2026_augmentation.md) §2.4, [fp_taxonomy](../research/fp_taxonomy_literature.md) §2 |

**HARCHOC positioning (one paragraph in §2 closing):** benchtop **developed vs aborted** seed boxes on dried heads; success = **test count MAE** at val-locked conf, not val mAP ([`val_test_map_gap.md`](val_test_map_gap.md)). Closest peers = dense **crop kernel** counters (GrainNet, LWCD-YOLO); sunflower **head** and **phenotyping** papers are analogues for methodology and limitations, not duplicate tasks.

---

## 2. Recommended §2 subsection outline

Use **§2.1–§2.6** (renumber to match journal style). Each subsection: 1–2 synthesis paragraphs + 3–8 cites; point to registry IDs where validated.

### §2.1 Plant phenotyping and dense organ counting

- **GWHD** wheat heads: dense overlap, mAP@0.5, duplicate FPs, counting RMSE in challenge retrospectives (`gwhd2020`, `gwhd2021` in corpus — [fp_taxonomy #3–4, #12](../research/fp_taxonomy_literature.md)).
- **Point / density counting** (P2PNet-Soy, SoyCountNet): overlap and shadow FN patterns — cite when contrasting bbox+count pipeline vs point heads ([fp_taxonomy #9, #15](../research/fp_taxonomy_literature.md)).
- **Counting-first metrics:** MAE, rRMSE, R² vs manual — GrainNet, cover-crop clustering, in-situ seed density ([fp_taxonomy #5, #7–8](../research/fp_taxonomy_literature.md)).
- **Gap we fill:** two-class **viability** (developed/aborted) on **tray-scale** seeds, not head-level yield or single-class spike counts.

### §2.2 Sunflower imaging and Helianthus analytics

- **UAV head detection + yield** — Iamchuen et al. 2026, YOLOv11, tiled 512×512, conf/IoU grids (`iamchuen2026_sunflower_uav`; [domain_shift #10](../research/domain_shift_transfer_literature.md), [eval scan](../research/training_tech_scan_2026_eval_calibration.md)).
- **Disk inclination / geometry** — YOLO11-seg phenotyping tilt ([augmentation §1.3](../research/augmentation_robustness_literature.md)); segmentation counting alternative ([fp_taxonomy #14](../research/fp_taxonomy_literature.md)).
- **Disease / field TL** — Gulzar 2025 review: domain shift, need cross-regional validation (`gulzar2025_sunflower_tl`; [domain_shift #11](../research/domain_shift_transfer_literature.md)).
- **Explicit limitation sentence:** we study **benchtop seed** detection on dried heads; do not claim UAV head-counting or field disease pipelines without new data ([`architecture_recommendations.md`](architecture_recommendations.md) defer table).

### §2.3 Small-object and ultra-dense object detection

- **Scale and resolution:** COCO-small stratum, Applied Sciences 2025 SOD survey; repo `imgsz=1280`, peak **1015** GT/image ([detectors scan §2–3](../research/training_tech_scan_2026_detectors.md), [aug robustness §6](../research/augmentation_robustness_literature.md)).
- **Augmentation for tiny instances:** mosaic/`close_mosaic`, mixup=0 for counting ([aug scan §2.4](../research/training_tech_scan_2026_augmentation.md), [aug robustness §4](../research/augmentation_robustness_literature.md)); validated peer `lwcd_yolo2025` (mosaic disabled).
- **Transformer detectors:** RT-DETR `num_queries` vs tray density — structural mismatch, accepted for zoo row only ([detectors scan §2](../research/training_tech_scan_2026_detectors.md)).
- **Error types at density:** TIDE Bkg/Loc/Dupe dominate over Cls in agricultural FPs ([fp_taxonomy §2–3](../research/fp_taxonomy_literature.md)) — foreshadows our locked-conf results ([gap §11](reviewer_comments_backlog_gap.md#11-manuscript-draft--fp-localization-vs-classification-methods--results)).

### §2.4 Crop seed and kernel detection (closest peers)

| Registry ID | Role in §2.4 |
|-------------|----------------|
| `grainnet2025` | Wheat grains, YOLOv7+EMA, counting MAE/R², background FP — **primary SOTA peer** |
| `lwcd_yolo2025` | Corn seeds, mosaic off, Grad-CAM — **training/eval peer** |
| (corpus) Kumari et al. 2025 cover-crop mix | Centroid de-duplication for overlap ([fp_taxonomy #7](../research/fp_taxonomy_literature.md)) |
| (corpus) Bagherpour 2025 wheat impurities | Multi-class confusion for similar kernels ([fp_taxonomy #6](../research/fp_taxonomy_literature.md)) |

Contribution contrast for **MS-ORIG** (Introduction): developed/aborted **viability** classes + frozen HSP val→lock→test protocol + tray/session domain eval — not “another YOLOv8 on grains” alone.

### §2.5 Thresholds, calibration, and counting-oriented evaluation

- AP does not prescribe operating point (Oksuz et al. ECCV 2018 — [threshold_calibration](../research/threshold_calibration_literature.md)).
- Agri precedent: separate **conf** and **NMS/match IoU** sweeps (sunflower UAV; GWHD competitions) — maps to `threshold_sweep.py` / locked conf **0.15** ([eval scan](../research/training_tech_scan_2026_eval_calibration.md)).
- Val vs test mAP gap: label val as early-stop only ([`val_test_map_gap.md`](val_test_map_gap.md) — **MS-VAL-MAPDOWN** Done).

### §2.6 Transfer learning, explainability, and deployment patterns (short)

Keep brief; full Discussion drafts live in gap §12–15.

| Registry ID | §2.6 one-liner | Deep draft |
|-------------|----------------|------------|
| `yang2024_oct_tl`, `tfa2020`, `gandhi2025_yolov8_freeze` | Staged transfer to new trays — **analogy** | [gap §12](reviewer_comments_backlog_gap.md#12-manuscript-draft--domain-adaptation-plan-discussion) |
| `ren2025_scripta_interp` | Interpretability framing | [explainability](../research/explainability_uncertainty_literature.md), **MS-EXPLAIN** |
| `alshehri2025_uav` | Two-stage robustness pattern | [HSP_BASELINE_MODELS](../HSP_BASELINE_MODELS.md), **MS-DEPLOY-2STG** |
| `yao2025_hfuzzy` | Graded trust on outputs — **regression analogy only** | [gap §15](reviewer_comments_backlog_gap.md), **MS-FUZZY-BOUND** |

---

## 3. Cite table (§2 — registry + corpus anchors)

**Validated registry** entries (use in BibTeX / `\cite{}` when IDs are fixed):

| Registry ID | DOI | §2 subsection | Manuscript use | Research synthesis |
|-------------|-----|---------------|----------------|-------------------|
| `gwhd2020` | [10.34133/2020.3521832](https://doi.org/10.34133/2020.3521832) | §2.1 | Phenotyping benchmark | [fp_taxonomy #3](../research/fp_taxonomy_literature.md) |
| `grainnet2025` | [10.1186/s13007-025-01363-y](https://doi.org/10.1186/s13007-025-01363-y) | §2.4 | Closest counting peer | [fp_taxonomy #5](../research/fp_taxonomy_literature.md) |
| `lwcd_yolo2025` | [10.3390/agriculture15181968](https://doi.org/10.3390/agriculture15181968) | §2.3–2.4 | Mosaic off, benchtop seeds | [aug scan §2.6](../research/training_tech_scan_2026_augmentation.md#26-mosaic-ablation-evidence-arch-mosaic0-ab--p1-aug-mosaic) |
| `iamchuen2026_sunflower_uav` | [10.3390/su18021026](https://doi.org/10.3390/su18021026) | §2.2, §2.5 | Sunflower heads; threshold grids | [domain_shift #10](../research/domain_shift_transfer_literature.md) |
| `gulzar2025_sunflower_tl` | [10.55730/1300-0152.2763](https://doi.org/10.55730/1300-0152.2763) | §2.2 | Field TL / domain shift survey | [domain_shift #11](../research/domain_shift_transfer_literature.md) |
| `yang2024_oct_tl` | [10.1371/journal.pone.0296175](https://doi.org/10.1371/journal.pone.0296175) | §2.6 | Transfer analogy (OCT) | [domain_shift §10](../research/domain_shift_transfer_literature.md#10-reviewer-cite-yang-et-al-2024-transfer-learning) |
| `tfa2020` | [arXiv:2003.06957](https://arxiv.org/abs/2003.06957) | §2.6 | Staged freeze finetune | [domain_shift](../research/domain_shift_transfer_literature.md) |
| `gandhi2025_yolov8_freeze` | [arXiv:2505.01016](https://arxiv.org/abs/2505.01016) | §2.6 | YOLO freeze schedule (preprint) | [domain_shift](../research/domain_shift_transfer_literature.md) |
| `ren2025_scripta_interp` | [10.1016/j.scriptamat.2024.116350](https://doi.org/10.1016/j.scriptamat.2024.116350) | §2.6 | Interpretability framing | [explainability](../research/explainability_uncertainty_literature.md) |
| `alshehri2025_uav` | [10.3389/fnbot.2025.1582995](https://doi.org/10.3389/fnbot.2025.1582995) | §2.6 | Deploy two-stage analogy | [HSP_BASELINE_MODELS](../HSP_BASELINE_MODELS.md) |
| `yao2025_hfuzzy` | [10.1109/TFUZZ.2025.3549791](https://doi.org/10.1109/TFUZZ.2025.3549791) | §2.6 | Graded trust analogy | [explainability](../research/explainability_uncertainty_literature.md) |

**Corpus-only** (cite from synthesis bib lists; add to registry when manuscript BibTeX is frozen — **LIT-VALIDATE**):

| Source | §2 | Doc anchor |
|--------|-----|------------|
| TIDE (Bolya et al., ECCV 2020) | §2.3, §2.5 | [fp_taxonomy #1](../research/fp_taxonomy_literature.md) |
| Hoiem et al., ECCV 2012 | §2.3 | [fp_taxonomy #2](../research/fp_taxonomy_literature.md) |
| Applied Sciences SOD survey 2025 | §2.3 | [aug robustness #12](../research/augmentation_robustness_literature.md) |
| Oksuz et al., ECCV 2018 | §2.5 | [threshold_calibration](../research/threshold_calibration_literature.md) |
| Sunflower disk inclination (Front. Plant Sci. 2025) | §2.2 | [aug robustness §1.3](../research/augmentation_robustness_literature.md) |

---

## 4. Cross-links — `docs/research/*.md`

| Research doc | §2 subsections fed | Backlog hooks |
|--------------|-------------------|---------------|
| [fp_taxonomy_literature.md](../research/fp_taxonomy_literature.md) | §2.1, §2.3–2.4 | **MS-FP-LOC-NARR**, **MS-ORIG**, **P1-TIDE** |
| [threshold_calibration_literature.md](../research/threshold_calibration_literature.md) | §2.5 | **P1-FP-BUDGET**, **MS-VAL-MAPDOWN** |
| [training_tech_scan_2026_eval_calibration.md](../research/training_tech_scan_2026_eval_calibration.md) | §2.5 | HSP protocol, **R-SCI-1** |
| [training_tech_scan_2026_augmentation.md](../research/training_tech_scan_2026_augmentation.md) | §2.3 | **ARCH-MOSAIC0-AB**, **P1-AUG** |
| [training_tech_scan_2026_detectors.md](../research/training_tech_scan_2026_detectors.md) | §2.3, §2.4 | **P0-5**, **MS-SOTA** |
| [augmentation_robustness_literature.md](../research/augmentation_robustness_literature.md) | §2.2–2.3 | **P1-AUG**, sunflower analogues |
| [domain_shift_transfer_literature.md](../research/domain_shift_transfer_literature.md) | §2.2, §2.6 | **MS-GEN**, **MS-DOMAIN-ADAPT** |
| [explainability_uncertainty_literature.md](../research/explainability_uncertainty_literature.md) | §2.6 | **MS-EXPLAIN**, **MS-FUZZY-BOUND** |

**Manuscript companions:** [reviewer_comments_backlog_gap.md](reviewer_comments_backlog_gap.md) · [RESEARCH_AND_OPS.md](../RESEARCH_AND_OPS.md) § Manuscript reviewer response

---

## 5. Paste into §2 (LaTeX) — opening synthesis

**Repo draft complete (**MS-LIT**).** Adapt subsection titles to journal template.

```latex
\subsection{Plant phenotyping and dense counting}
High-throughput phenotyping increasingly relies on object detectors for dense organs (e.g., wheat heads, GWHD) where overlapping instances produce duplicate false positives and counting metrics (MAE, RMSE) complement ranking mAP \cite{gwhd2020,...}. Our task differs by targeting individual \emph{seeds} on benchtop head images with two viability classes (developed vs aborted) rather than panicle-level yield proxies.

\subsection{Sunflower imaging}
Sunflower research spans UAV head detection and yield estimation \cite{iamchuen2026_sunflower_uav}, geometric phenotyping of disk inclination, and field disease detection with transfer-learning domain shift \cite{gulzar2025_sunflower_tl}. We study dried-head \emph{seed} detection in controlled benchtop imaging; UAV head pipelines are methodological analogues for threshold tuning and generalization limits, not direct comparators.

\subsection{Small-object detection in ultra-dense scenes}
Tray images contain hundreds of small bounding boxes per frame, requiring high input resolution, elevated \texttt{max\_det} caps, and counting-aware augmentation (e.g., disabling mixup; reduced mosaic) \cite{lwcd_yolo2025,grainnet2025}. At operating points tuned for counting, background and localization errors dominate class confusion \cite{...}, consistent with TIDE-style analyses in dense agricultural scenes.

\subsection{Evaluation protocol}
We follow counting-first practice: tune confidence on validation, lock a single threshold, and report test count MAE; validation mAP is used for early stopping only \cite{val_test_map_gap narrative}. This mirrors conf/IoU grid searches in sunflower head detection \cite{iamchuen2026_sunflower_uav} while adopting HSP-style lock-then-test discipline.
```

Replace `\cite{...}` placeholders with your `.bib` keys mapped from [§3](#3-cite-table--2--registry--corpus-anchors).

---

## 6. Done criteria vs open work

| Item | Status |
|------|--------|
| §2 subsection outline (phenotyping / sunflower / small-object) | **Done** (this file) |
| Gap §6 cite table + cross-links | **Done** ([gap §6](reviewer_comments_backlog_gap.md#6-manuscript-draft--related-work--literature-review-depth)) |
| Registry keys for primary sunflower + GWHD anchors | **Done** (`iamchuen2026_sunflower_uav`, `gulzar2025_sunflower_tl`, `gwhd2020`) |
| External LaTeX §2 prose | **User action** |
| **MS-ORIG** contribution bullets | **Next** — uses §2.4 contrast table |
| Corpus-only bib entries → registry | **LIT-VALIDATE** when BibTeX stabilizes |

---

*Validated 2026-05-29. **MS-LIT** repo draft; not a substitute for journal bibliography management.*

**Validated literature registry:** [`literature_validated.md`](literature_validated.md)
