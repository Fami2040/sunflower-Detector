# Originality vs crop seed-detection peers (Introduction)

**Use for reviewer theme #2 (**MS-ORIG**).** Registry: [`literature_validated.json`](literature_validated.json) (`grainnet2025`, `lwcd_yolo2025`). FP / counting peers: [`fp_taxonomy_literature.md`](../research/fp_taxonomy_literature.md) §1, §5. Frozen metrics: [`p0_summary.md`](../../reports/hsp/p0_summary.md).

**Repo draft complete (2026-05-29).** External LaTeX paste into Introduction = **user action** (source `.tex` is outside this repo).

---

## 1. Positioning sentence (gap vs peers)

Recent **benchtop crop-seed** detectors optimize **kernel counting** on wheat or corn trays (GrainNet, LWCD-YOLO) or **field** phenotyping at head/plant scale (GWHD, sunflower UAV papers). None publish a **sunflower-seed** benchmark at ~500-instance tray density with **morphologically similar developed vs aborted** classes. HARCHOC contributes a **reproducible two-class counting pipeline** whose primary claims are **held-out test count MAE** at a **validation-locked** confidence, **TIDE-style false-positive taxonomy** (background/localization dominate class confusion), and **cross-tray** generalization reporting—not peak validation mAP alone.

---

## 2. Introduction contribution bullets (paste-ready)

Use as numbered contributions or a short bullet list after the problem statement.

1. **Task and dataset.** First open, reproducible **sunflower head seed** detection benchmark on fixed-camera benchtop trays: **developed** vs **aborted** boxes (~**500** instances/image), frozen train/val/test splits with **split-drift** audit ([`split_drift_p0.json`](../../reports/hsp/split_drift_p0.json)); no equivalent public dataset at GrainNet/GWHD scale ([`fp_taxonomy_literature.md`](../research/fp_taxonomy_literature.md) §5).

2. **Counting-first evaluation protocol.** Operating point chosen on **validation** by minimizing **count MAE**, then **locked** on held-out **test** (conf **0.15**; test MAE **61.3**, 95% CI **51.3–71.3**; [`p0_summary.md`](../../reports/hsp/p0_summary.md))—aligned with GWHD/GWC practice of pairing detection scores with **counting RMSE/MAE**, not reporting in-training val mAP as generalization ([`val_test_map_gap.md`](val_test_map_gap.md)).

3. **Class-similarity stress test.** Unlike single-class grain counters (GrainNet) or variety-focused impurity matrices (multi-class wheat debris), we explicitly score **developed↔aborted** confusion alongside **total** count MAE under natural **~55% / ~45%** class prevalence ([`asymmetric_seed_policy.json`](../../configs/eval/asymmetric_seed_policy.json); **MS-ASYM-NARR**).

4. **Reviewer-facing error analysis.** Automated **TIDE-aligned** `fp_breakdown` on test exports shows **localization + background** dominate **classification** at the locked point (**~15×** Loc+Bkg vs Cls mass; [`tide_bucket_summary.json`](../../reports/hsp/tide_bucket_summary.json); **MS-FP-LOC-NARR**)—matching dense-kernel literature (GWHD duplicates, GrainNet background EMA, cover-crop overlap clustering) but quantified for **two-class** tray counting.

5. **Evidence-based training choices for dense benchtop seeds.** Augmentation policy cites **LWCD-YOLO** (mosaic disabled on dense corn kernels) and **GrainNet** (heavy offline aug with counting caveats) as precedents; repo arms **ARCH-MOSAIC0-AB** (S2 `mosaic=0`) vs baseline S0 ([`training_tech_scan_2026_augmentation.md`](../research/training_tech_scan_2026_augmentation.md) §2.6).

6. **Session/tray heterogeneity and adaptation roadmap.** **52** tray keys, per-tray test **count MAE** spread at locked conf ([`domain_eval.json`](../../reports/domains/domain_eval.json); **MS-GEN**, **MS-DOMAIN-ADAPT**)—a reporting axis absent from single-benchtop grain papers.

7. **Graded trust without relabeling.** Low-confidence score band + `ambiguous_summary` on two-class exports (**MS-FUZZY-BOUND**); not a third detect class—contrasts with fuzzy **regression** analogies (Yao 2025) and impurity **multi-class** expansion (Bagherpour 2025).

*Optional eighth bullet (deploy vs manuscript):* Production **gate + detector** two-stage deploy documented separately from HSP single-model eval (**MS-DEPLOY-2STG**; [`HSP_BASELINE_MODELS.md`](../HSP_BASELINE_MODELS.md)).

---

## 3. Peer comparison cite table

| Peer | Registry / key cite | Imaging & task | Primary metrics | What they emphasize | HARCHOC delta |
|------|---------------------|----------------|-----------------|---------------------|---------------|
| **GrainNet** | `grainnet2025` · [DOI 10.1186/s13007-025-01363-y](https://doi.org/10.1186/s13007-025-01363-y) | Benchtop **wheat grains**; YOLOv7 + **EMA** for background | Counting **MAE, R²**; attention ablation confusion matrices | Single-class-style counting; offline mosaic/mixup | **Two-class** developed/aborted; val-locked conf; automated TIDE `fp_breakdown` + test MAE primary |
| **LWCD-YOLO** | `lwcd_yolo2025` · [DOI 10.3390/agriculture15181968](https://doi.org/10.3390/agriculture15181968) | Benchtop **corn seeds**; YOLOv11 lightweight | mAP + **Grad-CAM** on localization errors | **Mosaic off** in Ultralytics config | Same dense-benchtop regime; we adopt mosaic-off arm + count MAE headline; CAM narrative (**MS-EXPLAIN**) |
| **GWHD / GWC** | fp_taxonomy #3–4, #12 | **Field** wheat **heads**; multi-country | mAP@0.5 + **count RMSE / FPR/FNR** | Overlap/occlusion; competition post-mortems | Tray-scale **kernels** not heads; analogous **Dupe/Loc** taxonomy; no public sunflower-seed GWHD |
| **Cover-crop seed mix** | fp_taxonomy #7 · [DOI 10.3390/agriengineering4040059](https://doi.org/10.3390/agriengineering4040059) | Multi-class **small seeds** | Per-class P/R/mAP; **centroid clustering** post-process | Overlap duplicate FPs | Overlap/adhesion in Layer B; harchoc uses greedy match + `dupe` bucket |
| **Wheat impurity (11-class)** | fp_taxonomy #6 · [DOI 10.1038/s41598-025-23032-9](https://doi.org/10.1038/s41598-025-23032-9) | Kernels + debris classes | Multi-class **confusion matrix** | Morphologically similar grain confusion | Only **two** similar classes; same confusion reporting need, narrower taxonomy |
| **In-situ density counting** | fp_taxonomy #8 · Sung 2025 | Field stacks; density not bbox | Qualitative FP panel (overlap vs straw) | 3D overlap limits | Benchtop 2D trays; quantitative TIDE buckets |
| **Sunflower UAV head** | fp_taxonomy #10 · [DOI 10.3390/su18021026](https://doi.org/10.3390/su18021026) | **Head** detection in field | mAP; conf/IoU sweeps | Crop-specific sunflower but **not seed counting** | Seed-level **tray** phenotyping complement |
| **SoyCountNet / P2PNet-Soy** | fp_taxonomy #9, #15 | Field **soybean** points | Point MAE; overlap penalty / k-d tree dedupe | Point-based counting | Bbox **two-class** pipeline + optional point-head future |

---

## 4. Suggested Introduction paragraph (LaTeX-friendly prose)

*Crop seed and kernel detection has progressed from field head counting (e.g., Global Wheat Head Detection) to benchtop grain detectors that pair YOLO boxes with manual count error (GrainNet; LWCD-YOLO on corn seeds). These works optimize detection architecture and augmentation for dense, touching instances, but typically treat counting as a single target class or report peak validation mAP without a frozen held-out test counting protocol. Sunflower breeding trays differ: hundreds of morphologically similar developed and aborted seeds per image require simultaneous per-class localization and total count accuracy under a fixed camera. We introduce a reproducible two-class sunflower seed detection benchmark with validation-locked operating points, primary evaluation on held-out test count MAE (61.3 seeds/image at confidence 0.15), and TIDE-style error taxonomy showing that false positives are dominated by background and localization rather than developed–aborted confusion—positioning the study relative to grain-counting peers while filling the missing sunflower-seed dense-tray gap.*

Adjust numeric claims only after re-running the val-lock chain ([`p0_summary.md`](../../reports/hsp/p0_summary.md)).

---

## 5. What we explicitly do **not** claim (reviewer-safe)

| Claim | Status |
|-------|--------|
| SOTA over GrainNet/LWCD on their datasets | **No** — different crop, labels, and splits |
| New YOLO architecture | **No** — YOLOv8 baseline + planned zoo matrix (**MS-SOTA**, **P0-5**) |
| Field / UAV sunflower head SOTA | **No** — benchtop trays only; head papers are related work |
| GrainNet EMA module replicated in our YOLO | **No** — **ARCH-EMA-BG-SPIKE** Done: cite-only; see [arch_ema_bg_spike](../research/arch_ema_bg_spike_literature.md) |
| Manual hour baseline or n=50 audit | **Partial** — **MS-MANUAL-BASE** + **MS-MANUAL-N50** repo drafts ([gap §8](reviewer_comments_backlog_gap.md#8-manuscript-draft--manual-counting-baseline-methods--results), [§17](reviewer_comments_backlog_gap.md#17-manuscript-draft--manual-validation-n50-methods)); timed human study + recount Results still open |

---

## Cross-links

| Artifact | MS-ORIG use |
|----------|-------------|
| [`reviewer_comments_backlog_gap.md` §2](reviewer_comments_backlog_gap.md#2-manuscript-draft--originality-vs-crop-seed-detection-introduction) | Short mirror + gap table |
| [`literature_validated.json`](literature_validated.json) | `grainnet2025`, `lwcd_yolo2025` |
| [`fp_taxonomy_literature.md`](../research/fp_taxonomy_literature.md) | Peer table sources #3–15 |
| [`reports/hsp/p0_summary.md`](../../reports/hsp/p0_summary.md) | Headline MAE, FP taxonomy |
