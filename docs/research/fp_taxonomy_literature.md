# False-positive taxonomy and error analysis in agricultural small-object detection

Research note for harchoc (**developed / aborted** dense seed detection on benchtop trays, class ids 0/1 per `harchoc/sunflower_dataset.py`). Automated taxonomy: `scripts/error_analysis.py` + `harchoc/error_taxonomy.py`; experiment configs per [`configs/experiments/README.md`](../../configs/experiments/README.md).

---

## 1. Key sources

*Verification (2026-05-29): sources 1–15 spot-checked (abstracts/DOI landing pages). TIDE (#1): six error types, \(t_f=0.5\), \(t_b=0.1\), delta-AP oracles — confirmed against [arXiv:2008.08115](https://arxiv.org/abs/2008.08115) and ECCV 2020 PDF. Prior corrections retained: #4 GWC metrics; #5 Fig. 13 scope; #7 author + year.*

| # | Reference | Relevance |
|---|-----------|-----------|
| 1 | **TIDE: A General Toolbox for Identifying Object Detection Errors** — Bolya, Foley, Hays & Hoffman (ECCV 2020). [arXiv:2008.08115](https://arxiv.org/abs/2008.08115) · [ECVA page](http://www.ecva.net/papers/eccv_2020/papers_ECCV/html/849_ECCV_2020_paper.php) | Canonical 6-type detection error taxonomy (Cls, Loc, Both, Dupe, Bkg, Miss) with IoU thresholds \(t_f=0.5\), \(t_b=0.1\) and **delta-AP** weighting. Closest match to harchoc’s current `fp_breakdown`. |
| 2 | **Diagnosing Error in Object Detectors** — Hoiem, Chodpathumwan & Dai (ECCV 2012). [PDF](https://dhoiem.cs.illinois.edu/publications/eccv2012_detanalysis_derek.pdf) | Foundational FP split: localization error, confusion with similar/dissimilar objects, background. Also stratifies by occlusion, size, aspect ratio — useful for reviewer-facing “why” beyond counts. |
| 3 | **Global Wheat Head Detection (GWHD) Dataset** — David et al. (Plant Phenomics 2020). [PMC7706323](https://pmc.ncbi.nlm.nih.gov/articles/PMC7706323/) · [DOI:10.34133/2020.3521832](https://doi.org/10.34133/2020.3521832) | Defines mAP@0.5 matching (IoU≥0.5 TP; duplicate high-confidence boxes → FP). Highlights dense overlap/occlusion as unique benchmark challenge — directly analogous to seed piles. |
| 4 | **Global Wheat Head Detection 2021: An Improved Dataset for Benchmarking Wheat Head Detection Methods** — David et al. (Plant Phenomics 2021). [arXiv:2105.07660](https://arxiv.org/abs/2105.07660) · [DOI:10.34133/2021/9846158](https://doi.org/10.34133/2021/9846158) · [global-wheat.com](http://www.global-wheat.com/) | Expanded multi-country GWHD; **Global Wheat Challenge 2021** primary score is weighted domain accuracy (WDA). GWHD 2020 paper proposes future counting metrics (RMSE, rRMSE, R²); FPR/FNR and counting RMSE appear in GWC challenge retrospectives (see #12). |
| 5 | **GrainNet: efficient detection and counting of wheat grains based on an improved YOLOv7 modeling** — Wang et al. (Plant Methods 2025). [DOI:10.1186/s13007-025-01363-y](https://doi.org/10.1186/s13007-025-01363-y) | YOLOv7-based grain counter with EMA for **background noise**; aug includes rotation, flip, translation, brightness, cutout, mixup, mosaic; Fig. 13 = attention-ablation confusion matrices; counting vs manual (MAE, R²). Closest “kernel counting” peer to sunflower seeds. |
| 6 | **Enhanced YOLO-based framework for accurate detection and identification of common wheat impurities with distinct objects** — Bagherpour & Peyruo (Scientific Reports 2025). [DOI:10.1038/s41598-025-23032-9](https://doi.org/10.1038/s41598-025-23032-9) | Multi-class confusion matrix (Fig. 4, YOLOv8x) for 11 impurity classes; highest confusion among morphologically similar grains; lighting/size/resolution cited — template for multi-class seed vs debris. |
| 7 | **Computer Vision for Cover Crop Seed-Mix Detection and Quantification** — Kumari et al. (AgriEngineering 2025). [DOI:10.3390/agriengineering4040059](https://doi.org/10.3390/agriengineering4040059) | Mixed small seeds (YOLOv5/v7); **centroid clustering** (≤15 px) post-process for overlap/occlusion duplicate FPs; per-class precision/recall/mAP tables (Fig. 9–10). |
| 8 | **Semi-Supervised Density Estimation with Background-Augmented Data for In Situ Seed Counting** — Sung et al. (Agriculture 2025). [DOI:10.3390/agriculture15151682](https://doi.org/10.3390/agriculture15151682) | Qualitative failure taxonomy (Fig. 13): **overlapped seeds** vs **background noise (straw/debris)**; notes 2D limits on severe 3D stacks. |
| 9 | **SoyCountNet: counting and locating soybean seeds in field** — (Frontiers in Plant Science 2026). [DOI:10.3389/fpls.2026.1743104](https://doi.org/10.3389/fpls.2026.1743104) | Point-based counting; failure modes = missed detections under **shadow, low contrast, dense overlap**; overlap penalty in loss. |
| 10 | **Automated Sunflower Head Detection … YOLOv11** — (Sustainability 2026). [DOI:10.3390/su18021026](https://doi.org/10.3390/su18021026) | Sunflower-specific YOLO; sweeps **conf and IoU thresholds** (optimal conf=0.50, IoU=0.40) — supports pairing with `threshold_sweep.py`. |
| 11 | **Lightweight safflower cluster detection (SF-YOLO)** — (Scientific Reports 2024). [DOI:10.1038/s41598-024-69584-0](https://doi.org/10.1038/s41598-024-69584-0) | Small dense floral clusters in complex fields; data aug for **lighting/noise/angle**; NWD loss for small-target localization. |
| 12 | **Global Wheat Head Detection Challenges: Winning Models and Application for Head Counting** — David et al. (Plant Phenomics 2023). [DOI:10.34133/plantphenomics.0059](https://doi.org/10.34133/plantphenomics.0059) · [PMC10795497](https://pmc.ncbi.nlm.nih.gov/articles/PMC10795497/) | Competition post-mortem: reports **FPR/FNR**, counting bias/RMSE, occlusion-driven FN; IoU-threshold and FP/FN trade-offs (not NMS/conf sweeps). |
| 13 | **Towards Resilient Agriculture: UAV Wheat Head Detection** — (Mathematics 2025). [DOI:10.3390/math13233844](https://doi.org/10.3390/math13233844) | Manual FP decomposition on GWHD: **background hallucination**, **duplicate detection** (IoU>0.3 between preds), **ambiguous/occluded/unlabeled** — rare explicit agricultural FP sub-taxonomy. |
| 14 | **Enhancing Sunflower Head Segmentation and Counting Using Improved UNet** — Bai et al. (Applied Engineering in Agriculture 2025, 41:125–136). [DOI:10.13031/aea.16163](https://doi.org/10.13031/aea.16163) | Segmentation-based sunflower head counting; alternative to bbox pipeline when heads touch/overlap. |
| 15 | **Improved Field-Based Soybean Seed Counting and Localization (P2PNet-Soy)** — (Plant Phenomics 2023). [DOI:10.34133/plantphenomics.0026](https://doi.org/10.34133/plantphenomics.0026) · [PMC10019992](https://pmc.ncbi.nlm.nih.gov/articles/PMC10019992/) | **Over-prediction** (extra points far from GT); **k-d tree clustering** post-processing reduces MAE (105.55→12.94) — relevant if harchoc adds point/density heads. |

---

## 2. Recommended FP categories for seed/kernel counting

Literature uses three layers; harchoc should report **Layer A** automatically and optionally **Layer B** via crop review.

### Layer A — Geometry-aware (automated, TIDE-aligned)

Implemented in `analyze_errors()` (defaults match TIDE thresholds):

| Category | Rule (IoU vs GT) | Agricultural interpretation |
|----------|------------------|----------------------------|
| **Background (Bkg)** | No GT overlap (IoU = 0) | Debris, chaff, hull fragments, soil specks, stem bits, glare spots |
| **Localization (Loc)** | Same class, 0 < IoU < 0.5 | Edge of seed, partial box, stacked/adjacent kernel offset |
| **Classification (Cls)** | Wrong class, IoU ≥ 0.5 | Damaged vs intact seed, weed seed, foreign grain (cf. wheat impurity paper) |
| **Duplicate (Dupe)** | Same GT already matched (same class, IoU ≥ \(t_f\)) | Double-count on touching **developed** or **aborted** kernels — dominant failure mode in GWHD/GrainNet dense scenes |
| **Both (Cls+Loc)** *(not split)* | TIDE: \(t_b \le\) IoU \(\le t_f\) vs wrong-class GT | Rare in two-class tray data; harchoc routes wrong-class at IoU ≥ \(t_f\) to **Cls**, else **Loc**/**Bkg** |
| **Miss (FN)** | Unmatched GT | Occluded/stacked seed, motion blur, out-of-focus |

### Layer B — Semantic (manual or semi-automated on `--export-fp-crops`)

Papers rarely label these automatically; GWHD 2025 and seed-counting work use expert review of sampled FPs:

| Sub-type | Typical cause | Mitigation cited |
|----------|---------------|------------------|
| **Debris / foreign material** | Texture/color similar to seed | Hard-negative mining; impurity class (Sci Reports 2025) |
| **Shadow / illumination** | Low contrast, specular reflection | Augmentation; adaptive contrast (soybean/sunflower papers) |
| **Overlap / adhesion** | 3D stack, touching kernels | Centroid clustering (cover-crop seeds); NWD/GIoU losses (safflower, wheat head) |
| **Background texture** | Soil, tray pattern, hull surface | Attention/EMA modules (GrainNet); background-augmented training |
| **Boundary / partial object** | Crop edge, cut-off seed | Separate “ambiguous” bucket (GWHD UAV paper) |
| **Label noise / unlabeled GT** | Missed annotation in dense piles | Manual audit; affects FP not model (noted in jujube & GWHD discussions) |

For **two-class** tray counting, expect **Cls** (developed ↔ aborted) and **Dupe** to matter as much as **Bkg** once conf is tuned; **Bkg** still captures tray texture, hull chips, and glare. Layer B labels should tag class-confusion vs adhesion separately.

### Applicability: developed / aborted dense detection

| Factor | Implication for taxonomy |
|--------|---------------------------|
| **~500 GT boxes / image** | **Dupe** and overlap/adhesion (GrainNet, GWHD, cover-crop clustering) are first-class; eval `max_det` caps can inflate **Miss** before taxonomy runs — parity at 3000 documented in [`s14_maxdet_truncation.json`](../../reports/hsp/s14_maxdet_truncation.json) (backlog **P0-1**). |
| **Two classes (morphologically similar)** | **Cls** = predicted **aborted** on **developed** GT (or vice versa); use per-class confusion overlays (impurity / variety papers) alongside TIDE buckets. |
| **Benchtop vs field** | Layer B **debris/foreign material** less central than **shadow, specular glare, 3D stack overlap** (Sung, SoyCountNet, safflower cluster papers). |
| **Counting-first metrics** | Report `counting_metrics` (MAE/RMSE/rRMSE) with `fp_breakdown`; GWHD/GWC retrospectives pair FPR/FNR with counting RMSE (#12). |
| **Size strata** | Most seeds fall in COCO **small** stratum (sqrt area under 32 px at native resolution); stratify `bbox_area_strata` when comparing aug or `imgsz`. |
| **Threshold coupling** | Run taxonomy at locked conf from `threshold_sweep.py` on **val**; manuscript numbers on **test** exports only. |

---

## 3. How papers structure confusion matrices and failure modes

### Detection (bbox) papers

- **Ultralytics/YOLO-style confusion matrix**: rows = GT class, cols = predicted class; diagonal = TP classifications; last column/row often aggregates **background FPs** and **missed GT**.
- **Per-class AP + FPR/FNR**: GWHD competitions and GWC 2021 paper report FP rate = FP/(TP+FP+FN) alongside counting RMSE.
- **Qualitative panels**: side-by-side GT vs pred overlays on worst images (GrainNet Fig. 8; sunflower YOLO papers).

### Classification / variety papers (kernels)

- **Square confusion matrix** over seed varieties (LWheatNet, maize quality): off-diagonal = class confusion; authors discuss **visual similarity** between varieties.
- **Impurity detection**: multi-class matrix including stones, chaff, pest-damaged grain (Sci Reports 2025).

### Error taxonomy papers (preferred for reviewers)

- **TIDE**: stacked bar or pie of six error types; **dAP** shows impact on AP if each error class were oracle-fixed.
- **Hoiem**: FP pie chart by error type; separate curves for object size/occlusion strata.
- **Manual FP sampling** (GWHD Mathematics 2025): N=500 random FPs → 3-way bar chart (background / duplicate / ambiguous).

**Reviewer expectation**: combine (1) numeric taxonomy counts, (2) top-K example crops, (3) threshold sensitivity — not raw mAP alone.

---

## 4. Code cross-check (`error_analysis.py`, `error_taxonomy.py`)

| Literature / doc claim | Repo state (2026-05-29) |
|------------------------|-------------------------|
| TIDE \(t_f=0.5\), \(t_b=0.1\) | Defaults `--iou 0.5`, `--iou-bg 0.1` in `analyze_errors()` |
| Six buckets incl. **Dupe** | `fp_breakdown`: `background`, `localization`, `classification`, `dupe`; `counts["dupe"]` separate from `counts["fp"]` |
| **Cls** cross-class | `fp_breakdown["classification"]` when IoU ≥ \(t_f\) vs wrong-class GT |
| Area strata + conf grid | `build_bbox_area_strata` / `build_conf_taxonomy_grid` on `instance_events` (`fp_*`, `dupe`, `tp`, `fn`) |
| `--export-fp-crops` | `export_topk_fp_crops()` — top-K by score, PIL crops under `--fp-crops-dir`; needs `DATASET_ROOT` + `file_name` in exports |
| TIDE **delta-AP** / **Both** oracle | **Not implemented** — counts only; optional `tidecv` cross-check (backlog **P1-TIDECV**) |
| Layer B `review_category` | **Not implemented** — crops carry `error_type` only |
| FN pairing grid | Instance-level `fn` events + strata; no TIDE-style FN↔FP partner tags yet |

### Remaining gaps (backlog-aligned)

**High priority**

1. **Real test preds** — `eval.py` export → `error_analysis.py --export-fp-crops` (blocked on trained weights / HSP exports).
2. **TIDE delta-AP per bucket** — reviewer-facing impact bars (literature §3).

**Medium priority**

3. **Layer B manifest** — `review_category` CSV for debris/shadow/overlap/class-confusion on exported crops.
4. **Threshold grid** — re-run taxonomy across conf grid from val sweep; lock conf for test report.
5. **Per-class `fp_breakdown`** — split **Cls** by developed vs aborted.
6. **FN↔FP pairing** — TIDE Miss vs partner Loc/Cls (partial: FN in strata only).

### `--light` vs full GT/pred pipeline

| Aspect | `--light` (`data/examples/gt.json`, `preds.json`) | Full mode (`--gt-json`, `--preds-json` from eval) |
|--------|---------------------------------------------------|---------------------------------------------------|
| Purpose | CI-safe scaffold; schema/taxonomy tests | Production error reports on real val/test split |
| Data | Tracked minimal examples | Export from `scripts/eval.py` on `DATASET_ROOT` |
| Images | May lack `DATASET_ROOT`; crops skipped | Requires dataset root for `--export-fp-crops` |
| Weights | Not loaded (no inference) | Weights path recorded in metadata only |
| Recommendation | Keep for unit tests & config dry-runs | Use for reviewer artifacts: run eval → export preds JSON → `error_analysis.py --export-fp-crops` |

**Full pipeline sketch:**

```bash
# 1. Export GT + preds JSON (val for sweep/tuning; test for manuscript)
python scripts/eval.py --weights models/best.pt \
  --split-file data/splits/test.txt \
  --export-gt-json reports/hsp/gt_test.json \
  --export-preds-json reports/hsp/preds_test.json \
  --export-conf 0.001 --export-iou 0.3

# 2. Error analysis on real data (test for reviewer artifacts)
python scripts/error_analysis.py \
  --gt-json reports/hsp/gt_test.json \
  --preds-json reports/hsp/preds_test.json \
  --export-fp-crops --fp-crops-topk 200 \
  --out reports/error_analysis/summary.json \
  --report reports/error_analysis/report.json
```

`eval.py` is **test-only** for mAP; do not use it to export val preds. CI: `--light` + `configs/experiments/error_analysis_light.json`.

---

## 5. Similar benchmarks and datasets

| Benchmark | Domain | URL | Notes for harchoc |
|-----------|--------|-----|-------------------|
| **Global Wheat Head Detection (GWHD)** | Dense small heads | [global-wheat.com](http://www.global-wheat.com/) | Gold standard for overlap/occlusion; counting metrics |
| **GrainNet wheat grain dataset** | Loose/adhered kernels | [Plant Methods paper](https://doi.org/10.1186/s13007-025-01363-y) | Closest scale to seed counting; adhesion levels |
| **Global Wheat 2020 (Ultralytics)** | Wheat heads | [Ultralytics docs](https://docs.ultralytics.com/datasets/detect/globalwheat2020/) | YOLO training baseline |
| **Cover crop seed mix** | Multi-class seeds | [AgriEngineering 2025](https://doi.org/10.3390/agriengineering4040059) | Overlap clustering post-process |
| **COCO / LVIS** | General detection | [cocodataset.org](https://cocodataset.org/) | TIDE reference implementation; not agricultural |

No public **sunflower seed** detection benchmark was found at GWHD/GrainNet scale; sunflower head datasets (UAV YOLO papers, UNet segmentation) are the closest crop-specific peers.

---

## 6. Summary mapping: literature → harchoc `error_taxonomy`

```
TIDE type          harchoc today          Seed-counting label
─────────────────────────────────────────────────────────────
Bkg                background (IoU < t_b) debris, texture, glare
Loc                localization           offset box, partial seed
Cls                classification         wrong class / variety
Both               (not split)            wrong box + wrong class @ mid IoU
Dupe               dupe                   double count on touch
Miss               fn                     occluded / stacked / max_det cap
```

Priority for reviewer feedback: run **full-mode FP crops** on real **test** exports; report **developed/aborted** confusion matrix plus TIDE-style `fp_breakdown` at locked conf.

### Manuscript narrative (**MS-FP-LOC-NARR**)

At locked operating conf (~0.15), **background and localization FPs dominate** cross-class confusion on test exports — state explicitly in Results (not only figure captions). Evidence: [`error_test_report.json`](../../reports/hsp/error_test_report.json) `fp_breakdown` + FP crop taxonomy (**P1-TIDE**). Closest peers: GWHD overlap/duplicate FPs, GrainNet background EMA (`grainnet2025` in [literature_validated.json](../manuscript/literature_validated.json)).

### ARCH-EMA-BG-SPIKE (GrainNet EMA)

**Done (2026-05-29):** Literature review; **implementation cancelled**. EMA is a YOLOv7 **head** module (not Ultralytics aug/conf); test **Bkg** ~35% vs **Loc** ~58% of FP buckets — threshold/aug/deploy levers rank higher. Decision matrix: [`arch_ema_bg_spike_literature.md`](arch_ema_bg_spike_literature.md).

### Manuscript narrative (**MS-ORIG**)

Introduction contribution bullets vs GrainNet, LWCD-YOLO, and fp_taxonomy peers: [`originality_contribution_peers.md`](../manuscript/originality_contribution_peers.md) (**MS-ORIG** Done); gap mirror [§2](../manuscript/reviewer_comments_backlog_gap.md#2-manuscript-draft--originality-vs-crop-seed-detection-introduction).

---

*Validated 2026-05-29. Generated for harchoc reviewer response. Not committed.*

**Validated literature registry:** [`docs/manuscript/literature_validated.md`](../manuscript/literature_validated.md) · **Related Work (MS-LIT):** [`related_work_outline.md`](../manuscript/related_work_outline.md)
