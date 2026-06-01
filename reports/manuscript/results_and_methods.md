# Methods and Results (HSP reproduction)

This document is the canonical Methods and Results text for journal submission. It describes the frozen **counting-first eval protocol** (repo shorthand **HSP** — internal label for `reports/hsp/` exports, not a published acronym) and primary quantitative results for `models/best2.pt` on the CVAT-annotated corpus (1093 images; train/validation/test 875/109/109). Tables and figures are regenerated from those exports (`experiment.py manuscript-docx-repro`).

We retained the upstream production checkpoint from the public [sunflower-Detector](https://github.com/Fami2040/sunflower-Detector) repository and re-evaluated it under a pre-registered counting protocol on versioned splits. Comparative training (augmentation grid and detector zoo) tests whether any recipe beats that anchor on the **same** test split at the **same** validation-locked confidence—not whether a new architecture was invented.

**Literature registry:** Peer metadata below is copied from [`docs/manuscript/literature_validated.json`](../../docs/manuscript/literature_validated.json) (`literature_validated.v1`, DOI audit **2026-06-01**, [`literature_doi_audit_2026-06-01.md`](literature_doi_audit_2026-06-01.md), per-paper reports [`lit_audit/`](lit_audit/README.md)). Do not add DOIs without matching that registry.

---

## Methods

### Dataset

See [`dataset.md`](dataset.md) for the full Methods subsection (corpus tiers, imaging, annotations, splits, tray structure, and provenance vs upstream code).

### Detection and counting protocol

Inference uses YOLOv8m weights at input size 1280 px. Predictions are exported at confidence 0.001 with non-maximum suppression IoU 0.3 and up to 3000 detections per image. Per-image seed counts are obtained by matching predictions to ground truth with category-aware IoU 0.3.

A single global confidence is chosen on the validation split by minimizing count MAE (`min_count_mae`). That value (approximately 0.15) is applied unchanged on the test split. No test image influences threshold choice. The operating point optimizes total seed count error rather than ranking mAP or per-class F1; elevated false positives at this threshold are expected in dense scenes ([GWHD benchmark framing for dense overlap and duplicate false positives](https://doi.org/10.34133/2020/3521852)).

Production deployment (sunflower gate plus sliced inference) follows a separate protocol and is not used for the numbers below. The two-stage gate→detector pattern is discussed **by analogy** to staged UAV **action-recognition** pipelines ([Alshehri et al., 2025](https://doi.org/10.3389/fnbot.2025.1582995)—not a detect-then-classify seed method); HSP evaluation remains single-stage full-frame export.

### Comparative design rationale (literature-guided)

Augmentation ablations and model-zoo training follow benchtop **seed/kernel** precedents and phenotyping evaluation practice:

| Topic | HARCHOC choice | Verified source (DOI) |
|-------|----------------|-------------------------|
| Dense kernel counting + heavy offline aug | Count MAE / R² vs manual; YOLOv7 + attention | [10.1186/s13007-025-01363-y](https://doi.org/10.1186/s13007-025-01363-y) (GrainNet, 2025, *Plant Methods*) |
| Benchtop corn kernels; LWCD reports mosaic off in their training config | Dense-benchtop **detection** peer; our S2 mosaic=0 **worsened** MAE (see Results) | [10.3390/agriculture15181968](https://doi.org/10.3390/agriculture15181968) (LWCD-YOLO, 2025, *Agriculture*) |
| Field wheat heads; dense overlap | Positions tray-scale **seeds** vs field **heads** | [10.34133/2020/3521852](https://doi.org/10.34133/2020/3521852) (GWHD, 2020, *Plant Phenomics*) |
| Sunflower **head** UAV detection | Related Work + limitations; not direct SOTA peer | [10.3390/su18021026](https://doi.org/10.3390/su18021026) (Iamchuen et al., 2026, *Sustainability*) |
| Sunflower disease TL; domain shift | Discussion limitations (field/variety) | [10.55730/1300-0152.2763](https://doi.org/10.55730/1300-0152.2763) (Gulzar, 2025, *Turkish J. Biol.*) |
| Tray/session transfer (planned finetune) | Two-stage freeze + low LR on holdout tray (**analogy**) | [arXiv:2003.06957](https://arxiv.org/abs/2003.06957) (TFA, ICML 2020); [arXiv:2505.01016](https://arxiv.org/abs/2505.01016) (Gandhi & Gandhi, 2025 **preprint**) |
| Graded trust at ambiguous boundaries | Score bands + FP taxonomy (**analogy**; not a third YOLO class) | [10.1109/TFUZZ.2025.3549791](https://doi.org/10.1109/TFUZZ.2025.3549791) (Yao et al., 2025, *IEEE TFS* — **tabular fuzzy regression**, not vision) |
| Interpretability framing | Grad-CAM panels alongside error crops (**not** Ren's materials method) | [10.1016/j.scriptamat.2024.116350](https://doi.org/10.1016/j.scriptamat.2024.116350) (Ren et al., 2025, *Scripta Materialia* — property prediction) |
| OCT transfer-learning analogy (Discussion) | Staged adaptation narrative (**classification**, not OD) | [10.1371/journal.pone.0296175](https://doi.org/10.1371/journal.pone.0296175) (Yang et al., 2024, *PLoS ONE*) |

Deeper research notes (not all DOI-verified in the registry): [`docs/research/augmentation_robustness_literature.md`](../../docs/research/augmentation_robustness_literature.md), [`docs/research/training_tech_scan_2026_augmentation.md`](../../docs/research/training_tech_scan_2026_augmentation.md), [`docs/manuscript/originality_contribution_peers.md`](../../docs/manuscript/originality_contribution_peers.md).

### Augmentation comparison (closed program)

**Anchor.** Production `best2.pt` was trained in upstream `main` (YOLOv8m, 100 epochs, `imgsz=1280`, conservative online augmentation: low mosaic, photometric jitter, no mixup). Provenance: [`docs/ORIGIN_MAIN_AND_DATASET.md`](../../docs/ORIGIN_MAIN_AND_DATASET.md), [`configs/origin/public_yolov8_train_reference.json`](../../configs/origin/public_yolov8_train_reference.json). HARCHOC codifies a similar recipe in `configs/aug/robustness_minimal.yaml`, aligned with GrainNet and LWCD-YOLO precedents above.

**Protocol (identical to anchor for all rows).** The anchor was re-evaluated on frozen CVAT splits (test count MAE **61.3** at validation-locked conf **≈0.15**). We then trained:

- **Fifteen 15-epoch** YOLOv8m smokes (**S0–S13**, plus sweeps), each with literature-guided augmentation overrides ([`configs/experiments/aug_smoke_index.json`](../../configs/experiments/aug_smoke_index.json)).
- One **100-epoch** confirmatory train (`aug_confirm_winner_100ep`) on `robustness_minimal.yaml`.
- **S14** — eval-only control (`max_det=300`); not an augmentation competitor.

Every train was scored on `data/splits/test.txt` (*n*=109) at the **same** locked confidence (no test leakage). Primary metric: **test count MAE**.

**Results (augmentation).** No alternative train beat the anchor.

| Condition | Test count MAE | Δ vs anchor 61.3 | Notes |
|-----------|---------------:|-----------------:|-------|
| Anchor `best2.pt` | **61.3** | — | Production reference |
| 100-ep confirm (`robustness_minimal`) | **64.1** | +2.9 | [`aug_confirm_winner_100ep_summary.json`](../../reports/aug_smoke/aug_confirm_winner_100ep_summary.json) |
| Best 15-ep smoke (**S1** / S0 cluster) | **68.9** | +7.6 | `close_mosaic=3`; S0/S13 equivalent |
| **S2** mosaic=0 | **147.4** | +86.1 | Rejected arm — supports **low** mosaic on sunflower trays, not LWCD-style mosaic-off ([LWCD-YOLO](https://doi.org/10.3390/agriculture15181968) used mosaic-off on corn; our S2 did not transfer) |
| S14 eval `max_det=300` | 265.8 | — | Truncation control only |

Full rankings: [`reports/aug_smoke/leaderboard.md`](../../reports/aug_smoke/leaderboard.md), [`reports/manuscript/tables/aug_smoke_top_n.md`](tables/aug_smoke_top_n.md), programmatic summary [`reports/aug_smoke/comparative_analysis.json`](../../reports/aug_smoke/comparative_analysis.json).

**Conclusion.** Retain **`best2.pt`** for manuscript headline metrics; augmentation program **closed** ([`backlog.md`](../../backlog.md) § Aug).

### Model zoo (SOTA comparison on 8 GiB GPU)

Group **`zoo_yolo_only`**: four Ultralytics YOLO M-scale rows × 100 epochs @ 1280, `robustness_minimal` aug, identical splits and counting protocol. See [`docs/zoo_comparison_design.md`](../../docs/zoo_comparison_design.md).

| Model | Test count MAE | mAP50 (test) | Status |
|-------|---------------:|-------------:|--------|
| Anchor `best2.pt` | **61.3** | 0.18 | Production |
| YOLOv8m retrain | 111.9 | 0.064 | Complete |
| YOLO11m | 119.6 | 0.077 | Complete |
| YOLO26m | 95.3 | 0.406 | Complete |
| YOLOv10m | — | — | In progress at draft (not OOM-deferred; ~3.3 GiB 1-ep probe) |

**Ultralytics RT-DETR-L** @ 1280 batch=1: training **OOM** on 8 GiB (measured VRAM probe). **External DETR** (D-FINE, DEIM, RT-DETRv2): same splits; D-FINE trained (~6.7 GiB peak); DEIM/rtdetrv2 blocked on integration/port issues — not a blanket >8 GiB exclusion.

Table: [`tables/zoo_core.md`](tables/zoo_core.md) · JSON: [`reports/hsp/matrix_train.json`](../../reports/hsp/matrix_train.json) · Status: [`FRESHNESS.md`](FRESHNESS.md).

---

## Results

### Primary counting and detection metrics

Table 1 summarizes pooled metrics at the validation-locked confidence.

| Split | Count MAE | 95% CI | mAP50 | mAP50-95 | Conf |
|-------|----------:|--------|------:|---------:|-----:|
| test | 61.3 | 51.3–71.3 | 0.180 | 0.060 | 0.15 |
| val | 71.0 | 58.8–84.2 | — | — | 0.15 |

Validation metrics are reported for threshold-selection transparency only; the manuscript primary endpoint is test count MAE. Evidence: [`reports/hsp/dual_metric.json`](../../reports/hsp/dual_metric.json), [`reports/hsp/eval_test_map.json`](../../reports/hsp/eval_test_map.json).

### Per-image relative counting error (test)

| Metric | Value |
|--------|------:|
| Mean relative error (%) | 12.0 |
| Median relative error (%) | 9.3 |
| Share of images with relative error &lt; 2% | 13.8% |
| Share with relative error &lt; 5% | 31.2% |
| Share with relative error &lt; 10% | 53.2% |

A smaller blinded audit (*n* = 50) in the earlier submitted draft reported different distribution statistics. On the full held-out test set (*n* = 109) at the locked operating point, the table above applies; the two samples must not be conflated.

### Detection mAP versus counting

Test ranking mAP50 under the canonical HSP evaluation path is **0.18**. An earlier submitted draft cited **~0.793**; that value is **not** reproduced on current held-out test exports ([`docx_vs_submission.md`](docx_vs_submission.md), [`docs/manuscript/val_test_map_gap.md`](../../docs/manuscript/val_test_map_gap.md)). Peak training-validation mAP near **0.97** reflects in-training validation geometry and must not be substituted for test ranking mAP. Counting MAE remains the primary metric for model comparison.

### Augmentation and model zoo (summary)

See Methods subsections above. No augmentation-only or completed Ultralytics zoo row beat the anchor on test count MAE at the locked confidence.

### Error taxonomy and generalization

At the locked operating point, false-positive contribution to error share is dominated by **localization and background** relative to class confusion ([`reports/hsp/error_test_report.json`](../../reports/hsp/error_test_report.json), [`reports/hsp/tide_bucket_summary.json`](../../reports/hsp/tide_bucket_summary.json)), consistent with dense small-object scenes and GWHD-style duplicate/overlap challenges.

Tray-level count MAE varies widely across acquisition sessions on held-out tray slices ([`reports/domains/domain_eval.json`](../../reports/domains/domain_eval.json)), while pooled test MAE is **61.3**. Single-site benchtop imaging does not establish field or multi-site generalization ([Gulzar, 2025](https://doi.org/10.55730/1300-0152.2763)).

---

## Pending scientific tasks (backlog snapshot)

Active manuscript-facing work from [`backlog.md`](../../backlog.md) and [`docs/manuscript/reviewer_comments_backlog_gap.md`](../../docs/manuscript/reviewer_comments_backlog_gap.md). **Do not cite results until the listed artifact exists.**

| ID | Priority | Task | Literature / design basis | Blocker / next artifact |
|----|----------|------|---------------------------|-------------------------|
| **P0-5** | P0 | Complete **`zoo_yolo_only`** (YOLOv10m 100 ep) | Same as zoo table above | `runs/hsp_zoo/yolov10m_e100_s0/weights/best.pt` → refresh [`tables/zoo_core.md`](tables/zoo_core.md) |
| **P1-ZOO-PARITY** | P1 | Count MAE columns for all matrix rows | Counting-first gate ([GrainNet](https://doi.org/10.1186/s13007-025-01363-y)) | After P0-5 |
| **P1-RTDETR-COUNT-REFRESH** | P1 | RT-DETR count MAE or document 8 GiB skip | RT-DETR query limits — [`zoo_comparison_design.md`](../../docs/zoo_comparison_design.md) | VRAM probe log; or Methods footnote |
| **MS-SOTA** | P1 | Manuscript SOTA table prose | §19 gap map | Blocked on P0-5 |
| **P1-FINETUNE-TRAY** | P1 | Staged finetune on weak trays (`3a5-9`, `350`, `200-3-1`) | [TFA](https://arxiv.org/abs/2003.06957), [Gandhi preprint](https://arxiv.org/abs/2505.01016); [`FINETUNE_WEAK_TRAYS.md`](../../docs/FINETUNE_WEAK_TRAYS.md) | GPU queue `gpu_queue_post_zoo.json`; `reports/transfer/finetune_*_s2.json` |
| **P1-DOMAIN-TAGS** | P2 | Variety / maturity / lighting / site metadata | [Gulzar](https://doi.org/10.55730/1300-0152.2763) domain-shift review | CSV scaffold TBD |
| **DATA-ACQ-GEN** | P1 | New multi-condition acquisition | Field/generalization reviewer theme #4 | Not started |
| **MS-GEN** | P1 | Generalization Discussion from tray spread | `domain_eval.json` | Partial — narrative open |
| **MS-MANUAL-BASE** | P2 | Timed human counting baseline | Peer counting studies (GrainNet manual comparison) | **No completed human study in repo** — placeholders only |
| **LIT-VALIDATE** | P2 | Registry + doc sweep **2026-06-01**; new cites via JSON + audit | [`literature_validated.json`](../../docs/manuscript/literature_validated.json) | Ongoing for new papers |
| **External DETR stack** | P1 | DEIM / RT-DETRv2 train + HSP eval | [`detector_sources.v1.json`](../../configs/external/detector_sources.v1.json) | Port 29500 / `engine` import fixes |
| **Aug program** | — | S0–S14 + 100 ep | §18 above | **Closed** |

**Regenerate after GPU work:** `mamba run -n harchoc python scripts/experiment.py manuscript-preflight`

---

## Generated tables and figures

- `reports/manuscript/docx/tables/table_01_detection_metrics.md`
- `reports/manuscript/docx/tables/table_02_counting_summary.md`
- `reports/manuscript/docx/tables/table_03_error_bins.md`
- `reports/manuscript/docx/figures/` (regenerate via `experiment.py manuscript-docx-repro`)

## Cross-links (agents and co-authors)

| Document | Role |
|----------|------|
| [`response_to_reviewers.md`](response_to_reviewers.md) | Point-by-point Reviewer 2 responses |
| [`reviewer2_rebuttal_for_coauthor.md`](reviewer2_rebuttal_for_coauthor.md) | Shareable rebuttal draft |
| [`docs/manuscript/reviewer_comments_backlog_gap.md`](../../docs/manuscript/reviewer_comments_backlog_gap.md) | Full gap map (§0, §18–§20) |
| [`reports/hsp/p0_summary.md`](../../reports/hsp/p0_summary.md) | One-page headline card |
