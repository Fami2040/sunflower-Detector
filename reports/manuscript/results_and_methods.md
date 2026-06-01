# Methods and Results (HSP reproduction)

This document is the canonical Methods and Results text for journal submission. It describes the frozen HARCHOC Sunflower Phenotyping (HSP) protocol and primary quantitative results for `models/best2.pt` on the CVAT-annotated corpus (1093 images; train/validation/test 875/109/109). Tables and figures are regenerated from HSP exports (`experiment.py manuscript-docx-repro`).

We retained the upstream production checkpoint from the public sunflower-Detector repository and re-evaluated it under a pre-registered counting protocol on versioned splits. Comparative training (augmentation grid and detector zoo) tests whether any recipe beats that anchor on the **same** test split at the **same** validation-locked confidence—not whether a new architecture was invented.

## Methods

### Dataset

See [`dataset.md`](dataset.md) for the full Methods subsection (corpus tiers, imaging, annotations, splits, tray structure, and provenance vs upstream code).

### Detection and counting protocol

Inference uses YOLOv8m weights at input size 1280 px. Predictions are exported at confidence 0.001 with non-maximum suppression IoU 0.3 and up to 3000 detections per image. Per-image seed counts are obtained by matching predictions to ground truth with category-aware IoU 0.3.

A single global confidence is chosen on the validation split by minimizing count MAE (`min_count_mae`). That value (approximately 0.15) is applied unchanged on the test split. No test image influences threshold choice. The operating point optimizes total seed count error rather than ranking mAP or per-class F1; elevated false positives at this threshold are expected in dense scenes.

Production deployment (sunflower gate plus sliced inference) follows a separate protocol and is not used for the numbers below.

### Comparative design rationale

Augmentation ablations (15 epochs each, plus one 100-epoch confirmatory run) and model-zoo training (multiple YOLO generations at 100 epochs, with external DETR-family detectors on the same splits) follow literature-guided choices for dense benchtop kernels: conservative mosaic, counting-first model selection, and tray-level reporting for session shift.

## Results

### Primary counting and detection metrics

Table 1 summarizes pooled metrics at the validation-locked confidence.

| Split | Count MAE | 95% CI | mAP50 | mAP50-95 | Conf |
|-------|----------:|--------|------:|---------:|-----:|
| test | 61.3 | 51.3–71.3 | 0.180 | 0.060 | 0.15 |
| val | 71.0 | 58.8–84.2 | — | — | 0.15 |

Validation metrics are reported for threshold-selection transparency only; the manuscript primary endpoint is test count MAE.

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

Test ranking mAP50 under the canonical HSP evaluation path is 0.18. An earlier submitted draft cited a higher test mAP; that value is not reproduced on current held-out test exports (see `docx_vs_submission.md` for snapshot drift). Peak training-validation mAP near 0.97 reflects in-training validation geometry and must not be substituted for test ranking mAP. Counting MAE remains the primary metric for model comparison.

### Augmentation and model zoo (counting gate)

| Model / condition | Test count MAE @ locked conf | Status |
|-------------------|-----------------------------:|--------|
| Anchor (`best2.pt`) | **61.3** | Production checkpoint |
| 100-epoch aug confirm | 64.1 | Complete |
| Best 15-epoch aug smoke (S1) | 68.9 | Complete |
| YOLOv8m, 100 epochs (`zoo_yolo_only`) | 111.9 | Complete |
| YOLO11m, 100 epochs | 119.6 | Complete |
| YOLO26m, 100 epochs | 95.3 | Complete |
| YOLOv10m, 100 epochs | *(pending)* | In progress at draft date |

No augmentation-only or completed Ultralytics zoo row beat the anchor on test count MAE at the locked confidence. See `reports/manuscript/tables/zoo_core.md` for mAP columns and `FRESHNESS.md` for GPU queue status.

### Error taxonomy and generalization

At the locked operating point, false-positive contribution to error share is dominated by localization and background relative to class confusion, consistent with dense small-object scenes. Tray-level count MAE varies widely across acquisition sessions on held-out tray slices, while pooled test MAE is 61.3. Single-site benchtop imaging does not establish field or multi-site generalization.

## Generated tables and figures

- `reports/manuscript/docx/tables/table_01_detection_metrics.md`
- `reports/manuscript/docx/tables/table_02_counting_summary.md`
- `reports/manuscript/docx/tables/table_03_error_bins.md`
- `reports/manuscript/docx/figures/` (regenerate via `experiment.py manuscript-docx-repro`)
