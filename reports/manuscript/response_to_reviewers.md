# Response to Reviewer 2

We thank the reviewer for a careful reading. Below we address each major comment. Verbatim reviewer text is in `reports/reviewer2.md`. Quantitative claims below refer to Tables 1–3 and the Methods/Results narrative in `reports/manuscript/results_and_methods.md`.

**Snapshot (2026-06-01):** Headline metrics and aug comparisons are final on frozen splits. Multi-detector zoo (`zoo_yolo_only`) has three completed 100-epoch Ultralytics rows (none beat anchor **61.3**); **YOLOv10m** 100-epoch training was in progress on the same 8 GiB GPU (VRAM probe ~3.3 GiB peak at 1 epoch—not deferred for OOM). See `reports/manuscript/FRESHNESS.md` before paste.

---

## 1. Abstract standardization

**Comment.** The abstract should follow purpose, methods, results, and conclusions with complete key information.

**Response.** We rewrote the abstract in IMRaD form (Background, Methods, Results, Conclusions), reporting the counting-first protocol, held-out test count MAE (61.3 seeds per image, 95% CI 51.3–71.3), relative error statistics on the full test set, augmentation and zoo comparisons, reconciled test mAP50 (0.18), and explicit scope limits (single-site benchtop imaging). See `reports/manuscript/abstract.md` for the replacement text.

**Evidence.** Table 1; Table 2.

---

## 2. Originality and contribution

**Comment.** The work applies existing YOLOv8 without substantive architectural innovation for dense sunflower seeds.

**Response.** We do not claim a new backbone. The contribution is an operational and evaluative one: (i) a two-class, seed-level sunflower tray benchmark with frozen splits; (ii) a counting-first protocol that separates threshold selection on validation from held-out test reporting; (iii) systematic augmentation and multi-detector comparisons gated on count MAE rather than peak validation mAP alone; and (iv) tray-level domain reporting and error taxonomy for dense counting. Positioning relative to benchtop kernel detectors (wheat, corn) and field sunflower head imaging is expanded in the revised Related Work section.

**Evidence.** Methods section; supplementary domain metrics (tray-level MAE spread).

---

## 3. Lack of SOTA comparison

**Comment.** Only YOLOv8 was tested; comparisons to recent crop small-object methods are missing.

**Response.** We ran a structured augmentation grid (15-epoch smokes plus 100-epoch confirm) and a counting-gated model zoo on identical frozen splits (`zoo_yolo_only`: YOLOv8m, YOLOv10m, YOLO11m, YOLO26m at 1280 px, 100 epochs, robustness-minimal aug). Completed retrains at the validation-locked confidence yielded test count MAE **111.9** (YOLOv8m), **119.6** (YOLO11m), and **95.3** (YOLO26m)—all worse than the production anchor **61.3**. **YOLOv10m** was queued but not finished in the first matrix pass; a 100-epoch retrain was underway when this rebuttal was drafted (1-epoch VRAM probe ~3.3 GiB on 8 GiB hardware). Ultralytics RT-DETR-L training at 1280 batch=1 OOMs on 8 GiB (measured probe); external DETR stacks (D-FINE, DEIM, RT-DETRv2) use the same splits but depend on integration scheduling, not blanket >8 GiB exclusion. All comparisons use count MAE as the primary gate, with test ranking mAP reported separately.

**Evidence.** Table 1; augmentation and zoo summary table in Results; Figure 6 (metrics panels) where regenerated.

---

## 4. Single scenario and generalization

**Comment.** Data come from one site, growth stage, and indoor lighting; field and multi-condition validation are absent.

**Response.** We agree and state this limitation explicitly. Supplementary tray-level evaluation shows wide spread in count MAE across acquisition sessions (approximately 17–244 seeds per image on tray slices versus pooled test MAE 61.3). We do not claim field robustness, variety coverage, or cross-regional validity without new data. Planned acquisition and optional leave-one-tray-out evaluation are noted as future work.

**Evidence.** Discussion paragraph on domain spread; supplementary domain table.

---

## 5. Reproducibility

**Comment.** Training parameters, thresholds, and procedures are insufficiently documented.

**Response.** We document the full HSP protocol: split files, export hyperparameters (confidence 0.001, IoU 0.3, max_det 3000), validation-based `min_count_mae` threshold lock, test-only headline metrics, and reproducible experiment bundles. Frozen weights (`best2.pt`) and dataset manifest are cited with provenance. Machine-readable exports support independent recomputation of all headline numbers.

**Evidence.** Methods in `reports/manuscript/results_and_methods.md`; Table 1 footnotes.

---

## 6. Literature review

**Comment.** Crop phenotyping, sunflower detection, and small-object recognition are not systematically reviewed; gaps and motivation are unclear.

**Response.** Related Work is reorganized into plant phenotyping and dense counting, sunflower imaging (heads versus seeds), small-object detection, crop kernel peers, counting-first evaluation, and transfer or deployment analogies. We state the gap explicitly: prior benchtop kernel work rarely addresses two-class sunflower seed viability on fixed-camera trays with a locked-confidence test protocol and tray-level error reporting.

**Evidence.** Revised §2 in the manuscript file (prose drafted in the rebuttal and full outline maintained in repository literature notes for editors).

---

## 7. Figure standardization

**Comment.** Fonts, resolution, and annotations do not meet journal requirements.

**Response.** Quantitative figures (detection example, training curves, confusion matrices, metrics panels) are regenerated at 300 DPI with a single journal style from HSP exports. Dataset spatial panels and photographic setup figures remain manual. We will apply the journal template font and size requirements on export to Word.

**Evidence.** `reports/manuscript/docx/figures/`; Figure catalog in `reports/manuscript/docx/README.md`.

---

## Additional clarifications (metrics in submitted manuscript)

**mAP50 0.793.** Held-out test mAP50 under the canonical protocol is 0.18 (Table 1). The submitted value is not reproduced on current test exports; we separate ranking mAP from counting MAE and do not use peak training-validation mAP as a generalization claim.

**Relative error and “80% below 2%”.** On the full test set at locked confidence, mean relative error is 12.0% and 13.8% of images fall below 2% relative error (Table 2–3). The prior 80% figure applied to a smaller blinded audit and must be reported separately if retained.

**Telegram deployment statistics.** Success rate and latency cited in the submitted text are not validated in the scientific artifact bundle; production metrics require a dedicated deploy audit before citation.
