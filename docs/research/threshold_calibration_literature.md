# Threshold sweeps, calibration, and PR curves for dense small-object detection

Literature review for **harchoc** benchtop **sunflower-seed** counting on **sunflower-cvat-2500** (frozen splits in `data/splits/`). Covers conf/NMS sweeps, post-hoc calibration, PR-curve operating points, and the val≫test mAP gap. Implementation cross-check: `scripts/threshold_sweep.py`, `harchoc/threshold_protocol.py`, `harchoc/dual_metric_report.py`, `scripts/experiment.py dual-metric`.

**Classes (repo only):** id **0 = developed**, id **1 = aborted** (`harchoc/sunflower_dataset.py`, `data/README.md`). Agri-UAV papers below count **plants or heads**, not developed vs aborted—cite them for threshold/counting **methodology**, not label semantics.

**In scope:** train defaults in `train_yolov8m_baseline.json` (`conf=0.05`, `iou=0.3`, `max_det=3000`); **val sweep → lock conf → test report** (no threshold re-tuning on test).

**Out of scope here:** `telegram_bot.py` / SAHI deploy thresholds as the primary manuscript path; closing large mAP gaps by threshold alone (run `split_drift.py` first).

**GPU / env:** use `mamba run -n harchoc python …` for `eval.py`, `threshold_sweep.py`, and exports (see `.cursor/rules/gpu-env-and-dataset.mdc`).

---

## Executive summary

| Question | Literature consensus | harchoc implication |
|----------|-------------------|-------------------|
| What metric for **ranking** detectors? | mAP / AP (area under PR curve, all conf thresholds) | `eval.py` → Ultralytics `model.val()` on **test** split |
| What metric for **counting / deployment**? | F1, rRMSE, R² at a **single operating point** | `threshold_sweep.py` → maximize F1 or constrain FP/image |
| How to pick conf threshold? | Sweep on held-out set; F1-max, oLRP, or constraint-based (min recall) | Sweep **val**; lock threshold; report **test** counts |
| Does calibration change mAP? | Not if scores are remapped **monotonically** (Kuzucu et al. 2024, Tab. 5: AP unchanged at 44.1); Guo (2017) only proves TS preserves **classification** argmax | Optional monotonic post-hoc (IR/PS) before sweep if scores miscalibrated |
| NMS IoU vs conf? | Separate knobs; low IoU NMS hurts dense recall | Train/val export: `iou=0.3` per baseline; re-export if deploy NMS differs |
| val ≫ test? | Domain shift / split leakage, not threshold | Run `split_drift.py` first; do not tune on test |

---

## Applicability to HSP (sunflower-cvat-2500)

| Aspect | Literature (UAV / field) | HSP benchtop |
|--------|-------------------------|--------------|
| Object | Sunflower **heads** or early **plants** | Hundreds of **seeds** per head (`developed` / `aborted`) |
| Metric for acceptance | rRMSE, R², F1 @ tuned conf | **test mAP** (`eval.py`) + **locked-count** MAE/F1 (`threshold_sweep` / `error_analysis`) |
| Threshold protocol | Grid conf ± NMS IoU on a holdout set | **val** export + sweep; **test** with `--locked-conf-from` only |
| Split discipline | Often single holdout | `train.txt` / `val.txt` / `test.txt` — manuscript mAP on **test**; tuning on **val** only |

**Manuscript chain (code-aligned):** `eval.py` export (val, then test) → `threshold_sweep.py` (val: `--select` / optional `--iou-grid` / `--calibrate`) → `threshold_sweep.py --locked-conf-from` (test) → `error_analysis.py --locked-conf-from` → `experiment.py dual-metric` → `dual_metric_report.v1`. Guardrails: `harchoc/threshold_protocol.py` rejects `--select`, `--iou-grid`, and `--calibrate` on **test** unless locked.

Companion scan: [`training_tech_scan_2026_eval_calibration.md`](training_tech_scan_2026_eval_calibration.md). Task tracker: [`backlog.md` § Work queue](../../backlog.md#work-queue-p0--p2); commands: [`EXPERIMENTS.md`](../EXPERIMENTS.md).

---

## 1. mAP vs counting: two different questions

### 1.1 Average Precision integrates over thresholds

Object-detection benchmarks (COCO, Pascal VOC) define **AP** as the area under the precision–recall curve: detections are ranked by confidence, TP/FP are assigned as the score cutoff moves, and AP integrates precision over recall (COCO uses 101 interpolated recall points; see [Lin et al., COCO](https://arxiv.org/abs/1405.0312) and [Ultralytics metrics guide](https://docs.ultralytics.com/guides/yolo-performance-metrics/)). AP answers: *“How well does the detector rank true boxes above false ones across operating points on that curve?”*—not *“what single conf should we deploy?”*

For **counting**, practitioners need one operating point. MathWorks and Ultralytics both document sweeping confidence to trace a PR curve, then choosing a threshold that balances precision and recall for the application ([MATLAB detector evaluation](https://www.mathworks.com/help/vision/ug/evaluate-object-detector-performance.html); [Ultralytics issue #7307](https://github.com/ultralytics/ultralytics/issues/7307)).

**Takeaway:** Reporting test mAP≈0.79 (`eval.py`) and separately reporting seed counts at conf=0.25 is valid — they measure different things. A model can have high mAP but suboptimal count error if the deployed threshold is wrong.

### 1.2 Optimal LRP (oLRP) — class-specific best confidence

Oksuz et al. (ECCV 2018) introduce **Localization Recall Precision (LRP)** and **optimal LRP (oLRP)**: the minimum LRP error over confidence scores, yielding an explicit best threshold per class ([LRP paper](https://openaccess.thecvf.com/content_ECCV_2018/papers/Kemal_Oksuz_Localization_Recall_Precision_ECCV_2018_paper.pdf)). Key claims relevant to counting:

- AP “does **not** compare the maximum but the overall capability” and “does **not** suggest a confidence score threshold” for a practical operating point (Oksuz et al., ECCV 2018, §2).
- oLRP yields a class-specific confidence threshold; on COCO, applying those thresholds vs one global threshold raised mAP by **2.3%** in their RefineNet ablation (Table 2: detector S vs G)—a re-scoring experiment, not the oLRP metric value itself.
- LRP decomposes error into localization, FP, and FN components — useful for dense scenes.

Our `threshold_sweep.py` uses a **global** F1-max (`--select best_f1`) with greedy matching (`--iou` default 0.5; optional `--class-agnostic`). Training is **two-class** (developed / aborted), but the default sweep picks **one conf** for both classes. Per-class operating points are **not implemented** (deploy may tune separately in `telegram_bot.py`).

### 1.3 Counting metrics in agricultural detection papers

| Paper | Task | Threshold strategy | Count metric |
|-------|------|-------------------|--------------|
| [David et al., plant detection & counting (bioRxiv 2021)](https://doi.org/10.1101/2021.04.27.441631) | Maize / sugar beet / **sunflower** UAV (early stage) | Faster R-CNN + active learning; density via detection—not conf grid | **rRMSE** below 5% on plant density after AL |
| [Wang et al., maize stand counting (*Agronomy* 2023)](https://doi.org/10.3390/agronomy13071728) | Early maize UAV | **NMS IoU swept** (0.1–0.5+) for YOLOv5 stand counting | R²=0.936, MAE=1.958 |
| [Iamchuen et al., sunflower UAV + YOLOv11 (2026)](https://doi.org/10.3390/su18021026) | Sunflower heads | Grid search **conf=0.50**, **IoU=0.40** (abstract optimum; F1 peak at conf≈0.543 on PR curves) | P=0.84, R=0.95, F1=0.90, mAP@0.5=0.95 |
| [GWHD wheat heads (David et al. 2020/2021)](https://www.global-wheat.com/gwhd.html) | Wheat head detection | Challenge metric **mAP@0.5**; counting scored via **RMSE / rRMSE** on head counts | mAP@0.5 + density/count error |

Agri papers rarely report mAP alone for counting acceptance; they report **density error (rRMSE)**, **R²**, or **F1 at tuned conf/NMS**. Iamchuen et al. (2026, Thailand UAV sunflower heads, 1290 tiles from 215 images) is the closest analogue: they sweep confidence and IoU thresholds separately, then report **conf=0.50** and **IoU=0.40** in the abstract (authors: Iamchuen, Hongpradit, Puttinaovarat, Anucharn—not “Kassim et al.”).

---

## 2. Confidence threshold sweeps (protocol patterns from literature)

### 2.1 Standard sweep

1. Run inference with **low conf** (e.g. 0.001–0.01) and fixed NMS IoU so the score distribution is preserved.
2. For each conf ∈ [0.05, 0.95], match predictions to GT (IoU≥0.5), compute TP/FP/FN, precision, recall, F1.
3. Select operating point:
   - **F1-max** (default in our script),
   - **Constraint-based**: min recall ≥ 0.95, max FP/image ≤ τ ([MATLAB](https://www.mathworks.com/help/vision/ug/evaluate-object-detector-performance.html)),
   - **oLRP-min** (literature alternative).

Our implementation: `scripts/threshold_sweep.py` steps 1–3 on exported JSON (`match_counts_for_threshold`, `select_operating_point`).

### 2.2 NMS IoU is a separate sweep dimension

Dense small objects create a known trade-off ([Learning NMS, Hosang et al. 2017](https://arxiv.org/abs/1705.02950)):

- **Low NMS IoU** → suppresses adjacent true instances (hurts recall in seed grids).
- **High NMS IoU** → duplicate boxes (hurts precision / over-counting).

**Literature note:** adaptive/Soft-NMS can help dense agri scenes; not planned in repo unless a sweep shows benefit on exported preds.

harchoc defaults (train / eval pipeline):

| Stage | conf | NMS IoU | Source |
|-------|------|---------|--------|
| Training / Ultralytics val | 0.05 | 0.3 | `train_yolov8m_baseline.json`, `train.py` |
| Error analysis (fixed point) | 0.25 | 0.5 match IoU | `error_analysis.py` CLI defaults |
| Deploy bot (optional) | per-class ~0.06 | ~0.50 | `telegram_bot.py` — tune separately from manuscript sweeps |

**Recommendation:** `threshold_sweep.py` operates on **post-NMS** exported boxes; changing NMS requires re-exporting preds. Optional 2D grid on **val** exports only.

### 2.3 Per-class thresholds (optional)

For **developed** (0) vs **aborted** (1), literature rarely reports separate conf per class on dense trays; repo uses one global conf. Deploy (`telegram_bot.py`) may tune per class separately from manuscript sweeps.

---

## 3. Calibration in object detection

### 3.1 Why calibration matters for counting (not for mAP)

**Guo et al. (ICML 2017)** — temperature scaling: single parameter T on logits; **does not change the argmax class** (“temperature scaling does not affect the model’s **accuracy**” in their classification setting) ([paper](https://proceedings.mlr.press/v70/guo17a/guo17a.pdf)). Guo does **not** discuss detection AP.

**Detection (Kuzucu et al., ECCV 2024):** post-hoc calibrators use **monotonically increasing** maps ζ so they “do not affect the ranking of the detections significantly and to keep their accuracy”; on COCO, D-DETR **AP@top-100 stays 44.1** before vs after PS/IR in Tab. 5 ([PDF](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/03148.pdf)). Non-monotonic or score-inverting calibration can change AP.

**Platt scaling** maps raw scores to probabilities via logistic regression on TP/FP labels ([MATLAB calibration example](https://www.mathworks.com/help/vision/ug/calibrate-object-detection-confidence-scores.html)).

For detection, calibration target is often **P(correct | score)** or alignment of score with **IoU** (localization-aware), not just class correctness.

### 3.2 Detection-specific calibration (2024–2025)

**Kuzucu et al. (ECCV 2024)** — *On Calibration of Object Detectors* ([PDF](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/03148.pdf)):

- Critiques D-ECE + AP as joint metrics (different thresholds for calibration vs AP); proposes **LaECE0 / LaACE0** (confidence should match IoU at τ=0).
- Post-hoc **Platt scaling (PS)**, **temperature scaling (TS)**, **isotonic regression (IR)** tailored to detection.
- On COCO minitest (Tab. 8, D-DETR): baseline TS **LaECE0=12.3** → PS **9.6** (~2.7 better); **IR=7.7** best among post-hoc baselines.
- Tab. 5: **AP@top-100 unchanged (44.1)** for PS/IR vs uncalibrated D-DETR; **LRP** also preserved under their monotonic maps.

**IJCAI 2025 PRISM (palm UAV)** ([paper](https://www.ijcai.org/proceedings/2025/1067)): compares LR, IR, TS, PS using Kuzucu’s **LaECE0**; **IR best LaECE0** for most detectors; applies **LRP-based thresholding** to drop scores **<0.05** before calibration ([arXiv:2502.13023](https://arxiv.org/abs/2502.13023)).

### 3.3 harchoc calibration (implemented on val)

`threshold_sweep.py` supports **`--calibrate none|isotonic|platt`** (val only; blocked on test by `threshold_protocol`) and **`--calibration-metrics`** (reliability bins + ECE). Workflow:

1. Export val preds at low conf (`eval.py --export-conf 0.001`).
2. Sweep with `--calibrate isotonic` (or Platt); compare val `locked.counting_metrics` / ECE vs `--calibrate none`.
3. Lock conf on val; report test with `--locked-conf-from` (no re-fit on test).

Keep **uncalibrated mAP** in `eval.py` for detector ranking; report **calibrated counting** separately if scores are remapped before the sweep.

---

## 4. Relation to harchoc tooling

| Module / script | Role in val→lock→test |
|-----------------|----------------------|
| `scripts/eval.py` | Test **mAP** (`model.val`); **export** GT/preds JSON (`--export-only`, `--split-file`) |
| `harchoc/threshold_protocol.py` | Split-role inference; **block test tuning** without `--locked-conf-from` |
| `scripts/threshold_sweep.py` | Conf grid, `--select`, `--iou-grid`, `--calibrate`; `locked.counting_metrics` on test |
| `harchoc/threshold_lock.py` | Read `conf_thr` from val sweep; build locked row |
| `scripts/error_analysis.py` | MAE / rRMSE + FP taxonomy at `--locked-conf-from` |
| `harchoc/dual_metric_report.py` | Merge mAP + counting + operating point → `dual_metric_report.v1` |
| `scripts/experiment.py dual-metric` | CLI entry for manuscript table JSON |

### 4.1 `scripts/eval.py` — mAP + export

- Default split: `data/splits/test.txt` (manuscript **mAP** ranking).
- **`--split-file data/splits/val.txt`** for val exports used in threshold tuning (split role recorded in `eval_target.split_role`).
- **`--export-gt-json` / `--export-preds-json`** (+ **`--export-only`**): low-conf preds for sweep (`--export-conf 0.001`, `--export-iou 0.3`, `--export-max-det` / `--max-det`; use **`--export-device cpu`** or `HARCHOC_EXPORT_DEVICE=cpu` if OOM @ 1280).
- **`--imgsz`**: passed to `model.val()` / export for train parity (e.g. 1280).

**Role:** test mAP for model zoo; val/test JSON export for HSP counting protocol.

### 4.2 `scripts/threshold_sweep.py` — counting operating point

- Input: `{"images":[{..., "annotations"|"detections"}]}` JSON ([`data/examples/README.md`](../../data/examples/README.md)).
- Sweeps **conf** on exported detections; greedy match at `--iou` (default 0.5); optional **`--iou-grid`** on val.
- **`--select best_f1`** or **`constraints`**; **`--calibrate isotonic|platt`**; **`--calibration-metrics`** for ECE.
- **`--locked-conf-from`**: test reporting only (no `--select`); writes **`locked.row`** and **`locked.counting_metrics`** (MAE/rRMSE + bootstrap CI).
- **`--fixed-conf`**: evaluate one conf without re-selection.
- Output: `threshold_sweep_run.v1` (`selected` on val; `locked` on test).
- **`--light`**: `data/examples/gt.json` + `preds.json` — CI plumbing only.

### 4.3 `scripts/error_analysis.py` — failure taxonomy at locked conf

- Use **`--locked-conf-from reports/hsp/threshold_val.json`** on val and test after sweep.
- Counting block feeds `dual_metric_report` (`counting_metrics`: MAE, rRMSE, `mae_ci`).

### 4.4 `experiment.py dual-metric` + `harchoc/dual_metric_report.py`

- Merges **val/test eval JSON**, **sweep val** (+ optional **sweep test locked**), **error val/test** into one table row per split.
- Extracts `mAP50` / `mAP50-95`, counting MAE, `selected_conf` / `locked_conf` (`extract_operating_point`).
- Schema: **`dual_metric_report.v1`**. Counting today comes from **error_analysis**, not `locked.counting_metrics` (known P1 gap in eval scan).

### 4.5 `--light` mode purpose

| Mode | Data | Validates |
|------|------|-----------|
| `--light` | `data/examples/*.json` | Schema, matching, F1 selection, CI |
| Real | Exported test/val preds | Scientific threshold for deployment |

Do **not** treat `--light` sweep results as sunflower operating points.

---

## 5. val≈0.97 vs test≈0.79 — literature vs our narrative

### 5.1 What the gap is not

- **Not** fixable by lowering conf alone: mAP integrates the full PR curve over confidence-ranked boxes (Oksuz: AP does not pick an operating threshold); a large val–test mAP50 gap indicates ranking/localization/domain shift on test, not only a wrong single cutoff.
- **Not** explained by train defaults (conf=0.05, iou=0.3): those affect val during training similarly.

### 5.2 What the literature attributes to similar gaps

| Cause | Evidence | harchoc check |
|-------|----------|---------------|
| **Domain / covariate shift** | Ag DA review: good source-domain fit, poor target ([arXiv 2025](https://arxiv.org/html/2506.05972v1)) | `split_drift.py`, `describe_split.py` |
| **Split leakage / easy val** | Early-stopping on val inflates perceived progress | Report test only in papers; val for model selection |
| **Geographic / session bias** | GWHD: performance varies by country and growth stage ([GWHD 2021](https://arxiv.org/pdf/2105.07660)) | Compare proxy stats train/val/test |
| **OOD deployment** | CVPR 2025 satellite DG benchmark: mAP_ID vs mAP_OOD, performance drop metric ([CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/papers/Al-Emadi_Benchmarking_Object_Detectors_under_Real-World_Distribution_Shifts_in_Satellite_Imagery_CVPR_2025_paper.pdf)) | Holdout test frozen; optional `eval_domains.py` |
| **Weed / clutter / occlusion** | Plant counting: weeds limit DL more than HC ([bioRxiv 2021](https://www.biorxiv.org/content/10.1101/2021.04.27.441631v3)) | Error analysis FP taxonomy |

**Workflow implication:** run split drift **before** threshold sweeps. If val and test proxies diverge, threshold tuned on val may **over-fit val counting** and still fail on test — same as mAP gap.

### 5.3 Threshold sweep’s role in the gap story

Sweep clarifies: *“Given fixed weights, what is the best achievable F1 on each split?”*

- If val F1@best ≈ 0.95 but test F1@best ≈ 0.82 → **generalization** problem (data/model).
- If val mAP high but val F1@best low at all conf → **dense-NMS / matching** problem.
- If test mAP low but test F1@best reasonable at low conf → detector ranks poorly at default Ultralytics conf but counting might work with lower threshold (rare; investigate).

---

## 6. External datasets and integration-test imagery

### 6.1 Wheat / grain detection datasets

| Dataset | Scale | Use for harchoc |
|---------|-------|-----------------|
| [GWHD 2021](https://www.global-wheat.com/gwhd.html) ([David et al. 2021 PDF](https://arxiv.org/pdf/2105.07660)) | 6500 images, 275k heads, 12 countries | Cross-domain dense-head detection; overlap/occlusion stress |
| [WGDB](https://doi.org/10.1016/j.compag.2022.107426) (Wang et al., *Computers and Electronics in Agriculture* 2022) | 1746 images, 7844 grain boxes | Belt/conveyor grain; quality classes |
| [YOLO-SDL wheat grain (2024)](https://www.frontiersin.org/journals/plant-science/articles/10.3389/fpls.2024.1495222/full) | 1170 images, 4 grain conditions | Small object + class imbalance |

### 6.2 Sunflower datasets

| Dataset | Notes |
|---------|-------|
| **sunflower-cvat-2500** (ours) | Primary; frozen splits in `data/splits/` |
| [Plant Detection and Counting](https://datasetninja.com/plant-detection-and-counting) | 189 UAV images, sunflower/maize/beet; early-stage plants |
| [Iamchuen et al., sunflower UAV YOLOv11 (2026)](https://doi.org/10.3390/su18021026) | 1290 tiles from 215 UAV images; conf/IoU grid (optimum **0.50 / 0.40**) |
| BARI-Sunflower | Leaf disease close-ups — **not** head counting ([systematic review](https://www.mdpi.com/2504-4990/7/4/130)) |

No public dataset matches our macro seed-on-head density; external sets are useful for **methodology** and **failure modes**, not direct transfer.

### 6.3 Integration testing in agri-CV (failure images)

Literature and industry patterns for **regression / edge-case** tests:

1. **Input health gates** — blur, exposure, misalignment before inference ([AgrigateVision case notes](https://axiscoretech.com/blog/case-notes/agrigatevision-notes/)).
2. **Declarative safety rules** on histogram/exposure ([EPSE field robot vision](https://arxiv.org/abs/1601.02778)).
3. **Automatic capture of low-confidence predictions** for human review (same case notes).
4. **Session / season stratification** — GWHD and plant-counting papers show performance swings by acquisition session.
5. **Known hard scenes**: overlap, blur, tray edge, low-contrast seeds (tag in integration set).

**harchoc today:** `tests/assets/*.ppm` are plumbing-only ([`tests/assets/README.md`](../../tests/assets/README.md)). **Needed:** curated real failure set (10–30 images) with tags: `overlap`, `blur`, `lighting`, `partial_head`, `class_confusion`, stratified from `error_analysis.py` exports.

**Suggested external failure proxies:**

- GWHD images with high head density / overlap (field occlusion).
- Plant-counting sunflower splits with high weed infestation (bioRxiv 2021).
- Downsampled / blurred variants of val images (simulate focus/exposure drift).

---

## 7. Source bibliography (15)

1. Oksuz, K. et al. **Localization Recall Precision (LRP)**. ECCV 2018. [PDF](https://openaccess.thecvf.com/content_ECCV_2018/papers/Kemal_Oksuz_Localization_Recall_Precision_ECCV_2018_paper.pdf)
2. Guo, C. et al. **On Calibration of Modern Neural Networks** (temperature scaling). ICML 2017. [PDF](https://proceedings.mlr.press/v70/guo17a/guo17a.pdf)
3. Kuzucu, A. et al. **On Calibration of Object Detectors: Pitfalls, Evaluation and Baselines**. ECCV 2024. [PDF](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/03148.pdf)
4. Cui, K. et al. **Detection and Geographic Localization of Natural Objects in the Wild: A Case Study on Palms** (PRISM; TS/PS/IR/LR calibration). IJCAI 2025. [PDF](https://www.ijcai.org/proceedings/2025/1067.pdf)
5. Hosang, J. et al. **Learning non-maximum suppression**. 2017. [arXiv:1705.02950](https://arxiv.org/abs/1705.02950)
6. Liu, S. et al. **Adaptive NMS: Refining Pedestrian Detection in a Crowd**. CVPR 2019. [arXiv:1904.03629](https://arxiv.org/abs/1904.03629)
7. David, E. et al. **Global Wheat Head Detection (GWHD)**. 2020/2021. [Site](https://www.global-wheat.com/gwhd.html) · [2021 PDF](https://arxiv.org/pdf/2105.07660)
8. David, E. et al. Plant detection and counting from UAV RGB (maize, sugar beet, **sunflower**). bioRxiv 2021. [DOI](https://doi.org/10.1101/2021.04.27.441631)
9. Iamchuen, N. et al. **Automated Sunflower Head Detection and Yield Estimation from High-Resolution UAV Imagery Using YOLOv11**. *Sustainability* 2026, 18(2), 1026. [DOI](https://doi.org/10.3390/su18021026)
10. Wang, B. et al. Plot-level maize early-stage stand counting (NMS IoU sweep). *Agronomy* 2023, 13(7), 1728. [DOI](https://doi.org/10.3390/agronomy13071728)
11. Domain adaptation in agricultural image analysis (review). 2025. [HTML](https://arxiv.org/html/2506.05972v1)
12. Al-Emadi et al. Benchmarking detectors under distribution shift. CVPR 2025. [PDF](https://openaccess.thecvf.com/content/CVPR2025/papers/Al-Emadi_Benchmarking_Object_Detectors_under_Real-World_Distribution_Shifts_in_Satellite_Imagery_CVPR_2025_paper.pdf)
13. Ultralytics **YOLO performance metrics** (mAP, F1, PR curves). [Docs](https://docs.ultralytics.com/guides/yolo-performance-metrics/)
14. MathWorks **Evaluate object detector performance** (PR sweep, threshold selection). [Docs](https://www.mathworks.com/help/vision/ug/evaluate-object-detector-performance.html)
15. Shi, H. et al. **Enhancing detection accuracy of highly overlapping targets in agricultural imagery using IoA-SoftNMS algorithm across diverse image sizes**. *Computers and Electronics in Agriculture*, 227, 109475 (2024). [DOI](https://doi.org/10.1016/j.compag.2024.109475) · [MARI summary](http://mari.hzau.edu.cn/info/1249/5633.htm)

---

## 8. Actionable threshold sweep protocol (eval.py pipeline)

### Phase A — Baseline test mAP (unchanged)

```bash
export DATASET_ROOT=/path/to/extracted/dataset

# Manuscript metric — test split only
mamba run -n harchoc python scripts/eval.py \
  --weights models/best2.pt \
  --out reports/hsp/eval_test.json

# Optional faster smoke (scratch path; do not overwrite canonical eval_test.json)
mamba run -n harchoc python scripts/eval.py \
  --weights models/best2.pt \
  --max-det 300 \
  --out /tmp/hsp_eval_test_smoke.json
```

Record mAP50 / mAP50-95 for model comparison. Do not tune thresholds here.

### Phase B — Split diagnostics (before threshold work)

```bash
mamba run -n harchoc python scripts/split_drift.py --with-ks --out reports/hsp/split_drift_p0.json
python scripts/describe_split.py --split val --out reports/split_val.json
python scripts/describe_split.py --split test --out reports/split_test.json
```

If val/test proxy stats diverge strongly, interpret threshold results cautiously (Section 5).

### Phase C — Export preds for sweep (val = tune, test = report)

Low conf preserves the PR tail; **export NMS IoU** should match train baseline (`--export-iou 0.3`) and be documented in the JSON metadata.

```bash
export DATASET_ROOT=/path/to/extracted/dataset

# Val — tuning split (no test leakage)
mamba run -n harchoc python scripts/eval.py \
  --weights models/best2.pt \
  --split-file data/splits/val.txt \
  --export-only \
  --export-gt-json reports/hsp/gt_val.json \
  --export-preds-json reports/hsp/preds_val.json \
  --export-conf 0.001 --export-iou 0.3 --export-max-det 3000 \
  --export-device cpu \
  --out reports/hsp/eval_val.json

# Test — repeat after val threshold locked (same export hyperparams)
mamba run -n harchoc python scripts/eval.py \
  --weights models/best2.pt \
  --split-file data/splits/test.txt \
  --export-only \
  --export-gt-json reports/hsp/gt_test.json \
  --export-preds-json reports/hsp/preds_test.json \
  --export-conf 0.001 --export-iou 0.3 --export-max-det 3000 \
  --export-device cpu \
  --out reports/hsp/eval_test.json
```

Schema: [`data/examples/README.md`](../../data/examples/README.md); `category_id` 0/1 = developed / aborted.

### Phase D — Confidence sweep on val

```bash
mamba run -n harchoc python scripts/threshold_sweep.py \
  --gt-json reports/hsp/gt_val.json \
  --preds-json reports/hsp/preds_val.json \
  --split-file data/splits/val.txt \
  --min 0.01 --max 0.60 --steps 60 \
  --iou 0.5 \
  --iou-grid 0.3 0.5 0.7 \
  --select best_f1 \
  --calibrate isotonic --calibration-metrics \
  --csv-out reports/hsp/threshold_val.csv \
  --out reports/hsp/threshold_val.json
```

**Constraint-based alternative** (dense trays; tune on val only):

```bash
mamba run -n harchoc python scripts/threshold_sweep.py \
  --gt-json reports/hsp/gt_val.json \
  --preds-json reports/hsp/preds_val.json \
  --split-file data/splits/val.txt \
  --select constraints \
  --min-recall 0.90 --max-fp-per-image 0.5 \
  --out reports/hsp/threshold_val_constrained.json
```

Read `selected.row.conf_thr` from val JSON. Per-class conf for **developed** vs **aborted** is not in-repo (single global conf).

### Phase E — Lock threshold; evaluate on test (no re-tuning)

```bash
# Test at val-locked conf only (guardrail enforced):
mamba run -n harchoc python scripts/threshold_sweep.py \
  --locked-conf-from reports/hsp/threshold_val.json \
  --gt-json reports/hsp/gt_test.json \
  --preds-json reports/hsp/preds_test.json \
  --split-file data/splits/test.txt \
  --out reports/hsp/threshold_test_locked.json

# Manuscript mAP (test split):
mamba run -n harchoc python scripts/eval.py \
  --weights models/best2.pt \
  --out reports/hsp/eval_test_map.json
```

Report:

- **Detection ranking:** test mAP50 / mAP50-95 (`eval.py`).
- **Counting:** `locked.row` + `locked.counting_metrics` on test; developed+aborted totals via error analysis.
- **Calibration (optional):** compare val sweeps with `--calibrate none` vs `isotonic|platt`; lock best val config before test export path above.

### Phase F — Error analysis + dual-metric table

```bash
mamba run -n harchoc python scripts/error_analysis.py \
  --locked-conf-from reports/hsp/threshold_val.json \
  --gt-json reports/hsp/gt_val.json --preds-json reports/hsp/preds_val.json \
  --out reports/hsp/error_val.json

mamba run -n harchoc python scripts/error_analysis.py \
  --locked-conf-from reports/hsp/threshold_val.json \
  --gt-json reports/hsp/gt_test.json --preds-json reports/hsp/preds_test.json \
  --out reports/hsp/error_test.json

mamba run -n harchoc python scripts/experiment.py dual-metric \
  --eval-val reports/hsp/eval_val.json --eval-test reports/hsp/eval_test.json \
  --sweep reports/hsp/threshold_val.json \
  --sweep-test reports/hsp/threshold_test_locked.json \
  --error-val reports/hsp/error_val.json --error-test reports/hsp/error_test.json \
  --out reports/hsp/dual_metric.json
```

Use FP/FN exports to grow a curated failure set (`tests/assets/` → future `tests/integration/`).

### Phase G — CI / regression (`--light`)

```bash
mamba run -n harchoc python scripts/threshold_sweep.py --light \
  --iou-grid 0.3 0.5 0.7 --out reports/hsp/threshold_val.json
mamba run -n harchoc python scripts/threshold_sweep.py --light \
  --locked-conf-from reports/hsp/threshold_val.json \
  --out reports/hsp/threshold_test_locked.json
mamba run -n harchoc python scripts/experiment.py dual-metric --dry-run \
  --eval-val reports/hsp/eval_val.json --eval-test reports/hsp/eval_test.json \
  --sweep reports/hsp/threshold_val.json --error-val reports/hsp/error_val.json \
  --error-test reports/hsp/error_test.json --out reports/hsp/dual_metric.json
```

Validates schema, guardrails, and merge only—not sunflower operating points.

### Phase H — Optional 2D sweep (export NMS IoU)

If count error is sensitive to **export** NMS: re-export val preds at `--export-iou` ∈ {0.30, 0.40, 0.50} (0.40 pairs with conf=0.50 in [Iamchuen et al. 2026](https://doi.org/10.3390/su18021026) for **UAV heads**, not benchtop seeds), pick on val, lock for test. Skip if train baseline 0.3 is sufficient.

---

## 9. Open implementation gaps in repo

| Gap | Priority |
|-----|----------|
| Full HSP chain on **real** GPU exports (not `--light`) | **P0** |
| `dual_metric_report` prefer `locked.counting_metrics` over error_analysis when present | P1 |
| Per-class conf (**developed** / **aborted**) in `threshold_sweep.py` | P1 |
| `--select min_count_mae` on val (count-first) | **Partial** (P1-FP-BUDGET) |
| Eval `max_det` 300 vs train 3000 on export | **Done** (P0-1) |
| Curated integration failure images from error_analysis | P1 |
| PR curve figure (`make_figures.py` todo) | P2 |

**Done (2026-05-29):** `eval.py` export; `--fixed-conf` / `--locked-conf-from`; test guardrails (`threshold_protocol`); `--calibrate` + ECE; `experiment.py dual-metric`.

---

## 10. Decision checklist before 100-epoch runs

- [ ] Test mAP reported via `eval.py` (not Ultralytics val console alone).
- [ ] Split drift reviewed (`split_drift.py`).
- [ ] Val threshold sweep complete; test evaluated at locked conf.
- [ ] NMS IoU documented (train 0.3 vs deploy 0.5).
- [ ] Counting metric (F1 / MAE per head) reported alongside mAP.
- [ ] `--light` sweeps passing in CI (plumbing only).
- [ ] `dual_metric.json` produced from real val/test exports.

---

*Validated 2026-05-29.* Agri citations checked against publisher pages (bioRxiv 2021 plant counting; *Agronomy* 2023 maize stand; *Sustainability* 2026 sunflower UAV; GWHD 2021). Code paths verified against `threshold_sweep.py`, `threshold_protocol.py`, `dual_metric_report.py`, and [`backlog.md`](../../backlog.md) HSP protocol.

**Val vs test mAP:** [`val_test_map_gap.md`](../manuscript/val_test_map_gap.md). **Deploy vs manuscript thresholds:** **R-SCI-2** Done, **MS-DEPLOY-2STG** Done; Alshehri 2025 analogy in [literature_validated.json](../manuscript/literature_validated.json) · [gap §14](../manuscript/reviewer_comments_backlog_gap.md#14-manuscript-draft--two-stage-deploy-discussion).

**Validated literature registry:** [`docs/manuscript/literature_validated.md`](../manuscript/literature_validated.md) · **Related Work (MS-LIT):** [`related_work_outline.md`](../manuscript/related_work_outline.md) · [Model improvement stack](../../backlog.md#model-improvement-stack-test-count-mae) steps 1 + 3
