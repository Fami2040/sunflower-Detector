# Val vs test mAP gap (manuscript note)

**Audience:** reviewers and manuscript authors.  
**Question:** Why does peak **validation** mAP (~0.97 in training logs) differ from **test** mAP (~0.79)?  
**Canonical artifacts:** `reports/hsp/` (JSON cited below; generated 2026-05-29 on `models/best2.pt`).

---

## 1. What the repo can and cannot cite today

### 1.1 Training-time mAP (reviewer numbers)

Peak **val** mAP50 during Ultralytics training (≈ **0.97**) is **not** in HSP JSON — it lives in training logs only. Internal scans cite **test ≈ 0.79** under the training ranking convention ([`training_tech_scan_2026_augmentation.md`](../research/training_tech_scan_2026_augmentation.md)). The counting-protocol test ranking eval is [`eval_test_map.json`](../../reports/hsp/eval_test_map.json) (merged into `dual_metric.json` test row; may differ from training-log test mAP because of export/`imgsz` settings).

**Manuscript rule:** Label any **val** detection mAP as **in-training early-stop split (not generalization)**. Report **test** ranking mAP from `eval.py` without `--export-only` (**R-SCI-1** / **SCI-MAP-CPU** Done).

### 1.2 HSP eval JSON (present, mAP null)

Both canonical eval runs were **export-only** (preds for threshold sweeps), so ranking mAP is explicitly absent:

| File | `export_only` | `mAP50` | `mAP50_95` | `max_det` | `n_images` |
|------|---------------|---------|------------|-----------|------------|
| [`eval_val.json`](../../reports/hsp/eval_val.json) | `true` | `null` | `null` | 3000 | 109 |
| [`eval_test.json`](../../reports/hsp/eval_test.json) | `true` | `null` | `null` | 3000 | 109 |

[`dual_metric.json`](../../reports/hsp/dual_metric.json): test row includes detection from `eval_test_map.json`; val row `"detection": {}` until optional `eval_val_map.json`.

### 1.3 What *is* in JSON (counting / operating point)

Same weights (`models/best2.pt`), same locked conf **0.15** (val-selected), category-aware match IoU **0.3**:

| Split | Source | Count MAE | F1 | Recall | rRMSE |
|-------|--------|-----------|-----|--------|-------|
| Val | `dual_metric.json` → val row | **71.05** | **0.642** | **0.654** | **0.172** |
| Test | `dual_metric.json` → test row | **61.27** | **0.610** | **0.592** | **0.149** |

Test MAE CI (bootstrap): **51.30 – 71.29** (`threshold_test_locked.json` → `locked.counting_metrics.mae_ci`, n=109).

**Note:** Counting MAE is **lower on test than val** at the locked operating point; the reviewer gap is specifically about **ranking mAP**, not count MAE.

---

## 2. Split drift evidence (`split_drift_p0.json`)

Frozen lists: **875** train / **109** val / **109** test images. Leakage audit: **`ok`**. Acceptance: **`ok`** for train_vs_val, val_vs_test, train_vs_test.

### 2.1 Val vs test distributional checks

From `comparisons.val_vs_test` in [`split_drift_p0.json`](../../reports/hsp/split_drift_p0.json):

| Metric | Value |
|--------|-------|
| `boxes_per_image` mean ratio (val/test) | **0.980** |
| `class_dist_l1` | **0.054** |
| `class_jsd_nats` | **0.00037** |
| Width KS *p*-value | **0.422** |
| Height KS *p*-value | **0.751** |
| Boxes-per-image KS *p*-value | **0.422** |
| `boxes_per_megapixel` ratio (val/test) | **0.866** |

Per-split label density:

| Split | Images | Boxes/image mean (min–max) | Boxes / megapixel |
|-------|--------|----------------------------|-------------------|
| Val | 109 | **565.0** (221–1015) | **247.38** |
| Test | 109 | **553.8** (169–830) | **214.34** |

**Interpretation:** KS tests do **not** reject “same distribution” at α=0.05 for val vs test on the proxies we measure. That **does not** rule out a large **mAP** gap (mAP integrates the full PR curve; test can be harder in ways not captured by marginal box counts or image size).

### 2.2 Train vs test (borderline width)

`comparisons.train_vs_test.images.width_ks.pvalue`: **0.052** (marginal). Other train_vs_test checks remain **ok** in acceptance.

---

## 3. Hypotheses for val ≫ test mAP

Use these when discussing the reviewer concern; tie claims to evidence above.

| Hypothesis | Mechanism | Repo evidence |
|------------|-----------|---------------|
| **Early-stop / selection bias** | `best2.pt` chosen by peak metric on **val** during training | Backlog: Ultralytics early stop on `val.txt`; val mAP is **not** a generalization estimate |
| **Split difficulty (unmeasured)** | Session, focus, tray layout, or annotation style differ on test | Val/test KS **ok** on width/height/box counts; gap may be semantic not marginal |
| **Overfitting to val** | Model fits val-specific appearance while test F1/recall drop | Locked test F1 **0.610** vs val **0.642**; test recall **0.592** vs val **0.654** (`dual_metric.json`) |
| **`max_det` truncation** | Capping preds hides true recall on dense trays | [`s14_maxdet_truncation.json`](../../reports/hsp/s14_maxdet_truncation.json): at `max_det=300`, test locked MAE **261.7** vs **61.3** @ 3000; **94.5%** of test images have GT **> 300** boxes |
| **Eval protocol mismatch** | Training val uses Ultralytics defaults; manuscript eval uses `imgsz=1280`, export conf **0.001**, NMS IoU **0.3** | `eval_*.json` export settings; see [`p0_summary.md`](../../reports/hsp/p0_summary.md) |
| **Not explained by conf alone** | mAP integrates all confidences; a single operating point does not fix ranking gap | Threshold sweep: best val F1 **0.642** @ conf **0.15** does not imply high mAP50 on test |

---

## 4. Reporting checklist

1. **Never** present peak training val mAP as test performance or “model accuracy.”
2. **Do** report test ranking mAP from [`eval_test_map.json`](../../reports/hsp/eval_test_map.json) (full `eval.py`, no `--export-only`; merged into `dual_metric.json` test row). Val ranking mAP is optional; export-only `eval_val.json` still has `mAP50: null`.
3. **Do** use `dual_metric.json` `metric_roles` / per-row `split_role_label` (see `harchoc/dual_metric_report.py`).
4. **Primary manuscript metrics** remain **test** count MAE / locked operating point ([`p0_summary.md`](../../reports/hsp/p0_summary.md)).

---

## 5. Manuscript draft — val mAP vs test (§Results / §2.2)

**Use for reviewer lines 70–72 and §2.2 tone-down (**MS-SPLIT-MAPNARR**, **MS-VAL-MAPDOWN**).**

Peak **validation** mAP50 during Ultralytics training (early stop on `data/splits/val.txt`; internal logs ≈ **0.97**) must not be read as field performance. The held-out **test** split reports **ranking** mAP50 ≈ **0.79** in the same training/eval convention ([`training_tech_scan_2026_augmentation.md`](../research/training_tech_scan_2026_augmentation.md)); that gap is expected when the checkpoint is selected on val. Our **counting-first** HSP protocol (`conf=0.001` export, `imgsz=1280`, val-locked conf **0.15**) yields a separate test ranking mAP in [`eval_test_map.json`](../../reports/hsp/eval_test_map.json) — cite that file for manuscript detection tables, not peak training val mAP.

**Split drift does not explain the mAP gap by itself.** Frozen splits pass leakage and distributional acceptance on box counts, class mix, and image size ([`split_drift_p0.json`](../../reports/hsp/split_drift_p0.json): `acceptance.status` **ok**; `val_vs_test` width/height/boxes-per-image KS *p* > 0.42; `class_dist_l1` **0.054**). The reviewer concern is therefore **metric role and early-stop bias**, not a failed split audit.

**Primary claims (manuscript):** **test** per-image **count MAE** at the val-selected operating point — **61.3** (95% CI **51.3–71.3**, n=109) — with val MAE **71.0** reported only for threshold selection transparency ([`dual_metric.json`](../../reports/hsp/dual_metric.json), [`p0_summary.md`](../../reports/hsp/p0_summary.md)). Treat **detection mAP** as supplementary; do not headline val mAP or imply val≈test generalization from mAP alone.

Suggested §2.2 sentence: *“Validation mAP during training informed early stopping only; all generalization claims use the held-out test split and count MAE at a confidence threshold locked on validation.”*

### Paste into §2.2 (LaTeX)

**Repo draft complete (**MS-VAL-MAPDOWN** Done).** Paste the block below into manuscript §2.2. Split-drift evidence and full narrative: **MS-SPLIT-MAPNARR** Done ([§2](#2-split-drift-evidence-split_drift_p0json) above, [`split_drift_p0.json`](../../reports/hsp/split_drift_p0.json)). External LaTeX edit is a **user action** (source `.tex` is outside this repo).

> Peak validation mAP50 during Ultralytics training (early stop on `data/splits/val.txt`; internal logs ≈ 0.97) informed checkpoint selection only and must not be read as field performance. The held-out test split reports ranking mAP50 ≈ 0.79 under the training ranking convention; that gap is expected when the best checkpoint is chosen on validation. Frozen splits pass leakage and distributional acceptance on box counts, class mix, and image size (`split_drift_p0.json`: `val_vs_test` width/height/boxes-per-image KS *p* > 0.42), so the concern is metric role and early-stop bias—not a failed split audit (**MS-SPLIT-MAPNARR**).
>
> Generalization claims use the held-out test split and **count MAE** at confidence **0.15** locked on validation: test per-image count MAE **61.3** (95% CI **51.3–71.3**, *n*=109); validation MAE **71.0** is reported only for threshold-selection transparency. Detection mAP is supplementary; cite test ranking mAP from `eval_test_map.json` when needed, not peak training validation mAP.

Mirror: [`p0_summary.md` § Detection mAP](../../reports/hsp/p0_summary.md#detection-map).

---

## 6. Related docs

- [`reports/hsp/p0_summary.md`](../../reports/hsp/p0_summary.md) — frozen headline numbers; §2.2 paste mirror  
- [`threshold_calibration_literature.md`](../research/threshold_calibration_literature.md) §5 — literature on val≫test mAP  
- [`backlog.md`](../../backlog.md) — **MS-VAL-MAPDOWN** (done), **MS-VAL-MAP-CAVEAT** (done), **MS-SPLIT-MAPNARR** (done), **R-SCI-1** / **SCI-MAP-CPU** (test mAP eval)
