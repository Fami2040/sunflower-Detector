# Reviewer 2 — Rebuttal draft (for co-author)

**Paper:** Sunflower seed detection & counting (YOLOv8 pipeline, benchtop trays)  
**Reviewer:** Reviewer 2 (verbatim comments: `reviewer2.md` in bundle root)  
**Snapshot:** 2026-06-01 — refresh zoo row for YOLOv10m after training finishes  

**Verified literature (2026-06-01):** [`literature_doi_audit_2026-06-01.md`](literature_doi_audit_2026-06-01.md) · per-paper [`lit_audit/README.md`](lit_audit/README.md). Cite only from [`docs/manuscript/literature_validated.json`](../../docs/manuscript/literature_validated.json). **GWHD DOI was wrong** (`3521832` → `3521852`). Several reviewer-suggested papers are **analogies only** (Yang OCT, Ren materials, Yao fuzzy regression, Alshehri action recognition).

**Canonical numbers (held-out test, *n* = 109, locked conf ≈ 0.15):**

| Metric | Value |
|--------|------:|
| Count MAE | **61.3** (95% CI 51.3–71.3) |
| Test mAP50 | **0.18** (not 0.793 from submitted draft) |
| Mean relative counting error | **12.0%** |
| Share of images with rel. error < 2% | **13.8%** (not “80%”) |
| Aug confirm 100 ep MAE | **64.1** |
| Zoo retrains (complete) | YOLOv8m **111.9**, YOLO11m **119.6**, YOLO26m **95.3** — all worse than anchor |

---

## 1. Abstract not standardized / incomplete

**Reviewer says:** Abstract should follow purpose → methods → results → conclusions; key information missing.

**Our response:** We rewrote the abstract in IMRaD form (see `manuscript/abstract.md`):

- **Purpose:** High-throughput counting of **developed vs aborted** seeds on dense benchtop capitula (~500 instances/image).
- **Methods:** Frozen splits, 1280 px detection, **counting-first** protocol (val `min_count_mae` → lock conf ~0.15; export conf 0.001, IoU 0.3, max_det 3000).
- **Results:** Test MAE **61.3**; rel. error stats; aug/zoo comparisons; test mAP50 **0.18**.
- **Conclusions:** Single-site benchtop limits; deployment separate from science metrics.

---

## 2. Insufficient originality / limited academic contribution

**Reviewer says:** Only applies existing YOLOv8; no structural innovation for dense small seeds.

**Our response:** We **do not** claim a new backbone. Contributions: (1) two-class tray benchmark + frozen splits; (2) counting-first locked-confidence protocol; (3) aug + zoo comparisons gated on test count MAE; (4) error taxonomy + tray-level reporting; (5) positioning vs grain/head peers (`docs/originality_contribution_peers.md` in bundle).

---

## 3. Lack of SOTA comparison experiments

**Reviewer says:** Only YOLOv8; no recent crop small-object comparisons.

**Our response:** Same splits, count MAE primary gate — see `manuscript/tables/zoo_core.md` and `metrics/matrix_train.json`. None of the completed zoo rows beat anchor **61.3**. YOLOv10m was in progress (not OOM-deferred). RT-DETR-L @ 1280 OOMs on 8 GiB; external DETR partial.

---

## 4. Single scenario / insufficient generalization

**Reviewer says:** One site, indoor light; no field/multi-site validation.

**Our response:** We **agree** and state explicitly (`manuscript/dataset.md`). Tray/session spread is wide; pooled test MAE 61.3. No field or multi-variety claim without new data.

---

## 5. Poor research reproducibility

**Reviewer says:** Parameters and thresholds too brief.

**Our response:** Full protocol in `manuscript/results_and_methods.md`; provenance in `docs/ORIGIN_MAIN_AND_DATASET.md`; headline JSON in `metrics/`; reproduce via repo `experiment.py repro` (commands in `BUNDLE_README.md`).

---

## 6. Insufficient literature review

**Reviewer says:** Phenotyping / sunflower / small-object literature not systematic.

**Our response:** Reorganized outline + validated cites in bundle `docs/`; explicit gap: two-class **seed** viability on trays with locked test count MAE.

---

## 7. Figure standardization

**Reviewer says:** Fonts, resolution, annotations not journal-ready.

**Our response:** Quantitative figures at 300 DPI in `manuscript/docx/figures/` and `figures/`; manual setup photos still editorial; final publisher template on Word export.

---

## Metrics to fix in submitted manuscript

| Draft claim | HSP canonical | Notes |
|-------------|---------------|--------|
| Test mAP50 ~0.793 | **0.18** | See `docs/val_test_map_gap.md` |
| ~80% < 2% rel. error | **13.8%** on *n*=109 | n=50 audit is separate |
| Telegram stats | Not in bundle | Deploy audit needed |

---

## Optional closing paragraph

We thank Reviewer 2 for comments on abstract structure, SOTA breadth, generalization scope, reproducibility, literature, and figures. We revised around a pre-registered counting protocol (test MAE 61.3), added aug and zoo comparisons on frozen splits (none beating the anchor on count MAE), reconciled test mAP50 to 0.18, stated benchtop limits explicitly, and regenerated quantitative figures at journal DPI.
