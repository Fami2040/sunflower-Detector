# Literature DOI audit (2026-06-01)

**Scope:** All 11 entries in [`docs/manuscript/literature_validated.json`](../../docs/manuscript/literature_validated.json) and the literature table in [`results_and_methods.md`](results_and_methods.md) (lines 29–39).

**Methods:** `curl -sL -o /dev/null -w "%{http_code}"` on `https://doi.org/<doi>` (30–40 s timeout, browser User-Agent where noted); [Crossref REST API](https://api.crossref.org/works/) for metadata; WebFetch on publisher/arXiv pages where fetch succeeded.

**Important:** Several publishers return **403** (MDPI) or **418** (IEEE) to automated clients while the DOI is **registered and valid** in Crossref. Those are **not** broken DOIs—use publisher links in browsers or Crossref for verification.

**Per-paper reports:** [`lit_audit/README.md`](lit_audit/README.md) (11 files).

---

## Summary

| Metric | Count |
|--------|------:|
| Entries audited | 11 |
| DOIs wrong in registry (fixed) | **1** (`gwhd2020`: `3521832` → `3521852`) |
| DOIs valid; automated `doi.org` blocked | **3** (2× MDPI, 1× IEEE) |
| arXiv DataCite DOIs (`10.48550/arXiv.*`) | **2** — `doi.org` → 200; Crossref 404 (expected); **use `arxiv.org/abs` as `validation.url`** |
| Claims already honest in JSON | 7 |
| Claims tightened in this pass | 4 (Alshehri title, LWCD mosaic/S2, GWHD DOI, manuscript table wording) |

---

## Per-entry audit

One markdown report per id under [`lit_audit/`](lit_audit/) (see [index](lit_audit/README.md)).

| id | claimed DOI | status | corrected DOI | what paper ACTUALLY says | what we claimed | fit | recommendation |
|----|-------------|--------|---------------|--------------------------|-----------------|-----|----------------|
| `yang2024_oct_tl` | 10.1371/journal.pone.0296175 | **ok** (doi.org 200; Crossref ok) | — | OCT **classification** (normal / dry AMD / DME); AlexNet+VGG16+ResNet34 ensemble with ImageNet TL; Grad-CAM; **15 subjects/class** | OCT transfer + XAI analogy for tray/lighting adaptation | **ok** (analogy) | Keep as **Discussion analogy only**; never imply seed object detection |
| `ren2025_scripta_interp` | 10.1016/j.scriptamat.2024.116350 | **ok** (200) | — | **Materials property prediction**: hybrid physics + deep learning on material descriptors; interpretability for **regression**, not computer vision | Interpretability framing for breeding trust | **misleading** if cited alone | Cite only for “interpretable ML” prose; **must** pair with Grad-CAM on **our** crops; do not imply vision method |
| `alshehri2025_uav` | 10.3389/fnbot.2025.1582995 | **ok** (200) | — | UAV **multi-person action recognition** (MOD20, Okutama-Action ~91.5% / ~89.7%); skeleton/keypoint features + DBN/CNN/RNN classifiers — **not** detect-then-classify for objects | Two-stage deploy analogy (gate→detector) | **misleading** if “detection pipeline” implied | Title in registry was wrong (“multi-person **detection**”); fix to “action recognition”. Methods: “staged pipeline **by analogy**”; not HSP eval |
| `yao2025_hfuzzy` | 10.1109/TFUZZ.2025.3549791 | **ok** (Crossref; doi.org **418** bot) | — | **Hierarchical fuzzy topological system** for **high-dimensional tabular regression** (13 datasets); MO evolution of rule topology — **no images, no seeds** | Graded trust at ambiguous **boundaries** | **misleading** without “analogy” | Keep **regression analogy** label in Methods; implement via conf bands + FP taxonomy, not fuzzy YOLO |
| `grainnet2025` | 10.1186/s13007-025-01363-y | **ok** (200; Crossref ok) | — | YOLOv7 + EMA for **benchtop wheat grain** detection/counting; offline aug (cutout, mixup, mosaic); count MAE/R² vs manual | Closest kernel-counting peer; count MAE + EMA narrative | **ok** | Strong related-work peer; contrast two-class sunflower vs single-class wheat |
| `lwcd_yolo2025` | 10.3390/agriculture15181968 | **ok** (Crossref; doi.org **403** bot) | — | **LWCD-YOLO** on **corn seed kernels** (YOLOv11n, PConv, EMA, MSFFM); benchtop dataset; internal docs cite §3.1 **mosaic disabled** (not in Crossref abstract) | Mosaic-off precedent for ARCH-MOSAIC0-AB / S2 | **ok** peer / **misleading** for S2 | Peer for dense benchtop **detection**; cite mosaic-off only with “paper reports”; **our S2 (mosaic=0) worsened MAE** — do not say LWCD “supports” mosaic-off for sunflower |
| `tfa2020` | 10.48550/arXiv.2003.06957 | **ok** (doi.org **200** → arXiv; Crossref N/A) | — | **Few-shot object detection**: fine-tune **last layer** of Faster R-CNN–style detectors on rare COCO/VOC/LVIS classes | Staged freeze + low LR on tray holdout | **ok** (analogy) | Link `https://arxiv.org/abs/2003.06957`; ICML 2020; tray finetune is inspired analogy, not same task |
| `gandhi2025_yolov8_freeze` | 10.48550/arXiv.2505.01016 | **ok** (doi.org **200**; Crossref N/A) | — | YOLOv8n **freeze-depth sweep** on **fruit** dataset; freeze=10 → +10% mAP50 on fruit, minimal COCO drop | Operational freeze for `finetune.py` | **ok** (preprint) | Always label **(preprint)**; use abs URL; not peer-reviewed at audit date |
| `gwhd2020` | 10.34133/2020.**3521832** | **broken** (doi.org **404**) | **10.34133/2020/3521852** | GWHD: 4700 field RGB images, ~190k **wheat head** boxes; phenotyping benchmark; dense overlap / counting | Dense organ phenotyping; duplicate FP framing | **ok** after DOI fix | **Replace DOI everywhere**; optional journal URL `https://spj.science.org/journals/plantphenomics/2020/3521852/` |
| `iamchuen2026_sunflower_uav` | 10.3390/su18021026 | **ok** (Crossref; doi.org **403** bot) | — | YOLOv11 on **512×512 UAV tiles**; **sunflower head** detection + yield; conf/NMS grids; single-plot spatial split | Sunflower imaging; conf/IoU methodology | **ok** (scope) | Related Work + limitations; **not** developed/aborted **tray seeds** |
| `gulzar2025_sunflower_tl` | 10.55730/1300-0152.2763 | **ok** (200; Crossref ok) | — | **Systematic review** (30 papers): TL for **sunflower disease** in field imagery; domain shift | Field TL / domain-shift limitations | **ok** (scope) | Discussion limitations only; not counting/detection SOTA |

---

## `results_and_methods.md` cross-check (lines 29–39)

| Table row DOI | In JSON? | Match after fix |
|---------------|----------|-----------------|
| 10.1186/s13007-025-01363-y | yes | ok |
| 10.3390/agriculture15181968 | yes | ok (soften mosaic wording) |
| 10.34133/2020.3521832 | yes | **fixed → 3521852** |
| 10.3390/su18021026 | yes | ok |
| 10.55730/1300-0152.2763 | yes | ok |
| 10.48550/arXiv.* (×2) | yes | prefer **abs** links |
| 10.1109/TFUZZ.2025.3549791 | yes | ok + “regression analogy” |
| 10.1016/j.scriptamat.2024.116350 | yes | ok + “framing only” |
| 10.1371/journal.pone.0296175 | yes | ok + “analogy” |

---

## Files updated (2026-06-01)

- [`docs/manuscript/literature_validated.json`](../../docs/manuscript/literature_validated.json) — GWHD DOI; `checked_date`; validation HTTP notes; honest summaries; Alshehri title; arXiv `validation.url`
- [`results_and_methods.md`](results_and_methods.md) — GWHD DOI; arXiv abs links; registry date; LWCD/S2 mosaic honesty
- [`docs/manuscript/literature_validated.md`](../../docs/manuscript/literature_validated.md) — GWHD DOI; audit pointer (optional sync)

---

## HTTP probe log (2026-06-01)

```
200  10.1371/journal.pone.0296175
200  10.1016/j.scriptamat.2024.116350
200  10.3389/fnbot.2025.1582995
418  10.1109/TFUZZ.2025.3549791   # Crossref: registered
200  10.1186/s13007-025-01363-y
403  10.3390/agriculture15181968  # Crossref: registered
200  10.48550/arXiv.2003.06957
200  10.48550/arXiv.2505.01016
404  10.34133/2020.3521832        # WRONG
200  10.34133/2020/3521852        # CORRECT
403  10.3390/su18021026           # Crossref: registered
200  10.55730/1300-0152.2763
```
