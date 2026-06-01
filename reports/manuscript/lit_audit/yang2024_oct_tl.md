# Literature audit: `yang2024_oct_tl`

**Audit date:** 2026-06-01  
**Registry:** `docs/manuscript/literature_validated.json`  
**Sources checked:** `https://doi.org/10.1371/journal.pone.0296175` (HTTP 200 → PLOS ONE article), `https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0296175`, Crossref API `10.1371/journal.pone.0296175`

---

## DOI resolves?

**Yes.** `https://doi.org/10.1371/journal.pone.0296175` returns **HTTP 200** and lands on the PLOS ONE article page. Crossref indexes the same DOI as a 2024 PLOS ONE research article (`type: journal-article`).

---

## Title / authors / year vs `literature_validated.json`

| Field | Registry | Publisher (PLOS / Crossref) | Match |
|-------|----------|-------------------------------|-------|
| **DOI** | `10.1371/journal.pone.0296175` | `10.1371/journal.pone.0296175` | Yes |
| **Title** | Explainable ensemble learning method for OCT detection with transfer learning | Same (citation_title / Crossref) | Yes |
| **Authors** | Yang J, Wang G, Xiao X, Bao M, Tian G | Jiasheng Yang; Guanfang Wang; Xu Xiao; Meihua Bao; Geng Tian | Yes (abbreviated vs full given names) |
| **Year** | 2024 | Published **22 Mar 2024** (PLOS ONE vol 19, issue 3, e0296175) | Yes |
| **Venue** | PLoS ONE | PLOS ONE | Yes |

Bibliographic metadata in the registry is **correct**. No DOI correction needed.

---

## What the paper actually says

Yang et al. (PLoS ONE, 2024) address **retinal optical coherence tomography (OCT) image classification**, not object detection and not agricultural or seed imaging.

**Task and data**

- **Three-class whole-image classification:** normal, dry age-related macular degeneration (AMD), diabetic macular edema (DME).
- **Public OCT dataset** with **15 images per class** (45 images total in the study description).
- Goal: reduce clinician labor for fundus lesion screening via OCT.

**Method**

- Compare **AlexNet, VGG16, and ResNet34** with vs without **ImageNet** pretrained weights.
- **Ensemble** the three CNNs with **majority soft voting** (abstract wording: “majority soft polling”).
- **Explainability:** Grad-CAM and CAM to localize lesion-relevant regions; Grad-CAM reported as more accurate for lesion areas.

**Reported results (on their setup)**

- ImageNet transfer raises performance from **68.17%** to **92.89%** (individual-network comparison as stated in abstract).
- Best ensemble (three pretrained CNNs): **100%** correct discrimination among AMD / DME / normal in their reported evaluation.
- Framing is **accuracy + interpretability** for clinical OCT workflow, not domain adaptation to new acquisition batches or trays.

**Not in the paper**

- Bounding-box or instance **object detection** (no YOLO, no mAP, no trays).
- Sunflower, benchtop seed heads, or field/UAV imagery.
- **Staged backbone freeze** schedules, per-tray fine-tuning, or explicit “adapt to new domain/session” protocols (those are HARCHOC/TFA/Gandhi choices, not Yang’s experimental design).

---

## What HARCHOC claims

From `literature_validated.json`, `docs/research/domain_shift_transfer_literature.md` §10, and `docs/manuscript/reviewer_comments_backlog_gap.md` §12:

| Claim | Where |
|-------|--------|
| Registry `approach_summary`: OCT ensemble (AlexNet/VGG16/ResNet34), ImageNet transfer, majority soft voting, Grad-CAM; 15 samples/class | Matches paper abstract |
| `manuscript_use`: **discussion_analogy** — medical OCT, not seed OD | Explicit |
| Gap §12 / domain §10: cite Yang for **framing** adaptation to new **trays** as **staged transfer fine-tuning** from `models/best2.pt`, not scratch training; operational schedule from **`tfa2020`** and **`gandhi2025_yolov8_freeze`** | Yang cited as analogy; freeze schedule attributed to other sources |
| `architecture_takeaway`: “Pretrain + ensemble + XAI for domain shift to a new imaging modality; **tray/session finetune should use staged backbone freeze** … not full retrain” | **HARCHOC engineering recommendation**; only the “pretrain helps new modality” part is loosely parallel to Yang |

Reviewer-facing draft text correctly notes Yang is **“retinal OCT ensemble with ImageNet transfer and Grad-CAM, not seed detection.”**

---

## Fit: ok / misleading / wrong

| Verdict | Scope |
|---------|--------|
| **OK** | DOI, title, authors, year, venue; registry summary of OCT + ImageNet transfer + ensemble + Grad-CAM + tiny dataset; using Yang only as a **high-level transfer-learning + XAI analogy** in Discussion |
| **Misleading** | Implying Yang **validates** tray-key domain shift, YOLO staged freeze, or sunflower detection; treating `architecture_takeaway` freeze guidance as if it came from Yang; citing **100%** or **92.89%** OCT numbers as evidence for tray mAP or seed counting |
| **Wrong** | Stating Yang studied **object detection**, **trays/sessions**, or **sunflower/seed** imagery; claiming their ensemble is the HARCHOC finetune architecture |

**Overall for manuscript use:** **OK as analogy** if every mention states modality/task mismatch and points implementation to TFA/Gandhi/finetune configs. **Misleading** if the sentence structure suggests Yang motivated or evaluated tray fine-tuning.

---

## Corrected DOI

**None.** Keep `10.1371/journal.pone.0296175`.

**Canonical URL:** https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0296175
