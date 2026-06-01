# Literature audit: `iamchuen2026_sunflower_uav`

**Checked:** 2026-06-01  
**Registry:** [`docs/manuscript/literature_validated.json`](../../../docs/manuscript/literature_validated.json)  
**DOI:** [10.3390/su18021026](https://doi.org/10.3390/su18021026)

## Validation verdict

| Field | Result |
|-------|--------|
| **DOI resolves** | **Pass** — Crossref/OpenAlex index a 2026 *Sustainability* article at this DOI |
| **Title / authors / venue** | **Pass** — minor registry title shortening (see below) |
| **Peer-reviewed article** | **Pass** — MDPI journal article (CC BY 4.0) |
| **Full-text fetch** | **Partial** — publisher HTML returned HTTP 403 from this host; metadata + abstract verified via Crossref API and `doi.org` redirect target |

**Overall:** `ok` — safe to cite for Related Work (sunflower UAV **head** detection, conf/IoU tuning); not a benchtop seed-counting comparator.

---

## Verified bibliographic record

| Field | Verified value | Registry (`literature_validated.json`) |
|-------|----------------|----------------------------------------|
| **Title** | Automated Sunflower Head Detection and Yield Estimation from High-Resolution UAV Imagery Using YOLOv11 **for Precision Agriculture** | Omits subtitle “for Precision Agriculture” — cosmetic only |
| **Authors** | Niti Iamchuen; Phongsakorn Hongpradit; Supattra Puttinaovarat; **Thidapath** Anucharn | Listed as “Anucharn T” — same fourth author |
| **Year** | 2026 | 2026 ✓ |
| **Venue** | *Sustainability* **18**(2), article **1026** | *Sustainability* ✓ |
| **Published** | 2026-01-19 (online) | — |
| **ISSN** | 2071-1050 | — |
| **Publisher URL** | https://www.mdpi.com/2071-1050/18/2/1026 | Same via DOI ✓ |

---

## Fetch provenance (2026-06-01)

| Source | URL / method | Outcome |
|--------|----------------|---------|
| `WebFetch` | https://doi.org/10.3390/su18021026 | Timeout |
| `curl` | MDPI landing | **403** (bot block) |
| **Crossref API** | `https://api.crossref.org/works/10.3390/su18021026` | **200** — title, authors, abstract, issue metadata |
| **OpenAlex** | `https://openalex.org/works/...` | **200** — confirms 2026 article, keywords (sunflower, precision agriculture, UAV) |

Use Crossref/DOI for manuscript bibliography; re-check MDPI page manually if full-text quotes are needed.

---

## Paper summary (from publisher abstract via Crossref)

**Task:** Detect **sunflower heads** in high-resolution **UAV** imagery and support **yield estimation** (field-scale precision agriculture).

**Model:** **YOLOv11** on **512×512** tiles; performance tuned via **confidence** and **NMS IoU** thresholds.

**Data (abstract):** **1290** tiles from **215** UAV images; **80:20** train/test split (abstract does not detail multi-plot design).

**Reported optimum (abstract):** conf **0.50**, IoU **0.40** → P **0.84**, R **0.95**, mAP@0.5 **0.95**, F1 **0.90**.

**Claims:** UAV + YOLOv11 reduces manual yield-assessment effort; framework described as transferable to other crops.

**Repo nuance (from [`domain_shift_transfer_literature.md`](../../../docs/research/domain_shift_transfer_literature.md)):** Internal notes cite a **single ~900 m² plot** (Thailand), spatially separated train/val tiles, and discussion that a **homogeneous** plot can inflate yield correlation (R²≈0.984) — authors call for **multi-location / multi-season** work. Treat as **methodological analogue** for threshold grids and generalization caveats, not multi-field validation.

---

## Alignment with registry `approach_summary`

| Registry claim | Audit |
|----------------|-------|
| YOLOv11 on 512×512 UAV tiles | **Confirmed** (abstract) |
| Conf and NMS IoU grids | **Confirmed** |
| Single-plot / spatially separated tiles | **Consistent** with research doc; not in Crossref abstract alone |

`harchoc_fit`: **med** — sunflower + detection + threshold methodology; **out of scope** for developed/aborted **tray seeds** (HSP).

---

## Manuscript use (MS-LIT)

- **§2 Related Work:** Sunflower imaging axis — UAV **head** counting vs benchtop **seed** viability classes.
- **Methods / Discussion:** Precedent for **conf/IoU sweeps** then locking operating point ([`threshold_calibration_literature.md`](../../../docs/research/threshold_calibration_literature.md)); contrast with HSP lock-then-test on `test.txt`.
- **Do not** position as SOTA comparator on tray mAP/count MAE.

**BibTeX-style cite string:**

> Iamchuen, N.; Hongpradit, P.; Puttinaovarat, S.; Anucharn, T. Automated Sunflower Head Detection and Yield Estimation from High-Resolution UAV Imagery Using YOLOv11 for Precision Agriculture. *Sustainability* **2026**, *18*(2), 1026. https://doi.org/10.3390/su18021026

---

## Actions

- [ ] Optional: refresh registry `title` to full Crossref string (subtitle).
- [ ] Optional: set `validation.checked_date` → `2026-06-01` in JSON after manuscript freeze.
