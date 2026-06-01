# Literature audit: `gulzar2025_sunflower_tl`

**Checked:** 2026-06-01  
**Registry:** [`docs/manuscript/literature_validated.json`](../../../docs/manuscript/literature_validated.json)  
**DOI:** [10.55730/1300-0152.2763](https://doi.org/10.55730/1300-0152.2763)

## Validation verdict

| Field | Result |
|-------|--------|
| **DOI resolves** | **Pass** — redirects to TÜBİTAK *Turkish Journal of Biology* article page |
| **Title / authors / venue** | **Pass** — author listed as **Yonis GULZAR** (registry “Gulzar Y” is citation-style abbreviation) |
| **Peer-reviewed article** | **Pass** — systematic **review** (not primary detection experiment) |
| **Full-text fetch** | **Pass** — publisher page HTTP 200; abstract in HTML meta; PMC mirror listed in research docs |

**Overall:** `ok` — cite for **field transfer learning**, **domain shift**, and **cross-regional validation** narrative; not seed counting or viability detection.

---

## Verified bibliographic record

| Field | Verified value | Registry (`literature_validated.json`) |
|-------|----------------|----------------------------------------|
| **Title** | Applications of transfer learning in sunflower disease detection: advances, challenges, and future directions | Match (case only) ✓ |
| **Author** | **Yonis GULZAR** (single author) | “Gulzar Y” — same person ✓ |
| **Year** | 2025 | 2025 ✓ |
| **Venue** | *Turkish Journal of Biology* **49**(5), **534–549** | *Turkish Journal of Biology* ✓ |
| **Published online** | 2025-10-28 (Crossref) | — |
| **Publisher URL** | https://journals.tubitak.gov.tr/biology/vol49/iss5/6 | DOI landing ✓ |
| **PMC** | [PMC12614360](https://pmc.ncbi.nlm.nih.gov/articles/PMC12614360/) (per [`domain_shift_transfer_literature.md`](../../../docs/research/domain_shift_transfer_literature.md)) | — |

*Note:* PMC citation meta once listed “2025 Oct 6”; Crossref/TÜBİTAK deposit date **2025-10-28** — use Crossref for bibliography date unless journal states otherwise.

---

## Fetch provenance (2026-06-01)

| Source | URL / method | Outcome |
|--------|----------------|---------|
| `WebFetch` | https://doi.org/10.55730/1300-0152.2763 | Timeout |
| `curl` | https://journals.tubitak.gov.tr/biology/vol49/iss5/6/ | **200** — title, DOI, structured abstract in meta description |
| **Crossref API** | `https://api.crossref.org/works/10.55730/1300-0152.2763` | **200** — title, author, volume/issue/pages, online date |
| **OpenAlex** | DOI work | **200** — title/year; no abstract text in API |
| PMC | https://pmc.ncbi.nlm.nih.gov/articles/PMC12614360/ | Intermittent bot challenge from automation; DOI/citation tags previously matched |

---

## Paper summary (publisher abstract / PMC-style text)

**Type:** **Systematic review** of **transfer learning (TL)** for **sunflower disease** classification from field imagery (not object-detection counting).

**Scope:** Structured **Scopus** search, papers **2021–2025**; **30** studies included after inclusion/exclusion criteria.

**Methods reviewed:** CNN families (ResNet, VGG, Inception, EfficientNet), transformers/hybrids, lightweight and federated variants; preprocessing and reported metrics compared across studies.

**Findings:** Strong reliance on ImageNet-pretrained CNNs; recurring limits — **small / imbalanced datasets**, limited **explainability**, shift from “apply deep learning” toward **XAI** and **edge** deployment; keyword co-occurrence analysis in paper.

**Conclusion (reviewers’ message):** Progress in TL for sunflower pathology, but need **larger standardized datasets** and **cross-regional validation**; future work should emphasize **interpretable**, field-ready models.

**Harchoc mapping:** Supports **Discussion** limitations on **field vs lab**, lighting, variety, and domain shift — analogous rigor for tray/session holdout — **not** a detection-counting peer.

---

## Alignment with registry `approach_summary`

| Registry claim | Audit |
|----------------|-------|
| Systematic review, 30 papers | **Confirmed** |
| TL for sunflower **disease** in field imagery | **Confirmed** |
| Stresses domain shift, cross-regional validation | **Confirmed** |

`harchoc_fit`: **low** (review, disease classification axis) — still valuable for **MS-LIT** / **MS-GEN** limitation framing.

---

## Manuscript use (MS-LIT / MS-GEN)

- **§2 Related Work:** Sunflower + **field imaging** + **TL/domain shift** survey (disease classification literature).
- **Discussion:** Cite when arguing that models trained under one acquisition protocol need **held-out conditions** (tray/session analogues to cross-regional field validation).
- **Do not** cite for YOLO zoo comparison, mAP@0.5 leaderboards, or HSP counting metrics.

**BibTeX-style cite string:**

> Gulzar, Y. Applications of Transfer Learning in Sunflower Disease Detection: Advances, Challenges, and Future Directions. *Turk. J. Biol.* **2025**, *49*(5), 534–549. https://doi.org/10.55730/1300-0152.2763

---

## Actions

- [ ] Optional: expand registry `authors` to “Gulzar, Y. (Yonis)” if coauthor confusion arises.
- [ ] Optional: add `pmc_id: PMC12614360` to JSON for mirror link.
- [ ] Optional: set `validation.checked_date` → `2026-06-01` in JSON after manuscript freeze.
