# Literature audit: `ren2025_scripta_interp`

**Registry ID:** `ren2025_scripta_interp`  
**DOI:** [10.1016/j.scriptamat.2024.116350](https://doi.org/10.1016/j.scriptamat.2024.116350)  
**Audited:** 2026-06-01  
**Auditor inputs:** Crossref REST (`api.crossref.org`), `curl -sL` redirect check to Elsevier linking hub; `WebFetch` on `https://doi.org/...` timed out (publisher HTML not ingested).

---

## Resolve status

| Check | Result |
|-------|--------|
| DOI registered (Crossref) | **OK** — `journal-article`, Elsevier, deposited 2025-01-24 |
| DOI HTTP redirect | **OK** — `200` → `https://linkinghub.elsevier.com/retrieve/pii/S1359646224003853` |
| Title / venue match registry | **OK** |
| Year | **2025** (print date-parts `2025-01`; journal issue Scripta Materialia **255**, article **116350**) |

**Verdict:** Citation is **real and resolvable**. Safe to keep the DOI; update prose so it does not over-claim methods overlap with HARCHOC.

---

## Bibliographic record (authoritative)

| Field | Value |
|-------|--------|
| **Title** | Harmonizing physical and deep learning modeling: A computationally efficient and interpretable approach for property prediction |
| **Authors** | Da Ren; Chenchong Wang; Xiaolu Wei; Yuqi Zhang; Siyu Han; Wei Xu |
| **Journal** | *Scripta Materialia* |
| **Volume / article** | 255, 116350 |
| **Publisher** | Elsevier (Acta Materialia Inc.) |
| **PII** | S1359646224003853 |

**Registry correction:** `docs/manuscript/literature_validated.json` lists authors as `Ren D, Wei WC, Zhang Y, Han S, Xu W` — **incorrect** (omits **Wang C**; mis-attributes **Wei**). Use the Crossref author list above in manuscript references.

---

## Actual topic (paper vs our stack)

**Domain:** Computational materials science — **dual-phase (DP) steel** microstructure–mechanics, not plant phenotyping, not object detection, not sunflower seeds.

**Problem:** Predict **mechanical properties / constitutive response** (e.g. stress–strain behavior, work hardening, necking) across composition/processing conditions where classical **crystal plasticity (CP)** models need heavy fitting and re-parameterization.

**Approach (from title, Crossref reference graph, and the same group’s extended CP–CNN line in *Acta Materialia*):**

- **Hybrid physics + deep learning:** concise, fitting-free CP theory used to **constrain / guide** a **convolutional neural network** trained on simulation-derived fields (e.g. local stress “nephograms”), not a generic black-box regressor alone.
- **“Interpretable”** in the paper means **physics-aligned, structure-aware modeling** (known constitutive structure + learned correction/surrogate), plus efficiency/generality claims — **not** post-hoc class-saliency maps on natural RGB crop images.
- **Inputs/outputs:** metallic microstructure / mechanics simulation tensors → scalar or curve-level **property prediction**; **no** YOLO boxes, **no** developed/aborted seed classes, **no** counting MAE.

**Grad-CAM in this paper:** Crossref lists Selvaraju et al. (Grad-CAM, ICCV 2017) in the **reference list** only. That supports “interpretability” as a **literature context** citation inside materials ML; it does **not** establish that Ren et al. applied Grad-CAM to our detection task or that their main result is Grad-CAM-based XAI.

**Relation to longer sibling work:** Same author team publishes a fuller CP-guided CNN study for DP steels in *Acta Materialia* (separate DOI). The Scripta item is a **short, high-impact materials letter** in the same research program; treat Scripta as the **reviewer-facing DOI**, not as an agricultural vision paper.

---

## Our claim vs paper (HARCHOC)

| Our usage (repo) | Supported? |
|------------------|------------|
| Reviewer **explainability / breeding trust** framing (`MS-EXPLAIN`, §13 backlog) | **Partial** — paper argues for **trustworthy, interpretable ML** in a scientific domain; analogy to “show why the model decides” is **conceptual only**. |
| `results_and_methods.md` row: “**Grad-CAM panels** alongside error crops” tied to this DOI | **No / misleading** — our Grad-CAM figure (`fig_gradcam_panel`, `harchoc/gradcam_panel.py`) is **our** YOLO attribution; Ren et al. do **not** document sunflower Grad-CAM. |
| “Interpretable ML; pair with Grad-CAM panel” (`literature_validated.*`, `explainability_uncertainty_literature.md`) | **Split the cites:** Ren = **hybrid interpretability framing**; Grad-CAM methodology = **Selvaraju et al.** + agricultural precedent (**LWCD-YOLO**, already in peer table) + our panel. |
| `approach_summary` “tabular/material descriptors” in JSON | **Imprecise** — primary modality is **simulation/image fields + CP physics**, not tray seed descriptors or YOLO labels. |

---

## Fit rating

| Dimension | Rating | Notes |
|-----------|--------|--------|
| **DOI / bibliographic validity** | **High** | Resolved, consistent metadata. |
| **Task alignment (dense seed detection)** | **None** | Different problem, data, and metrics. |
| **Method alignment (Grad-CAM / detection XAI)** | **Low** | At most background citation in their refs; not our method stack. |
| **Interpretability *framing* for Discussion** | **Low–medium** | Usable **one sentence** in Related Work / Discussion if labeled as **cross-domain analogy** (physics-informed trust), not as technical precedent for YOLO Grad-CAM. |
| **Overall HARCHOC fit** | **Low** | Keep registry `harchoc_fit: low`; do not elevate to methods anchor. |

---

## Recommended corrections (manuscript & registry)

1. **Decouple Grad-CAM from Ren in tables.** In `reports/manuscript/results_and_methods.md`, change the interpretability row so Ren supports **“interpretable / trustworthy ML framing”** only; cite **LWCD-YOLO** (or Selvaraju) for **Grad-CAM on crop/seed imagery**, and state that **`fig_gradcam_panel` is our analysis**.
2. **Fix author string** in `literature_validated.json` / `.md` to the six Crossref authors (include **Wang C**; use **Wei X** for Xiaolu Wei, not “Wei WC”).
3. **Tighten `approach_summary`** to: CP-constrained CNN for **DP steel mechanical property** prediction from **microstructure/simulation** inputs; interpretability = **physics–DL hybrid**, not seed saliency maps.
4. **Prose template (safe):**  
   *“Following calls for interpretable models in breeding-adjacent decision support, we report post-hoc Grad-CAM on detector crops (Selvaraju et al.; cf. hybrid physics–ML interpretability in materials property modeling [Ren et al., 2025]).”*  
   Do **not** write that Ren et al. used Grad-CAM on sunflower seeds.
5. **Optional stronger ag XAI cite** for the same reviewer thread: keep **LWCD-YOLO** (`10.3390/agriculture15181968`) as the **direct** crop-vision Grad-CAM peer; Ren stays **secondary framing**.
6. **Re-validation date:** set `validation.checked_date` to **2026-06-01** when registry is next edited (this audit file is the source of truth until then).

---

## Evidence log

- Crossref work record: `GET https://api.crossref.org/works/10.1016/j.scriptamat.2024.116350` (2026-06-01).
- Redirect: `curl -sL -o /dev/null -w "%{http_code} %{url_effective}" https://doi.org/10.1016/j.scriptamat.2024.116350` → `200` Elsevier linking hub.
- `WebFetch(https://doi.org/10.1016/j.scriptamat.2024.116350)` → timeout (no abstract text captured).
- Internal registry cross-check: `docs/manuscript/literature_validated.json`, `docs/research/explainability_uncertainty_literature.md`, `reports/manuscript/results_and_methods.md` (interpretability row).
