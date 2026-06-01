# Literature audit: `gandhi2025_yolov8_freeze` (preprint status)

**Checked:** 2026-06-01  
**Registry ID:** `gandhi2025_yolov8_freeze`  
**DOI:** [10.48550/arXiv.2505.01016](https://doi.org/10.48550/arXiv.2505.01016)  
**Primary URL:** [arXiv:2505.01016](https://arxiv.org/abs/2505.01016)

## Verification method

- **WebFetch** of `https://doi.org/10.48550/arXiv.2505.01016` (resolves to arXiv abstract page).
- **WebFetch** of `https://arxiv.org/abs/2505.01016` (abstract + submission history).
- Spot check of [arXiv HTML v1](https://arxiv.org/html/2505.01016v1) for venue/journal signals (none found).

## Bibliographic identity (confirmed)

| Field | Value |
| --- | --- |
| Title | Fine-Tuning Without Forgetting: Adaptation of YOLOv8 Preserves COCO Performance |
| Authors | Vishal Gandhi; Sagar Gandhi (Joyspace AI) |
| arXiv ID | `2505.01016` |
| Subject | `cs.CV` (also tagged `cs.AI` on DOI landing page) |
| DOI type | arXiv-issued DataCite DOI (`10.48550/arXiv.…`) — **not** a journal publisher DOI |

## Preprint status (audit conclusion)

| Question | Finding |
| --- | --- |
| **Peer-reviewed journal?** | **No** — available only as an **arXiv preprint** as of 2026-06-01. |
| **Conference proceedings?** | **No** — no proceedings link, `Journal-ref`, or published-version DOI on the arXiv record. |
| **Withdrawn / replaced?** | **No** — record is active; not marked withdrawn. |
| **Latest arXiv version** | **v1 only** — submitted **Fri, 2 May 2025** (05:27:14 UTC); no v2+ in submission history. |
| **Venue to cite in manuscript** | **arXiv preprint** (year **2025**). Do **not** cite as a journal article or peer-reviewed conference paper. |

### Evidence summary

1. arXiv abstract page lists a single submission: `[v1] Fri, 2 May 2025`.
2. DOI landing page (`doi.org → arXiv`) shows category **Computer Science > Computer Vision and Pattern Recognition**, cite-as `arXiv:2505.01016 [cs.CV]`, and **no** journal name, volume, or acceptance date.
3. No Crossref/journal container metadata was found tied to this work (distinct from the arXiv DataCite registration).
4. Repo registry [`docs/manuscript/literature_validated.json`](../../../docs/manuscript/literature_validated.json) already records `venue: "arXiv preprint"` and `reviewer_alignment: "Not peer-reviewed at check date — label as preprint in manuscript."` — **still accurate** after this re-check.

### Related but distinct work

[MDPI *Mathematics* 2025 layer-freezing study on YOLO](https://doi.org/10.3390/math13152539) (peer-reviewed, different authors/title) is **not** this Gandhi & Gandhi preprint. Do not merge citations.

## Manuscript labeling (required)

Use explicit preprint wording everywhere this source appears, consistent with [`results_and_methods.md`](../results_and_methods.md) and [`domain_shift_transfer_literature.md`](../../../docs/research/domain_shift_transfer_literature.md):

- Example: *Gandhi & Gandhi (2025, **arXiv preprint** arXiv:2505.01016)* or *Gandhi & Gandhi, 2025 — preprint*.
- **Methods analogy only** (YOLOv8 `freeze` depth / staged unfreeze); not sunflower-seed evidence.

## Quality note (does not change preprint status)

The arXiv HTML v1 includes a **Keywords** block about emotion-aware language modeling / LLMs that is unrelated to the CV abstract — likely template contamination. This reinforces treating the work as an **unreviewed preprint** when weighing evidentiary strength; it does not affect DOI/arXiv identity or preprint classification.

## Status for HARCHOC registry

| Item | Status |
| --- | --- |
| DOI resolves | **OK** |
| Title/authors match registry | **OK** |
| Preprint-only | **Confirmed** |
| Safe to cite for freeze-schedule heuristic | **Yes**, with **preprint** label and domain caveat |
| Safe to cite as peer-reviewed evidence | **No** |
