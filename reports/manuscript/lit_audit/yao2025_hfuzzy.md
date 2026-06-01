# Literature audit: `yao2025_hfuzzy`

**Registry ID:** `yao2025_hfuzzy`  
**DOI:** [10.1109/TFUZZ.2025.3549791](https://doi.org/10.1109/TFUZZ.2025.3549791)  
**Checked:** 2026-06-01  
**HARCHOC hook:** reviewer §270 / **MS-FUZZY-BOUND** — graded trust on two-class detections (not a third YOLO class)

## Source check

| Source | Result |
|--------|--------|
| `https://doi.org/10.1109/TFUZZ.2025.3549791` | Resolves; redirects to IEEE Xplore doc **10925448** |
| IEEE Xplore HTML (automated fetch) | **Blocked** — HTTP **418** / “Unable to Load Page” (bot protection); `WebFetch` timed out |
| [Crossref API](https://api.crossref.org/works/10.1109/TFUZZ.2025.3549791) | **OK** — bibliographic record matches registry |
| [OpenAlex](https://openalex.org/W4408399762) | **OK** — abstract text (no paywall) |

**Bibliographic record (Crossref):** Mengxue Yao, Taoyan Zhao, Jiangtao Cao, Jinna Li. *Hierarchical Fuzzy Topological System for High-Dimensional Data Regression Problems.* **IEEE Transactions on Fuzzy Systems**, 33(7):2084–2095, July 2025.

**Abstract gist (OpenAlex, 2026-06-01):** Proposes a **hierarchical fuzzy topological system (HFTS)** for **high-dimensional data regression** (strong nonlinearity, rule explosion). Modular fuzzy layers; graph-based feature grouping into a topological structure; cross-layer rule sharing; evaluated on **KEEL regression datasets** (abstract: *eleven*; related MOEHFRS line in prior work often reports *thirteen* tabular sets — both are **tabular**, not imaging).

**Abstract keyword scan:** `regression`, `fuzzy`, `keel`, `interpretab` — present. `seed`, `image`, `vision`, `boundary`, `detection`, `yolo`, `plant`, `agricult`, `kernel` — **absent**.

---

## Tabular regression vs seed-boundary claim

| Dimension | What Yao et al. (2025) actually address | What HARCHOC cites / implements | Audit verdict |
|-----------|------------------------------------------|----------------------------------|---------------|
| **Problem domain** | High-dimensional **tabular regression**; KEEL-style feature vectors | Tray **image** detection; developed vs aborted **bounding boxes** | **Different domain** — cite as analogy only |
| **Output type** | Continuous **regression** targets via hierarchical fuzzy rules (HFTS) | Discrete **detections** + class scores; count MAE on boxes | **Analogous “graded output,” not same task** |
| **“Boundary” / ambiguous seeds** | **Not claimed** in title/abstract; no seed, kernel, or phenotype boundary task | Reviewer **§270** asks about **boundary-ambiguous seeds**; repo uses **low-confidence score band** + FP taxonomy (`ambiguous_summary`, `fig_ambiguous_panel`) | **Do not attribute seed-boundary methods to Yao** — our boundary handling is **HARCHOC protocol**, not their contribution |
| **Hierarchical / fuzzy** | **Hierarchical fuzzy topology** for rule structure and feature grouping | **Hierarchical** only in rhetorical parallel; **no fuzzy head**, no third class, no relabel | **Conceptual parallel** — fuzzy **membership on regression outputs** ↔ **confidence band on detections** |
| **Graded trust** | Fuzzy rules + interpretability / accuracy trade-off on regression | Val-locked conf + band **(0.15, 0.30]** on exports; cross-tab vs TIDE FP buckets | **Valid analogy** if manuscript states Yao does **not** study vision or seeds |
| **Evidence in paper for our use case** | KEEL regression benchmarks; GNN feature topology | Test *n*=109, ambiguous-band counts in `error_test_report.json` | **Our numbers are internal** — not from Yao |

---

## Claim validation summary

| Claim | Supported by paper? | Safe manuscript wording |
|-------|---------------------|-------------------------|
| Yao et al. propose hierarchical fuzzy methods for **high-dimensional regression** | **Yes** (title + abstract + venue) | “Yao et al. (2025) address tabular high-dimensional **regression** with a hierarchical fuzzy topological system.” |
| Yao et al. study **sunflower**, **seeds**, or **boundary-ambiguous kernels** | **No evidence** (abstract; metadata topics: fuzzy control, regression, TDA — not plant phenotyping) | **Do not** write “Yao et al. on boundary seeds” or imply their SOTA applies to our labels. |
| Yao et al. justify a **third detect class** or fuzzy YOLO head | **No** | “We keep two hard classes; graded trust is post-hoc on scores.” |
| Yao et al. support **graded trust / gradual membership** as a **discussion analogy** | **Reasonable** (fuzzy regression → non-binary output trust) | “By **analogy** to gradual membership in fuzzy **regression** outputs (Yao et al., 2025), we report graded trust on two-class **detections**…” |

**Overall:** Registry entry `yao2025_hfuzzy` is **bibliographically valid**. The **seed-boundary** problem is **reviewer-driven** and **HARCHOC-specific**; Yao is **not** a direct empirical peer for ambiguous kernel boundaries. Mis-citation risk: treating Yao as a **methods source for seed-boundary labeling** — **reject**; treating Yao as **related work on fuzzy graded outputs** mapped to **confidence bands** — **accept** with explicit “analogy” language.

---

## Repo pointers

- Registry: [`docs/manuscript/literature_validated.json`](../../../docs/manuscript/literature_validated.json) (`yao2025_hfuzzy`)
- Discussion draft: [`docs/manuscript/reviewer_comments_backlog_gap.md`](../../../docs/manuscript/reviewer_comments_backlog_gap.md) §15
- Synthesis: [`docs/research/explainability_uncertainty_literature.md`](../../../docs/research/explainability_uncertainty_literature.md#reviewer-cites-ren-2025-yao-2025)
