# Literature audit: `alshehri2025_uav`

| Field | Value |
|-------|--------|
| **Registry id** | `alshehri2025_uav` |
| **DOI** | [10.3389/fnbot.2025.1582995](https://doi.org/10.3389/fnbot.2025.1582995) |
| **Title** | Unmanned aerial vehicle based multi-person detection via deep neural network models |
| **Authors** | Alshehri M, Zahoor L, AlQahtani Y, et al. |
| **Venue** | *Frontiers in Neurorobotics* |
| **Published** | 17 April 2025 (Vol. 19, 2025) |
| **Checked** | 2026-06-01 (WebFetch: [Frontiers full text](https://www.frontiersin.org/journals/neurorobotics/articles/10.3389/fnbot.2025.1582995/full); DOI redirect timed out) |
| **Validation** | **OK** — DOI resolves; abstract, methods, results, and tables match registry entry |

---

## Verdict (one line)

**UAV multi-person action recognition (HAR), not detect-then-classify and not comparable object detection.** Safe only as a **loose** “staged perception before specialized output” analogy to HARCHOC deploy; unsafe as architectural precedent.

---

## What the paper actually does

### Task and metrics

- **Task:** Multi-person **human action recognition** from **UAV video** (Abstract, §1, §3.1).
- **Datasets:** MOD20, Okutama-Action (clip-/frame-level action labels).
- **Reported results:** **Classification accuracy** — 91.50% (MOD20), 89.71% (Okutama-Action) (Abstract, Table 5). No COCO-style detection mAP, no bounding-box seed counts.

### Proposed pipeline (authors’ system)

Sequential **preprocessing → segmentation → pose → handcrafted features → classifier**, not an off-the-shelf detector cascade:

1. Frame extraction, Gaussian blur, grayscale, background removal (§3.2).
2. **GMM** segmentation for human silhouettes (§3.3–3.4).
3. **MediaPipe Pose** — 33 landmarks, skeletal graph (§3.5).
4. **Handcrafted features:** full-body (AKAZE, distance transform, Fourier descriptors) + keypoint-based (0–180° intensity, motion histograms, multi-point autocorrelation) (§3.6).
5. **Classifiers:** DBN, CNN, or RNN with gradient-descent training; CNN reported best (§3, Table 6 ablation, Table 8).

Figure 1 and §3.1 describe this as a single **action recognition** framework. There is **no** YOLO (or similar) detector in the **proposed** method.

### Title vs body

The title emphasizes “multi-person **detection**,” but the abstract, introduction, methods, and evaluation consistently frame **action recognition** and **classification accuracy**. Treat the title as **marketing/scope drift**, not a description of the evaluated system.

---

## Detect-then-classify?

| Question | Answer |
|----------|--------|
| Is the **proposed** method detect-then-classify? | **No.** It is segment/pose → engineer features → classify **action class**. “Detection” in the title does not mean a detector→classifier product pipeline. |
| Does detect-then-classify appear at all? | **Only in related work** (§2), e.g. Abbas & Jalal (2024): YOLOv5 humans → pose → SVM actions; Khan et al. (2024): YOLOv8 + tracking + transformer for drone HAR. Those are **cited comparators**, not the authors’ contribution. |
| Closest honest label | **Classical CV + pose + handcrafted spatiotemporal features + shallow/deep classifiers** for UAV HAR. |

---

## Brutally honest vs HARCHOC two-stage deploy

HARCHOC production (see `docs/HSP_BASELINE_MODELS.md`, reviewer gap §14):

| | **Alshehri et al. (2025)** | **HARCHOC deploy** |
|--|---------------------------|-------------------|
| **Input** | UAV **video** frames | User **still** images (Telegram) |
| **Stage 1** | Blur, grayscale, GMM **silhouette** isolation | **`classifier.pt`** — top-1 sunflower vs other, conf ≥ 0.5 |
| **Stage 2** | MediaPipe + AKAZE/Fourier/etc. → **action label** (skiing, paddling, …) | **`best2.pt`** + **SAHI** → **developed/aborted** boxes & counts |
| **Stage 1 role** | Improve subject isolation for features | **Reject non-target uploads** (gate) |
| **Stage 2 role** | Recognize **verb** (activity) | **Count/localize** **nouns** (seeds) |
| **Learning** | Handcrafted features + CNN/RNN/DBN on action datasets | End-to-end YOLO detector (+ separate gate model) |
| **Manuscript metrics** | N/A (different project) | **Single-stage** `best2.pt` full-frame eval; gate **not** used |

### What the analogy gets right (minimal)

Both systems apply **more than one processing stage** before the final user-facing decision: something **coarse** (scene/subject handling) before something **specialized** (fine-grained label or boxes). That matches a reviewer’s informal “robustness pattern” language — **not** a claim of shared architecture.

### What the analogy gets wrong (say this in Discussion)

1. **Different problem:** HAR on people in aerial video vs static **seed instance detection** on benchtop heads.
2. **Different “stage 1”:** Image-level **learned binary/multiclass gate** vs **unsupervised** segmentation/pose prep — not interchangeable.
3. **Different “stage 2”:** **Bounding-box detector + counting** vs **global action classification** without box metrics.
4. **No literal cascade:** HARCHOC is **gate → detector** (two models, deploy-only). Alshehri is **one integrated HAR pipeline** (ablated in Table 6), not “detector weights then classifier head.”
5. **Title trap:** Citing this paper as “multi-person **detection**” beside a seed detector invites reviewers to think you copied a **YOLO-style** two-stage paper — you did not; their own method is not that.

### Recommended manuscript wording

- **Do:** “By analogy to staged perception in UAV action recognition (Alshehri et al., 2025), production uses a sunflower gate before SAHI seed detection; reported HSP metrics omit the gate.”
- **Do not:** “Following Alshehri et al.’s two-stage detect-then-classify framework…” or “similar detection pipeline.”

---

## Registry alignment

Matches `docs/manuscript/literature_validated.json` (`approach_summary`, `architecture_takeaway`, `manuscript_use: methods_analogy`). No change required to registry claims; this file adds **audit depth** and explicit **anti-overclaim** language for Discussion §14 / `results_and_methods.md` deploy sentence.

---

## Sources

- Full text: https://www.frontiersin.org/journals/neurorobotics/articles/10.3389/fnbot.2025.1582995/full  
- DOI: https://doi.org/10.3389/fnbot.2025.1582995  
