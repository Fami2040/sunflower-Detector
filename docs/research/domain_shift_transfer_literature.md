# Domain shift, cross-condition evaluation, and transfer learning

Literature notes for **harchoc** (benchtop **sunflower-seed** detection, frozen `data/splits/test.txt` for manuscript metrics). Use for optional cross-session slices and `finetune.py`—not as a mandate to adopt UAV head-counting workflows.

**Out of scope unless explicitly tagged in data:** unsupervised domain adaptation across unlabeled farms, satellite/TimeMatch-style temporal DA, multi-cultivar UAV LOFO.

Tied to scaffolds `scripts/eval_domains.py`, `scripts/finetune.py`, and `configs/transfer/`.

---

## Applicability to harchoc (HSP)

Benchtop **sunflower-seed** detection on fixed head photos (`imgsz=1280`, ~500 boxes/image, classes **developed** / **aborted**). Domain shift here is **session / tray / lighting**, not UAV plot blocks or cultivar phenology.

| Dimension | HSP (this repo) | Closest literature peers | Repo status (2026-05-29) |
|-----------|-----------------|--------------------------|--------------------------|
| **Domain axis** | `tray_key` from CVAT-style stems (`349-10-2`, `3a2-2`; aug suffix stripped) | Session / field holdout (Habibi et al. 2024); stratified UAV tiles (Iamchuen et al. 2026) | **`eval_domains.py` catalog Done** via `harchoc/domain_tags.py` |
| **Split lists** | Frozen `data/splits/{train,val,test}.txt`; optional `data/domains/{split}_{tray_key}.txt` from `--write-domain-splits` | LOFO / spatial CV for extrapolation claims | Writer **Done**; per-tray mAP **Partial** ([`domain_eval.json`](../../reports/domains/domain_eval.json), 41/52 trays) |
| **Manuscript metric** | Canonical `test.txt` only; val for selection / threshold lock | Random CV optimistic for new fields [12] | `eval.py` + HSP protocol ([eval scan](training_tech_scan_2026_eval_calibration.md)) |
| **Transfer / fine-tune** | Same two classes; optional tray holdout for adaptation | TFA two-stage [6]; YOLO freeze table [8]; Gandhi staged unfreeze [9] | **`finetune.py` Done** (`--stage 1|2`, `finetune_tray_stage{1,2}.json`); GPU 25+25 ep metrics open |
| **Out of scope (P1)** | Unlabeled target farms, satellite TimeMatch [4], crop-weed UDA [3] | Field UDA pipelines | No UDA script; collect labels or add tray slice to train/val first |

**Tray keys:** parsed by `_TRAY_KEY_RE` in `harchoc/domain_tags.py` — leading alphanumeric token with `-` segments (e.g. `349-10-2__________aug0` → `349-10-2`). Catalog JSON records per-key `n_images`, `n_boxes`, `class_counts`, and split membership.

**Commands (catalog today):**

```bash
export DATASET_ROOT="$(pwd)/data/raw/extracted/dataset"
python scripts/eval_domains.py --catalog reports/domains/catalog.json --out reports/domains/domain_eval.json
python scripts/eval_domains.py --dry-run --catalog reports/domains/catalog.json --out reports/domains/domain_eval.json
python scripts/eval_domains.py --write-domain-splits --domains-dir data/domains
# Per-tray mAP (GPU): mamba run -n harchoc python scripts/eval.py \
#   --weights models/best.pt --split-file data/domains/test_<tray_key>.txt --imgsz 1280
python scripts/finetune.py --dry-run   # CI; live train loop not implemented
```

**Open before domain science:** align eval `max_det` with train (300 vs 3000 on dense trays — [aug scan](training_tech_scan_2026_augmentation.md) S14 / backlog P0) so per-tray count/mAP comparisons are not capped artificially.

---

## 1. What “domain” means in agricultural vision

Agricultural images shift along axes that rarely appear in generic benchmarks (COCO, ImageNet):

| Axis | Examples | Effect on detectors |
|------|----------|---------------------|
| **Acquisition** | UAV nadir vs ground RGB, resolution, sensor | Scale, blur, GSD, occlusion patterns |
| **Illumination** | Time of day, cloud, flash vs ambient | Color texture cues dominate; false positives on soil/shadow |
| **Environment** | Soil, weeds, residue, background clutter | Background “looks like” heads or grains |
| **Biology** | Cultivar/hybrid, phenology (R-stages), head tilt, senescence | Appearance and size distribution of targets |
| **Protocol** | Lab/controlled vs operational field | Clean backgrounds vs heterogeneous scenes |

Reviews and field studies agree: models trained on one condition (greenhouse, single farm, single season) **overestimate** performance when evaluated with random image splits; **spatially aware** or **leave-condition-out** protocols are needed for honest transfer claims [1, 9, 10, 11].

Sunflower **head** UAV work [10] is the closest analogue for lighting/clutter and tiled orthomosaics; disease TL reviews [11] stress field domain shift and cross-regional validation. Our primary images are fixed-tray seeds, not nadir heads.

---

## 2. Cited literature (12 sources)

### Domain shift and adaptation (agriculture)

1. **Hu et al., 2025** — *Domain Adaptation for Big Data in Agricultural Image Analysis: A Comprehensive Review* ([arXiv:2506.05972](https://arxiv.org/abs/2506.05972); **preprint**). Taxonomy of DA (supervised / semi- / unsupervised, adversarial, generative). Stresses illumination, season, crop type, and sensor diversity as primary shift drivers; surveys public ag-vision datasets and compares shallow vs deep DA strategies.

2. **Vu et al., 2019** — *ADVENT: Adversarial Entropy Minimization for Domain Adaptation in Semantic Segmentation* (CVPR 2019; [arXiv:1811.12833](https://arxiv.org/abs/1811.12833)). UDA for semantic segmentation via entropy minimization and adversarial alignment (synthetic→real benchmarks). Widely cited in ag segmentation work as a domain-shift baseline—not crop-type classification.

3. **Ilyas et al., 2023** (often “Won et al.”) — *Overcoming field variability: unsupervised domain adaptation for enhanced crop–weed recognition in diverse farmlands* (*Frontiers in Plant Science* 14:1234616; [10.3389/fpls.2023.1234616](https://www.frontiersin.org/journals/plant-science/articles/10.3389/fpls.2023.1234616/full)). Segmentation + adversarial feature alignment + entropy minimization on unlabeled target farms; gains on four held-out fields. Template for “source farm labeled, target farm unlabeled” — **out of harchoc scope** unless unlabeled target trays are added.

4. **Nyborg et al., 2022** — *TimeMatch: Unsupervised Cross-Region Adaptation by Temporal Shift Estimation* ([arXiv:2111.02682](https://arxiv.org/abs/2111.02682); *ISPRS J. Photogramm. Remote Sens.* 188, [doi:10.1016/j.isprsjprs.2022.04.018](https://doi.org/10.1016/j.isprsjprs.2022.04.018)). Satellite **crop-type** classification from image time series; explicitly models **phenological temporal shift** between regions—a loose metaphor for sunflower growth-stage domains (early vs peak flowering), not tray imaging.

5. **Arad et al., 2019** — *Controlled lighting and illumination-independent target detection…* (sweet-pepper greenhouse harvesting; *Sensors* [10.3390/s19061390](https://www.mdpi.com/1424-8220/19/6/1390)). Flash–No-Flash (FNF) imaging subtracts ambient light to stabilize color-based detection. Shows illumination as a first-class domain axis; practical for lab vs field lighting gaps (even if we stay RGB-only).

### Object detection, few-shot, and YOLO transfer

6. **Wang et al., 2020** — *Frustratingly Simple Few-Shot Object Detection* (TFA; ICML 2020; [arXiv:2003.06957](https://arxiv.org/abs/2003.06957), [PMLR](http://proceedings.mlr.press/v119/wang20j/wang20j.pdf)). **Two-stage fine-tune**: train full Faster R-CNN on base classes → freeze feature extractor (backbone + RPN) → tune box heads only on a K-shot balanced set; stage-2 LR **÷20** vs stage 1. Strong baseline when target labels are scarce (few-shot **classes**; adapt the freeze/head-only idea for same-class domain shift).

7. **Zhao et al., 2022** — *Exploring Effective Knowledge Transfer for Few-Shot Object Detection* (ACM MM; [arXiv:2210.02021](https://arxiv.org/abs/2210.02021)). Builds on TFA with RoI **distribution calibration** (low-shot) and ImageNet-guided regularization (high-shot); stage-2 fine-tuning still **freezes the backbone** and updates only the detection head.

8. **Ultralytics** — [Fine-tuning guide](https://docs.ultralytics.com/guides/finetuning-guide) (YOLO26 docs, 2026). Operational YOLO guidance: `freeze` first N layers, two-stage train (frozen backbone then full model at lower `lr0`), table of freeze depth vs dataset size / domain distance from COCO.

9. **Gandhi & Gandhi, 2025** — *Fine-Tuning Without Forgetting: Adaptation of YOLOv8 Preserves COCO Performance* ([arXiv:2505.01016](https://arxiv.org/abs/2505.01016); **preprint, not peer-reviewed**). Empirical sweep of `freeze` depth on YOLOv8n; unfreezing through layer 10 (`freeze=10`) reports **~10 absolute mAP50** gain on a 6-class fruit set with **&lt;0.1 absolute mAP** COCO change—heuristic support for staged unfreeze in `finetune_minimal.yaml` (fruit domain, not sunflower seeds).

### Sunflower, multi-condition detection, and evaluation rigor

10. **Iamchuen et al., 2026** — Sunflower head detection + yield from UAV imagery with YOLOv11 (*Sustainability* **18**(2), 1026; [10.3390/su18021026](https://www.mdpi.com/2071-1050/18/2/1026)). **Single** ~900 m² plot (Thailand); 1290 tiles (512×512) from 215 UAV frames; 80:20 train/test (also 60:40, 70:30). Methods note **spatially separated** train vs validation tiles and stratified 2×2 m ground plots; discussion cautions that a **homogeneous** plot can inflate yield correlation (R²=0.984) and calls for **future** multi-location / multi-season validation—not a multi-season study itself.

11. **Gulzar, 2025** — *Applications of transfer learning in sunflower disease detection* (*Turkish Journal of Biology* 49(5):534–549; [10.55730/1300-0152.2763](https://doi.org/10.55730/1300-0152.2763), [PMC12614360](https://pmc.ncbi.nlm.nih.gov/articles/PMC12614360/)). Systematic review (30 papers): **domain shift** when ImageNet/lab models move to field imagery; urges larger, **cross-regional** datasets and field validation—analogous rigor needs for detection, not head-counting UAV work.

12. **Habibi et al., 2024** — *Critical evaluation of cross-validation strategy … UAV-based soybean yield prediction* (*Journal of Agriculture and Food Research* 16:101096; [10.1016/j.jafr.2024.101096](https://doi.org/10.1016/j.jafr.2024.101096)). Compares random, cluster spatial, and **leave-one-field-out** CV on 10 fields; **random CV is optimistic** for extrapolation to a held-out field; spatial and LOFO CV better match independent-field error. Analogue for **tray-key holdout** reporting, not yield ML directly.

**Honorable mentions (not counted in 12):** MAR-YOLOv9 multi-dataset ag detection [PMC11521258]; optimal YOLO dataset size/labels for ag objects [MDPI Agronomy 15(7):731]; 3D ag point-cloud UDA [PMC12647907].

---

## 3. Implications for sunflower / grain detection

- **Acquisition session / lighting** on the same tray setup—most plausible domain axis for harchoc; **`tray_key`** in catalog + optional `data/domains/*_{tray_key}.txt` lists.
- **Lighting** drives color-based FPs; aug (`configs/aug/robustness_minimal.yaml`) helps but does not replace held-out session eval.
- **UAV head / variety / growth stage**—only relevant if future data collection adds those tags; do not block current prep on LOFO field protocols [12].

---

## 4. `eval_domains.py` harness (implemented catalog + eval scaffold)

**Partial (2026-05-29):** `scripts/eval_domains.py` builds a tray-key catalog (`eval_domains_run.v1` → `--catalog`) and `domain_eval.v1` (`--out`, `--dry-run`, `--run-all-trays`). Live CPU per-tray mAP in [`domain_eval.json`](../../reports/domains/domain_eval.json) (**41/52** trays ok). Per-tray **counting MAE** still **Next** (**MS-GEN**).

**Next:** loop `eval.py` (or matrix helper) over domain split files for per-tray mAP50 / count MAE; aggregate deltas vs canonical test in one JSON (design below).

### 4.1 Catalog and split lists (as implemented)

| Piece | Location / behavior |
|-------|---------------------|
| **Tray key** | `harchoc/domain_tags.tray_key_from_stem` — e.g. `349-10-2`, `3a2-2` |
| **Catalog fields** | `tray_key`, `n_images`, `n_boxes`, `class_counts` (0=developed, 1=aborted), `splits`, `example_images` |
| **Subset lists** | `--write-domain-splits` → `data/domains/{train,val,test}_{tray_key}.txt` (one path per line, same format as `data/splits/test.txt`) |
| **Weights in JSON** | `--weights` metadata only until per-domain eval is wired |

Optional future: `data/domains/manifest.json` with free-form tags (`lighting`, `session`) when metadata exists beyond filename stems — not required for current catalog.

### 4.2 Per-domain evaluation loop (mirror `eval.py`)

For each domain split file (e.g. `data/domains/test_349-10-2.txt`):

1. Resolve dataset via `resolve_dataset_args` / `DATASET_ROOT` (same precedence as training).
2. Run `scripts/eval.py` with `--split-file`, `--imgsz 1280` (wired to `model.val()`), and **`--max-det` aligned with train** (3000 until P0 parity fix).
3. Record: `mAP50`, `mAP50-95`, per-class AP, weights path, **split SHA256**, `eval_target.split_role: "test"`.
4. Aggregate: domain × metric table; **delta vs canonical** `data/splits/test.txt` on the same weights.

### 4.3 Splitting discipline (avoid optimistic bias)

| Protocol | Use for | Avoid |
|----------|---------|--------|
| Canonical `test.txt` | Primary manuscript metric (frozen seed 0) | Tuning thresholds or early stopping |
| `val.txt` | Training-time model selection only | Reporting final generalization |
| Domain `test_{tray_key}.txt` | Cross-tray / cross-session reporting | Training unless explicitly a *target adaptation* experiment |
| Leave-one-`tray_key`-out | New trays with few labels | Pooled random splits that mix trays |

- Enforce **spatial disjointness** only if domain lists come from tiled UAV mosaics (not default for tray images).
- Prefer **leave-one-condition-out** when condition tags exist (e.g. session id); otherwise use canonical `test.txt` only.
- Store **before/after** metrics when `finetune.py` runs: baseline weights vs fine-tuned on the same domain lists.

### 4.4 JSON output shape (target when per-domain eval is wired)

```json
{
  "schema_version": "eval_domains_run.v1",
  "weights": "models/best.pt",
  "catalog": { "n_domains": 0, "domains": [] },
  "canonical_test": { "split_file": "data/splits/test.txt", "metrics": { } },
  "domains": [
    {
      "tray_key": "349-10-2",
      "split_file": "data/domains/test_349-10-2.txt",
      "split_sha256": "...",
      "metrics": { "mAP50": 0.0, "mAP50-95": 0.0 },
      "delta_vs_canonical": { "mAP50": 0.0 }
    }
  ]
}
```

Today’s catalog-only payload omits `domains[].metrics`; see `scripts/eval_domains.py` notes field.

---

## 5. Fine-tuning workflow (`finetune.py`, `configs/transfer/`)

**Status:** scaffold only — `finetune.py` writes `finetune_run.v1` metadata (`build_versioned_scaffold_payload`); `--dry-run` for CI. Training loop and before/after eval **not implemented**.

Placeholder config:

```yaml
# configs/transfer/finetune_minimal.yaml
name: transfer_finetune_minimal
freeze_backbone: true   # maps to Ultralytics freeze=N when implemented
unfreeze_epoch: 10
epochs: 50
lr: 0.001
seed: 0
```

### 5.1 Recommended stages (align with literature + Ultralytics)

| Stage | Settings | Rationale |
|-------|----------|-----------|
| **0 – Baseline** | Eval canonical + domain tests with `--base-weights` | Establish transfer gap |
| **1 – Head only** | `freeze=10` or `freeze=23` (YOLO), `lr0 ≈ 1e-3`, short epochs | TFA / small-target-set practice [6, 8] |
| **2 – Partial backbone** | `unfreeze_epoch: 10`, lower `lr0` (1e-4–5e-4) | Gandhi et al.: mid-backbone adapts without catastrophic forgetting [9] |
| **3 – Full (optional)** | `freeze=None`, lowest LR, only if target set > ~500 labels and domain far from pretrain | Hu et al.: large shift may need backbone relearning [1] |

**Learning rate:** stage-2 typically **10×–20× below** stage-1 [6, 8]; use cosine or linear decay with patience on **target-domain val**, not canonical test.

**Few-shot:** keep a **balanced** mini-set (base + novel shots per class) when adding new classes; freeze backbone, tune head [6, 7]. For **same-class** domain adaptation (sunflower only), still use small LR and early stopping on target val to avoid erasing COCO-like features [9] (fruit fine-grain transfer study; not sunflower-specific).

### 5.2 Config knobs to add (future `configs/transfer/`)

```yaml
name: transfer_sunflower_field_b
base_weights: models/best.pt
freeze: 10                    # Ultralytics layer index
unfreeze_epoch: 15
epochs: 30
lr0: 0.001
lrf: 0.01
optimizer: AdamW
warmup_epochs: 3
patience: 10
target_split: data/domains/train_<tray_key>.txt   # adaptation train (not canonical test)
target_val_split: data/domains/val_<tray_key>.txt
shots: null                   # or 50 for few-shot experiment
seed: 0
eval_before_after: true
canonical_test: data/splits/test.txt           # never train on this
```

### 5.3 `finetune.py` contract (scaffold → implementation)

1. Load YAML; resolve dataset; verify target splits do not intersect canonical `test.txt` (SHA256 check).
2. Run `train` (Ultralytics) from `base_weights` with freeze schedule (`freeze`, `unfreeze_epoch` → staged `model.train()`).
3. Write `reports/transfer/finetune.json`: config hash, split SHA256s, train metrics, **post-hoc** `eval.py` on canonical test + each `test_{tray_key}.txt`.
4. Support `--dry-run` (**Done**) for CI.

### 5.4 When *not* to fine-tune

- Target domain has **no labels** → out of repo scope for now (no UDA pipeline); collect labels or add a domain slice to train/val first.
- Target set **< ~30 images** → use `cv_eval.py` or report wide CIs; avoid overfitting canonical test.

---

## 6. Test-only evaluation discipline (checklist)

Aligned with `data/splits/README.md`, `scripts/eval.py`, and `scripts/train.py` metadata.

### Canonical benchmark (manuscript)

- [ ] Final numbers only from `scripts/eval.py` on `data/splits/test.txt` (or explicit `--split-file` documented in run metadata).
- [ ] Record `eval_target.split_role: "test"` and split SHA256 via `collect_run_metadata(include_repo_splits=True)`.
- [ ] Never use `val.txt` for reported generalization metrics.
- [ ] Do not tune detection confidence on `test.txt`; use `val.txt` or a dedicated calibration split.

### Training vs selection

- [ ] Ultralytics validation during training uses `val.txt` only (`split_roles.ultralytics_val` in train metadata).
- [ ] Post-train auto-eval in `train.py` targets **test** for the recorded benchmark run—do not reuse that loop for hyperparameter search without a separate protocol.

### Domain / transfer experiments

- [ ] Register cross-tray sets via `eval_domains.py --write-domain-splits`; git-ignore raw images, track list format in docs.
- [ ] **Before fine-tune:** run `eval_domains.py` + canonical `eval.py` with source weights.
- [ ] **After fine-tune:** re-run same splits; store deltas in `reports/transfer/finetune.json`.
- [ ] Fine-tune train/val must not include paths from canonical `test.txt` (assert in script).
- [ ] For LOFO-style claims, train without any image from the held-out location/stage in **both** train and val.

### Reporting

- [ ] Report per-domain metrics, not only pooled domain average.
- [ ] Document tags (lighting, stage, location) in JSON for figure scripts (`make_figures.py`).
- [ ] Note spatial tile deduplication when UAV mosaics are used [10].
- [ ] Compare random vs spatial/domain holdout when publishing transfer conclusions [12].

---

## 7. Quick reference: freeze / LR heuristics

| Target labels | Domain vs pretrain | Backbone | Initial LR (order of magnitude) |
|---------------|-------------------|----------|----------------------------------|
| &lt; 100 | Similar | Freeze (`freeze≥10`) | 1e-3 → 1e-4 |
| 100–500 | Moderate shift | Unfreeze mid layers @ epoch 10–15 | 1e-3 then 1e-4 |
| 500+ | Large shift (field vs COCO) | Full fine-tune, strong aug | 1e-3 with decay |
| K-shot / class | Novel class | Freeze backbone, head only [6] | 1e-3 (10–20× below base train) |

---

## 8. References (formatted)

1. Hu, X., Chen, S., Duan, Q., Ahn, C. K., Shang, H., & Zhang, D. (2025). Domain adaptation for big data in agricultural image analysis: A comprehensive review. *arXiv:2506.05972*. https://arxiv.org/abs/2506.05972
2. Vu, T.-H., Jain, H., Bucher, M., Cord, M., & Pérez, P. (2019). ADVENT: Adversarial entropy minimization for domain adaptation in semantic segmentation. *CVPR*. https://arxiv.org/abs/1811.12833
3. Ilyas, T., Lee, J., Won, O., Jeong, Y., & Kim, H. (2023). Overcoming field variability: Unsupervised domain adaptation for enhanced crop-weed recognition in diverse farmlands. *Frontiers in Plant Science*, 14, 1234616. https://doi.org/10.3389/fpls.2023.1234616
4. Nyborg, J., Pelletier, C., Lefèvre, S., & Assent, I. (2022). TimeMatch: Unsupervised cross-region adaptation by temporal shift estimation. *ISPRS Journal of Photogrammetry and Remote Sensing*, 188, 301–313. https://doi.org/10.1016/j.isprsjprs.2022.04.018
5. Arad, B., Kurtser, P., Barnea, E., Harel, B., Edan, Y., & Ben-Shahar, O. (2019). Controlled lighting and illumination-independent target detection for real-time cost-efficient applications: The case study of sweet pepper robotic harvesting. *Sensors*, 19(6), 1390. https://doi.org/10.3390/s19061390
6. Wang, X., Huang, T. E., Darrell, T., Gonzalez, J. E., & Yu, F. (2020). Frustratingly simple few-shot object detection. *ICML*. https://arxiv.org/abs/2003.06957
7. Zhao, Z., Liu, Q., & Wang, Y. (2022). Exploring effective knowledge transfer for few-shot object detection. *Proc. ACM Multimedia (MM ’22)*; *arXiv:2210.02021*.
8. Ultralytics. Fine-tuning YOLO on a custom dataset. https://docs.ultralytics.com/guides/finetuning-guide
9. Gandhi, V., & Gandhi, S. (2025). Fine-tuning without forgetting: Adaptation of YOLOv8 preserves COCO performance. *arXiv:2505.01016*.
10. Iamchuen, N., Hongpradit, P., Puttinaovarat, S., & Anucharn, T. (2026). Automated sunflower head detection and yield estimation from high-resolution UAV imagery using YOLOv11. *Sustainability*, 18(2), 1026. https://doi.org/10.3390/su18021026
11. Gulzar, Y. (2025). Applications of transfer learning in sunflower disease detection: Advances, challenges, and future directions. *Turkish Journal of Biology*, 49(5), 534–549. https://doi.org/10.55730/1300-0152.2763
12. Habibi, L. N., Matsui, T., & Tanaka, T. S. T. (2024). Critical evaluation of the effects of a cross-validation strategy and machine learning optimization on the prediction accuracy and transferability of a soybean yield prediction model using UAV-based remote sensing. *Journal of Agriculture and Food Research*, 16, 101096. https://doi.org/10.1016/j.jafr.2024.101096

---

## 9. Open implementation gaps

| Gap | Priority |
|-----|----------|
| Per-tray `eval.py` loop inside `eval_domains.py` | P1 |
| `finetune.py` Ultralytics train + before/after eval | P1 |
| Eval `max_det` parity (300 vs 3000) for fair tray counts | **P0** ([backlog](../../backlog.md)) |
| Optional `manifest.json` tags beyond `tray_key` | P2 |
| LOFO tray-key re-split for adaptation experiments | P2 |

**Done (2026-05-29):** tray-key catalog (`domain_tags.py`, `eval_domains.py`); `--write-domain-splits`; `eval.py --imgsz`; `finetune.py` / `finetune_minimal.yaml` scaffolds + dry-run.

---

## 10. Reviewer cite: Yang et al. 2024 (transfer learning)

**Registry:** [`yang2024_oct_tl`](../manuscript/literature_validated.json) · DOI [10.1371/journal.pone.0296175](https://doi.org/10.1371/journal.pone.0296175)

Yang et al. (PLoS ONE 2024) ensemble CNNs with ImageNet transfer and Grad-CAM on **retinal OCT** — not seed detection. For HARCHOC, use as **future-work analogy** for adapting to new trays/lighting: pretrain on `train.txt`, staged freeze finetune on held-out `tray_key` ([TFA](https://arxiv.org/abs/2003.06957), Gandhi YOLOv8 freeze preprint). **Generalization limits** (single site, indoor dried heads) must cite [`val_test_map_gap.md`](../manuscript/val_test_map_gap.md) and split drift JSON — KS proxies do not guarantee similar ranking mAP.

**Backlog:** **MS-DOMAIN-ADAPT** Done (repo draft [gap §12](../manuscript/reviewer_comments_backlog_gap.md#12-manuscript-draft--domain-adaptation-plan-discussion)); **P1-FINETUNE-LOOP** Done; **P1-DOMAIN-EVAL** Partial ([`domain_eval.json`](../../reports/domains/domain_eval.json)); **MS-GEN** Next.

---

**Validated 2026-05-29.** Cross-checked `scripts/eval_domains.py`, `scripts/finetune.py`, `configs/transfer/finetune_minimal.yaml`, `harchoc/domain_tags.py` against [`backlog.md` § Work queue](../../backlog.md#work-queue-p0--p2) (**P1-DOMAIN-EVAL** Partial) and [backlog stack step 6](../../backlog.md#model-improvement-stack-test-count-mae) (tray / domain shift). Citations spot-checked via arXiv, Frontiers, MDPI, Elsevier / ScienceDirect snippets; Gandhi [9] and Hu [1] remain **arXiv preprints**.

**Validated literature registry:** [`docs/manuscript/literature_validated.md`](../manuscript/literature_validated.md) · **Related Work (MS-LIT):** [`related_work_outline.md`](../manuscript/related_work_outline.md)

### Changelog

| Date | Change |
|------|--------|
| 2026-05-29 | **MS-DOMAIN-ADAPT** repo draft Done: [gap §12](../manuscript/reviewer_comments_backlog_gap.md#12-manuscript-draft--domain-adaptation-plan-discussion), [p0_summary § domain adapt](../../reports/hsp/p0_summary.md#domain-adaptation-plan-reviewer-306308--360); live `domain_eval.json`; finetune stage configs cited. |
| 2026-05-29 | Added § Applicability to harchoc (tray keys, catalog Done, finetune scaffold); aligned §4–5 with implemented `eval_domains.py` / placeholder transfer config; fixed stale `eval.py --imgsz` claim; noted Gandhi preprint status; full Habibi title; Ilyas et al. author order for [3]; validation footer + gaps table. |
| (prior) | Initial 12-source literature + recommended harness design. |
