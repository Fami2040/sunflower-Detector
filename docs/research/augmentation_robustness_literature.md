# Data augmentation and robustness for grain/seed/head detection and counting

Literature review and practical recommendations for **harchoc** (dense **benchtop sunflower-seed** detection/counting: class **0 = developed**, class **1 = aborted**; YOLO @ `imgsz=1280`; frozen `data/splits/`; HSP threshold protocol on val→test). Training uses `configs/experiments/train_*.json` + `scripts/train.py` under `mamba run -n harchoc`.

**Related (repo):** [Training tech scan — augmentation (2026)](training_tech_scan_2026_augmentation.md) (S0–S14 smokes, `max_det` parity, `close_mosaic` on 15-ep runs).

**Out of scope here:** UAV head counting as primary task, synthetic/GenAI datasets, SAHI/telegram deploy tuning, new model architectures.

---

## Top recommendations (1280 YOLO, counting-first)

These align with `configs/aug/robustness_minimal.yaml` and `scripts/train.py` `_BASELINE_DEFAULTS` (`mosaic=0.1`, moderate HSV, low translate/scale) but tighten them for **count integrity** and the observed **val ≫ test** gap.

| Priority | Recommendation | Rationale |
|:--:|---|---|
| 1 | **Keep `mosaic` low (0.0–0.15), not field-default 1.0** | Grain/seed papers use mosaic, but counting metrics are sensitive to synthetic multi-scene tiles; Ultralytics warns mosaic only when partial occlusion does not change label semantics ([Ultralytics aug guide](https://docs.ultralytics.com/guides/yolo-data-augmentation/)). |
| 2 | **Set `mixup=0` (and `cutmix=0`) for counting runs** | Ultralytics mixup blends pixels and **concatenates all boxes from both images**—train count per image is not physical; set explicitly in experiment JSON or aug YAML ([aug scan](training_tech_scan_2026_augmentation.md) §2.3). |
| 3 | **Use `close_mosaic=10–20` on 100-epoch runs; scale down for 15-ep smokes** | Ultralytics disables mosaic/mixup/cutmix/copy-paste in the last N **epochs** ([#18013](https://github.com/ultralytics/ultralytics/issues/18013)). Repo default **15** in `robustness_minimal.yaml`; use **`close_mosaic=3`** on 15-ep smokes ([aug scan](training_tech_scan_2026_augmentation.md) Win 2). [Yolo-pest](https://www.nature.com/articles/s41598-025-97825-3) uses `close_mosaic=35` with `epochs=200` (paper text says “iterations”). |
| 4 | **Bias augmentation toward photometric (HSV, blur, noise, shadow)** | Fixed camera, top-down disk: lighting dominates; avoid large `degrees` / `perspective` ([Frontiers corn-on-ear 2021](https://www.frontiersin.org/journals/robotics-and-ai/articles/10.3389/frobt.2021.627009/full)). |
| 5 | **Moderate geometric jitter only: `translate≤0.1`, `scale≤0.2`, `fliplr=0.5`, `degrees=0`** | Matches legacy sunflower recipe; wheat spike UAV work uses 30–60° rotation—not benchtop seeds. |
| 6 | **Tune inference for density: `conf≈0.05`, `iou≈0.3`, `max_det=3000` on train and test eval** | `train_bench_base.json` sets train and **`eval.max_det: 3000`**; use `eval.py --max-det 3000` for ad-hoc exports ([aug scan](training_tech_scan_2026_augmentation.md) S14). |
| 7 | **Evaluate aug ablations on test count error, not val mAP alone** | Val inflation under heavy aug is common; use `scripts/eval.py` on `data/splits/test.txt` + `scripts/error_analysis.py` count MAE. |
| 8 | **Collect session/lighting diversity in real captures** | Practical domain randomization for fixed-camera trays—no sim pipeline required. |

**Suggested recipe (matches committed `robustness_minimal.yaml`):**

```yaml
# ultralytics: block in configs/aug/robustness_minimal.yaml
mosaic: 0.1
close_mosaic: 15
mixup: 0.0
cutmix: 0.0
hsv_h: 0.02
hsv_s: 0.35
hsv_v: 0.35
translate: 0.05
scale: 0.15
degrees: 0.0
fliplr: 0.5
flipud: 0.0
erasing: 0.2
```

---

## Applicability to harchoc

How literature recommendations map to **current repo entrypoints** (verified against code/configs 2026-05-29).

| Literature theme | Repo mechanism | Notes |
|------------------|----------------|-------|
| Conservative aug recipe | `configs/aug/robustness_minimal.yaml` | Referenced by `"aug_config"` in `configs/experiments/train_bench_base.json` and all `train_bench_*.json`; merged in `harchoc/aug_config.py` |
| Train | `mamba run -n harchoc python scripts/train.py --config … [--aug-config …] [--name …]` | `--aug-config` overrides/extends JSON `aug_config`; keys forwarded via `harchoc/train_kwargs.py` (`close_mosaic`, `mixup`, `erasing`, …) |
| 15-ep aug smokes | `configs/experiments/train_aug_s1_close3_smoke.json` | Same aug YAML; see [aug scan §5](training_tech_scan_2026_augmentation.md) S0–S14 table |
| Mosaic-off / photometric-only | Duplicate aug YAML or inline `ultralytics:` overrides | `ablation_variants` in YAML are **documentation only**—not auto-selected; copy keys into `configs/aug/robustness_mosaic_off.yaml` etc. |
| `close_mosaic` vs early stop | `harchoc/train_config.validate_epochs_patience_close_mosaic` | Bench JSONs tested for `epochs - patience >= close_mosaic`; 15-ep smoke with `patience=50` never early-stops but **`close_mosaic=15` disables mosaic from epoch 0** on 15-ep runs |
| Test metrics (count-first) | `scripts/eval.py --imgsz 1280 --max-det 3000` on `test.txt` | Post-train hook reads `eval.max_det` from JSON (**3000** in `train_bench_base.json`) |
| Count MAE / error buckets | `scripts/error_analysis.py` | After eval export; primary metric for aug decisions per [aug scan](training_tech_scan_2026_augmentation.md) |
| HSP thresholds | `scripts/threshold_sweep.py`, `scripts/experiment.py` | Separate from aug; val lock → test |
| Zoo matrix | `scripts/benchmark_matrix.py` + `train_bench_*.json` | Shared aug YAML; RT-DETR query-cap policy separate ([detectors scan](training_tech_scan_2026_detectors.md)) |

**Before interpreting aug sweeps:** `eval.max_det` matches train (`3000` in bench base); run S14 negative control to quantify truncation if you change caps ([aug scan](training_tech_scan_2026_augmentation.md)).

**Not wired:** Albumentations hooks, Select-Mosaic, online copy-paste (box-only labels), Ultralytics `copy_paste` (seg-only).

---

## 1. Augmentation policies in related papers

### 1.1 Wheat grain (benchtop, dense, adhesion)

| Study | Methods | Notes |
|-------|---------|-------|
| [GrainNet, Plant Methods 2025](https://plantmethods.biomedcentral.com/articles/10.1186/s13007-025-01363-y) | Rotation, flip, translation, brightness, **cutout**, **mixup**, **mosaic** | Offline expansion to **4,552** images (8:2 train/val); benchtop wheat with adhesion. Mixup/mosaic used for **detection** training—counting eval should stay on unmixed test imagery. |
| [YOLO-SDL, Front Plant Sci 2024](https://www.frontiersin.org/journals/plant-science/articles/10.3389/fpls.2024.1495222/full) | Random scale, rotation, flip, **noise**, lighting | Offline dataset aug; classes include perfect, germinated, **diseased**, damaged grains (imbalanced). |
| [Wheat spikes YOLOv10, Agronomy 2024](https://www.mdpi.com/2073-4395/14/9/1936) | Flip 180°, rotation 30–60°, brightness ±30%, Gaussian noise, cutout, mosaic (4-image) | **Field** wheat spikes (GWHD), not benchtop grains; stronger geometry than grain papers. |

### 1.2 Maize kernel (on-ear and loose kernels)

| Study | Methods | Notes |
|-------|---------|-------|
| [On-ear corn kernels, Front Robot AI 2021](https://www.frontiersin.org/journals/robotics-and-ai/articles/10.3389/frobt.2021.627009/full) | Albumentations: rotation p=0.9, V/H flip, brightness/contrast, gamma, HSV, flare, shadow, rain, Gaussian noise | On-ear kernel counting + localization (YOLOv5 and density models). |
| [Maize kernel CNN, Sensors 2020](https://www.mdpi.com/1424-8220/20/9/2721) | Sliding-window CNN + NMS + center regression; ~70% training images with flip/color aug | Overlap/NMS on dense on-ear kernels, not YOLO online aug. |
| [Maize embryo segmentation, Front Plant Sci 2023](https://www.frontiersin.org/journals/plant-science/articles/10.3389/fpls.2023.1108355/full) | Mask-coupled H-flip; brightness/contrast ±0.2; scale 0.75–1; shear ±π/6 | Segmentation (embryo ID); aug paired on image+mask. |
| [LWCD-YOLO corn seeds, Agriculture 2025](https://www.mdpi.com/2077-0472/15/18/1968) | Offline rotation, scale, flip, contrast, crop (**10×** on training set); **Ultralytics mosaic disabled** in §3.1 | Benchtop corn-seed platform; mosaic off in training config. |

### 1.3 Sunflower and close analogues

| Study | Methods | Notes |
|-------|---------|-------|
| [Sunflower heads UAV YOLOv11, Sustainability 2025](https://www.mdpi.com/2071-1050/18/2/1026) | Rotation, H/V flip, brightness, minor perspective; conf/IoU tuning | **Head** detection (512×512 tiles); custom aug pipeline, not stock Ultralytics mosaic/mixup defaults. |
| [Sunflower disk inclination, Front Plant Sci 2025](https://www.frontiersin.org/journals/plant-science/articles/10.3389/fpls.2025.1614898/full) | YOLO11-seg + geometry | Phenotyping disk **tilt**; geometric post-processing, not aug-heavy training doc. |

### 1.4 Panicle / spike (field heads — analogues only)

| Study | Methods | Notes |
|-------|---------|-------|
| [RICE-YOLO, Agronomy 2024](https://www.mdpi.com/2073-4395/14/4/836) | YOLOv5 online: HSV (1.5%, 70%, 40%), flip 50%, translate 10%, perspective 0.05, **mosaic 100%** | Field rice spikes; mosaic default “on” for UAV small-object pipelines. |
| [FEWheat-YOLO, Plants 2025](https://www.mdpi.com/plants14193058) ([PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC12526082/)) | Mosaic throughout training | Lightweight spike detector; field complex scenes. |

**Pattern:** Benchtop **grain/seed** papers combine photometric + mild geometry + sometimes mosaic/mixup; **field head/spike** papers push mosaic and rotation harder. Harchoc is closer to **benchtop dense seeds** (fixed viewpoint, hundreds of instances per image).

---

## 2. Photometric vs geometric augmentation for small dense objects

### Photometric (prefer for harchoc)

- **What:** HSV jitter, brightness/contrast, gamma, noise, blur, shadows, color temperature.
- **Why:** Seeds on a disk share geometry; appearance shifts from exposure, white balance, moisture, and shadow dominate ([Datature aug guide](https://datature.com/blog/image-augmentation-for-machine-learning-techniques-examples-code); [Applied Sciences SOD survey 2025](https://www.mdpi.com/2076-3417/15/22/11882)).
- **Practice:** Keep `hsv_h` small (0.01–0.03); `hsv_s`/`hsv_v` moderate (0.25–0.4). Ultralytics defaults (`hsv_s=0.7`, `hsv_v=0.4`) are **stronger** than `robustness_minimal.yaml`—justify increases with **test** count MAE.
- **Risk:** Extreme color jitter can merge seed/background contrast; validate on held-out lighting.

### Geometric (use conservatively)

- **What:** Flip, translate, scale, rotation, shear, perspective.
- **Why:** Position/scale invariance; partial objects at frame edge ([Albumentations choosing guide](https://albumentations.ai/docs/3-basic-usage/choosing-augmentations/)).
- **Risk for counting:** Rotation/perspective on near-circular 5–15 px boxes change aspect ratios and neighbor distances; can inflate val mAP while hurting **test count MAE** when inference is fixed top-down.
- **Rule of thumb:** If camera pose is fixed at inference, cap `degrees` at 0–5° and skip `perspective` unless validated on test counts.

### Occlusion-specific

- **Random erasing / cutout:** Simulates missing seeds or debris; Ultralytics `erasing` (default 0.4) and GrainNet-style cutout appear in grain literature. Repo uses **`erasing: 0.2`** in `robustness_minimal.yaml`.
- **Constrained dropout:** Region-aware occlusion for tiny objects ([Albumentations](https://albumentations.ai/docs/3-basic-usage/choosing-augmentations/))—future custom hook.

### Composition / instance level

- **Mosaic:** Scale diversity and more objects per step ([Ultralytics](https://docs.ultralytics.com/guides/yolo-data-augmentation/)); default `mosaic=1.0` in generic YOLO recipes vs **0.1** here.
- **Select-Mosaic:** Highest object-density tile in largest quadrant ([arXiv:2406.05412](https://arxiv.org/abs/2406.05412))—not in stock Ultralytics; low `mosaic` is the practical substitute.

---

## 3. Domain randomization

**Core idea:** Randomize visual factors during training so deployment looks like one more sample from a broad distribution ([Tobin et al., 2017](https://arxiv.org/abs/1703.06907)).

| Mechanism | Relevance to harchoc |
|-----------|---------------------|
| Lighting / exposure randomization | **High**—simulate sun/cloud/shadow across disk sessions |
| Texture / background diversity | **Medium**—vary tray/cloth/head background in real captures |
| Camera noise & blur | **Medium**—motion blur less critical for static benchtop |

**Practical DR without a simulator:** Widen `hsv_*` within `robustness_minimal.yaml` bounds and collect more tray sessions—validate on **test** count MAE ([aug scan](training_tech_scan_2026_augmentation.md) S3, S8).

---

## 4. Mosaic, Mixup, and counting tasks — cautions

| Technique | Detection benefit | Counting risk | Mitigation |
|-----------|-------------------|---------------|------------|
| **Mosaic** | More objects/scales per step | Image-level count sums multiple scenes; density statistics shift | Low `mosaic` (≤0.15); `close_mosaic`; report counts on **test**, not mosaic-augmented val |
| **Mixup** | Regularization | Blends pixels + concatenates boxes—non-physical counts | **`mixup=0`** (repo default in aug YAML) |
| **CutMix** | Local features vs mixup | Label concatenation alters global count | **`cutmix=0`** |

Ultralytics: use mosaic only if multi-image composition matches label semantics ([docs](https://docs.ultralytics.com/guides/yolo-data-augmentation/)).

**Eval discipline (harchoc):** Ultralytics val during train drives early stopping; **manuscript metrics** from `eval.py` on **`data/splits/test.txt`** with **`--imgsz 1280`** and **`--max-det 3000`** (default from bench JSON `eval` block). Compare **count MAE / MAPE** with mAP when tuning aug ([aug scan](training_tech_scan_2026_augmentation.md) §3–5).

---

## 5. `configs/aug/robustness_minimal.yaml` (committed)

Live file at `configs/aug/robustness_minimal.yaml` (referenced from `train_bench_base.json` and every `train_bench_*.json`). Key `ultralytics:` keys match the recipe in §Top recommendations; `ablation_variants` documents mosaic-off and photometric-only overrides for smoke YAML copies.

**Merge path:** `scripts/train.py` → `merge_aug_yaml()` → `ultralytics_train_kwargs()`; all `ultralytics:` keys in the aug YAML (including `close_mosaic`, `mixup`, `erasing`) are forwarded—no passthrough gap. `_BASELINE_DEFAULTS` still supplies `mosaic=0.1` when no YAML is used (`train.py` without `aug_config`).

---

## 6. Training hyperparameters (beyond aug)

| Parameter | Repo baseline | Literature / guidance |
|-----------|---------------|----------------------|
| `imgsz` | 1280 | Small seeds need resolution ([Applied Sciences 2025](https://www.mdpi.com/2076-3417/15/22/11882)) |
| `batch` | 1 (yolov8m) | Accept; accumulation optional later |
| `optimizer` / `lr0` | AdamW, 2e-4 | Stable until aug ablations converge |
| `iou` (train NMS) | 0.3 | Dense overlap |
| `max_det` | 3000 train + eval | Required for full-disk counts |
| `patience` | 50 | With strong aug, use **test** count MAE for science conclusions |
| `workers` | 2 | Raise if CPU aug bound |

**Ablation order (matches [aug scan](training_tech_scan_2026_augmentation.md)):** (1) fix `eval.max_det`; (2) 15-ep S0–S14 smokes with scaled `close_mosaic`; (3) sweep `mosaic` ∈ {0, 0.1, 0.3}; (4) `close_mosaic` ∈ {10, 15, 25} on 100-ep runs.

---

## 7. References (URLs)

1. Wang et al., *GrainNet* (Plant Methods, 2025): https://plantmethods.biomedcentral.com/articles/10.1186/s13007-025-01363-y  
2. YOLO-SDL (Front. Plant Sci., 2024): https://www.frontiersin.org/journals/plant-science/articles/10.3389/fpls.2024.1495222/full  
3. Wheat spikes YOLOv10 (Agronomy, 2024, 14(9):1936): https://www.mdpi.com/2073-4395/14/9/1936  
4. Hobbs et al., on-ear corn kernels (Front. Robot. AI, 2021): https://www.frontiersin.org/journals/robotics-and-ai/articles/10.3389/frobt.2021.627009/full  
5. Khaki et al. (Sensors, 2020, 20(9):2721): https://www.mdpi.com/1424-8220/20/9/2721  
6. Maize embryo segmentation (Front. Plant Sci., 2023): https://www.frontiersin.org/journals/plant-science/articles/10.3389/fpls.2023.1108355/full  
7. LWCD-YOLO (Agriculture, 2025, 15(18):1968): https://www.mdpi.com/2077-0472/15/18/1968  
8. RICE-YOLO (Agronomy, 2024, 14(4):836): https://www.mdpi.com/2073-4395/14/4/836  
9. FEWheat-YOLO (Plants, 2025, 14(19):3058): https://www.mdpi.com/plants14193058 · PMC: https://pmc.ncbi.nlm.nih.gov/articles/PMC12526082/  
10. Sunflower head UAV YOLOv11 (Sustainability, 2025, 18(2):1026): https://www.mdpi.com/2071-1050/18/2/1026  
11. Sunflower disk inclination (Front. Plant Sci., 2025): https://www.frontiersin.org/journals/plant-science/articles/10.3389/fpls.2025.1614898/full  
12. Small-object detection survey: https://www.mdpi.com/2076-3417/15/22/11882  
13. Select-Mosaic: https://arxiv.org/abs/2406.05412  
14. Copy-paste / RS (example): https://www.mdpi.com/2072-4292/18/4/647  
15. Domain randomization (Tobin et al., 2017): https://arxiv.org/abs/1703.06907  
16. Ultralytics YOLO data augmentation: https://docs.ultralytics.com/guides/yolo-data-augmentation/  
17. Yolo-pest, `close_mosaic` (Sci. Rep., 2025): https://www.nature.com/articles/s41598-025-97825-3  

**Supplementary:** SoyCountNet (Front. Plant Sci., 2026): https://www.frontiersin.org/journals/plant-science/articles/10.3389/fpls.2026.1743104/full · PBR vs DR (Sensors, 2021): https://www.mdpi.com/1424-8220/21/23/7901 · Albumentations: https://albumentations.ai/docs/3-basic-usage/choosing-augmentations/

---

## 8. Gaps and follow-up experiments

- Run [aug scan](training_tech_scan_2026_augmentation.md) **S0–S14** (15-ep) with `mamba run -n harchoc`; primary metric: **test count MAE**.
- Bench JSON already sets `eval.max_det: 3000`; override with `eval.py --max-det` only for ablations.
- For mosaic-off: `configs/aug/robustness_mosaic_off.yaml` with `mosaic: 0`, `close_mosaic: 0` (see YAML `ablation_variants`).
- If val/test gap persists with mild aug, prioritize split/drift (`scripts/split_drift.py`) and collection diversity before raising mosaic.

---

## Validated 2026-05-29

**Changelog**

- Cross-checked claims against `configs/aug/robustness_minimal.yaml`, `scripts/train.py`, `harchoc/aug_config.py`, `harchoc/train_kwargs.py`, and `configs/experiments/train_bench_*.json` (aug merge forwarded; `eval.max_det: 3000` aligned with train).
- Removed stale note that `eval.py --imgsz` was unwired; added `eval.max_det` / aug-scan S0–S14 cross-links.
- Added **Applicability to harchoc** table (scripts, flags, smoke JSON, ablation YAML caveat).
- Fixed FEWheat citation (Plants 2025, 14(19):3058, not “Agriculture 2025”); softened GrainNet pre-augment count (paper states 4,552 post-augmentation).
- Clarified Yolo-pest `close_mosaic=35` with `epochs=200` (paper wording “iterations” vs Ultralytics epochs).
- Linked throughout to [training_tech_scan_2026_augmentation.md](training_tech_scan_2026_augmentation.md).
- Re-validated primary paper URLs via publisher pages / PMC (2026-05-29).

**Could not fully verify**

- GrainNet **pre-augmentation** image count (1,198→4,552 ratio): paper reports 4,552 after enhancement; Table 1 numeric breakdown not extracted from HTML.
- [Wheat spikes Agronomy 2024](https://www.mdpi.com/2073-4395/14/9/1936): fetch timed out; MDPI listing and prior validation retained.
- [Ultralytics aug guide](https://docs.ultralytics.com/guides/yolo-data-augmentation/): fetch timed out; URL matches Ultralytics docs site structure.
- [Datature aug guide](https://datature.com/blog/image-augmentation-for-machine-learning-techniques-examples-code): not re-fetched (blog; non-peer-reviewed).
- MDPI RS copy-paste reference (#14): not re-fetched; URL format consistent with MDPI.

**Validated literature registry:** [`docs/manuscript/literature_validated.md`](../manuscript/literature_validated.md) · **Related Work (MS-LIT):** [`related_work_outline.md`](../manuscript/related_work_outline.md) · [`architecture_recommendations.md`](../manuscript/architecture_recommendations.md) · [Model improvement stack](../../backlog.md#model-improvement-stack-test-count-mae) steps 1 + 4
