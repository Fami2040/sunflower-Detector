# Training tech scan (2026): augmentation, schedule, and dense-counting integrity

> **Backlog alignment:** [Model improvement stack](../../backlog.md#model-improvement-stack-test-count-mae) steps **1** (`max_det=3000`, P0-1 **Done**, S14) and **4** (**P1-AUG** S0–S14, **ARCH-MOSAIC0-AB**). Primary metric: test **count MAE** at val-locked conf — not val mAP alone.

Research assignment for **harchoc** (benchtop sunflower-seed YOLO @ `imgsz=1280`, ~500 boxes/image, val≈0.97 mAP50 vs test≈0.79). Compares our committed recipe against Ultralytics YOLO11/YOLO26 defaults and 2025–2026 literature.

**Related:** [`augmentation_robustness_literature.md`](augmentation_robustness_literature.md), [`threshold_calibration_literature.md`](threshold_calibration_literature.md), [`backlog.md` § Work queue](../../backlog.md#work-queue-p0--p2).

---

## Executive summary

1. **Our recipe is already far more conservative than 2026 Ultralytics defaults** (`mosaic=0.1`, `mixup=0`, photometric-biased HSV) — appropriate for dense counting. **`eval.max_det=3000` parity is shipped** (P0-1 **Done**); remaining schedule gaps (`close_mosaic` on 15-ep smokes, early-stop interaction) can still confound aug sweeps.
2. **Five high-leverage, underused wins** for val≫test: (a) **S14 negative control** documents truncation if caps regress (P0-1 **Done**), (b) rescale `close_mosaic` for short smokes and guard early-stop interaction ([#18013](https://github.com/ultralytics/ultralytics/issues/18013)), (c) run mosaic-off / photometric-only ablations, (d) tighten patience + log actual stop epoch from `results.csv`, (e) add photometric diversity (erasing/HSV) without geometric distortion — **not** field-default mosaic/mixup/copy-paste.
3. **15 mapped 15-epoch smokes** below use only existing `scripts/train.py`, `scripts/eval.py`, `scripts/error_analysis.py`, and JSON/YAML configs — no new scripts required (optional inline aug YAML copies for smoke-specific `close_mosaic`).

---

## 1. Current recipe snapshot

### 1.1 Code defaults vs committed configs

| Parameter | `train.py` `_BASELINE_DEFAULTS` | `robustness_minimal.yaml` | `train_bench_*.json` / baseline |
|-----------|-----------------------------------|---------------------------|--------------------------------|
| `epochs` | 100 | — | 100 (15 for smoke) |
| `imgsz` | 1280 | — | 1280 |
| `batch` | 1 | — | model-specific (mostly 1) |
| `optimizer` | AdamW | — | AdamW |
| `lr0` | 2e-4 | — | 2e-4 |
| `patience` | 50 | — | 50 |
| `mosaic` | 0.1 | 0.1 | via aug YAML |
| `close_mosaic` | *(unset in defaults)* | **15** | via aug YAML |
| `mixup` / `cutmix` | *(unset → Ultralytics default 0)* | 0 / 0 | via aug YAML |
| `hsv_h/s/v` | 0.02 / 0.3 / 0.3 | 0.02 / 0.35 / 0.35 | via aug YAML |
| `translate` / `scale` | 0.05 / 0.15 | 0.05 / 0.15 | via aug YAML |
| `degrees` / `perspective` | *(unset)* | 0 / 0 | via aug YAML |
| `erasing` | *(unset → ~0.4 Ultralytics)* | **0.2** | via aug YAML |
| `fliplr` | *(unset)* | 0.5 | via aug YAML |
| `conf` / `iou` / `max_det` (train) | 0.05 / 0.3 / **3000** | — | same |
| `eval.max_det` | — | — | **3000** (P0-1 **Done**; S14 @ 300 = negative control) |

All `train_bench_*.json` extend `train_bench_base.json`, which references `configs/aug/robustness_minimal.yaml`. Aug keys are merged in `harchoc/aug_config.py` and forwarded via `harchoc/train_kwargs.py` (`close_mosaic`, `mixup`, `erasing`, etc. are wired).

### 1.2 Ultralytics YOLO11 / YOLO26 defaults (2025–2026)

From [YOLO26 training recipe](https://docs.ultralytics.com/guides/yolo26-training-recipe/) and standard YOLO11 train args:

| Setting | YOLO11 generic default | YOLO26-S/M recipe (640 COCO) | harchoc |
|---------|------------------------|------------------------------|---------|
| `mosaic` | **1.0** | 0.99 | **0.1** |
| `close_mosaic` | **10** | 10 | **15** |
| `mixup` | 0 (detect) / up to 0.43 (large) | 0.05–0.43 by size | **0** |
| `copy_paste` | 0 (detect boxes) | 0.30–0.40 | **not forwarded** |
| `scale` | 0.5 | **0.90–0.95** | **0.15** |
| `hsv_s` / `hsv_v` | 0.7 / 0.4 | 0.35 / 0.19 | **0.35 / 0.35** |
| `translate` | 0.1 | 0.27 | **0.05** |
| `degrees` | 0 | ~0 (S/M) | **0** |
| `optimizer` | auto (AdamW short / MuSGD long) | **MuSGD** | **AdamW** |
| `lr0` | task-dependent | ~3.8e-4 (S/M) | **2e-4** |
| `patience` | 100 (default) | run-specific | **50** |

**Interpretation:** COCO-scale recipes assume moderate object counts and benefit from heavy composition aug. Our fixed-camera, ~500-instance disks are closer to **benchtop grain** papers (LWCD-YOLO: mosaic off; GrainNet: mosaic with counting caveats) than to RICE-YOLO / FEWheat field pipelines.

---

## 2. 2026 best-practice themes (web + literature)

### 2.1 Mosaic-off late training (`close_mosaic`)

- Ultralytics disables mosaic (and mixup/cutmix/copy-paste in the dataloader) for the **last N epochs** of the **scheduled** `epochs`, not relative to early stop ([#18013](https://github.com/ultralytics/ultralytics/issues/18013)).
- Trigger: `epoch == epochs - close_mosaic` → `_close_dataloader_mosaic()`.
- YOLO26 COCO recipe: `close_mosaic=10` with mosaic≈1.0 — ~10% of training in “calibration” mode.
- Dense small-object papers: [Yolo-pest (2025)](https://www.nature.com/articles/s41598-025-97825-3) uses `close_mosaic=35` on 140 epochs (~25%) for bbox calibration on tiny pests.
- **harchoc guard:** `validate_epochs_patience_close_mosaic()` in `harchoc/train_config.py` requires `epochs - patience >= close_mosaic` (CI-tested on `train_bench_*.json` + baseline; **not** called from `scripts/train.py` at runtime). Smoke JSON with `epochs=15`, `patience=50` would fail this check if validated — use scaled `close_mosaic` (S1+).

### 2.2 Early stopping vs counting integrity

- Val mAP drives Ultralytics early stopping; our **manuscript metric is test-only** (`scripts/eval.py` on `test.txt`).
- Heavy aug + val early stop → classic **val inflation** (documented in our aug literature §4).
- `patience=50` on 100-epoch runs: earliest stop at epoch 50; with `close_mosaic=15`, mosaic-off starts epoch 85 — **only runs that reach epoch 85+ get the tail**. Early stop at 55 → **no mosaic-off ever** (Ultralytics behavior).
- Workaround (not implemented): callback calling `_close_dataloader_mosaic()` when val plateaus, or reduce `patience` / increase `close_mosaic` fraction for counting runs.

### 2.3 Mixup / CutMix / Copy-Paste

| Technique | 2026 Ultralytics status | Counting risk | harchoc |
|-----------|-------------------------|---------------|---------|
| **Mixup** | Default 0 (detect); up to 0.43 in YOLO26 large models | Concatenates all boxes from two images — **non-physical counts** | **0** (correct) |
| **CutMix** | Default 0 | Alters local density | **0** (correct) |
| **Copy-Paste** | Requires **segmentation labels** ([#18073](https://github.com/ultralytics/ultralytics/issues/18073)); YOLO26 uses 0.3–0.4 on seg | Can boost rare/small objects if masks exist | **Not in kwargs whitelist**; box-only labels |

Copy-paste remains a **future win only if** we add polygon/mask labels or offline paste (GrainNet-style offline expansion is an alternative without online copy-paste).

### 2.4 Small-object / dense-scene augmentation

- **Photometric-first** for fixed top-down cameras: HSV, brightness, noise, blur, shadow ([Frontiers corn-on-ear 2021](https://www.frontiersin.org/journals/robotics-and-ai/articles/10.3389/frobt.2021.627009/full); [Applied Sciences SOD survey 2025](https://www.mdpi.com/2076-3417/15/22/11882)).
- **Low geometric distortion** when inference pose is fixed: `degrees=0`, `perspective=0` (our default).
- **Select-Mosaic** ([arXiv:2406.05412](https://arxiv.org/abs/2406.05412)): places high-density quadrant first — relevant to ~500 boxes/image but **not available in stock Ultralytics**; low-probability mosaic is our practical substitute.
- **YOLO26 STAL / ProgLoss**: small-target-aware label assignment and progressive loss balancing — **YOLO26-only** training internals; not applicable to YOLOv8m baseline without architecture change.

### 2.5 Optimizer: MuSGD

- [MuSGD](https://docs.ultralytics.com/reference/optim/muon/) (SGD + Muon orthogonalized updates) ships with YOLO26; `optimizer=auto` picks MuSGD for runs >10k iterations.
- Fine-tuning guidance (Ultralytics): **reduce aug**, **lower lr**, shorter patience — opposite of large-dataset pretraining.
- Our AdamW @ `lr0=2e-4` matches legacy sunflower recipe; MuSGD is an **untested** axis for `yolo11s` / longer runs, not a counting-specific fix.

### 2.6 Mosaic ablation evidence (ARCH-MOSAIC0-AB / P1-AUG-MOSAIC)

Peer-reviewed and official sources supporting mosaic-off and `close_mosaic` ablations for dense small-object / seed counting:

| Source | Claim | harchoc mapping |
|--------|-------|-----------------|
| [Ultralytics YOLO data augmentation](https://docs.ultralytics.com/guides/yolo-data-augmentation/) | Mosaic stitches four images; **“highly effective for improving small object detection”**; `close_mosaic` disables mosaic for the last N **scheduled** epochs so the model stabilizes on deployment-like frames | S0/S4/S5 keep low mosaic + scaled tail; **S2** tests full mosaic-off |
| [Ultralytics PR #3525](https://github.com/ultralytics/ultralytics/pull/3525) | Default `close_mosaic=10` added to improve **validation accuracy** and reduce overfitting by ending training without mosaic | **P1-AUG-CLOSE** sweep `{10, 15, 25}` via [`robustness_close10.yaml`](../../configs/aug/robustness_close10.yaml) / [`robustness_minimal.yaml`](../../configs/aug/robustness_minimal.yaml) / [`robustness_close25.yaml`](../../configs/aug/robustness_close25.yaml) |
| [LWCD-YOLO (Agriculture 2025)](https://doi.org/10.3390/agriculture15181968) (`lwcd_yolo2025`) | Dense **corn seed kernel** benchtop detection: **“Mosaic data augmentation was disabled to ensure environmental consistency”** (§3.1) | **S2** (`mosaic=0`) — **ARCH-MOSAIC0-AB** arm; **HARCHOC test MAE 147.4** (rejected) — LWCD reports their choice; does **not** validate mosaic-off for sunflower ([`lit_audit/lwcd_yolo2025.md`](../../reports/manuscript/lit_audit/lwcd_yolo2025.md)) |
| [Small-object aug framework (Iran J. Computer Science 2025)](https://doi.org/10.1007/s42044-025-00384-z) | YOLOv8m on SODA-A aerial small objects: baseline without augmentation **11.2% AP50**; compositional aug (Mosaic + adaptive cropping) improves sparsely distributed small-object recall in ablation Table 7 | Motivates low/non-zero mosaic sweeps (**S4/S5**) before committing to mosaic=0 |
| [Scale-aware Mosaic (Electronics 2026)](https://doi.org/10.3390/electronics15051075) | Ablation: scale-aware mosaic **+1.09 AP on small objects** vs baseline RT-DETRv2 on HRSC2016-MS; addresses scale-frequency imbalance | Supports mosaic as a tunable axis, not a fixed default — compare **S0** vs **S2** on **test count MAE**, not val mAP alone |

**Ablation protocol (15-ep smokes):**

| Smoke | Mosaic | close_mosaic | Role |
|-------|--------|--------------|------|
| **S0** | 0.1 | 3 (scaled) | Baseline — low mosaic + valid short tail |
| **S2** | 0 | 0 | **ARCH-MOSAIC0-AB** — LWCD-style mosaic-off |
| **S4** | 0.1 | 3 | Same mosaic prob as S0 (control) |
| **S5** | 0.3 | 3 | Upper-bound mosaic sweep arm |

Sweep constants and smoke scaling: `harchoc.train_config` (`MOSAIC_SWEEP_VALUES`, `CLOSE_MOSAIC_SWEEP_100EP`, `scale_close_mosaic_for_epochs`). 100-ep template (archived): [`train_aug_mosaic_sweep_template.json`](../../configs/experiments/archive/templates/train_aug_mosaic_sweep_template.json); winner: [`train_aug_winner_100ep.json`](../../configs/experiments/train_aug_winner_100ep.json).

**Decision rule:** If **S2** (or **S3** photometric-only) beats **S0/S4/S5** on test count MAE at val-locked conf → promote mosaic=0 (or photometric-only) to 100-ep; otherwise run full mosaic / `close_mosaic` sweeps from the template.

---

## 3. Gap diagnosis: val≫test + ~500 boxes/image

| Symptom | Likely cause | Aug/schedule lever |
|---------|--------------|-------------------|
| val mAP50 ≈ 0.97, test ≈ 0.79 | Split/session drift, val used for early stop | `split_drift.py` first; report **test** count MAE for aug decisions |
| Count errors on test (historical) | `eval.max_det=300` capped predictions when GT ≈ 500 | **Fixed:** `train_bench_base.json` + smoke JSONs use `eval.max_det=3000`; S14 @ 300 quantifies truncation |
| 15-ep smokes “show no mosaic benefit” | `close_mosaic=15` on 15-ep run → mosaic **never active** | Use `close_mosaic=3` (or 5) on smokes |
| 100-ep run stops at ~55 | `close_mosaic` never fires | Lower `patience` or callback; mine `results.csv` |
| Val improves, test flat under mild mosaic | Mosaic shifts density statistics | Try `mosaic=0` (LWCD-YOLO precedent) |
| Duplicate/missed seeds | NMS + threshold, not only aug | Keep train `iou=0.3`; threshold protocol on val→test |

---

## 4. Underused wins (5–8 concrete items)

Priority order for **our** val≫test gap and dense counting — all runnable with existing tooling.

### Win 1 — `max_det` train/eval parity (eval plumbing, not aug) — **Done** (P0-1)

**Issue (historical):** Train `max_det=3000` vs eval `max_det=300` silently truncated predictions (~500 GT/image) → inflated FN and count MAE.

**Shipped:** `"eval": {"max_det": 3000}` in `train_bench_base.json`, [`train_smoke_rank_15ep.json`](../../configs/experiments/train_smoke_rank_15ep.json), and bench YAML `infer.max_det: 3000`. **S14** re-evaluates at `--max-det 300` as negative control ([`s14_maxdet_truncation.json`](../../reports/hsp/s14_maxdet_truncation.json)).

**Expected impact:** Honest test baseline before interpreting aug changes; S14 guards against cap regression.

### Win 2 — Smoke-scaled `close_mosaic` (schedule)

**Issue:** `robustness_minimal.yaml` sets `close_mosaic=15`. On `epochs=15`, condition `epoch == 0` from start → **zero mosaic training** in smokes, confounding ablations.

**Action:** For 15-ep smokes use `close_mosaic=3` (~20% tail, mirrors 15/100 ratio) via a smoke aug YAML copy or inline experiment block *without* full aug file (see experiments table).

**Expected impact:** Valid mosaic vs mosaic-off comparisons in cheap runs.

### Win 3 — Mosaic-off and photometric-only ablations (aug)

**Issue:** We have not empirically confirmed whether even `mosaic=0.1` hurts test counts. LWCD-YOLO (corn seeds, dense benchtop) trains with **mosaic disabled**.

**Action:** Compare `robustness_minimal` vs `ablation_variants.robustness_mosaic_off` (documented in aug YAML; requires separate YAML or inline keys) vs baseline `_BASELINE_DEFAULTS` without aug file.

**Expected impact:** If test count MAE drops with `mosaic=0`, reduce val inflation from synthetic multi-scene tiles.

### Win 4 — Early-stop / mosaic-off interaction (schedule)

**Issue:** [Ultralytics #18013](https://github.com/ultralytics/ultralytics/issues/18013): early stop at epoch 55 on a 100-epoch schedule skips the epoch-85 mosaic-off phase.

**Action:** (a) For 100-ep production: try `patience=20–30` so more runs reach the tail, or set `close_mosaic=10` with guard satisfied. (b) Post-hoc: parse `results.csv` → record `stop_epoch` and whether `stop_epoch >= epochs - close_mosaic`. (c) Future: optional callback (backlog).

**Expected impact:** Bbox calibration on real single-scene statistics late in training — reduces val/test calibration gap.

### Win 5 — Conservative photometric sweep (aug)

**Issue:** We under-use Ultralytics photometric knobs vs defaults (`hsv_s=0.35` vs 0.7) but never swept **erasing** or HSV on **test count MAE**.

**Action:** Sweep `erasing ∈ {0.0, 0.1, 0.2, 0.3}` with `mosaic=0`, fixed geometry. Optional: `hsv_v=0.45` for lighting DR without geometry.

**Expected impact:** Better lighting robustness without val-inflating composition aug — matches domain-randomization literature for fixed-camera benchtop.

### Win 6 — Explicit mixup/cutmix=0 on all configs (counting hygiene)

**Issue:** Ultralytics defaults mixup to 0 for detection, but YOLO26 large models use mixup heavily; easy to accidentally inherit if aug YAML omitted.

**Action:** Already in `robustness_minimal.yaml`; verify any config **without** aug_config sets `mixup=0, cutmix=0` explicitly.

**Expected impact:** Prevents silent count-label corruption if defaults change upstream.

### Win 7 — Optimizer smoke on YOLO11 (`optimizer=MuSGD`)

**Issue:** Matrix includes `yolo11s` but we still train with AdamW @ legacy lr. YOLO26 docs recommend MuSGD for longer fine-tunes.

**Action:** 15-ep smoke: `yolo11s`, `optimizer=MuSGD`, optionally `lr0=1e-4`. Compare test mAP50 and count MAE vs AdamW.

**Expected impact:** Unclear for counting; low-cost exploration before 100-ep matrix commits.

### Win 8 — Test-count-aware model selection (eval protocol)

**Issue:** Ultralytics saves `best.pt` on **val** fitness; aug that helps val can hurt test counts.

**Action:** After each smoke/full run: `eval.py` on test + `error_analysis.py` count MAE; keep a spreadsheet of val vs test delta. Do **not** pick production weights on val count alone.

**Expected impact:** Aligns model selection with manuscript metrics; surfaces aug that widens val≫test.

**Explicitly deprioritized (for now):** field-default `mosaic=1.0`, mixup>0, copy-paste (no masks), large `scale`/`degrees` (fixed camera), Albumentations pipeline (no hook yet).

---

## 5. Mapped 15-epoch smoke experiments

All use `mamba run -n harchoc python scripts/train.py` + manual `eval.py` on **test** + `error_analysis.py` for count MAE. Set `DATASET_ROOT` first. Budget caps: [`docs/training_budget.md`](../training_budget.md).

**Committed smoke configs (P1-AUG, dry-run safe):** S0–S14 — see [`aug_smoke_index.json`](../../configs/experiments/aug_smoke_index.json). S14 is **eval-only** on historical **`models/best2.pt`** (`max_det=300`); queue job `aug_smoke_S14` sets `eval_only: true` (no GPU train).

**`eval.max_det` (P0-1 **Done**):** Smoke JSON and bench base set `"eval": {"max_det": 3000}`; `train.py` forwards to post-train eval. **S14** alone uses `--max-det 300` as negative control vs S1@3000 ([`s14_maxdet_truncation.json`](../../reports/hsp/s14_maxdet_truncation.json)).

**Shared eval (S0–S13):**

```bash
mamba run -n harchoc python scripts/eval.py --weights runs/<name>/weights/best.pt \
  --imgsz 1280 --max-det 3000 \
  --export-gt-json reports/aug_smoke/<name>_gt.json \
  --export-preds-json reports/aug_smoke/<name>_preds.json \
  --out reports/aug_smoke/<name>_eval.json

mamba run -n harchoc python scripts/error_analysis.py \
  --gt-json reports/aug_smoke/<name>_gt.json \
  --preds-json reports/aug_smoke/<name>_preds.json \
  --out reports/aug_smoke/<name>_error.json
```

**S14 eval only** (historical HSP baseline; truncation control vs [`s14_maxdet_truncation.json`](../../reports/hsp/s14_maxdet_truncation.json)):

```bash
mamba run -n harchoc python scripts/eval.py --weights models/best2.pt \
  --imgsz 1280 --max-det 300 \
  --export-gt-json reports/aug_smoke/aug_smoke_eval300_gt.json \
  --export-preds-json reports/aug_smoke/aug_smoke_eval300_preds.json \
  --out reports/aug_smoke/aug_smoke_eval300_eval.json
```

Per-smoke aug YAMLs are committed under `configs/aug/` (e.g. [`robustness_smoke_close3.yaml`](../../configs/aug/robustness_smoke_close3.yaml), [`robustness_mosaic_off.yaml`](../../configs/aug/robustness_mosaic_off.yaml)). Aug YAML **overwrites** same keys from JSON train block.

| ID | Name | Train config | Key overrides vs baseline smoke | Hypothesis | Primary metric |
|----|------|--------------|--------------------------------|------------|----------------|
| S0 | `aug_smoke_baseline` | [`train_smoke_rank_15ep.json`](../../configs/experiments/train_smoke_rank_15ep.json) | [`robustness_minimal.yaml`](../../configs/aug/robustness_minimal.yaml) (`close_mosaic=15` → **3** @ 15 ep) | Production minimal baseline smoke | test count MAE |
| S1 | `aug_smoke_close3` | [`train_smoke_rank_15ep.json`](../../configs/experiments/train_smoke_rank_15ep.json) | [`robustness_smoke_close3.yaml`](../../configs/aug/robustness_smoke_close3.yaml) | Mosaic active ep 0–11, off ep 12–14 | test count MAE |
| S2 | `aug_smoke_mosaic0` | [`train_smoke_rank_15ep.json`](../../configs/experiments/train_smoke_rank_15ep.json) | [`robustness_mosaic_off.yaml`](../../configs/aug/robustness_mosaic_off.yaml) | No composition aug | test count MAE |
| S3 | `aug_smoke_photometric` | [`train_smoke_rank_15ep.json`](../../configs/experiments/train_smoke_rank_15ep.json) | [`robustness_photometric_only.yaml`](../../configs/aug/robustness_photometric_only.yaml) | Photometric only (`hsv_s=0.45`, `hsv_v=0.40`) | test count MAE |
| S4 | `aug_smoke_mosaic01` | [`train_smoke_rank_15ep.json`](../../configs/experiments/train_smoke_rank_15ep.json) | [`robustness_smoke_mosaic01.yaml`](../../configs/aug/robustness_smoke_mosaic01.yaml) | `mosaic=0.1`, `close_mosaic=3`, **`translate=0.10`** (≠ S1 @ 0.05) | test count MAE |
| S5 | `aug_smoke_mosaic03` | [`train_smoke_rank_15ep.json`](../../configs/experiments/train_smoke_rank_15ep.json) + [`robustness_smoke_mosaic03.yaml`](../../configs/aug/robustness_smoke_mosaic03.yaml) | `mosaic: 0.3, close_mosaic: 3` | Upper bound mosaic sweep | test count MAE |
| S6 | `aug_smoke_erasing0` | [`train_smoke_rank_15ep.json`](../../configs/experiments/train_smoke_rank_15ep.json) + [`robustness_photometric_erasing0.yaml`](../../configs/aug/robustness_photometric_erasing0.yaml) | S3 + `erasing: 0` | Is erasing hurting small seed recall? | test count MAE |
| S7 | `aug_smoke_erasing03` | [`train_smoke_rank_15ep.json`](../../configs/experiments/train_smoke_rank_15ep.json) + [`robustness_photometric_erasing03.yaml`](../../configs/aug/robustness_photometric_erasing03.yaml) | S3 + `erasing: 0.3` | Occlusion robustness vs FN rate | test count MAE |
| S8 | `aug_smoke_hsv_v045` | [`train_smoke_rank_15ep.json`](../../configs/experiments/train_smoke_rank_15ep.json) + [`robustness_photometric_hsv_v045.yaml`](../../configs/aug/robustness_photometric_hsv_v045.yaml) | S3 + `hsv_v: 0.45` | Session lighting DR | test count MAE |
| S9 | `aug_smoke_no_aug_yaml` | [`train_aug_s9_no_aug_yaml_smoke.json`](../../configs/experiments/train_aug_s9_no_aug_yaml_smoke.json) **without** `aug_config`; inline `_BASELINE_DEFAULTS` only | Legacy inline mosaic 0.1, no close_mosaic | Compare YAML merge vs code defaults | test count MAE |
| S10 | `aug_smoke_yolo11s` | [`train_aug_s10_yolo11s_smoke.json`](../../configs/experiments/train_aug_s10_yolo11s_smoke.json) (`train_smoke_rank_yolo11s_15ep` + close3 aug) | YOLO11s backbone | Architecture vs aug interaction | test mAP50 |
| S11 | `aug_smoke_musgd` | [`train_aug_s11_musgd_smoke.json`](../../configs/experiments/train_aug_s11_musgd_smoke.json) | S10 + `"optimizer": "MuSGD", "lr0": 0.0001` | MuSGD optimizer | test count MAE |
| S12 | `aug_smoke_amp_off` | [`train_aug_s12_amp_off_smoke.json`](../../configs/experiments/train_aug_s12_amp_off_smoke.json) | S1 + `"amp": false` | Numerical stability @ 1280 dense | test count MAE |
| S13 | `aug_smoke_patience5` | [`train_aug_s13_patience5_smoke.json`](../../configs/experiments/train_aug_s13_patience5_smoke.json) | S1 + `"patience": 5` only | Stress early-stop + close_mosaic interaction | **Same test MAE as S1 expected when `stop_epoch=15`** (observed 68.908…); parse `results.csv` for actual stop |
| S14 | `aug_smoke_eval300` | eval-only (no train) | **`models/best2.pt`**; `--max-det 300` | **Negative control** for P0-1 (HSP historical baseline) | count MAE delta vs best2@3000 |

### Example train commands

```bash
# S1 — close_mosaic=3 smoke
mamba run -n harchoc python scripts/train.py --name aug_smoke_close3 \
  --config configs/experiments/train_smoke_rank_15ep.json \
  --aug-config configs/aug/robustness_smoke_close3.yaml

# S2 — mosaic off
mamba run -n harchoc python scripts/train.py --name aug_smoke_mosaic0 \
  --config configs/experiments/train_smoke_rank_15ep.json \
  --aug-config configs/aug/robustness_mosaic_off.yaml

# Dry-run any smoke (CI-safe)
mamba run -n harchoc python scripts/train.py --dry-run --name aug_smoke_close3 \
  --config configs/experiments/train_smoke_rank_15ep.json \
  --aug-config configs/aug/robustness_smoke_close3.yaml
```

### Decision gate after smokes

1. If **S2 or S3** beats **S4/S5** on test count MAE → adopt mosaic=0 or photometric-only for 100-ep run.
2. If **S14 ≫ S1** on count MAE → confirms P0-1 cap matters; do not regress `eval.max_det` below 3000.
3. If val mAP ranks differ from test count MAE ranking → **freeze aug choice on test count**, not val.
4. Promote winner to 100-ep with `close_mosaic=15`, `patience=30`, full `eval.py` + threshold lock workflow.

---

## 6. 100-epoch follow-ups (post-smoke)

| Experiment | Config base | Overrides | When |
|------------|-------------|-----------|------|
| Full mosaic sweep | `train_yolov8m_baseline.json` | `mosaic ∈ {0, 0.1, 0.3}` via aug YAML | Smoke winner unclear |
| `close_mosaic` sweep | baseline | `{10, 15, 25}`, `patience=30` | Smoke shows mosaic helps |
| Patience ablation | `configs/ablation/early_stopping.yaml` pattern | `patience ∈ {10, 20, 50}` | Early stop skips mosaic tail |
| Matrix parity | [`train_smoke_rank_yolo11s_15ep.json`](../../configs/experiments/train_smoke_rank_yolo11s_15ep.json) (runtime zoo overlay) | winner aug YAML | Zoo comparison |

---

## 7. References

1. Ultralytics YOLO26 training recipe: https://docs.ultralytics.com/guides/yolo26-training-recipe/
2. Ultralytics YOLO data augmentation: https://docs.ultralytics.com/guides/yolo-data-augmentation/
3. Ultralytics MuSGD / Muon: https://docs.ultralytics.com/reference/optim/muon/
4. close_mosaic vs early stopping (#18013): https://github.com/ultralytics/ultralytics/issues/18013
5. copy-paste for detection (#18073): https://github.com/ultralytics/ultralytics/issues/18073
6. Internal: [`augmentation_robustness_literature.md`](augmentation_robustness_literature.md)
7. Internal: [`threshold_calibration_literature.md`](threshold_calibration_literature.md) (val≈0.97 vs test≈0.79)
8. YOLO26 arXiv (STAL, ProgLoss, MuSGD): https://arxiv.org/html/2509.25164v1
9. Small-object aug survey 2025: https://www.mdpi.com/2076-3417/15/22/11882
10. Yolo-pest close_mosaic: https://www.nature.com/articles/s41598-025-97825-3
11. LWCD-YOLO (mosaic off, dense seeds): https://doi.org/10.3390/agriculture15181968
12. Small-object aug framework ablation (SODA-A): https://doi.org/10.1007/s42044-025-00384-z
13. Scale-aware mosaic ablation (Electronics 2026): https://doi.org/10.3390/electronics15051075
14. Ultralytics close_mosaic default PR #3525: https://github.com/ultralytics/ultralytics/pull/3525

---

---

## Source verification (top external links)

| # | Source | Claim used in this doc | Verified 2026-05-29 |
|---|--------|------------------------|---------------------|
| 1 | [YOLO26 training recipe](https://docs.ultralytics.com/guides/yolo26-training-recipe/) | S/M/MuSGD: `mosaic≈0.99`, `close_mosaic=10`, `mixup` up to 0.427 (L/X), `scale` 0.9–0.95, `hsv_s`≈0.35, `translate`≈0.27, MuSGD + `lr0≈3.8e-4` | **Match** — per-size table on official doc |
| 2 | [YOLO data augmentation](https://docs.ultralytics.com/guides/yolo-data-augmentation/) | `close_mosaic` disables mosaic for last N **scheduled** epochs (`epochs - close_mosaic`); `close_mosaic=0` keeps mosaic on | **Match** — § Mosaic / `close_mosaic` |
| 3 | [MuSGD / Muon](https://docs.ultralytics.com/reference/optim/muon/) | MuSGD in YOLO26; `optimizer=auto` → MuSGD for long runs | **Match** — linked from recipe FAQ |
| 4 | [Ultralytics #18013](https://github.com/ultralytics/ultralytics/issues/18013) | Early stop at epoch 55 on 100-ep schedule skips epoch-85 `close_mosaic` tail; no built-in fix | **Match** — reporter scenario; maintainer: use callbacks; closed `not_planned` |
| 5 | [Ultralytics #18073](https://github.com/ultralytics/ultralytics/issues/18073) | Stock `copy_paste` needs segment/mask labels; xywh box-only detect labels cannot use it | **Match** — Ultralytics staff: segment labels required for copy-paste on detect |

---

*Validated 2026-05-30. Code-checked: `scripts/train.py`, `scripts/eval.py`, [`train_smoke_rank_15ep.json`](../../configs/experiments/train_smoke_rank_15ep.json), [`harchoc/aug_smoke_train.py`](../../harchoc/aug_smoke_train.py), `configs/aug/robustness_minimal.yaml`, `harchoc/train_config.py`, `harchoc/train_kwargs.py`. Aligns with [`backlog.md` model improvement stack](../../backlog.md#model-improvement-stack-test-count-mae) steps 1 + 4.*

**Val vs test mAP gap:** Peak training val mAP can exceed test ranking mAP even when split drift proxies look similar — see [`val_test_map_gap.md` §5](../manuscript/val_test_map_gap.md#5-manuscript-draft--val-map-vs-test--results--22) + [`split_drift_p0.json`](../../reports/hsp/split_drift_p0.json) (**MS-SPLIT-MAPNARR** Done; **MS-VAL-MAPDOWN** Done). **Mosaic-off precedent:** `lwcd_yolo2025` → **ARCH-MOSAIC0-AB** (smoke S2).

**Validated literature registry:** [`docs/manuscript/literature_validated.md`](../manuscript/literature_validated.md) · **Related Work (MS-LIT):** [`related_work_outline.md`](../manuscript/related_work_outline.md)
