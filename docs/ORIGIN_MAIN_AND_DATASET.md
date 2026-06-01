# Origin `main` weights & canonical dataset

**Purpose:** Single source of truth for where `models/best2.pt` came from, how it relates to HARCHOC experiments, and which dataset this repo actually uses.

---

## Public upstream repo (small)

| Item | Value |
|------|--------|
| **Repository** | [Fami2040/sunflower-Detector](https://github.com/Fami2040/sunflower-Detector) (`main`) |
| **Scope** | Telegram bot, `run_infer_once.py`, `models/best2.pt`, `classifier.pt`, minimal training notebook/script |
| **Not in upstream** | HARCHOC `scripts/`, frozen `data/splits/`, aug smokes, zoo matrix, manuscript preflight |

Reviewer criticisms of the **submitted manuscript** are captured in [`reports/reviewer2.md`](../reports/reviewer2.md) (read-only docx snapshot: [`reports/plants-4336582.docx`](../reports/plants-4336582.docx)). Human submission text: [`reports/manuscript/`](../reports/manuscript/). Agent gap map: [`manuscript/reviewer_comments_backlog_gap.md`](manuscript/reviewer_comments_backlog_gap.md).

---

## Manuscript integration (fork vs upstream `main`)

We **do not** re-train `best2.pt` in this fork for the headline result. We **do** re-measure it under HSP on frozen CVAT splits, then run comparative experiments (aug grid, detector zoo, tray eval) that ask whether any recipe beats the anchor on the **same** test count MAE gate.

| Layer | Upstream `main` | This fork |
|-------|-----------------|-----------|
| Weights | Trained/shipped `best2.pt` | Same file; SHA in `baseline_models_manifest.json` |
| Data / splits | Informal | CVAT manifest + `data/splits/*.txt` |
| Metrics | Mixed in draft | Val-lock conf → test MAE (+ test mAP separate) |
| Comparisons | Not systematic | Aug S0–S14, zoo, domain trays |

Module map and pipeline diagram notes: [`manuscript/fork_integration.md`](manuscript/fork_integration.md).  
Experiment–literature graph: [`manuscript/study_lineage.md`](manuscript/study_lineage.md).

---

## `models/best2.pt` — production detector

**Provenance:** Checkpoint shipped from **upstream `main`**, not produced by `scripts/train.py` in this fork.

Upstream training (representative; see public repo history) used Ultralytics YOLOv8m:

- `epochs=100`, `imgsz=1280`, `batch=1`, `optimizer=AdamW`
- `mosaic=0.1`, `hsv_h/s/v`, `translate`, `scale`, `lr0=0.0002`, `max_det=3000`, `patience=50`
- Run name historically: `sunflower_seed_detection_v1`

Machine-readable snapshot of those args (for diff vs HARCHOC YAML): [`configs/origin/public_yolov8_train_reference.json`](../configs/origin/public_yolov8_train_reference.json).

**SHA256 (this workspace):** `f2cff94070296a2b4733d988c9edf50df9450aa8bff664190417133c654cdc79` — [`reports/hsp/baseline_models_manifest.json`](../reports/hsp/baseline_models_manifest.json).

## Study logic (what we actually did)

The submitted article had a **legacy `best2.pt`** checkpoint (previous maintainer, [origin repo](https://github.com/Fami2040/sunflower-Detector)) with **inconsistent or unclear metrics** in the draft. This fork **re-measures everything on one frozen protocol** so reviewer responses are apples-to-apples.

| Step | What | Same split / protocol? |
|------|------|-------------------------|
| **A. Anchor `best2`** | Export preds on CVAT corpus; val sweep → lock conf → **test count MAE** | **`data/splits/test.txt`**, HSP @ 1280, locked conf **≈0.15** |
| **B. Aug program (S0–S14 + 100 ep)** | Train **new** YOLOv8m weights with **literature-guided** aug tactics ([`docs/research/`](../docs/research/)); rank by **test count MAE** vs anchor | **Same** test split + **same** locked conf as step A |
| **C. SOTA zoo (reviewer)** | Train YOLOv10/11/26m (etc.) @ 1280 with shared bench recipe | **Same** HSP test eval |
| **Decision** | Nothing in B or C beat anchor **61.3** → **keep `best2.pt`** for deploy + manuscript headline | — |

**Aug was not an afterthought:** it was the **structured comparison** “can literature tactics or a full retrain beat the legacy checkpoint on **our** held-out test?” — parallel to the **SOTA zoo** work for reviewer #3.

### Artifact roles

| Work | Role |
|------|------|
| **`best2.pt`** | **Anchor weights** (unchanged file); **61.3** MAE is HARCHOC’s fair re-measure on CVAT test, not the old article table |
| **S0–S14** | Literature aug ablations (15 ep each); **same test metric** as anchor |
| **`aug_confirm_winner_100ep`** | Full-budget (100 ep) train with production-like minimal aug on CVAT splits → **64.1** MAE — still loses to anchor |
| **Zoo rows** | Other architectures / retrains — e.g. v8m zoo **111.9**, 11m **119.6**, 26m **95.3** — all ≫ **61.3** |
| [`robustness_minimal.yaml`](../configs/aug/robustness_minimal.yaml) | Documents ~what legacy train used + what we used for confirm/zoo bench |

**Headline number:** test count MAE **61.3** (95% CI **51.3–71.3**, *n*=109) — [`dual_metric.json`](../reports/hsp/dual_metric.json).

**Reproduce anchor on same split as aug/zoo:**

```bash
export DATASET_ROOT="$(pwd)/data/raw/extracted/dataset"
mamba run -n harchoc python scripts/eval.py --weights models/best2.pt --max-det 3000 \
  --split-file data/splits/test.txt --export-only --out reports/hsp/eval_test.json
# then threshold_sweep on val, error_analysis on test @ locked conf → dual_metric
```

---

## Dataset — CVAT 1093 (canonical here)

**This fork does not treat Kaggle `linaaabrahim/dataset1` as the published dataset.**

The **canonical** corpus for HARCHOC is the **CVAT-annotated sunflower seed dataset (1093 images)**:

| Field | Value |
|-------|--------|
| **Share URL** | https://izba-memes.ru/share/y9xGFqCW |
| **Manifest** | [`data/manifest.json`](../data/manifest.json) (`sunflower-cvat-1093`) |
| **Local root** | `data/raw/extracted/dataset` (via `DATASET_ROOT` / manifest) |
| **Modeling splits** | Tracked [`data/splits/{train,val,test}.txt`](../data/splits/) (875 / 109 / 109) |

The public README links the same CVAT share (`y9xGFqCW`). Some upstream drafts reference **Kaggle** for convenience during early training; that hub dataset is **not** separately published as the manuscript corpus. **Reproducibility in this repo** = manifest + split SHA256 + HSP eval on `best2.pt`.

See [`data/README.md`](../data/README.md).

---

## Reviewer theme ↔ reproducibility (comment #5)

| Reviewer concern | HARCHOC response (repo) |
|------------------|-------------------------|
| Training params too brief | Origin train reference JSON + `robustness_minimal.yaml` + [`EXPERIMENTS.md`](EXPERIMENTS.md) |
| Threshold basis unclear | Val `threshold_sweep.py --select min_count_mae` → lock → test ([`threshold_val.json`](../reports/hsp/threshold_val.json)) |
| Hard to reproduce | [`manuscript_repro_bundle.json`](../configs/experiments/manuscript_repro_bundle.json), `experiment.py repro` / `manuscript-preflight` |
| Only YOLOv8 tested | **P0-5** `zoo_yolo_only` (YOLO M-scale rows; Ultralytics RT-DETR train OOM @ 1280 on 8 GiB) — [gap map §19](manuscript/reviewer_comments_backlog_gap.md#19-manuscript-draft--model-selection--sota-zoo-methods--results) |

---

## Quick links

- Weights & deploy: [`HSP_BASELINE_MODELS.md`](HSP_BASELINE_MODELS.md)
- Aug comparison program: [`reports/aug_smoke/README.md`](../reports/aug_smoke/README.md)
- Reviewer gap map: [`manuscript/reviewer_comments_backlog_gap.md`](manuscript/reviewer_comments_backlog_gap.md)
