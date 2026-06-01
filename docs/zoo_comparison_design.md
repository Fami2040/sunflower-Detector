# Zoo comparison design

How we choose **which** detectors enter the matrix and **which group** to train/eval when GPU budget is limited.

## Primary metric

**Test count MAE** at val-locked confidence (HSP protocol). Architecture zoo ranks backbones; aug smokes rank augmentation. Do not use val mAP alone to pick counting models.

## Production baseline (selection anchor)

| Field | Value |
|-------|--------|
| Weights | `models/best2.pt` (`HSP_DETECTION_WEIGHTS`) |
| Architecture | YOLOv8-class detector (2-class developed/aborted) |
| Production anchor | `models/best2.pt` from [origin `main`](https://github.com/Fami2040/sunflower-Detector) — [`ORIGIN_MAIN_AND_DATASET.md`](ORIGIN_MAIN_AND_DATASET.md) |
| Matrix retrain (v8m row) | Fresh `yolov8m.pt` @ 1280 on CVAT splits — compare to **best2**, not a copy of it |
| Headline test MAE | ~**61.3** @ val-locked conf ~0.15 ([`threshold_test_locked.json`](../reports/hsp/threshold_test_locked.json)) |

`zoo_core` includes only rows that **plausibly beat this anchor** on test count MAE after the shared 100-ep @ 1280 recipe—not every hub row we can train.

## Source registry (non-Ultralytics)

Canonical repos and COCO checkpoints: [`configs/external/detector_sources.v1.json`](../configs/external/detector_sources.v1.json).

| `source_id` | Stack | Upstream repo | Role in comparison |
|-------------|--------|---------------|-------------------|
| `rtdetrv2_l` | `rtdetrv2_pytorch` | [lyuwenyu/RT-DETR](https://github.com/lyuwenyu/RT-DETR) (`rtdetrv2_pytorch`) | RT-DETRv2 **without** DEIM matching |
| `dfine_l` | `dfine` | [Peterande/D-FINE](https://github.com/Peterande/D-FINE) | D-FINE **without** DEIM |
| `deim_rtdetrv2_l` | `deim` | [Intellindust-AI-Lab/DEIM](https://github.com/Intellindust-AI-Lab/DEIM) | DEIM recipe on RT-DETRv2-L |
| `deim_dfine_l` | `deim` | same | DEIM recipe on D-FINE-L |

DEIM re-implements RT-DETRv2 and D-FINE training; prefer **one checkout** (`Intellindust-AI-Lab/DEIM`) for both DEIM rows. Upstream rows isolate architecture vs matching/loss changes.

Ultralytics rows remain separate: `rtdetr-x.pt` (v1 hub large), `rtdetr-l_nq1024` (scratch query-cap ablation). Vanilla `rtdetr-l.pt` (300 queries) stays in **`sota_2026`** only.

Prep weights and upstream repos (`harchoc.bench_assets.build_weights_prep_report`, CLI [`scripts/check_weights_cache.py`](../scripts/check_weights_cache.py)):

```bash
mamba run -n harchoc pip install gdown   # once, for DEIM Drive checkpoints
mamba run -n harchoc python scripts/check_weights_cache.py --sync-repos-manifest
mamba run -n harchoc python scripts/check_weights_cache.py --download --strict --out reports/hsp/weights_cache.json
```

This updates [`data/weights/weights_manifest.json`](../data/weights/weights_manifest.json) with:

- Ultralytics `.pt` files under `data/weights/ultralytics/`
- External `.pth` under `data/weights/external/`
- Cloned repos under `external/` (clone specs derived from [`detector_sources.v1.json`](../configs/external/detector_sources.v1.json); optional generated mirror [`external_repos.v1.json`](../configs/external/external_repos.v1.json), git-ignored tree)

After clone, each repo is validated (train script + per-row `config_relpath` from `detector_sources.v1.json`). No manual `git clone` steps.

## Matrix groups (tags overlap — not additive)

There are **26 bench YAML files** total. Groups are **filters**, not disjoint buckets.

```text
                    ┌── sota_deim (4 external DETR)
                    │
  26 bench rows ──────┼── sota_2026 (~22 Ultralytics + NAS)
                    │
                    └── zoo_core (10) ⊂ beat-yolov8m candidates
                            │
                            └── zoo_scale (14) = scale variants NOT in zoo_core
```

| Group | What it selects | Count | GPU if you run *only this filter* |
|-------|-----------------|-------|-----------------------------------|
| **`zoo_core`** | Mid+ YOLO gens, counting-oriented DETR, external DETR stack | **10** | **10 × 100 ep** (full zoo; post–P0-5; **Ultralytics RT-DETR** train needs >8 GiB on this machine) |
| **`zoo_yolo_only`** | Four M-scale Ultralytics YOLO rows only | **4** | **4 × 100 ep** (**P0-5** on 8 GiB — [`gpu_queue_full.json`](../configs/experiments/archive/gpu_queue_full.json) `zoo_matrix_p0_5`) |
| **`sota_2026`** | Every hub-backed Ultralytics row + NAS | **~22** | ~22 × 100 ep |
| **`zoo_scale`** | n/s/l/x/b rows **excluding** the single “m” pick per YOLO family | **14** | 14 × 100 ep (optional) |
| **`sota_deim`** | `rtdetrv2_l`, `dfine_l`, `deim_*` | **4** | 4 × 100 ep (subset of `zoo_core`) |
| **`zoo_detr_stack`** | Same four external DETR rows as **`sota_deim`** | **4** | Filter alias for Tier D ablations |

**Run one filter at a time**, not 10+14+22:

Manifest: [`configs/zoo/matrix_rows.v1.json`](../configs/zoo/matrix_rows.v1.json) (`zoo_matrix_rows.v1`, **26** `rows[]`). Group tag counts (rows may carry multiple tags): `zoo_core` **10**, `zoo_scale` **14**, `sota_2026` **22**, `sota_deim` **4**, `zoo_detr_stack` **4** (last two are the same external DETR filter names).

Regenerate or validate bench YAML + `train_bench_*.json` (`harchoc.zoo_matrix_scaffold`):

```bash
python scripts/benchmark_matrix.py --scaffold-zoo --out reports/benchmarks/zoo_scaffold_report.json
python scripts/benchmark_matrix.py --validate-zoo   # exit non-zero on drift
```

Shared infer/epochs/patience/seed: [`configs/bench/_defaults.yaml`](../configs/bench/_defaults.yaml) (included via `include: _defaults.yaml`).

1. **On 8 GiB (P0-5):** `--group zoo_yolo_only` → **4** trains (`yolov8m`, `yolov10m`, `yolo11m`, `yolo26m`) — queue job `zoo_matrix_p0_5`.
2. **When queue/integration allows:** `--group zoo_core` → **10** trains (not 26). **Ultralytics RT-DETR** @ 1280 batch=1 OOMs on 8 GiB; external DETR may run (e.g. D-FINE ~6.7 GiB peak) but DEIM/rtdetrv2 need working imports and non-conflicting distributed ports.
3. **Only if needed:** `--group zoo_scale` on the **one** YOLO family that wins on test count MAE.
4. **Full** `sota_2026` only for a complete hub leaderboard — rarely needed if core + scale answer the science question.

Example: `yolov8m` is in `zoo_core`, `sota_2026`, and `yolov8_scales` — it is **one** row, not three.

Train the lean set:

```bash
python scripts/benchmark_matrix.py --group zoo_core --no-dry-run ...
```

List counts:

```bash
python scripts/benchmark_matrix.py --list-groups --out reports/hsp/zoo_groups.json
```

## `zoo_core` membership (10 rows)

Objective rule: **capacity ≥ production anchor** (YOLOv8m) or **architecture/decoder change aimed at dense-tray counting**, wired for matrix train @ 1280.

| `model_id` / bench stem | Why in core |
|-------------------------|-------------|
| `yolov8m` | Matrix retrain of production anchor; direct comparison to `best2.pt` |
| `yolov10m` | Newer YOLO gen, same m-scale |
| `yolo11m` | Newer YOLO gen, same m-scale |
| `yolo26m` | Newest Ultralytics YOLO gen @ m-scale |
| `rtdetr_l_nq1024` | Query-cap ablation (1024) for peak GT ~1015; replaces vanilla `rtdetr-l` in core |
| `rtdetr-x` | Larger Ultralytics DETR if DETR family competes |
| `rtdetrv2_l` | External v2 baseline (no DEIM) |
| `dfine_l` | External D-FINE baseline (no DEIM) |
| `deim_rtdetrv2_l` | DEIM on RT-DETRv2-L |
| `deim_dfine_l` | DEIM on D-FINE-L |

### `zoo_yolo_only` (4 rows, 8 GiB P0-5)

On **8 GiB** GPUs, Ultralytics RT-DETR @ `imgsz=1280` `batch=1` OOMs (see `train_batch_probe_rtdetr-l.json`). **P0-5** uses **`--group zoo_yolo_only`** (four M-scale YOLO rows only — no external DETR stack):

| Row | In `zoo_yolo_only` |
|-----|---------------------|
| `yolov8m`, `yolov10m`, `yolo11m`, `yolo26m` | Yes |
| External DETR (`rtdetrv2_l`, `dfine_l`, `deim_*`) | **No** in P0-5 — optional via `zoo_core` / `zoo_core_8gb` (VRAM often OK; integration/port scheduling is the usual blocker) |
| `rtdetr_l_nq1024`, `rtdetr_x` | **No** — Ultralytics RT-DETR train OOM @ 1280 batch=1 on 8 GiB |

Queue default on this machine: `gpu_queue_full.json` job `zoo_matrix_p0_5` uses `matrix_group: zoo_yolo_only` (~480 min).

### `zoo_core_8gb` (8 rows, optional 8 GiB superset)

Same four YOLO rows as **`zoo_yolo_only`**, plus external DETR four-pack when integration is ready — still **no** Ultralytics RT-DETR **train** on 8 GiB (vram probe OOM). External stacks: verify imports and exclusive GPU (avoid `EADDRINUSE` on port 29500).

**Excluded from core** (still in `sota_2026` / `zoo_scale` where applicable):

| Row | Why not core |
|-----|----------------|
| `rtdetr-l` (300 queries) | Structurally capped vs dense trays; `nq1024` is the counting-oriented Ultralytics DETR pick |
| `yolo_nas_s` | s-scale; below m-anchor capacity—unlikely to beat `yolov8m` on count MAE |
| All `n`/`s`/`l`/`x` YOLO scales | Deferred to `zoo_scale` after a family wins core |

## Logical comparison tiers

```text
Tier A — Production anchor (not always in matrix)
  yolov8m  best2.pt  @ minimal aug

Tier B — zoo_core YOLO lineage (one mid-scale per generation)
  yolov8m, yolov10m, yolo11m, yolo26m
  → answers: does latest YOLO beat our v8m on count MAE?

Tier C — zoo_core Ultralytics DETR
  rtdetr_l_nq1024, rtdetr-x
  → answers: query-cap + large v1 hub vs dense trays

Tier D — zoo_core transformer stack (external)
  rtdetrv2_l → dfine_l → deim_rtdetrv2_l / deim_dfine_l
  → answers: v2 vs D-FINE vs DEIM; is DEIM worth separate venv?

Tier E — zoo_scale (optional)
  Full n/s/m/l/x only for the family that wins Tier B at 15-ep smoke or 30-ep shortlist
```

## Pruning rules (avoid 25×100 ep)

1. **Never** run full scale ladders for every YOLO generation up front — O(~20) × 100 ep.
2. **On 8 GiB:** run **`zoo_yolo_only`** first (P0-5, 4 × 100 ep). **Full `zoo_core`** (10 × 100 ep) when GPU and integration allow.
3. After `zoo_core`, promote **one** YOLO family to `zoo_scale` if it beats `yolov8m` on test MAE by a pre-registered margin (e.g. ≥5% relative MAE reduction).
4. **DETR external rows** share high integration cost — run 15-ep smokes on `deim_dfine_l` and `rtdetrv2_l` before committing 100 ep on all four.
5. Vanilla **`rtdetr-l`** is **`sota_2026` only**; core uses **`rtdetr_l_nq1024`** for the query-cap hypothesis.
6. **`rtdetr-x`** stays in core as the large Ultralytics DETR candidate; drop from a repeat matrix only if Tier C+D fail vs `yolov8m`.

## What we deliberately omit from core

| Omitted | Why |
|---------|-----|
| Every `n/s/l/x` variant in core | Redundant until family winner known |
| `rtdetr-l` @ 300 queries | Superseded by `nq1024` for counting |
| `yolo_nas_s` | s-scale reference; not a plausible beat over `yolov8m` |
| DEIMv2 / DINOv3 teachers | Separate env, manuscript side track |
| RT-DETRv4 | Not wired; zero-shot distillation story ≠ counting fine-tune |
| YOLO26 MuSGD / recipe defaults | Aug parity uses `robustness_minimal`; optimizer smoke is separate |

## Wiring status

| Backend | Train in `benchmark_matrix.py` | Weights prep |
|---------|-------------------------------|--------------|
| `ultralytics` | Yes (`train.py`) | `check_weights_cache.py` → `harchoc.bench_assets` |
| `supergradients` | Yes | Train-time download |
| `external` | Yes (`harchoc/external_detector_train.py`) | `bench_assets` + `--download` → `external/` |

**External train:** runs after `check_weights_cache.py --download` has cloned repos. Optional env overrides: `HARCHOC_DEIM_REPO`, `HARCHOC_DFINE_REPO`, `HARCHOC_RTDETR_REPO`. Registry docs: [`configs/external/README.md`](../configs/external/README.md).

Exports sunflower splits to COCO under `runs/<run_name>/coco_export/`, writes a YAML overlay, runs upstream `train.py -t <cached_ckpt>`. Post-train HSP eval: `harchoc/external_detector_eval.py` (test export + `error_analysis.py` @ locked conf → `test_count_mae` in `matrix_train.json`).

## Aug dedup before zoo

15-ep aug smokes finished before **`zoo_core`** trains. Duplicate recipe classes (**S0≡S1≡S13**, **S3≡S6≡S7**) must not consume GPU ahead of matrix work — audit: [`aug_smoke_index.json`](../configs/experiments/aug_smoke_index.json) `equivalence_classes`. GPU tier order: [backlog § GPU execution tiers](../backlog.md#gpu-execution-tiers-post-aug_pending).
