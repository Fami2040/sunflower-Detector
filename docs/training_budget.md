# Training budget caps (`HARCHOC_MAX_*`)

## Mamba env (required for GPU / torch)

All **live train**, **eval**, **check_gpu**, matrix planning, and smokes use the project env — **not** base Python:

```bash
mamba run -n harchoc python scripts/<script>.py ...
# or: conda activate harchoc
```

Override env name: `HARCHOC_MAMBA_ENV` (default `harchoc`). `scripts/check_gpu.py` and `scripts/rtdetr_smoke.py` **re-exec** under mamba when the current interpreter has no torch.

CI unittest only: `HARCHOC_ALLOW_BASE_PYTHON=1` (no GPU).

**Convention:** every GPU command block below is shown as `mamba run -n harchoc python …`. Export `HARCHOC_MAX_*` (and `DATASET_ROOT`) in the same shell before those commands.

---

Export these environment variables **before** sweeps or live training so oversized configs fail fast instead of starting multi-day runs.

Implementation: [`harchoc/training_budget.py`](../harchoc/training_budget.py) (`enforce_budget`, defaults below).

## Variables

| Variable | Default | Caps |
|----------|---------|------|
| `HARCHOC_MAX_EPOCHS` | `500` | `epochs` in bench YAML / train JSON |
| `HARCHOC_MAX_IMGSZ` | `2048` | `infer.imgsz` in bench YAML; `imgsz` in live train |
| `HARCHOC_MAX_BATCH` | `16` | `batch` in committed `train_bench_*.json` and live train |

Values must be positive integers. Unset or empty env vars use the default.

**Smoke vs full runs:** Export tighter caps before short GPU probes and aug ablations; raise them for production training in the [model improvement stack](EXPERIMENTS.md#model-improvement-count-mae). Typical pattern: `HARCHOC_MAX_EPOCHS=15` for RT-DETR / aug smokes (**P0-4**, **P1-AUG**); `HARCHOC_MAX_EPOCHS=100` (or `120` for matrix headroom) for the full YOLO baseline and **P0-5** zoo sweep. Keep `HARCHOC_MAX_IMGSZ=2048` and tune `HARCHOC_MAX_BATCH` to VRAM (often `1` @ 1280 on 8 GiB). Violations exit before Ultralytics starts a multi-day run.

**Smoke epoch tiers** (`harchoc.train_config`): **micro** = `SMOKE_EPOCHS_MICRO` (3) for CI/unittest and 1-ep VRAM probes; **rank** = `SMOKE_EPOCHS_RANK` (15) for GPU aug smokes and RT-DETR rank-tier configs. Use `scale_close_mosaic_for_epochs(SMOKE_EPOCHS_MICRO)` → `1` in tests; rank smokes keep `close_mosaic=3` @ 15 ep.

## Where enforced

| Entrypoint | When |
|------------|------|
| `harchoc/bench_config.py` | Matrix dry-run / plan: `epochs`, `infer.imgsz`, committed `train_bench_*.json` `batch` |
| `scripts/benchmark_matrix.py` | Invokes bench validation above |
| `scripts/train.py` | Live training: `enforce_budget` on merged train kwargs before Ultralytics / SG |

Unit tests: `tests/test_benchmark_matrix.py` (`test_budget_caps_are_enforced_via_env`).

## Post-train eval (`train.py` → `eval.py`)

After Ultralytics training, `scripts/train.py` may chain a test-split eval. Policy is in [`harchoc/post_train_eval.py`](../harchoc/post_train_eval.py).

| Control | Effect |
|---------|--------|
| Train config `"eval": {"device": "cpu"}` | Default for `train_bench_base.json` / `train_yolov8m_baseline.json` (8 GiB-safe) |
| `"eval": {"skip": true}` | No chained eval (RT-DETR 15-ep smoke) |
| `HARCHOC_POST_TRAIN_EVAL_DEVICE` | Overrides config `eval.device` |
| `HARCHOC_EXPORT_DEVICE` | Fallback if post-train env unset |
| *(auto)* | If neither set and **&lt;2 GiB** CUDA free after train → `cpu`, else `cuda` |
| `--skip-eval` | CLI skip (same as `eval.skip`) |

Train releases the training model and calls `torch.cuda.empty_cache()` before eval. **Training stays on GPU** (`device: 0` in JSON); only the chained eval defaults to CPU on this repo’s bench recipes.

**100-epoch full runs (recommended on 8 GiB):**

```bash
export DATASET_ROOT="$(pwd)/data/raw/extracted/dataset"
# Committed configs already set eval.device=cpu; optional explicit override:
export HARCHOC_POST_TRAIN_EVAL_DEVICE=cpu
mamba run -n harchoc python scripts/train.py \
  --config configs/experiments/train_yolov8m_baseline.json \
  --name yolov8m_baseline_full
```

HSP manuscript metrics (threshold sweep, locked test, `dual_metric`) remain a **separate** `eval.py` / `experiment.py` chain — see stack step 3 in [backlog](../backlog.md#model-improvement-stack-test-count-mae).

## Export device (VRAM)

When `eval.py` exports GT/preds JSON at `imgsz=1280`, GPU VRAM may be insufficient. Set before export:

```bash
export HARCHOC_EXPORT_DEVICE=cpu   # used when --export-device is omitted
mamba run -n harchoc python scripts/eval.py ...   # example; same env as train
```

Matrix test eval forwards this env to `eval.py` when set.

### Test mAP on 8 GiB GPU (**SCI-MAP-CPU** / **R-SCI-1**)

Stack **step 3** support: after val-locked threshold + error analysis, regen test mAP and the detection row in `dual_metric.json` (**R-SCI-1**, **MS-SOTA**). Ranking mAP needs a separate eval pass **without** `--export-only` (export-only JSON leaves `mAP50: null`). On memory-limited GPUs use the wired **`map-cpu`** subcommand (defaults: test split, `imgsz=1280`, `max_det=3000`, `--device cpu`):

```bash
export DATASET_ROOT="$(pwd)/data/raw/extracted/dataset"
mamba run -n harchoc python scripts/experiment.py map-cpu
mamba run -n harchoc python scripts/experiment.py map-cpu --dry-run   # prints eval argv only

# Equivalent manual eval.py (avoid hand-assembly errors):
# mamba run -n harchoc python scripts/eval.py \
#   --weights models/best2.pt --split-file data/splits/test.txt \
#   --device cpu --imgsz 1280 --max-det 3000 \
#   --out reports/hsp/eval_test_map.json

mamba run -n harchoc python scripts/experiment.py dual-metric \
  --eval-val reports/hsp/eval_val.json \
  --eval-test reports/hsp/eval_test.json \
  --eval-test-map reports/hsp/eval_test_map.json \
  --sweep reports/hsp/threshold_val.json \
  --sweep-test reports/hsp/threshold_test_locked.json \
  --error-val reports/hsp/error_val.json \
  --error-test reports/hsp/error_test.json \
  --out reports/hsp/dual_metric.json
```

Optional agent strictness: `HARCHOC_STRICT_ML=1` (see [`scripts/strict_ml_smoke.py`](../scripts/strict_ml_smoke.py)).

## VRAM probes (1 epoch @ imgsz 1280)

Short GPU memory probes use non-canonical `configs/experiments/train_batch_probe_*.json` (`epochs: 1`, `imgsz: 1280`, `batch` from matching `train_bench_*.json`). On 8 GiB-class GPUs keep **`batch=1`** for `yolov8m` and `rtdetr-l`; skip chained eval (`eval.skip` or `--skip-eval`).

| Model | Bench batch | 1-ep probe config | Peak VRAM (this machine) |
|-------|-------------|-------------------|--------------------------|
| `yolov8m.pt` | 1 | `train_batch_probe_yolov8m.json` | **7416 MiB** @ batch=1 (V100S-8Q, 2026-05-29) |
| `rtdetr-l.pt` | 1 | `train_batch_probe_rtdetr-l.json` | *(pending)* |

```bash
export DATASET_ROOT="$(pwd)/data/raw/extracted/dataset"
export HARCHOC_MAX_EPOCHS=15
mamba run -n harchoc python scripts/train.py \
  --config configs/experiments/train_batch_probe_yolov8m.json \
  --name batch_probe_yolov8m --skip-eval --out-dir runs/batch_probe
# Log peak: nvidia-smi during run, or see reports/hsp/train_batch_probe.json
```

Aggregated results: [`reports/hsp/train_batch_probe.json`](../reports/hsp/train_batch_probe.json).

## Micro-batch (`nbs`)

Ultralytics uses nominal batch size `nbs=64` for loss scaling when training with `batch=1`. Document in train configs; no code change required.

## Examples

Typical multi-model sweep (tune caps to your GPU and schedule):

```bash
export DATASET_ROOT="$(pwd)/data/raw/extracted/dataset"
export HARCHOC_MAX_EPOCHS=120
export HARCHOC_MAX_IMGSZ=2048
export HARCHOC_MAX_BATCH=2   # e.g. 8 GiB GPUs @ imgsz 1280

mamba run -n harchoc python scripts/benchmark_matrix.py \
  --out reports/hsp/matrix_plan.json
```

Short GPU smoke (RT-DETR, aug ablation, etc.):

```bash
export HARCHOC_MAX_EPOCHS=15
export HARCHOC_MAX_IMGSZ=2048
# HARCHOC_MAX_BATCH optional
```

### RT-DETR 15-epoch smoke (`rtdetr-l`)

Config: [`configs/experiments/train_rtdetr_smoke_15ep.json`](../configs/experiments/train_rtdetr_smoke_15ep.json) (extends `train_bench_rtdetr-l.json` → `train_bench_base.json`: `epochs=15`, inherited `imgsz=1280`, `batch=1`).

```bash
export DATASET_ROOT="$(pwd)/data/raw/extracted/dataset"
export HARCHOC_MAX_EPOCHS=15
export HARCHOC_MAX_IMGSZ=2048

# Probe GPU + write reports/hsp/rtdetr_smoke_15ep.json (re-execs into mamba if needed)
mamba run -n harchoc python scripts/rtdetr_smoke.py

# Full 15-epoch train + update same JSON
mamba run -n harchoc python scripts/rtdetr_smoke.py --run-train
```

Equivalent manual train:

```bash
mamba run -n harchoc python scripts/check_gpu.py --json-out reports/hsp/gpu_check.json
mamba run -n harchoc python scripts/train.py --name rtdetr_smoke_15ep \
  --config configs/experiments/train_rtdetr_smoke_15ep.json
```

Status metadata: `reports/hsp/rtdetr_smoke_15ep.json` (`gpu_ok_pending_train` until `--run-train` completes).

## RT-DETR query cap (dense trays)

Sunflower frozen splits peak **1015** GT boxes/image (`harchoc/rtdetr_limits.py`, `SUNFLOWER_DOCUMENTED_PEAK_GT_BOXES_PER_IMAGE`). Ultralytics RT-DETR-L uses decoder **`num_queries=300`** by default; extra GT slots are not matched at train time.

**Repo policy (matrix + live train):**

1. Keep **`num_queries=300`** in committed `train_bench_rtdetr-l.json` until a GPU ablation proves a higher cap is worth VRAM/latency.
2. Set **`accept_rtdetr_query_truncation: true`** on RT-DETR train JSON after review (required when `num_queries` &lt; peak; without it, `train.py` exits unless overridden).
3. Optional **`documented_peak_gt_boxes_per_image`** (default 1015); override scan with **`HARCHOC_RTDETR_PEAK_GT_BOXES_PER_IMAGE`**.
4. **`HARCHOC_RTDETR_QUERY_CAP=warn`** — log only, do not exit (CI / exploratory runs). Unset = strict (fail if `accept_rtdetr_query_truncation` is false).

Enforced in `scripts/train.py`, `harchoc/bench_config.py` / `harchoc/rtdetr_limits.py` (matrix dry-run), and `scripts/validate_splits.py --check-rtdetr-query-cap`.

**Eval/infer `max_det` (P1-RTDETR-MAXDET):** For RT-DETR matrix rows, bench YAML `infer.max_det` and train JSON `eval.max_det` must equal **`num_queries`** (300 for stock `rtdetr-l.pt`, 1024 for `rtdetr-l_nq1024.yaml`). YOLO rows keep `max_det=3000`. Validated by `validate_rtdetr_infer_max_det()` at bench load.

Raising `num_queries` above 300 for dense trays requires a **custom Ultralytics model YAML** (`RTDETRDecoder` third argument) or post-load `model.model[-1].num_queries`; the flat train field documents intent but does not change a frozen `.pt` head. See Ultralytics RT-DETR docs (`num_queries`, `max_det` at eval).

Higher batch ceiling when VRAM allows:

```bash
export HARCHOC_MAX_EPOCHS=120
export HARCHOC_MAX_IMGSZ=2048
export HARCHOC_MAX_BATCH=16

mamba run -n harchoc python scripts/train.py --config configs/experiments/train_bench_yolov8m.json ...
```

On violation, scripts exit with a message like `epochs=200 exceeds HARCHOC_MAX_EPOCHS=120`.

## Research and ops

- **Consolidated research & ops:** [`docs/RESEARCH_AND_OPS.md`](RESEARCH_AND_OPS.md) (layered roadmap, P0–P2, env, refactors).
- **2026 training tech scans (interim):**
  - [`docs/research/training_tech_scan_2026_detectors.md`](research/training_tech_scan_2026_detectors.md)
  - [`docs/research/training_tech_scan_2026_augmentation.md`](research/training_tech_scan_2026_augmentation.md)
  - [`docs/research/training_tech_scan_2026_eval_calibration.md`](research/training_tech_scan_2026_eval_calibration.md)
- Disk planning, run naming `{model}_e{N}_s{seed}`, feature-branch training: [`docs/EXPERIMENTS.md`](EXPERIMENTS.md#ops-disk-naming-branches-budgets).

---

*Validated 2026-05-29.*
