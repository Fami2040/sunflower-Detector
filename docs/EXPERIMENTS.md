# Experiments & scripts

Entrypoints for training, evaluation, threshold calibration, and the model-zoo benchmark matrix. They share dataset resolution, config layout (`configs/`), and artifact locations (`reports/`, `figures/`, `runs/`).

## GPU environment (required for torch work)

All Ultralytics / PyTorch scripts call `require_conda_env()` and expect the **`harchoc`** mamba env (override with `HARCHOC_MAMBA_ENV`):

```bash
mamba run -n harchoc python scripts/train.py --help
```

- Create or refresh the env: `python scripts/bootstrap_env.py --env harchoc --create`
- External DETR zoo rows: add `--with-external-detr` (see [`configs/external/README.md`](../configs/external/README.md))
- YOLO-NAS: add `--with-super-gradients`
- CUDA sanity: `mamba run -n harchoc python scripts/check_gpu.py`
- Some helpers (`check_gpu.py`, `rtdetr_smoke.py`) re-exec into mamba via `harchoc/ml_env.py` when torch is missing on base Python.
- **CI / dry imports:** GitHub Actions sets `PYTHONPATH=.` and `HARCHOC_ALLOW_BASE_PYTHON=1` on the unit-test job so `unittest` can import modules without a GPU env.

Run repo scripts from the repository root as `python scripts/<name>.py` (each imports `scripts._path`); prefer the `mamba run` prefix above for any live train/eval/export.

## IDE / Pyright (basedpyright)

Type checking expects the **`harchoc`** interpreter (torch, ultralytics, optional `super_gradients` live there — not base Python).

- Root [`pyrightconfig.json`](../pyrightconfig.json): `extraPaths` for vendored detectors; `ignore` = `external/**`; `typeCheckingMode` = `basic`. Do **not** put `venvPath` with `~` in that file — Pyright treats `~` as a literal path under the repo. Point the env via `.vscode/settings.json` (`python.defaultInterpreterPath`, `basedpyright.analysis.venvPath` / `venv`). JSON merge helpers: `harchoc.config_coerce` (`pick_int`, `optional_str`, …).
- Copy [`.vscode/settings.json.example`](../.vscode/settings.json.example) → `.vscode/settings.json`; set `python.defaultInterpreterPath` to your `harchoc` env; `basedpyright.analysis.typeCheckingMode` = `basic`. Per-rule severities use flat `report*` keys in `pyrightconfig.json`, not nested `diagnosticSeverityOverrides`.

Reload the window after changing interpreter or Pyright settings so `reportMissingImports` and stale severity-8 diagnostics clear.

## Agent runtime (Cursor agents)

Permanent rules: `.cursor/rules/gpu-env-and-dataset.mdc`, `.cursor/rules/ml-strict-runtime.mdc` (`alwaysApply: true`).

| | CI | GPU dev machine (agents) |
|--|----|---------------------------|
| Run prefix | `PYTHONPATH=. HARCHOC_ALLOW_BASE_PYTHON=1 python -m unittest ...` | `mamba run -n harchoc python ...` |
| Torch/CUDA | Not required for import tests | Available in `harchoc`; verify with `mamba run -n harchoc python scripts/check_gpu.py` |
| Missing ML deps | Deferred imports OK | `mamba run -n harchoc pip install <pkg>` for well-known deps; torch via `bootstrap_env.py` |
| HSP / `reports/hsp/` artifacts | N/A in CI | **Strict:** fail loudly — no placeholder JSON or import-only "success" |

Never infer "no GPU" from bare `python` on base. CI lightweight import is for GitHub Actions only; on this machine, prefer explicit errors over silent empty results when ML outputs are requested.

## Consolidated guide

**Start here:** [`docs/RESEARCH_AND_OPS.md`](RESEARCH_AND_OPS.md) — layered ops + research roadmap (P0–P2), env, refactors, links to all scans below.

**Related docs**

| Doc | Role |
|-----|------|
| [backlog.md](../backlog.md) | Task status — [§ Work queue](../backlog.md#work-queue-p0--p2) |
| [HSP_BASELINE_MODELS.md](HSP_BASELINE_MODELS.md) | Weights, deploy SAHI vs manuscript eval |
| [reports/README.md](../reports/README.md) | **Scientific artifact tree** (HSP → aug → manuscript → reviewer2; anti-sprawl) |
| [reports/hsp/README.md](../reports/hsp/README.md) | Canonical HSP JSON paths (cite for paper numbers) |
| [reports/hsp/p0_summary.md](../reports/hsp/p0_summary.md) | One-page headline metrics (local) |
| [reports/manuscript/README.md](../reports/manuscript/README.md) | Preflight, tables, docx exports |

## Research scans (2026)

Literature-backed plans referenced by backlog and bench recipes. **Artifact paths:** [`docs/research/README.md`](research/README.md) → [`reports/README.md`](../reports/README.md).

| Scan | Doc |
|------|-----|
| Detectors / RT-DETR / zoo | [`docs/research/training_tech_scan_2026_detectors.md`](research/training_tech_scan_2026_detectors.md) |
| Augmentation / S0–S14 smokes | [`docs/research/training_tech_scan_2026_augmentation.md`](research/training_tech_scan_2026_augmentation.md) |
| Eval / thresholds / dual-metric | [`docs/research/training_tech_scan_2026_eval_calibration.md`](research/training_tech_scan_2026_eval_calibration.md) |

Supporting literature notes: [`docs/research/augmentation_robustness_literature.md`](research/augmentation_robustness_literature.md), [`docs/research/threshold_calibration_literature.md`](research/threshold_calibration_literature.md).

## Dataset resolution (required convention)

**Environment variables** (when CLI dataset flags are unset):

1. `DATASET_ROOT` — dataset root directory.
2. `YOLO_DATA_YAML` — path to `data.yaml` when it is not at dataset root.
3. `DATASET_NAME` — selects an entry in `data/manifest.json`.

**CLI flags** (`--dataset-root`, `--yolo-data-yaml`, `--dataset-name`, `--manifest`) override env. Shared helper: `harchoc/datasets.py` (`resolve_dataset`).

### Classes and on-disk layout

Sunflower seed detection uses **two classes** only:

| id | name |
|----|------|
| 0 | **developed** |
| 1 | **aborted** |

Labels are Ultralytics YOLO format (`class_id cx cy w h`, normalized). Each image under `images/` has a matching `labels/` file with the same stem. Dense heads often have hundreds of boxes per image.

Full tree, split policy, and install steps: [`data/README.md`](../data/README.md). Constants: `harchoc/sunflower_dataset.py`.

Modeling splits live in repo-tracked `data/splits/{train,val,test}.txt`. Ultralytics early stopping uses **val**; manuscript metrics use **test** via `scripts/eval.py` (default split file).

## Entry points

| Script | Role |
|--------|------|
| `scripts/train.py` | Ultralytics training + optional post-train test eval (`--config`, `--aug-config`) |
| `scripts/eval.py` | Test-split mAP + optional GT/preds export (`--export-*`, `--max-det`, `--imgsz`) |
| `scripts/benchmark_matrix.py` | Model-zoo plan / train / test eval (`configs/bench/*.yaml`) |
| `scripts/experiment.py` | Unified CLI: `splits`, `describe`, `eval`, `benchmark`, `train`, `hpo`, **`cv-eval`**, `map-cpu`, `dual-metric`, `repro`, **`reviewer2-repro`**, **`figures-repro`**, **`tables-repro`**, **`manuscript-docx-repro`**, **`manuscript-preflight`**, `aug-compare`, `backlog-narrative`, `gradcam`, `deploy-parity` |
| `scripts/threshold_sweep.py` | Confidence sweep (default `--out reports/thresholds/sweep.json`; HSP → `reports/hsp/threshold_*.json`) |
| `scripts/error_analysis.py` | Counting / TIDE-style summary (HSP → `reports/hsp/error_*.json`; see [`reports/error_analysis/README.md`](../reports/error_analysis/README.md)) |
| `scripts/describe_split.py` | Split stats → `reports/split_stats.json` |
| `scripts/split_drift.py` | Train/val/test drift report → `reports/hsp/split_drift_p0.json` (P0); rich proxies → `--extended` → `reports/hsp/split_drift_rich.json` |
| `scripts/make_figures.py` | Figures + `reports/figures/run.json` |
| `scripts/check_weights_cache.py` | Pre-flight Ultralytics weights under `data/weights/ultralytics/` |
| `scripts/check_gpu.py` | PyTorch/CUDA probe (`check` default), `sanity`, `smoke-ultralytics` subcommands (legacy `gpu_sanity.py` / `gpu_smoke_ultralytics.py` **removed**) |
| `scripts/finetune.py` | Transfer fine-tune → `train.main` (`configs/transfer/finetune_minimal.yaml`, `configs/experiments/finetune_tray.json`; default `--dry-run`) |
| `scripts/strict_ml_smoke.py` | Agent debug-all: `HARCHOC_STRICT_ML=1 mamba run -n harchoc python scripts/strict_ml_smoke.py` → `reports/hsp/strict_ml_smoke.json` |
| `scripts/pre_train_gate.py` | Pre-train preflight: manifest + env reminders + unittest (`--quick`); `--full` + `HARCHOC_STRICT_ML=1` runs strict ML smoke (no reports unless `--json-out`) |
| `scripts/rtdetr_smoke.py` | RT-DETR GPU probe + optional 15-epoch smoke train |

**Split drift (rich proxies, CPU):**

```bash
mamba run -n harchoc python scripts/split_drift.py --with-ks --extended --out reports/hsp/split_drift_rich.json
```

**Pre-train gate** (before long GPU trains):

```bash
python scripts/pre_train_gate.py --quick
HARCHOC_STRICT_ML=1 mamba run -n harchoc python scripts/pre_train_gate.py --full
```

## Model improvement (count MAE)

Ordered workflow to improve **test count MAE** at val-locked conf — not val mAP alone. Stack definition: [backlog § Model improvement stack](../backlog.md#model-improvement-stack-test-count-mae). Layered roadmap: [`docs/RESEARCH_AND_OPS.md`](RESEARCH_AND_OPS.md).

| Step | Action | Backlog |
|------|--------|---------|
| 1 | Train/export parity (`max_det=3000`, frozen splits, HSP protocol) | P0-1 **Done** — [`s14_maxdet_truncation.json`](../reports/hsp/s14_maxdet_truncation.json) |
| 2 | Full YOLO @ 1280 (`train_yolov8m_baseline.json`, 100 ep, `robustness_minimal`) | CLI below |
| 3 | Val sweep → lock conf → test error analysis + `dual_metric` | **P1-FP-BUDGET**, **MS-FUZZY-BOUND** — [§ Threshold sweep](#threshold-sweep--error-analysis-real-preds) |
| 4 | Aug ablations (test MAE primary) | **P1-AUG** S0–S14 — [aug scan §5](research/training_tech_scan_2026_augmentation.md), [aug smoke index](../configs/experiments/aug_smoke_index.json) |

### P1-AUG smoke index (S0–S14)

Canonical registry: [`configs/experiments/aug_smoke_index.json`](../configs/experiments/aug_smoke_index.json). All smokes: 15 epochs, YOLOv8m @ 1280 (except S10 YOLO11s), test **count MAE** primary. **Runtime train:** [`train_smoke_rank_15ep.json`](../configs/experiments/train_smoke_rank_15ep.json) + index `aug_config` for S0–S8; committed train JSON for S9–S13 only. Aug YAMLs, status, and summaries live in the index — do not duplicate rows here.

Dry-run any pending smoke (CI-safe):

```bash
mamba run -n harchoc python scripts/train.py --dry-run --name aug_smoke_close3 \
  --config configs/experiments/train_smoke_rank_15ep.json \
  --aug-config configs/aug/robustness_smoke_close3.yaml
```

Post-train test eval: [aug scan §5 shared eval](research/training_tech_scan_2026_augmentation.md#5-mapped-15-epoch-smoke-experiments) → `reports/aug_smoke/<name>_error.json`. Automated via [`harchoc/aug_smoke_runner.py`](../harchoc/aug_smoke_runner.py) and the GPU queue below.

### GPU sequential queue

**Canonical ops:** [`./scripts/run_gpu_queue.sh`](../scripts/run_gpu_queue.sh) (`dry-run` | `run` | `resume`). Direct / CI: [`scripts/run_gpu_queue.py`](../scripts/run_gpu_queue.py) (`--manifest`, `--job`). `experiment.py gpu-queue` is a deprecated alias (config-file workflows only).

**Default manifest:** [`configs/experiments/gpu_queue_aug_pending.json`](../configs/experiments/gpu_queue_aug_pending.json) (preflight + index-expanded pending smokes). **Full backlog** (RT-DETR probes, close25 sweep, **`zoo_core` 10×100 ep**, CV folds): [`gpu_queue_full.json`](../configs/experiments/gpu_queue_full.json) via `GPU_QUEUE_MANIFEST=configs/experiments/gpu_queue_full.json`. Mosaic `aug_sweep_15_*` jobs are **removed** from manifests (covered by smokes **S2/S4/S5**).

```bash
export DATASET_ROOT="$(pwd)/data/raw/extracted/dataset"
export HARCHOC_EXPORT_DEVICE=cpu

# 1) Validate all stage commands (CPU-safe)
./scripts/run_gpu_queue.sh dry-run

# 2) Unattended live run (default: gpu_queue_aug_pending.json)
./scripts/run_gpu_queue.sh run

# Full backlog:
# GPU_QUEUE_MANIFEST=configs/experiments/gpu_queue_full.json ./scripts/run_gpu_queue.sh run

# 3) Monitor / resume
tail -f reports/gpu_queue/nohup.log
python -m json.tool reports/gpu_queue/run_state.json
./scripts/run_gpu_queue.sh resume
```

Per-job logs: `reports/gpu_queue/logs/{job_id}/{stage_id}.log`. Completed aug smokes update [`aug_smoke_index.json`](../configs/experiments/aug_smoke_index.json) and refresh [`reports/aug_smoke/leaderboard.md`](../reports/aug_smoke/leaderboard.md) (per-smoke `s{N}_summary.json` are written but ranking uses the leaderboard). On failure the queue stops; fix the issue and `--resume`.

**Live queue snapshot** (pending jobs, dedup skips, run state): [backlog § Runbook (GPU)](../backlog.md#runbook-gpu).

### GPU queue manifest map

One sequential GPU — pick **one** manifest per run via `GPU_QUEUE_MANIFEST` (default: aug pending, now **complete**). Job order is manifest order; recipe dedup skips trains when a complete smoke owns the same [`effective_train_recipe_fingerprint`](../harchoc/train_config.py). Rankings / equivalence / preds dedup audit: [`aug_smoke_index.json`](../configs/experiments/aug_smoke_index.json) (`equivalence_classes`), [`leaderboard.md`](../reports/aug_smoke/leaderboard.md).

| Manifest | Tier | Purpose | Notable jobs |
|----------|------|---------|--------------|
| [`gpu_queue_aug_pending.json`](../configs/experiments/gpu_queue_aug_pending.json) | **Done** | Index-expanded S0–S14 smokes | `aug_smoke_from_index: true` — finished 2026-05-30 15:08 UTC |
| [`gpu_queue_aug_confirm.json`](../configs/experiments/gpu_queue_aug_confirm.json) | **1** | 100-ep aug winner confirm | `aug_confirm_winner_100ep` (**P1-AUG-100EP-WINNER**) — not in full manifest |
| [`gpu_queue_full.json`](../configs/experiments/gpu_queue_full.json) | **1–2**, **P0-5** | Post-smoke backlog | RT-DETR refresh (**P1-RTDETR-COUNT-REFRESH**), amp/sg HSP (**P1-AMP-HSP-EVAL**, **P1-SG-HSP-EVAL**), close10/close25 sweeps, **`zoo_matrix_p0_5`** (`zoo_core` 10×100 ep), CV folds |

```bash
# Tier 1 — 100-ep winner (separate manifest)
GPU_QUEUE_MANIFEST=configs/experiments/gpu_queue_aug_confirm.json ./scripts/run_gpu_queue.sh dry-run

# Tier 1–2 + P0-5 — full backlog
GPU_QUEUE_MANIFEST=configs/experiments/gpu_queue_full.json ./scripts/run_gpu_queue.sh dry-run
```

**P0-5 job:** `zoo_matrix_p0_5` in full manifest — `kind: zoo_matrix_train`, `matrix_group: zoo_core`, out `reports/hsp/matrix_train.json` (~2000 min). Run after Tier 1 RT-DETR + Tier 2 eval/sweeps unless manually reordered. Zoo design: [zoo_comparison_design.md](zoo_comparison_design.md).

**`aug_smoke_from_index`** (on `gpu_queue_manifest.v1`): when `true`, `load_gpu_queue_manifest()` replaces inline `aug_smoke` jobs with one queue job per [`aug_smoke_index.json`](../configs/experiments/aug_smoke_index.json) row in `status: gpu_pending` (optional `"aug_smoke_index"` path override). Train/`--aug-config` for those jobs come from the index entry, not duplicated in the manifest. Parity: `harchoc.aug_smoke_runner.aug_smoke_index_queue_parity_errors()`. Summaries: `finalize_smoke_job()` in [`aug_smoke_runner.py`](../harchoc/aug_smoke_runner.py).

Single job dry-run: `mamba run -n harchoc python scripts/run_gpu_queue.py --manifest configs/experiments/gpu_queue_aug_pending.json --dry-run --job aug_smoke_S3`.

### P1-AUG sweeps (15-ep, post-smoke)

Constants in `harchoc.train_config`: `MOSAIC_SWEEP_VALUES = (0.0, 0.1, 0.3)`, `CLOSE_MOSAIC_SWEEP_100EP = (10, 15, 25)`, `scale_close_mosaic_for_epochs(15) → 3`.

Template: [`train_aug_mosaic_sweep_smoke_15ep.json`](../configs/experiments/train_aug_mosaic_sweep_smoke_15ep.json) (`epochs=15`, `patience=12`). Pass `--aug-config` per arm:

| Sweep | `aug_config` | Backlog |
|-------|--------------|---------|
| mosaic=0 | `configs/aug/robustness_mosaic_off.yaml` | **ARCH-MOSAIC0-AB** / **P1-AUG-MOSAIC** |
| mosaic=0.1 | `configs/aug/robustness_smoke_mosaic01.yaml` | **P1-AUG-MOSAIC** |
| close_mosaic=15 | `configs/aug/robustness_smoke_close15.yaml` | **P1-AUG-CLOSE** |
| mosaic=0.3 | `configs/aug/robustness_smoke_mosaic03.yaml` | **P1-AUG-MOSAIC** |
| close_mosaic=10 | `configs/aug/robustness_smoke_close10.yaml` (+ [`train_aug_close10_sweep_smoke_15ep.json`](../configs/experiments/train_aug_close10_sweep_smoke_15ep.json)) | **P1-AUG-CLOSE** |
| close_mosaic=25 | `configs/aug/robustness_smoke_close25.yaml` (+ [`train_aug_close25_sweep_smoke_15ep.json`](../configs/experiments/train_aug_close25_sweep_smoke_15ep.json)) | **P1-AUG-CLOSE** |

Dry-run a 15-ep mosaic-off arm (CI-safe):

```bash
mamba run -n harchoc python scripts/train.py --dry-run --name aug_sweep_mosaic0_15ep \
  --config configs/experiments/train_aug_mosaic_sweep_smoke_15ep.json \
  --aug-config configs/aug/robustness_mosaic_off.yaml
```

Optional 100-ep winner confirm (**P1-AUG-100EP-WINNER**): production [`robustness_minimal.yaml`](../configs/aug/robustness_minimal.yaml) via [`train_aug_winner_100ep.json`](../configs/experiments/train_aug_winner_100ep.json) and [`gpu_queue_aug_confirm.json`](../configs/experiments/gpu_queue_aug_confirm.json) (`aug_confirm_winner_100ep` → `reports/aug_smoke/aug_confirm_winner_100ep_summary.json`). Runbook: [backlog § P1-AUG-100EP-WINNER](../backlog.md#p1-aug-100ep-winner-optional-manifest). General 100-ep sweep template: [`train_aug_mosaic_sweep_template.json`](../configs/experiments/train_aug_mosaic_sweep_template.json).

| 5 | Model zoo matrix (`zoo_core` 10×100 ep) | **P0-4** → **P0-5** — [§ Model zoo](#model-zoo-benchmark-matrix) |
| 6+ | Tray finetune, domain eval, RT-DETR query cap | **P1-FINETUNE-LOOP**, **P1-DOMAIN-EVAL**, **P1-RTDETR-Q** |

Set `DATASET_ROOT` and `HARCHOC_MAX_*` caps per [`docs/training_budget.md`](training_budget.md) (15-ep smokes vs 100-ep full runs).

**2 — Full baseline train**

```bash
export DATASET_ROOT=/path/to/dataset
export HARCHOC_MAX_EPOCHS=100
export HARCHOC_MAX_IMGSZ=2048
export HARCHOC_MAX_BATCH=1

mamba run -n harchoc python scripts/train.py \
  --config configs/experiments/train_yolov8m_baseline.json \
  --name yolov8m_baseline_e100
```

**3 — HSP eval / sweep chain** (after exporting preds from new weights; full detail below)

```bash
mamba run -n harchoc python scripts/eval.py --split-file data/splits/val.txt --export-only \
  --weights runs/yolov8m_baseline_e100/weights/best.pt --export-device cpu \
  --export-gt-json reports/hsp/gt_val.json --export-preds-json reports/hsp/preds_val.json \
  --export-conf 0.001 --export-iou 0.3 --out reports/hsp/eval_val.json

mamba run -n harchoc python scripts/eval.py --max-det 3000 \
  --weights runs/yolov8m_baseline_e100/weights/best.pt \
  --export-gt-json reports/hsp/gt_test.json --export-preds-json reports/hsp/preds_test.json \
  --out reports/hsp/eval_test.json

mamba run -n harchoc python scripts/threshold_sweep.py --config configs/experiments/threshold_sweep_val.json \
  --select min_count_mae
mamba run -n harchoc python scripts/threshold_sweep.py --config configs/experiments/threshold_sweep_test_locked.json
mamba run -n harchoc python scripts/error_analysis.py --config configs/experiments/error_analysis_test.json

mamba run -n harchoc python scripts/experiment.py dual-metric \
  --eval-val reports/hsp/eval_val.json \
  --eval-test reports/hsp/eval_test.json \
  --sweep reports/hsp/threshold_val.json \
  --sweep-test reports/hsp/threshold_test_locked.json \
  --error-val reports/hsp/error_val.json \
  --error-test reports/hsp/error_test.json \
  --out reports/hsp/dual_metric.json
```

**Test mAP on CPU** (**SCI-MAP-CPU** / **R-SCI-1**): after export-only val/test preds, run full test mAP without GPU OOM risk, then merge into `dual_metric`:

```bash
export DATASET_ROOT=/path/to/dataset
mamba run -n harchoc python scripts/experiment.py map-cpu
mamba run -n harchoc python scripts/experiment.py map-cpu --dry-run   # prints eval argv only

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

Use `--select min_count_mae` on the val sweep for count-first operating-point selection (**P1-FP-BUDGET**; see [§ Threshold sweep](#threshold-sweep--error-analysis-real-preds)).

**4 — Aug smokes** — 15-ep S0–S14 grid; compare **test** count MAE: [aug scan §5](research/training_tech_scan_2026_augmentation.md). Example: `train_smoke_rank_15ep.json` + aug YAML from [aug smoke index](../configs/experiments/aug_smoke_index.json) under [§ Augmentation](#augmentation-trainpy---aug-config).

**5 — Matrix zoo** (after **P0-4** RT-DETR smoke completes)

```bash
export HARCHOC_MAX_EPOCHS=15
mamba run -n harchoc python scripts/rtdetr_smoke.py --run-train

export HARCHOC_MAX_EPOCHS=100
mamba run -n harchoc python scripts/check_weights_cache.py --download --strict --out reports/hsp/weights_cache.json
mamba run -n harchoc python scripts/benchmark_matrix.py --no-dry-run \
  --runs-dir runs/hsp_zoo --train-out reports/hsp/matrix_train.json \
  --out reports/hsp/matrix_plan.json
```

## Configs

| Path | Purpose |
|------|---------|
| `configs/bench/` | Model-zoo matrix YAML (8 models: YOLOv8 scales, YOLO10/11, RT-DETR, YOLO-NAS) |
| `configs/experiments/` | Canonical `experiments.v1` specs + flat `train_*.json` / `train_bench_*.json` |
| `configs/aug/` | Augmentation recipes (e.g. `robustness_minimal.yaml` for counting-first training) |
| `configs/ablation/`, `configs/transfer/` | Referenced by path from train or analysis workflows |

See `configs/experiments/README.md` for the three config styles (`experiments.v1`, flat train JSON, bench YAML) and merge rules.

## Unified `scripts/experiment.py`

Thin wrapper around script entrypoints. Subcommands:

- `splits` → `scripts/make_splits.py`
- `describe` → `scripts/describe_split.py`
- `eval` → `scripts/eval.py`
- `benchmark` → `scripts/benchmark_matrix.py`
- `train` → `scripts/train.py`
- `hpo` → plans budget-capped search configs (execution via generated train configs)
- `cv-eval` → `scripts/cv_eval.py` (**preferred** entry for k-fold lists, `--write-fold-splits`, `--fold-metrics` aggregation; direct `scripts/cv_eval.py` is equivalent)
- `figures-repro` → all manuscript figures with journal style + `reports/figures/manifest.json` (`figures_repro_manifest.v1`; delegates to `make_figures.py`)
- `gradcam` → **canonical** Grad-CAM FP panel: `scripts/make_figures.py --figure fig_gradcam_panel` (`harchoc/gradcam_panel.py`; routing rationale in [`docs/manuscript/gradcam_routing.md`](manuscript/gradcam_routing.md))
- `dual-metric` → merges eval + threshold + error-analysis JSON (`dual_metric_report.v1`)

Accepts legacy CLI flags and/or `--config` (inline JSON or path). If both are provided, **CLI flags win**. Use top-level `--dry-run` to force dry-run on the selected subcommand.

### Examples (legacy args)

For **live HSP / manuscript metrics**, use paths under [`reports/hsp/`](../reports/hsp/README.md) (see [`reports/README.md`](../reports/README.md)). Generic `--out` paths below are valid for dry-run smoke only.

```bash
mamba run -n harchoc python scripts/experiment.py --dry-run splits
mamba run -n harchoc python scripts/experiment.py --dry-run describe --out reports/split_stats.json
mamba run -n harchoc python scripts/experiment.py --dry-run eval --out reports/hsp/eval_dry_run.json
mamba run -n harchoc python scripts/experiment.py --dry-run benchmark --out reports/hsp/matrix_plan.json
mamba run -n harchoc python scripts/experiment.py cv-eval --dry-run
mamba run -n harchoc python scripts/experiment.py --dry-run \
  --config configs/experiments/cv_eval_dry.json cv-eval
```

**Cross-validation:** use `experiment.py cv-eval` (not `scripts/cv_eval.py` directly). Dry-run writes `cv_eval_run.v1` scaffold JSON; real runs need `DATASET_ROOT` and optional `--write-fold-splits` / `--fold-metrics`. Canonical dry-run spec: [`configs/experiments/cv_eval_dry.json`](../configs/experiments/cv_eval_dry.json) (`experiments.v1`, `run.kind: cv_eval`).

**Figures reproduction (journal style)** — prefer `experiment.py figures-repro` for the full manuscript set. Writes `reports/figures/run.json` (render log) and `reports/figures/manifest.json` (per-file SHA256, DPI, pixel size). Bundle: [`configs/experiments/figures_repro.json`](../configs/experiments/figures_repro.json).

```bash
# Plan only (no matplotlib / GPU)
PYTHONPATH=. HARCHOC_ALLOW_BASE_PYTHON=1 python scripts/experiment.py \
  --config configs/experiments/figures_repro.json figures-repro --dry-run

# CPU panels + drift/PR (omit error_report to skip FP mosaics)
python scripts/make_figures.py --out-dir reports/figures --meta-out reports/figures/run.json \
  --split-drift-report reports/hsp/split_drift_p0.json \
  --threshold-csv reports/hsp/threshold_val.csv --figure all

# Full set (needs error_test_report.json + optional weights for Grad-CAM overlays)
mamba run -n harchoc python scripts/experiment.py \
  --config configs/experiments/figures_repro.json figures-repro
mamba run -n harchoc python scripts/experiment.py figures-repro \
  --weights models/best2.pt --error-report reports/hsp/error_test_report.json
```

**Grad-CAM panel** — prefer `experiment.py gradcam` over calling `make_figures.py` directly. Requires prior `error_analysis.py --export-fp-crops` on the test split (`reports/hsp/error_test_report.json` + crop files). `--dry-run` prints the delegated `make_figures.py` argv only (no torch/GPU). Overlays need `--weights`; without weights the mosaic renders crop thumbnails only (`status: partial`). See [`docs/manuscript/gradcam_routing.md`](manuscript/gradcam_routing.md).

```bash
mamba run -n harchoc python scripts/experiment.py gradcam --weights models/best2.pt
mamba run -n harchoc python scripts/experiment.py gradcam --dry-run
mamba run -n harchoc python scripts/experiment.py gradcam \
  --error-report reports/hsp/error_test_report.json --panel-size 12
```

### Examples (`--config` JSON)

```bash
mamba run -n harchoc python scripts/experiment.py --dry-run \
  --config '{"dataset":{"default_dataset_name":"sunflower-cvat-2500"},"eval":{"out":"reports/hsp/eval_dry_run.json"}}' \
  eval

mamba run -n harchoc python scripts/experiment.py --dry-run --config configs/experiment.json benchmark
```

## Test eval: `max_det` (faster smoke / matrix passes)

Training defaults use a high `max_det` (see `scripts/train.py` baseline, typically 3000). For **test-split** evaluation, NMS cost scales with `max_det`.

- **CLI:** `scripts/eval.py --max-det 300` (optional; omit for Ultralytics defaults).
- **Train config JSON:** `"eval": {"max_det": 3000, "device": "cpu"}` (bench base); post-train eval forwards device to `eval.py` ([`training_budget.md`](training_budget.md) § Post-train eval). RT-DETR smoke sets `"eval": {"skip": true}`.
- **Bench YAML:** `infer.max_det` caps matrix val/eval and is copied into generated train configs when the matrix trains.

YOLO bench configs use `infer.max_det: 3000` @ 1280 (train/eval parity). **RT-DETR rows** align eval/infer `max_det` with decoder **`num_queries`** (`300` for `rtdetr_l_default.yaml`, `1024` for `rtdetr_l_nq1024.yaml` / `train_bench_rtdetr-l_nq1024.json`) — enforced in `harchoc/rtdetr_limits.py` at bench load. Negative control (YOLO truncation): [`reports/hsp/s14_maxdet_truncation.json`](../reports/hsp/s14_maxdet_truncation.json) ([aug scan](research/training_tech_scan_2026_augmentation.md) S14).

## Ops: disk, naming, branches, budgets

Before multi-model sweeps:

- **Disk:** tens of GB per model (checkpoints, `runs/`, `reports/`). Both are git-ignored by default.
- **Run naming:** `{model}_e{N}_s{seed}` (e.g. `yolov8m_e100_s0`). `benchmark_matrix.py` derives this from bench `model:` / `model_id:`, `epochs`, and `seed`. `HARCHOC_BENCH_USE_LEGACY_NAME=1` uses each YAML `name:` instead.
- **Branches:** full-budget training on a feature branch, not `main`.
- **Budget caps:** `HARCHOC_MAX_EPOCHS`, `HARCHOC_MAX_IMGSZ`, `HARCHOC_MAX_BATCH` — [`docs/training_budget.md`](training_budget.md).

## Threshold sweep + error analysis (real preds)

Full val→lock→test workflow: [`data/examples/README.md`](../data/examples/README.md). Protocol detail: [eval scan](research/training_tech_scan_2026_eval_calibration.md).

**Export** (low conf preserves PR tail; document NMS IoU):

```bash
export DATASET_ROOT=/path/to/dataset

# Val preds for tuning (export-only; --export-device cpu if GPU OOM @ imgsz 1280)
mamba run -n harchoc python scripts/eval.py --split-file data/splits/val.txt --export-only \
  --weights runs/yolov8m_medium_smoke/weights/best.pt \
  --export-device cpu \
  --export-gt-json reports/hsp/gt_val.json \
  --export-preds-json reports/hsp/preds_val.json \
  --export-conf 0.001 --export-iou 0.3 \
  --out reports/hsp/eval_val.json

# Test preds + mAP (default split: data/splits/test.txt; use max-det 3000 for counting parity)
mamba run -n harchoc python scripts/eval.py --weights models/best2.pt --max-det 3000 \
  --export-gt-json reports/hsp/gt_test.json \
  --export-preds-json reports/hsp/preds_test.json \
  --out reports/hsp/eval_test.json
```

**HSP baseline** (`models/best2.pt`, export-only val): see [`configs/experiments/eval_hsp_baseline.json`](../configs/experiments/eval_hsp_baseline.json) and [HSP_BASELINE_MODELS.md](HSP_BASELINE_MODELS.md).

**Sweep val; report test at locked conf:**

```bash
# Count-first val selection (P1-FP-BUDGET): --select min_count_mae
mamba run -n harchoc python scripts/threshold_sweep.py --config configs/experiments/threshold_sweep_val.json \
  --select min_count_mae

# Or set "select": "min_count_mae" in threshold_sweep_val.json run block
mamba run -n harchoc python scripts/threshold_sweep.py --config configs/experiments/threshold_sweep_val.json

mamba run -n harchoc python scripts/threshold_sweep.py --config configs/experiments/threshold_sweep_test_locked.json
mamba run -n harchoc python scripts/error_analysis.py --config configs/experiments/error_analysis_test.json
```

**Operating-point selection:** `--select min_count_mae` picks the val confidence that minimizes count MAE (count-first; **P1-FP-BUDGET**, ~194–217 FP/img budget at locked conf). Default in committed `threshold_sweep_val.json` (`"select": "min_count_mae"`). Alternatives: `best_f1` or `constraints` (`--min-recall`, `--min-precision`, `--max-fp-per-image`). Test split uses `--locked-conf-from` only — no re-selection on test.

Locked confidence: `harchoc/threshold_lock.py` (`--fixed-conf`, `--locked-conf-from` on `threshold_sweep.py` and `error_analysis.py`). Prefer `--locked-conf-from reports/hsp/threshold_val.json` (or `threshold_test_locked.json` — both resolve to val-locked conf ~0.15). `experiments.v1` configs: `threshold_sweep_val.json`, `threshold_sweep_test_locked.json`, `error_analysis_val.json`, `error_analysis_test.json`.

### TIDE bucket ΔAP + FP crop manifest (P1-TIDE)

After val/test GT/preds exports exist under `reports/hsp/`, run error analysis on both splits. Count-share proxy ΔAP lands in `tide_bucket_summary*.json`; taxonomy + optional FP crops in `error_*_report.json` (Layer B review manifest for **P2-FIG** / **MS-FP-LOC-NARR**). Manuscript narrative (**MS-FP-LOC-NARR** Done, repo draft): [`localization_dominates_classification`](../../reports/hsp/tide_bucket_summary.json) in test summary + [reviewer gap §11](manuscript/reviewer_comments_backlog_gap.md#11-manuscript-draft--fp-localization-vs-classification-methods--results) + [p0_summary § FP localization](../../reports/hsp/p0_summary.md#fp-localization-vs-classification-reviewer-142145).

```bash
export DATASET_ROOT=/path/to/dataset

# Val (tuning split; writes tide_bucket_summary_val.json)
mamba run -n harchoc python scripts/error_analysis.py --config configs/experiments/error_analysis_val.json

# Test (manuscript primary; writes tide_bucket_summary.json + FP crops)
mamba run -n harchoc python scripts/error_analysis.py --config configs/experiments/error_analysis_test.json
```

**Outputs:**

| Artifact | Path |
|----------|------|
| Val summary / report | `reports/hsp/error_val.json`, `error_val_report.json` |
| Test summary / report | `reports/hsp/error_test.json`, `error_test_report.json` |
| TIDE bucket ΔAP (count-share proxy) | `reports/hsp/tide_bucket_summary_val.json`, `tide_bucket_summary.json` |
| FP crop manifest + PNGs | `error_test_report.json` → `fp_crops`; files under `reports/error_analysis/fp_crops/` |
| Error taxonomy figure (CPU) | `reports/figures/fig_error_taxonomy.png` via `make_figures.py --figure fig_error_taxonomy` |

Optional official `tidecv` cross-check (GPU when installed): add `"tidecv": true` to the experiment config run block, or pass `--tidecv` — writes `*_tidecv_compare.json` sidecar (**P1-TIDECV** Done).

**Manuscript table merge:**

```bash
mamba run -n harchoc python scripts/experiment.py dual-metric \
  --eval-val reports/hsp/eval_val.json \
  --eval-test reports/hsp/eval_test.json \
  --sweep reports/hsp/threshold_val.json \
  --sweep-test reports/hsp/threshold_test_locked.json \
  --error-val reports/hsp/error_val.json \
  --error-test reports/hsp/error_test.json \
  --out reports/hsp/dual_metric.json
```

### Manuscript reproducibility bundle

Canonical path index: [`configs/experiments/manuscript_repro_bundle.json`](../configs/experiments/manuscript_repro_bundle.json) (`manuscript_repro_bundle.v1`). Lists train config for `best2.pt`, experiment configs, artifact outputs, and frozen `data/splits/*.txt` SHA256 (same shape as `collect_run_metadata(..., include_repo_splits=True)`).

**One-command P0 regen** (split drift → val/test exports → threshold lock → error analysis → dual-metric):

```bash
export DATASET_ROOT=/path/to/extracted/dataset
mamba run -n harchoc python scripts/experiment.py repro
```

Preview commands without running ML:

```bash
mamba run -n harchoc python scripts/experiment.py repro --dry-run
```

Optional test mAP + detection row in `dual_metric.json` (CPU via `map-cpu`; GPU path may OOM on 8 GiB):

```bash
mamba run -n harchoc python scripts/experiment.py map-cpu
mamba run -n harchoc python scripts/experiment.py dual-metric \
  --eval-test-map reports/hsp/eval_test_map.json \
  ...  # same paths as default dual-metric above

# Or append to repro chain (runs map-cpu + dual-metric regen):
mamba run -n harchoc python scripts/experiment.py repro --include-test-map
```

Headline numbers: [`reports/hsp/p0_summary.md`](../reports/hsp/p0_summary.md). Backlog: **MS-REPRO** (Done).

### Post-zoo reviewer-2 pack (before Word paste)

After P0 / zoo matrix train and HSP exports exist under `reports/hsp/`, regenerate reviewer-2 JSON in one chain:

```bash
mamba run -n harchoc python scripts/experiment.py reviewer2-repro
```

Preview (CI-safe; counting/confusion steps auto `--dry-run` if test GT/preds exports are missing):

```bash
PYTHONPATH=. HARCHOC_ALLOW_BASE_PYTHON=1 python scripts/experiment.py reviewer2-repro --dry-run
```

Same stage via manuscript repro: `experiment.py repro --stage post-zoo`. **`repro --stage full`** runs the HSP chain then **manuscript-preflight** (figures, tables, aug compare, backlog narrative, reviewer2 — not reviewer2 alone). Bundle: [`configs/experiments/manuscript_repro_bundle.json`](../configs/experiments/manuscript_repro_bundle.json); reviewer2 subset: [`reviewer2_repro.json`](../configs/experiments/reviewer2_repro.json). Shared runner: `harchoc/repro_chain.py`. Index: [`reports/reviewer2_index.md`](../reports/reviewer2_index.md).

### Publication preflight (before Word paste)

One ordered CPU chain after HSP exports (and optionally post-zoo reviewer-2): **reviewer2-repro** (skipped with warning if test GT/preds exports are missing) → **figures-repro** → **tables-repro** → **manuscript-docx-repro** → **aug-compare** → **backlog-narrative**. Writes step status to [`reports/manuscript/preflight_manifest.json`](../reports/manuscript/preflight_manifest.json). Implementation: `harchoc/manuscript_preflight.py` (`_cfg_section` / `_mapping` for nested bundle dicts).

```bash
mamba run -n harchoc python scripts/experiment.py manuscript-preflight
# alias:
mamba run -n harchoc python scripts/experiment.py repro --stage preflight
```

Preview (CI-safe):

```bash
PYTHONPATH=. HARCHOC_ALLOW_BASE_PYTHON=1 python scripts/experiment.py manuscript-preflight --dry-run
```

Config block: `manuscript_preflight` in [`manuscript_repro_bundle.json`](../configs/experiments/manuscript_repro_bundle.json) (`steps`: reviewer2 → figures → tables → docx → aug → narrative). Subcommands (also runnable standalone): `reviewer2-repro`, `figures-repro`, `tables-repro`, `manuscript-docx-repro`, `aug-compare`, `backlog-narrative`. Docx outputs: `reports/manuscript/docx/catalog.json`; tables: `reports/manuscript/tables/`.

CI-light path: `threshold_sweep.py --light`, `error_analysis.py --light` (tracked `data/examples/{gt,preds}.json`).

### Manuscript and reviewer traceability

| Artifact | Path |
|----------|------|
| Reviewer comment → backlog gap | [`docs/manuscript/reviewer_comments_backlog_gap.md`](manuscript/reviewer_comments_backlog_gap.md) |
| Val vs test mAP narrative | [`docs/manuscript/val_test_map_gap.md`](manuscript/val_test_map_gap.md) |
| Validated literature (JSON + index) | [`docs/manuscript/literature_validated.json`](manuscript/literature_validated.json), [`literature_validated.md`](manuscript/literature_validated.md) |
| Architecture recommendations | [`docs/manuscript/architecture_recommendations.md`](manuscript/architecture_recommendations.md) |
| Strict ML smoke (agents) | `HARCHOC_STRICT_ML=1 mamba run -n harchoc python scripts/strict_ml_smoke.py` |

Regenerate `dual_metric.json` after full test mAP eval so `metric_roles` / `split_role_label` label val as early-stop only (**SCI-MAP-CPU**, **R-SCI-1**).

### Asymmetric seed eval policy

Sunflower heads have **more developed than aborted** seeds (~55% / ~45% of boxes). This is biological prevalence, not class balancing. Manuscript metrics still use the frozen **test** split only (`data/splits/test.txt`, 109 images); val is for threshold tuning only.

Canonical policy: [`configs/eval/asymmetric_seed_policy.json`](../configs/eval/asymmetric_seed_policy.json) (`asymmetric_seed_policy.v1`). Class counts and split drift acceptance values are sourced from [`reports/hsp/split_drift_p0.json`](../reports/hsp/split_drift_p0.json). Loader + schema guard: `harchoc/asymmetric_seed_policy.py`.

| Split | Developed fraction | Aborted fraction | n images |
|-------|-------------------:|-----------------:|---------:|
| train | 0.551 | 0.449 | 875 |
| val | 0.527 | 0.473 | 109 |
| test | 0.554 | 0.446 | 109 |

**Reporting:** primary metric = **total count MAE** at val-locked conf; report per-class counts and `cls_confusion` in `error_analysis` JSON. Manuscript sweep uses one **global** conf for both classes (deploy may tune per-class conf separately — see [HSP_BASELINE_MODELS](HSP_BASELINE_MODELS.md)).

### Detection confusion matrix (developed / aborted / background)

3×3 matrix at the val-locked operating point (`detection_confusion_matrix.v1`).

| Path | Needs GPU | Notes |
|------|-----------|--------|
| `eval.py --confusion-matrix-only` (streaming) | **Yes** — Ultralytics `predict` | One model load; `--confusion-matrix-splits test,train` writes `{prefix}_{role}_confusion.json` |
| `eval.py --confusion-from-exports` | **No** | Reads existing `--export-gt-json` / `--export-preds-json`; match IoU from `--locked-conf-from` |
| `error_analysis.py --confusion-matrix-out` | **No** | Same matcher as §11 error report; with `--locked-conf-from`, confusion uses counting match IoU (0.3) |
| `experiment.py reviewer2-confusion` | **No** | CPU audit: IoU 0.5 vs stored error report + IoU 0.3 vs `best2_test_confusion.json` |

**GPU streaming** (no preds JSON on disk):

```bash
mamba run -n harchoc python scripts/eval.py \
  --weights runs/hsp_zoo/yolo26m_e100_s0/weights/best.pt \
  --confusion-matrix-only \
  --confusion-matrix-splits test,train \
  --confusion-matrix-out reports/hsp/yolo26m_e100_s0 \
  --locked-conf-from reports/hsp/threshold_val.json \
  --imgsz 1280 --export-max-det 3000 \
  --out reports/hsp/yolo26m_e100_s0_confusion_run.json
```

**CPU from frozen exports** (best2 / HSP parity, no dataset mount required):

```bash
PYTHONPATH=. HARCHOC_ALLOW_BASE_PYTHON=1 python scripts/eval.py \
  --confusion-matrix-only --confusion-from-exports \
  --export-gt-json reports/hsp/gt_test.json \
  --export-preds-json reports/hsp/preds_test.json \
  --confusion-matrix-out reports/hsp/best2_test_confusion.json \
  --locked-conf-from reports/hsp/threshold_val.json \
  --weights models/best2.pt \
  --out reports/hsp/best2_confusion_from_exports_run.json
```

```bash
PYTHONPATH=. HARCHOC_ALLOW_BASE_PYTHON=1 python scripts/experiment.py reviewer2-confusion
```

Library API: `harchoc/detection_confusion.py` (`confusion_matrix_streaming`, `confusion_matrix_multi_split`, `confusion_matrix_from_exports`).

Optional supplementary list for high-aborted or ambiguous images: `data/eval_sets/asymmetric.txt` (not required for primary test metrics; see [eval calibration scan §2.6](research/training_tech_scan_2026_eval_calibration.md)).

Backlog: **P2-ASYM-SEED** Done; manuscript sentence → **MS-ASYM-NARR**.

## Augmentation (`train.py --aug-config`)

Conservative counting-first recipe: `configs/aug/robustness_minimal.yaml` (`mixup=0`, low `mosaic`, `close_mosaic=15`). Merged into Ultralytics train kwargs and recorded in run `config.json`.

```bash
mamba run -n harchoc python scripts/train.py --dry-run --name yolov8m_aug_s1_close3_smoke \
  --config configs/experiments/train_smoke_rank_15ep.json \
  --aug-config configs/aug/robustness_smoke_close3.yaml

mamba run -n harchoc python scripts/train.py --name yolov8m_aug_s1_close3_smoke \
  --config configs/experiments/train_smoke_rank_15ep.json \
  --aug-config configs/aug/robustness_smoke_close3.yaml
```

Compare **test** count MAE (error analysis on exported preds), not val mAP alone. Plan: [aug scan](research/training_tech_scan_2026_augmentation.md).

## Model zoo benchmark matrix

### Plan only (default, CI-safe)

Dry-run parses bench YAML, resolves weights paths, validates budget caps — no ML imports required for the plan path:

```bash
export DATASET_ROOT=/path/to/dataset
mamba run -n harchoc python scripts/benchmark_matrix.py --out reports/benchmarks/matrix.json
```

Filter by group: `--group yolov8_scales` (repeatable). List groups: `--list-groups`.

### Weights resolution (no downloads in matrix)

`scripts/benchmark_matrix.py` **never downloads weights**. Ultralytics identifiers map to `data/weights/ultralytics/<identifier>` (override `WEIGHTS_CACHE_DIR`). SuperGradients uses backend-specific acquisition (not triggered by matrix).

Pre-cache:

```bash
mamba run -n harchoc python scripts/check_weights_cache.py --strict --out reports/hsp/weights_cache.json
mamba run -n harchoc python scripts/check_weights_cache.py --download --out reports/hsp/weights_cache.json
```

`--download` fetches missing Ultralytics weights and updates `data/weights/weights_manifest.json`. Use only during prep.

Bench validation (stdlib): required `model` / `model_id` per backend; budget caps from [`docs/training_budget.md`](training_budget.md).

### Training + test eval

When cached weights exist, `--no-dry-run` invokes `scripts/train.py` per ultralytics bench config, then **`scripts/eval.py` on the test split** unless `--no-eval`. Manuscript / zoo aggregates land in **`reports/hsp/matrix_train.json`** (not `reports/benchmarks/matrix_train.json`).

```bash
export DATASET_ROOT=/path/to/dataset

mamba run -n harchoc python scripts/benchmark_matrix.py --no-dry-run \
  --runs-dir runs/hsp_zoo \
  --train-out reports/hsp/matrix_train.json \
  --out reports/hsp/matrix_plan.json
```

- Missing cache → run `skipped` with `weights_not_cached`.
- Train recipes: `configs/experiments/train_bench_<stem>.json` (from bench `model:` stem).
- **SuperGradients** (`yolo_nas_s`): `harchoc/supergradients_train.py` / `supergradients_eval.py`; install via `python scripts/bootstrap_env.py --env harchoc --with-super-gradients`.
- **External DETR** (DEIM, D-FINE, RT-DETRv2): `harchoc/external_detector_train.py`; install via `--with-external-detr`, then `check_weights_cache.py --download --strict`.

Example HSP zoo sweep:

```bash
export DATASET_ROOT=/path/to/dataset

mamba run -n harchoc python scripts/check_weights_cache.py --download --strict --out reports/hsp/weights_cache.json
mamba run -n harchoc python scripts/benchmark_matrix.py --out reports/hsp/matrix_plan.json
mamba run -n harchoc python scripts/benchmark_matrix.py --no-dry-run \
  --runs-dir runs/hsp_zoo \
  --train-out reports/hsp/matrix_train.json \
  --out reports/hsp/matrix_plan.json
```

Train-only (skip post-train test eval): `--no-eval`. Train skipped: `--no-train`.

Seed comparison after a sweep: `--aggregate-seeds` (reads `--train-out`, writes `*.seed_stats.json` with mAP and count MAE when run rows reference `error_*_report.json` / `threshold_*_locked.json`, or via `--count-mae-json run_name=path`).

Count MAE across seeds (standalone, **P2-SEED-MAE**): point at per-run `error_*_report.json` or `threshold_*_locked.json` via run-row keys or `--count-mae-json`:

```bash
mamba run -n harchoc python scripts/matrix_seed_stats.py \
  --train-out reports/hsp/matrix_train.json \
  --count-mae-json yolov8m_e100_s0=reports/hsp/runs/yolov8m_e100_s0/error_test_report.json \
  --count-mae-json yolov8m_e100_s1=reports/hsp/runs/yolov8m_e100_s1/error_test_report.json \
  --out reports/hsp/matrix_seed_stats.json
```

Schema: `matrix_seed_stats.v1` (`count_mae_mean`, `count_mae_std` per model and globally). `--dry-run` writes the same shape (null MAE fields without `--train-out`).

### Eval-only

Already-trained weights on disk:

```bash
export DATASET_ROOT=/path/to/dataset
mamba run -n harchoc python scripts/benchmark_matrix.py --no-dry-run --no-train \
  --eval-out reports/benchmarks/matrix_eval.json
```

Ultralytics backend only; `model:` must be an **existing file path** (identifiers are not auto-resolved for eval-only).

### SAHI matrix eval protocol (scaffold)

Deploy uses SAHI sliced inference (`run_infer_once.py`, `harchoc/sahi_infer.py`); the zoo matrix default is full-frame Ultralytics val. **`--sahi-eval`** (or `experiments.v1` kind **`sahi_matrix_eval`**) writes a dry-run plan that expands each bench row with slice/conf params for deploy-parity eval — GPU execution is not wired yet.

```bash
export DATASET_ROOT=/path/to/dataset

# CLI: deploy-default slice params per bench row
mamba run -n harchoc python scripts/benchmark_matrix.py --sahi-eval \
  --group yolov8_scales --out reports/benchmarks/sahi_matrix_plan.json

# experiments.v1: cross-product bench rows × sahi_rows
mamba run -n harchoc python scripts/benchmark_matrix.py \
  --config configs/experiments/sahi_matrix_eval_plan_dry.json

mamba run -n harchoc python scripts/experiment.py benchmark \
  --config configs/experiments/sahi_matrix_eval_plan_dry.json
```

Plan schema: `sahi_matrix_eval.v1` with `eval_protocol: sahi`. Per-run `planned.sahi` holds `slice_size`, `overlap`, `nms_iou`, optional class conf thresholds. Bench YAML can override via `infer.tiling: sahi` and nested `infer.sahi:` (see below). Single-image SAHI grid tuning: `experiment.py tune-sahi --dry-run` only (live grid removed).

### Adding a bench entry

Add `configs/bench/<name>.yaml`:

- **backend:** `ultralytics` or `supergradients`
- **model:** weights id or file path
- **model_id:** SuperGradients architecture id (optional alternative to `model`)
- **infer:** `imgsz`, `max_det`, `tiling` (`none` | `sahi`), optional per-row SAHI params as JSON `sahi: {…}` or flat YAML keys `sahi_slice_size`, `sahi_overlap`, `sahi_nms_iou`, `sahi_conf_fertilized`, `sahi_conf_unfertilized`, `sahi_label`
- **groups:** tags for `--group` filtering

Top-level `imgsz` remains supported on older YAMLs.

## AutoBatch probe (`batch=-1`)

One-epoch Ultralytics AutoBatch probe for **yolov8m @ imgsz=1280** (train-only; no post-train eval). Logs resolved batch and exit codes to [`reports/hsp/autobatch_probe.json`](../reports/hsp/autobatch_probe.json). Run on an **idle GPU** — contention causes AutoBatch ladder OOM and fallback to batch 16.

Config: [`configs/experiments/train_autobatch_probe_yolov8m.json`](../configs/experiments/train_autobatch_probe_yolov8m.json).

```bash
export DATASET_ROOT="$(pwd)/data/raw/extracted/dataset"
export HARCHOC_MAX_EPOCHS=1
export HARCHOC_MAX_IMGSZ=2048

# Plan (CI-safe)
mamba run -n harchoc python scripts/train.py --dry-run \
  --config configs/experiments/train_autobatch_probe_yolov8m.json \
  --name autobatch_probe_yolov8m

# Live 1-epoch probe (GPU; update autobatch_probe.json with resolved batch + exit)
mamba run -n harchoc python scripts/train.py \
  --config configs/experiments/train_autobatch_probe_yolov8m.json \
  --name autobatch_probe_yolov8m
```

Re-run once per GPU class when scheduling production `batch` in bench configs. See also [`train_batch_probe_yolov8n.json`](../configs/experiments/train_batch_probe_yolov8n.json) (manual ladder) and [`docs/training_budget.md`](training_budget.md).

## GPU sanity check

```bash
mamba run -n harchoc python scripts/check_gpu.py
mamba run -n harchoc python scripts/check_gpu.py --json-out reports/hsp/gpu_check.json
mamba run -n harchoc python scripts/check_gpu.py --dry-run --json-out reports/hsp/gpu_check.json
mamba run -n harchoc python scripts/check_gpu.py sanity --dry-run --out reports/gpu_sanity.json
mamba run -n harchoc python scripts/check_gpu.py smoke-ultralytics --dry-run --out reports/gpu_smoke_ultralytics.json
```

## Domain catalog and eval scaffold (tray / session)

Build catalog from `DATASET_ROOT` (no GPU); dry-run writes `domain_eval.v1` tray groups without torch:

```bash
export DATASET_ROOT="$(pwd)/data/raw/extracted/dataset"
python scripts/eval_domains.py --catalog reports/domains/catalog.json --out reports/domains/domain_eval.json
python scripts/eval_domains.py --dry-run --catalog reports/domains/catalog.json --out reports/domains/domain_eval.json
```

Per-tray count MAE @ val-locked conf: `--merge-tray-count-mae` (export-only CPU). Per-tray mAP only: `eval.py --split-file data/domains/test_<tray_key>.txt`. See [domain_shift_transfer_literature.md](research/domain_shift_transfer_literature.md).

Optional acquisition metadata (variety / maturity / lighting / site) merges into `catalog.json` → `domain_metadata_tags.per_tray` and `domain_eval.json` when a CSV is supplied (no GPU):

```bash
python scripts/eval_domains.py \
  --catalog reports/domains/catalog.json \
  --out reports/domains/domain_eval.json \
  --import-domain-tags data/domain_tags.example.csv
```

CSV columns: `tray_key`, `variety`, `maturity`, `lighting`, `site`. Tray keys must match stems parsed by `harchoc/domain_tags.py` (e.g. `349-10-2`). Example: [`data/domain_tags.example.csv`](../data/domain_tags.example.csv); test fixture: [`tests/fixtures/domain_tags_sample.csv`](../tests/fixtures/domain_tags_sample.csv).

## Domain eval (per-tray)

Build catalog and per-tray split lists; run one tray or all trays on CPU (val-locked conf).
`DATASET_ROOT` is optional when `data/manifest.json` lists an on-disk extracted path (same resolution as `train.py` / `eval.py` via `harchoc.datasets.resolve_dataset`).

```bash
# Optional if manifest default (sunflower-cvat-2500) exists on disk:
export DATASET_ROOT=/path/to/dataset
mamba run -n harchoc python scripts/eval_domains.py \
  --write-domain-splits --catalog reports/domains/catalog.json
mamba run -n harchoc python scripts/eval_domains.py \
  --write-domain-splits --run-tray-eval --tray-key 200-3-1 \
  --device cpu --locked-conf-from reports/hsp/threshold_val.json \
  --out reports/domains/domain_eval.json
# Plan all tray evals without GPU/torch:
python scripts/eval_domains.py --dry-run --run-all-trays \
  --catalog reports/domains/catalog.json --out reports/domains/domain_eval.json
# All catalog trays → domain_eval.v1 (auto-writes per-tray split lists when needed):
mamba run -n harchoc python scripts/eval_domains.py --merge-tray-count-mae \
  --out reports/domains/domain_eval.json --device cpu

mamba run -n harchoc python scripts/eval_domains.py --run-all-trays \
  --device cpu --locked-conf-from reports/hsp/threshold_val.json \
  --catalog reports/domains/catalog.json --out reports/domains/domain_eval.json
```

## Transfer fine-tune (tray domain)

**Playbook:** [FINETUNE_WEAK_TRAYS.md](FINETUNE_WEAK_TRAYS.md) (weak-tray audit → `tray_adapt` splits → staged GPU).

`scripts/finetune.py` merges `configs/transfer/finetune_minimal.yaml` into a temp train JSON and calls `scripts/train.py`. Transfer fields:

| Field | Role |
|-------|------|
| `freeze_backbone` | When `true` and `freeze` unset → Ultralytics `freeze=10` (partial backbone) |
| `freeze` | Optional explicit Ultralytics `freeze` (int or list of layer indices) |
| `unfreeze_epoch` | **Meta only** — logged once per train run; staged unfreeze needs two train stages |
| `epochs`, `lr`, `seed` | Override experiment JSON for the finetune run |
| `train_mode` | `tray_adapt` (default when `--tray-key` set), `lofo_pool`, or `canonical` |

`train.py` records `freeze_policy` in `meta.json` (`train_meta.v1`). Dry-run finetune includes `transfer_policy`, `split_plan`, and `tray_eval_plan` in `finetune_run.v1` JSON.

| Flag | Role |
|------|------|
| `--train-mode` | `tray_adapt`: train on `data/domains/train_{tray}+val_{tray}` only (no `test.txt` leak) |
| `--tray-eval` / `--no-tray-eval` | Chain tray holdout eval before/after train (default on) |
| `--tray-key` | Holdout tray id(s); else transfer YAML or `--tray-catalog` |
| `--domains-dir` | `data/domains/{train,val,test}_{tray_key}.txt` (requires `--write-domain-splits`) |
| `--splits-dir` | Canonical `test.txt` for manuscript test eval |

`train.py` also accepts `--train-split-file` / `--val-split-file` (set automatically by finetune when `train_mode` ≠ `canonical`).

```bash
python scripts/experiment.py domain-tray-audit --out reports/domains/weak_tray_plan.json
mamba run -n harchoc python scripts/experiment.py finetune-tray --stage 1 --tray-key 349-10-2 \
  --dataset-root "$DATASET_ROOT"
```

Live runs record `tray_eval_before` / `tray_eval_after` output paths in `finetune.json`; device policy matches train config `eval` via `harchoc/post_train_eval.py`.

```bash
mamba run -n harchoc python scripts/finetune.py --dry-run --out reports/transfer/finetune.json
# Real train (GPU): omit --dry-run; set DATASET_ROOT or --dataset-root
mamba run -n harchoc python scripts/finetune.py --dataset-root "$DATASET_ROOT" --out reports/transfer/finetune.json

**Staged unfreeze (two train invocations):**

```bash
# Stage 1 — frozen backbone
mamba run -n harchoc python scripts/finetune.py --stage 1 --dataset-root "$DATASET_ROOT" \
  --out reports/transfer/finetune_stage1.json
# Stage 2 — full unfreeze from stage-1 best.pt
mamba run -n harchoc python scripts/finetune.py --stage 2 \
  --base-weights runs/transfer/finetune_tray_s1/weights/best.pt \
  --dataset-root "$DATASET_ROOT" --out reports/transfer/finetune_stage2.json
```

Configs: `configs/experiments/finetune_tray_stage{1,2}.json`, `configs/transfer/finetune_stage{1,2}.yaml`.
```

If installs are flaky: `python scripts/bootstrap_env.py --env harchoc --create`.

---

*Validated 2026-05-29.*
