# `configs/experiments/`

Canonical, reviewable **experiment specs** for this repo.

These JSON files are meant to be:

- easy to diff in PRs (no ad-hoc CLI flag sprawl)
- safe to load/normalize in CI (no ML deps required)
- reusable across scripts via a single schema

## Config decision tree

Three config styles exist; pick one — do not invent a fourth.

```
What are you configuring?
│
├─ Analysis / eval / reporting (no Ultralytics train loop)?
│  └─ experiments.v1 JSON here → scripts/experiment.py <subcommand> --config …
│     (eval, benchmark plan, split_drift, threshold_sweep, error_analysis, cv_eval)
│
├─ Ultralytics training hyperparameters?
│  └─ Flat train_*.json here → scripts/train.py --config …
│     (no schema_version; consumed directly by train.py)
│     Matrix live train: train_bench_<model>.json paired with bench YAML
│
└─ Model-zoo candidate (backend, model id, budget, groups)?
   └─ configs/bench/*.yaml → scripts/benchmark_matrix.py --bench-dir …
      (one file per model; matrix plan / train / eval chain)
```

| Style | Location | Entrypoint | When to use |
|-------|----------|------------|-------------|
| **`experiments.v1`** | `configs/experiments/*.json` with `"schema_version": "experiments.v1"` | `scripts/experiment.py` | Reviewable specs for eval, matrix **plan**, split drift, threshold sweep, error analysis; CI dry-runs via `normalize_experiment_spec()` |
| **Flat train JSON** | `configs/experiments/train_*.json` (no `schema_version`) | `scripts/train.py` | Ultralytics epochs, aug, optimizer, post-train eval caps; smoke/baseline/bench training recipes |
| **Bench YAML** | `configs/bench/*.yaml` | `scripts/benchmark_matrix.py` | Zoo matrix: backend, `model`, `imgsz`, `epochs`, `groups`; matrix generates or selects matching `train_bench_*.json` for live train |

Related YAML (not experiment specs): `configs/aug/` (aug recipes via `train.py --aug-config`), `configs/ablation/`, `configs/transfer/` — referenced by path from train or scaffold scripts, not a separate config system.

### When adding a feature

Extend existing entrypoints; **do not add a fourth config style**:

1. **`scripts/experiment.py`** — new subcommand or `run.kind` for analysis workflows.
2. **`scripts/train.py` / `scripts/eval.py` / `scripts/benchmark_matrix.py`** — new flags; keep flat JSON / bench YAML fields aligned with those flags.
3. **`harchoc/*`** — shared merge, parsing, dataset resolution (see [extend-before-add-script rule](../../.cursor/rules/extend-before-add-script.mdc)).

New top-level `scripts/*.py` only when the workflow is genuinely standalone and extending the above would harm clarity.

## Train-only configs (non-canonical)

Some JSON files under this directory are **not** `experiments.v1` specs. They are flat or `train`/`eval` blobs consumed directly by `scripts/train.py` (and optionally by `scripts/benchmark_matrix.py` via `train_bench_<model>.json`).

Examples:

- `train_yolov8m_baseline.json`, `train_smoke_yolov8n.json`
- **Bench matrix train** (`train_bench_<stem>.json`, paired with `configs/bench/*.yaml`):
  - Shared hyperparameters: `train_bench_base.json` (epochs, imgsz, optimizer, aug, eval caps, etc.)
  - Per-model overlays: `train_bench_<stem>.json` with `"extends": "configs/experiments/train_bench_base.json"` plus `model` / `model_id`, `batch`, and `notes` (RT-DETR adds query-cap fields on its overlay)
  - **RT-DETR query-cap policy** (`train_bench_rtdetr-l.json`): `num_queries` (default 300), `documented_peak_gt_boxes_per_image` (1015), `accept_rtdetr_query_truncation` (required when queries &lt; peak). Guarded by `harchoc/rtdetr_limits.py` in `train.py` and matrix bench load. Env: `HARCHOC_RTDETR_PEAK_GT_BOXES_PER_IMAGE`, `HARCHOC_RTDETR_QUERY_CAP=warn`. Smoke: `train_rtdetr_smoke_15ep.json` — see [`docs/training_budget.md`](../../docs/training_budget.md#rt-detr-query-cap-dense-trays).
  - Ultralytics YOLO: `yolov8n/s/m/l`, `yolov10s/m/b/l/x`, `yolo11s/m/l/x`, `yolo26n/s/m/l/x`
  - Ultralytics RT-DETR: `rtdetr-l`, `rtdetr-x` (+ scratch `rtdetr-l_nq1024` YAML ablation)
  - SuperGradients: `yolo_nas_s`
  - External DETR stack (`backend: external`, groups `zoo_core` / `sota_deim` / `zoo_detr_stack`): canonical registry [`detector_sources.v1.json`](../external/detector_sources.v1.json); generated clone mirror [`external_repos.v1.json`](../external/external_repos.v1.json) — see [`configs/external/README.md`](../external/README.md)
  - Matrix groups: **`zoo_core`** (10-row beat-`yolov8m` comparison), **`zoo_scale`** (size ladders), **`sota_2026`** (full Ultralytics set)
  - Shared aug: `configs/aug/robustness_minimal.yaml` via `"aug_config"` in the base file
  - Batch @ 1280: `yolov8n` → `2`, all other Ultralytics scales → `1`
  - `scripts/train.py` and `harchoc.bench_config` resolve `extends` via `harchoc.train_config.resolve_train_config_extends()` (deep-merge; overlay wins)
- `train_batch_probe_yolov8n.json`, `train_batch_probe_yolov8m.json`, `train_batch_probe_rtdetr-l.json` (`"_canonical": false` — 1-ep GPU VRAM probes; see [`docs/training_budget.md`](../../docs/training_budget.md#vram-probes-1-epoch--imgsz-1280))
- `train_autobatch_probe_yolov8m.json` (`"_canonical": false` — P1-AUTOBATCH: 1-ep Ultralytics AutoBatch `batch=-1` @ 1280, `eval.skip`)
- **Aug / ranking smokes (15 ep @ 1280):**
  - [`train_smoke_rank_15ep.json`](train_smoke_rank_15ep.json) — shared 15-ep parent for P1-AUG smokes (runtime via [`aug_smoke_index.json`](aug_smoke_index.json) + [`harchoc/aug_smoke_train.py`](../../harchoc/aug_smoke_train.py))
  - [`train_smoke_rank_yolo11s_15ep.json`](train_smoke_rank_yolo11s_15ep.json) — YOLO11s 15-ep parent (S10 committed chain)
  - Committed aug exceptions: `train_aug_s9` … `train_aug_s13` + close/mosaic sweep overlays; S0–S8/S1 use index `aug_config` only (no per-smoke train JSON)
  - Index + queue wiring: [`aug_smoke_index.json`](aug_smoke_index.json) (`aug_smoke_index.v1`); GPU queue expands pending rows when the manifest sets `"aug_smoke_from_index": true` (see below)

`normalize_experiment_spec()` skips files without `"schema_version": "experiments.v1"`.

### Zoo matrix manifest (`matrix_rows.v1`)

Canonical **26-row** zoo definition: [`configs/zoo/matrix_rows.v1.json`](../zoo/matrix_rows.v1.json) (`zoo_matrix_rows.v1`). Bench YAML + `train_bench_*.json` must stay aligned with this file.

| Group | Rows (filter, not additive) |
|-------|-----------------------------|
| `zoo_core` | 10 |

**RT-DETR gate (zoo_core):** 100-ep matrix rows `rtdetr_l_nq1024` and `rtdetr_x` are blocked until GPU queue 15-ep smokes finish with **test count MAE** summaries (`rtdetr_queries_smoke`, `rtdetr_imgsz1280`). Enforced in `harchoc/rtdetr_zoo_gate.py`, `benchmark_matrix.py`, and `zoo_matrix_train` job stage `rtdetr_15ep_gate` in [`gpu_queue_full.json`](gpu_queue_full.json).
| `zoo_scale` | 14 |
| `sota_2026` | 22 |
| `sota_deim` / `zoo_detr_stack` | 4 each (same external DETR rows) |

```bash
# Regenerate missing bench/train scaffolds from manifest (review diffs before commit)
python scripts/benchmark_matrix.py --scaffold-zoo --out reports/benchmarks/zoo_scaffold_report.json

# CI / pre-merge: bench + train_bench JSON match manifest (no ML deps)
python scripts/benchmark_matrix.py --validate-zoo

# CI / pre-merge: aug smoke index vs runtime train/aug configs (no ML deps)
python scripts/experiment.py validate-aug-smoke
```

Design and pruning: [`docs/zoo_comparison_design.md`](../../docs/zoo_comparison_design.md).

### Weights / repos prep (`harchoc.bench_assets`)

[`scripts/check_weights_cache.py`](../../scripts/check_weights_cache.py) is a thin CLI over `harchoc.bench_assets.build_weights_prep_report()` (alias `build_report`): Ultralytics `.pt` cache, external `.pth`, and `external/` git clones derived from bench configs.

```bash
python scripts/bootstrap_env.py --env harchoc --with-external-detr   # includes gdown for DEIM Drive checkpoints
mamba run -n harchoc python scripts/check_weights_cache.py --sync-repos-manifest
mamba run -n harchoc python scripts/check_weights_cache.py --download --strict \
  --out reports/hsp/weights_cache.json
```

- **`--sync-repos-manifest`** — refresh generated [`external_repos.v1.json`](../external/external_repos.v1.json) from canonical [`detector_sources.v1.json`](../external/detector_sources.v1.json) (no network).
- **`--download`** — fetch missing weights + clone upstream repos under `external/`; updates [`data/weights/weights_manifest.json`](../../data/weights/weights_manifest.json) when `--strict` or default manifest path is used.

## Schema (`experiments.v1`)

Each file is a single JSON object:

- **`schema_version`**: must be `experiments.v1`
- **`dataset`**: dataset selection inputs (shared)
- **`run`**: script selection + args

Minimal shape:

```json
{
  "schema_version": "experiments.v1",
  "dataset": {
    "manifest": "data/manifest.json",
    "default_dataset_name": "sunflower-cvat-2500"
  },
  "run": {
    "kind": "eval",
    "dry_run": true,
    "out": "reports/ci/eval.dry.json"
  }
}
```

### Dataset fields

- **`dataset.manifest`**: path to `data/manifest.json` (repo-relative or absolute)
- **`dataset.default_dataset_name`**: used when `DATASET_NAME` is unset
- **`dataset.dataset_env`** (optional): a dict of env overrides used only for normalization/testing:
  - `DATASET_ROOT`
  - `YOLO_DATA_YAML`
  - `DATASET_NAME`

### Run kinds

- **`eval`**: corresponds to `scripts/eval.py`
  - common keys: `weights`, `split_file`, `out`, `dry_run`
- **`benchmark_matrix`**: corresponds to `scripts/benchmark_matrix.py`
  - common keys: `bench_dir`, `bench_config`, `group`, `out`, `eval_out`, `dry_run`
- **`sahi_matrix_eval`**: SAHI deploy-parity matrix plan (alias for `benchmark_matrix` + `sahi_eval: true`)
  - common keys: `bench_dir`, `group`, `sahi_rows`, `out`, `dry_run`
- **`split_drift`**: corresponds to `scripts/split_drift.py`
  - common keys: `splits_dir`, `out`, `dry_run`
- **`threshold_sweep`**: corresponds to `scripts/threshold_sweep.py`
  - common keys: `gt_json`, `preds_json`, `out`, `csv_out`, `fixed_conf`, `locked_conf_from`, `dry_run`, `light`
- **`error_analysis`**: corresponds to `scripts/error_analysis.py`
  - common keys: `gt_json`, `preds_json`, `out`, `report`, `locked_conf_from`, `export_fp_crops`, `light`, `dry_run`
- **`cv_eval`**: corresponds to `scripts/cv_eval.py` via `experiment.py cv-eval`
  - common keys: `folds`, `seed`, `splits_dir`, `fold_metrics`, `write_fold_splits`, `weights`, `out`, `dry_run`

### GPU queue manifest (`gpu_queue_manifest.v1`)

Sequential one-GPU backlog runner: default [`gpu_queue_aug_pending.json`](gpu_queue_aug_pending.json); full backlog [`gpu_queue_full.json`](gpu_queue_full.json) (`GPU_QUEUE_MANIFEST=…`).

- Schema: `gpu_queue_manifest.v1` with ordered `jobs[]` (`kind`, `backlog`, `env`, `skip_if`)
- State: `reports/gpu_queue/run_state.json`, logs under `reports/gpu_queue/logs/`
- **Canonical ops:** [`./scripts/run_gpu_queue.sh`](../../scripts/run_gpu_queue.sh) (`dry-run` | `run` | `resume`; override manifest with `GPU_QUEUE_MANIFEST`)
- Direct / CI: [`scripts/run_gpu_queue.py`](../../scripts/run_gpu_queue.py) (`--manifest`, `--job`). `experiment.py gpu-queue` is a deprecated alias (config-file workflows only).
- Implementation: [`harchoc/gpu_queue.py`](../../harchoc/gpu_queue.py), [`harchoc/aug_smoke_runner.py`](../../harchoc/aug_smoke_runner.py)

Job kinds: `preflight`, `vram_probe`, `aug_smoke`, `rtdetr_smoke`, `train_compare`, `amp_smoke`, `sg_smoke`, `aug_sweep_15` (alias `aug_sweep_100`), `zoo_matrix_train`, `cv_fold_train`.

**`aug_smoke_from_index`** (optional manifest fields on `gpu_queue_manifest.v1`):

- `"aug_smoke_from_index": true` — at load time, [`harchoc/gpu_queue.py`](../../harchoc/gpu_queue.py) replaces inline `aug_smoke` jobs with one job per [`aug_smoke_index.json`](aug_smoke_index.json) row in `status: gpu_pending` (default index path: `"aug_smoke_index": "configs/experiments/aug_smoke_index.json"`). Each expanded job carries `train_config` / `aug_config` from the index (passed to `train.py` as `--aug-config` when set).
- Used by [`gpu_queue_aug_pending.json`](gpu_queue_aug_pending.json) and [`gpu_queue_full.json`](gpu_queue_full.json) so the queue stays DRY vs duplicating S0–S14 job blocks. Mosaic `aug_sweep_15_*` jobs are **removed** from manifests (covered by smokes S2/S4/S5).
- Parity check (tests / pre-run): `harchoc.aug_smoke_runner.aug_smoke_index_queue_parity_errors()`.
- Post-eval summaries for smokes, sweeps, and RT-DETR probes share `finalize_smoke_job()` in [`aug_smoke_runner.py`](../../harchoc/aug_smoke_runner.py); `aug_smoke` jobs also patch the index on success.

## Precedence (config < env < CLI)

**Order:** config defaults → environment overrides (dataset only) → CLI wins.

1. **Config** — `configs/experiments/*.json`, flat `train_*.json`, or `configs/bench/*.yaml` provides defaults.
2. **Environment** — dataset resolution only ([`docs/EXPERIMENTS.md` § Dataset resolution](../../docs/EXPERIMENTS.md#dataset-resolution-required-convention)):
   - `DATASET_ROOT` (highest)
   - `YOLO_DATA_YAML`
   - `DATASET_NAME`
3. **CLI flags** — override config when invoking scripts directly or via `experiment.py`.

Implementation:

- **`scripts/experiment.py`** — `merge_experiment_config()` applies **config < CLI** ([`harchoc/experiment_config.py`](../../harchoc/experiment_config.py)).
- **`harchoc/datasets.py`** — `resolve_dataset()` applies **env precedence** for dataset root / `data.yaml` / manifest name.
- **`scripts/train.py`**, **`scripts/benchmark_matrix.py`** — load JSON/YAML first; CLI flags override loaded fields.

## Normalization (dependency-light)

`harchoc/experiment_config.py` exposes `normalize_experiment_spec()` which resolves:

- `dataset.root` / `dataset.yolo_data_yaml`
- default split sources (`data/splits/*.txt` if present, else `<dataset_root>/images/<split>`)
- common output paths as repo-root absolute paths

