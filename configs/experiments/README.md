# `configs/experiments/`

Reviewable experiment specs: diff-friendly JSON, CI-safe normalization, shared via `scripts/experiment.py`.

## Config decision tree

```
Analysis / eval / reporting  →  experiments.v1 JSON  →  experiment.py
Ultralytics training         →  flat train_*.json     →  train.py
Model-zoo candidate          →  configs/bench/*.yaml  →  benchmark_matrix.py
```

| Style | Entrypoint | When to use |
|-------|------------|-------------|
| **`experiments.v1`** | `experiment.py` | Eval, matrix plan, drift, threshold sweep, error analysis |
| **Flat train JSON** | `train.py` | Epochs, aug, optimizer; `train_bench_*` for matrix live train |
| **Bench YAML** | `benchmark_matrix.py` | Zoo backend, model, groups |

Related paths only: `configs/aug/`, `configs/transfer/` — referenced by train scripts, not a fourth schema.

**GPU queues (active):** [`gpu_queue_zoo_p0_5.json`](gpu_queue_zoo_p0_5.json) · [`gpu_queue_post_zoo.json`](gpu_queue_post_zoo.json) · [`gpu_queue_post_zoo_smoke.json`](gpu_queue_post_zoo_smoke.json). **Archived (read-only):** [`archive/`](archive/README.md) — do not add jobs or revive `gpu_queue_full` on 8 GiB for RT-DETR-by-default.

**GPU runner:** [`scripts/run_gpu_queue.sh`](../../scripts/run_gpu_queue.sh) or [`scripts/run_gpu_queue.py`](../../scripts/run_gpu_queue.py) — not `experiment.py`.

Extend `experiment.py` / `train.py` / `benchmark_matrix.py` before new top-level scripts ([extend-before-add-script](../../.cursor/rules/extend-before-add-script.mdc) · [dry-refactor-plan](../../docs/plans/dry-refactor-plan.md)).

## Train JSON inventory (~30 active)

| Group | Files | Notes |
|-------|--------|--------|
| Baseline / hyperparams | `train_yolov8m_baseline.json`, `train_hyperparams_common.json`, `train_bench_base.json` | 100 ep production recipe |
| Rank smokes | `train_smoke_rank_15ep.json`, `train_smoke_rank_yolo11s_15ep.json` | 15 ep @ 1280 |
| Aug index | [`aug_smoke_index.json`](aug_smoke_index.json) | S0–S14 + sweeps; `train_overrides` for S9/S12/S13; aug YAML per arm |
| Aug committed | `train_aug_s10_yolo11s_smoke.json`, `train_aug_s11_musgd_smoke.json` | Model / optimizer exceptions only |
| Aug sweeps | `train_aug_mosaic_sweep_smoke_15ep.json`, `train_aug_winner_100ep.json`, `train_aug_schedule_patience25_100ep.json` | 15 ep / 100 ep shared bases |
| Probes | `train_batch_probe.template.json` + overlays, `train_amp_probe.template.json` + on/off | VRAM / AMP 1 ep |
| Bench matrix | `train_bench_*.json` | One overlay per zoo row |
| RT-DETR smokes | `train_rtdetr_*_smoke*.json` | Query / imgsz probes |

**Unused / archived:** [`archive/unused_train/`](archive/unused_train/) (legacy smokes with no references).

## Generated train configs (gitignored)

Index-only smokes with `train_overrides` materialize merged JSON under:

`configs/experiments/.aug_smoke_generated/<smoke_id>.json`

Written by `experiment.py validate-aug-smoke`, GPU queue aug_smoke stages, and tests. **Do not commit** — regenerate from [`aug_smoke_index.json`](aug_smoke_index.json).

## Manuscript / reviewer repro bundles

| File | Role |
|------|------|
| [`manuscript_repro_bundle.json`](manuscript_repro_bundle.json) | HSP repro + preflight |
| [`reviewer2_repro.json`](reviewer2_repro.json) | Post-zoo reviewer CPU chain |
| [`figures_repro.json`](figures_repro.json) | Figure manifest |
| [`reviewer_counting.json`](reviewer_counting.json) | Counting metrics JSON (top-level `reviewer_counting` section; not `experiments.v1`) |
| [`reviewer2_confusion_tide.json`](reviewer2_confusion_tide.json) | §11 confusion + TIDE parity (top-level `reviewer2_confusion` section) |
| [`cohort_zeroshot_eval.template.json`](cohort_zeroshot_eval.template.json) | Copy after year-cohort ingest — zero-shot `best2` eval |
| [`gpu_queue_post_zoo.json`](gpu_queue_post_zoo.json) | After P0-5 (`require_before.matrix_train` gate) |
| [`gpu_queue_post_zoo_smoke.json`](gpu_queue_post_zoo_smoke.json) | 1-ep finetune + domain audit smoke (test wiring) |
| [`now_todos_smoke_bundle.json`](now_todos_smoke_bundle.json) | `experiment.py now-todos-smoke` stage definitions |

Ops runbooks (GPU queue job matrix, zoo groups, weights prep): [`docs/EXPERIMENTS.md`](../../docs/EXPERIMENTS.md#gpu-sequential-queue).

## Schema (`experiments.v1`)

```json
{
  "schema_version": "experiments.v1",
  "dataset": {
    "manifest": "data/manifest.json",
    "default_dataset_name": "sunflower-cvat-1093"
  },
  "run": {
    "kind": "eval",
    "dry_run": true,
    "out": "reports/ci/eval.dry.json"
  }
}
```

Run kinds include `eval`, `benchmark_matrix`, `split_drift`, `threshold_sweep`, `error_analysis`, `cv_eval`, `sahi_matrix_eval`. Full field list: [`harchoc/experiment_config.py`](../../harchoc/experiment_config.py).

## Validation (CI, no GPU)

```bash
python scripts/benchmark_matrix.py --validate-zoo
python scripts/experiment.py validate-aug-smoke
PYTHONPATH=. HARCHOC_ALLOW_BASE_PYTHON=1 python scripts/run_tests.py -q
```

Zoo row manifest: [`configs/zoo/matrix_rows.v1.json`](../zoo/matrix_rows.v1.json). Design: [`docs/zoo_comparison_design.md`](../../docs/zoo_comparison_design.md).

## Precedence

Config defaults → env (`DATASET_ROOT`, `YOLO_DATA_YAML`, `DATASET_NAME`) → CLI. See [`docs/EXPERIMENTS.md` § Dataset resolution](../../docs/EXPERIMENTS.md#dataset-resolution-required-convention).
