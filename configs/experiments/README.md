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

**GPU queues (active):** [`gpu_queue_zoo_p0_5.json`](gpu_queue_zoo_p0_5.json) · [`gpu_queue_post_zoo.json`](gpu_queue_post_zoo.json) · [`gpu_queue_post_zoo_smoke.json`](gpu_queue_post_zoo_smoke.json). Closed aug/full queues: [`archive/`](archive/README.md).

Extend `experiment.py` / `train.py` / `benchmark_matrix.py` before new top-level scripts ([extend-before-add-script](../../.cursor/rules/extend-before-add-script.mdc) · [dry-refactor-plan](../../docs/plans/dry-refactor-plan.md)).

## Train-only configs

Files without `"schema_version": "experiments.v1"` are consumed by `train.py` (and matrix via `train_bench_<stem>.json`):

- Baselines: `train_yolov8m_baseline.json`, smokes, batch probes (`train_batch_probe.template.json` + thin `extends` overlays; `"_canonical": false`)
- AMP 1-ep probes: `train_amp_probe.template.json` + `train_amp_on_probe_1ep.json` / `train_amp_off_probe_1ep.json`
- Matrix: `train_bench_base.json` + per-model overlays; aug via `configs/aug/robustness_minimal.yaml`
- Aug smokes: [`aug_smoke_index.json`](aug_smoke_index.json) (canonical arms; `train_overrides` for S9/S12/S13) + [`train_smoke_rank_15ep.json`](train_smoke_rank_15ep.json) + S10/S11 committed JSON only

RT-DETR query-cap policy: `train_bench_rtdetr-l.json` — see [`docs/training_budget.md`](../../docs/training_budget.md#rt-detr-query-cap-dense-trays). External DETR: [`configs/external/README.md`](../external/README.md).

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
```

Zoo row manifest: [`configs/zoo/matrix_rows.v1.json`](../zoo/matrix_rows.v1.json). Design: [`docs/zoo_comparison_design.md`](../../docs/zoo_comparison_design.md).

## Precedence

Config defaults → env (`DATASET_ROOT`, `YOLO_DATA_YAML`, `DATASET_NAME`) → CLI. See [`docs/EXPERIMENTS.md` § Dataset resolution](../../docs/EXPERIMENTS.md#dataset-resolution-required-convention).
