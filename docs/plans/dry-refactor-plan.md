# DRY refactor plan (phased)

**Audit index:** [`refactor.md`](../../refactor.md) · **Status:** [`docs/manuscript/status.md`](../manuscript/status.md)

## Phase 1 (done)

- Canonical [`status.md`](../manuscript/status.md); slim `now_todos` / `FRESHNESS` / `STUBS`
- GPU queue archive under [`configs/experiments/archive/`](../../configs/experiments/archive/)
- Active queues: `gpu_queue_zoo_p0_5`, `gpu_queue_post_zoo`, `gpu_queue_post_zoo_smoke`
- `now-todos-smoke` harness + tests

## Phase 2 (in progress)

| ID | Task | Module / files |
|----|------|----------------|
| 2a | Config merge helper | [`harchoc/experiment_cli.py`](../../harchoc/experiment_cli.py) → `threshold_sweep`, `error_analysis`, `benchmark_matrix` |
| 2b | HSP subcommands | `experiment.py`: `threshold-sweep`, `error-analysis`, `split-drift`, `fp-budget-sweep` |
| 2c | Train template dedupe | `train_batch_probe.template.json`; aug via `aug_smoke_index.json` only |
| 2d | Mechanical | `describe_split` → `eval_export.read_image_size`; HSP default outs `reports/hsp/` |

## Phase 3 (optional)

- Reports `.gitignore` policy doc in [`reports/README.md`](../../reports/README.md)
- Untrack generated `narrative_from_backlog.md` if preflight always rebuilds

## Do not

- Merge 1093 test with future cohort without new split SHA
- Split `telegram_bot.py` monolith
- Re-enable full archived `gpu_queue_full` on 8 GiB for RT-DETR by default
