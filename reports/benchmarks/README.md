# Benchmark matrix reports

Artifacts produced by `scripts/benchmark_matrix.py`.

| File | When |
|------|------|
| `matrix.json` | **Dry-run** (default): parsed bench configs, planned train/eval commands, resolved paths. No training or eval execution. |
| `matrix_train.json` | After **`--no-dry-run`** with training enabled: aggregated train results (`--train-out`, default here). |
| `matrix_eval.json` | After **`--no-dry-run`** with eval enabled: aggregated eval metrics (`--eval-out`). |

Regenerate the plan (example):

```bash
DATASET_ROOT=/path/to/dataset \
  mamba run -n harchoc python scripts/benchmark_matrix.py --dry-run \
  --out reports/benchmarks/matrix.json
```

HSP workflow copies the same plan to `reports/hsp/matrix_plan.json`; full matrix training writes `reports/hsp/matrix_train.json`.
