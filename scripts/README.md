# Scripts

Prefer **`python scripts/experiment.py <subcommand>`** for eval, threshold sweep, error analysis, split drift, repro, and dataset resolution.

| Need | Use |
|------|-----|
| GPU sequential queue | [`run_gpu_queue.sh`](run_gpu_queue.sh) or [`run_gpu_queue.py`](run_gpu_queue.py) |
| Training | [`train.py`](train.py) + `configs/experiments/train_*.json` |
| Zoo matrix | [`benchmark_matrix.py`](benchmark_matrix.py) |
| Unit tests | [`run_tests.py`](run_tests.py) (summary + all failures in one run) |
| Dataset root only | `experiment.py dataset-root` |

Heavy logic lives in [`harchoc/`](../harchoc/). Do not add top-level scripts when a subcommand or library call suffices ([extend-before-add-script](../.cursor/rules/extend-before-add-script.mdc)).
