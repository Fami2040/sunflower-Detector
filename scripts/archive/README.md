# Archived scripts (read-only shims)

Deprecated entrypoints kept for backward-compatible invocations. **Do not** add new logic here — extend [`scripts/experiment.py`](../experiment.py) or [`harchoc/`](../harchoc/) instead.

## Policy

- Shims print a `DeprecationWarning` and delegate to the canonical path.
- Bootstrap via [`harchoc.script_entry`](../../harchoc/script_entry.py) (same as other `scripts/*.py`).
- CI and agents should prefer `experiment.py` subcommands or direct library calls.

## Replacements

| Removed / archived | Use instead |
|--------------------|-------------|
| `scripts/dataset_from_manifest.py` | [`dataset_from_manifest.py`](dataset_from_manifest.py) shim → `python scripts/experiment.py dataset-root` |
| `experiment.py gpu-queue` | [`run_gpu_queue.sh`](../run_gpu_queue.sh) or [`run_gpu_queue.py`](../run_gpu_queue.py) with `GPU_QUEUE_MANIFEST=...` |
| Split validation only | `python scripts/experiment.py validate-splits` (delegates to [`validate_splits.py`](../validate_splits.py)) |
