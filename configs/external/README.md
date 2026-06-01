# External detector registry

Non-Ultralytics matrix rows (RT-DETRv2, D-FINE, DEIM) are configured here. Ultralytics and SuperGradients rows use hub weights only.

## Files

| File | Role |
|------|------|
| **`detector_sources.v1.json`** | **Canonical** — per-`source_id` checkpoints, `train_stack`, upstream repo URLs, `config_relpath`, COCO pretrained URLs. Edit this when adding or changing external rows. |
| **`external_repos.v1.json`** | **Generated** — clone specs derived from unique `train_stack` values in `detector_sources`. Do not edit by hand. |

Runtime code loads **`detector_sources.v1.json`** via `harchoc.detector_sources`. Clone URLs, refs, and `cache_dirname` come from `train_stacks` + entry `repos` fields; `harchoc.external_repos.derive_external_repo_specs()` builds the in-memory spec (and optional JSON mirror).

## Regenerate `external_repos.v1.json`

After editing `detector_sources.v1.json`:

```bash
python scripts/check_weights_cache.py --sync-repos-manifest
```

Implementation: `harchoc.external_repos.write_external_repos_manifest()`.

## Prep weights and clone upstream repos

[`scripts/check_weights_cache.py`](../../scripts/check_weights_cache.py) wraps [`harchoc.bench_assets`](../../harchoc/bench_assets.py) (`build_weights_prep_report`):

```bash
python scripts/bootstrap_env.py --env harchoc --with-external-detr   # faster-coco-eval, calflops, transformers, loguru, gdown
mamba run -n harchoc python scripts/check_weights_cache.py --sync-repos-manifest
mamba run -n harchoc python scripts/check_weights_cache.py --download --strict \
  --out reports/hsp/weights_cache.json
```

- **`--download`** — cache external `.pth` under `data/weights/external/`, git-clone into `external/` (git-ignored), update [`data/weights/weights_manifest.json`](../../data/weights/weights_manifest.json).
- **`--strict`** — exit non-zero if bench-required assets or manifest entries are missing.

Bench configs reference external rows via `backend: external` and `source_id` (see [`configs/zoo/matrix_rows.v1.json`](../zoo/matrix_rows.v1.json)). Matrix train: `harchoc/external_detector_train.py` writes `harchoc_train_overlay.yml` per stack:

| Stack | Epoch key | Collate @ 1280 |
|-------|-----------|----------------|
| D-FINE | `epochs` | `base_size` + scaled `stop_epoch` |
| DEIM | `epoches` | `base_size` + scaled DEIM schedule (`flat_epoch`, mixup, …); overlay disables Mosaic and uses D-FINE-style `policy.epoch` (int). **Torchvision ≥0.21:** runtime shim in [`harchoc/deim_tv_compat.py`](../../harchoc/deim_tv_compat.py) via [`harchoc/run_external_train.py`](../../harchoc/run_external_train.py) — do not patch `external/DEIM`. |
| RT-DETRv2 | `epoches` | `scales` (not `base_size`; stale overlays with `base_size` fail at collate) |

Preflight: `train_bench_run` runs `verify_external_train_smoke()` (isolated subprocess per stack) before distributed train.

Also passes `-u epochs=…` / `-u epoches=…` and `eval_spatial_size=[1280,1280]` on the train CLI so overrides survive YAML merge quirks.

## Env overrides (local clones)

| Env | Stack |
|-----|--------|
| `HARCHOC_DEIM_REPO` | `deim` |
| `HARCHOC_DFINE_REPO` | `dfine` |
| `HARCHOC_RTDETR_REPO` | `rtdetrv2_pytorch` |

## Related docs

- Zoo groups and row counts: [`docs/zoo_comparison_design.md`](../../docs/zoo_comparison_design.md)
- Experiment / matrix entrypoints: [`configs/experiments/README.md`](../experiments/README.md)
