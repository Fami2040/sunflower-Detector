# Canonical train / val / test splits

Tracked split lists for dataset `sunflower-cvat-1093` (see `data/manifest.json`).

## Policy

- **Mode**: reproducible random holdout (`scripts/make_splits.py --mode random`)
- **Seed**: `0`
- **Fractions**: 80% train / 10% val / 10% test (`--val-frac 0.1 --test-frac 0.1`)
- **Pool**: all images under `images/**/*` (both `images/train/` and `images/val/` folders from CVAT export)
- **Counts**: 875 train, 109 val, 109 test (1093 total)

CVAT folder layout is **not** used as the train/val boundary for modeling; only the frozen lists in this directory define splits. Ultralytics in-training validation uses `val.txt`; manuscript / `scripts/eval.py` metrics use `test.txt` only.

## Regenerate (optional)

```bash
export DATASET_ROOT="$(python -c 'from harchoc.datasets import dataset_root_from_manifest; print(dataset_root_from_manifest())')"
PYTHONPATH=. python scripts/make_splits.py \
  --mode random --seed 0 --val-frac 0.1 --test-frac 0.1
cp data/splits/{train,val,test}.txt "$DATASET_ROOT/data/splits/"
```

## SHA256 (v1)

| file | sha256 |
|------|--------|
| train.txt | `48b433418e631e7a08854793a3240b01f060825e35574600319f8c95b060f342` |
| val.txt | `2514b7881cd0dc0d18fd8f05fde51bdaec4fd35ffda3b529bb1fc9221a721268` |
| test.txt | `49f5efc46732689f993aae8b2bdfed98695cee1db030708d96418d459bc5922a` |

Recorded in run metadata via `collect_run_metadata(..., include_repo_splits=True)`.
