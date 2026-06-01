# Data (local)

**Canonical corpus:** CVAT-annotated sunflower seed dataset (~**2500** images), share **`y9xGFqCW`** — see [`manifest.json`](manifest.json) (`sunflower-cvat-2500`). Modeling pool: **1093** images with frozen splits.

**Manuscript Methods text:** [`reports/manuscript/dataset.md`](../reports/manuscript/dataset.md).  
**Reproducibility:** [`manifest.json`](manifest.json) + [`splits/`](splits/) SHA256 + [`docs/ORIGIN_MAIN_AND_DATASET.md`](../docs/ORIGIN_MAIN_AND_DATASET.md).

Raw CVAT export and weights are **git-ignored**. Tracked files here define **where** data lives and **how** splits are frozen.

## Classes (canonical names)

| Class id | Name | Role |
|---------:|------|------|
| **0** | **developed** | Viable / developed seed on the head |
| **1** | **aborted** | Aborted seed on the head |

Use only **developed** / **aborted** in `data.yaml`, training configs, backlog, and reports. Numeric ids in label files are ground truth; names are for humans and Ultralytics plots.

## On-disk layout (`DATASET_ROOT`)

Typical root: `data/raw/extracted/dataset` (see `data/manifest.json`).

```text
DATASET_ROOT/
  images/
    train/          # CVAT export folder (not the modeling train split by itself)
    val/            # CVAT export folder
  labels/
    train/          # one .txt per image, same stem as under images/
    val/
  data.yaml         # install from data/data.yaml.example (developed / aborted)
  data/splits/      # optional mirror of repo lists
    train.txt
    val.txt
    test.txt
```

**Pairing:** `images/val/3a2-2___.jpg` ↔ `labels/val/3a2-2___.txt`.

**Label format (YOLO):** one box per line:

```text
<class_id> <cx> <cy> <w> <h>
```

Example (`0` = developed, `1` = aborted):

```text
0 0.791832 0.556161 0.026351 0.021794
1 0.519543 0.661358 0.017567 0.015088
```

Coordinates are **normalized** to image width/height. Images are dense tray/head photos (often hundreds of boxes per file).

## Modeling splits (tracked)

Canonical lists: **`data/splits/{train,val,test}.txt`** (~875 / 109 / 109 images).

- Lines are paths **relative to `DATASET_ROOT`**, e.g. `images/val/3a2-2___.jpg`.
- Entries may point at `images/train/` or `images/val/`; CVAT folder names are **not** the train/val boundary.
- **Ultralytics in-training validation:** `val.txt`. **Manuscript / `scripts/eval.py` test metrics:** `test.txt` only.

See `data/splits/README.md` for policy, seed, and SHA256.

After extract:

```bash
export DATASET_ROOT="$(python -c 'from harchoc.datasets import dataset_root_from_manifest; print(dataset_root_from_manifest(dataset_name=\"sunflower-cvat-2500\"))')"
mkdir -p "$DATASET_ROOT/data/splits"
cp data/splits/{train,val,test}.txt "$DATASET_ROOT/data/splits/"
cp data/data.yaml.example "$DATASET_ROOT/data.yaml"
```

## Manifest

Update `data/manifest.json` (tracked) with source URL, archive path, and `extracted_paths`.

## Resolution env vars

1. `DATASET_ROOT` — dataset root directory  
2. `YOLO_DATA_YAML` — path to `data.yaml`  
3. `DATASET_NAME` — entry in `data/manifest.json` (default `sunflower-cvat-2500`)

Helper: `harchoc.datasets.resolve_dataset`.

## Domain metadata tags (optional)

Per-tray acquisition metadata (variety, maturity, lighting, site) is **not** in image stems or YOLO labels. When available, supply a CSV and import via `scripts/eval_domains.py --import-domain-tags`.

**Schema** (`domain_metadata_tags.v0` in `reports/domains/catalog.json`):

| Column | Required | Description |
|--------|----------|-------------|
| `tray_key` | yes | Tray id from image stem (lowercase), e.g. `349-10-2` — see `harchoc/domain_tags.tray_key_from_stem` |
| `variety` | no | Cultivar / hybrid label |
| `maturity` | no | Maturity stage (e.g. `ripe`, `early`) |
| `lighting` | no | Imaging lighting (e.g. `benchtop`, `LED`, `field`) |
| `site` | no | Lab or field site id |

Example (git-tracked): [`domain_tags.example.csv`](domain_tags.example.csv). Test fixture: [`tests/fixtures/domain_tags_sample.csv`](../tests/fixtures/domain_tags_sample.csv).

Import merges into `catalog.json` → `domain_metadata_tags.per_tray` and propagates to `domain_eval.json` (top-level block + per-domain `tags` when present):

```bash
export DATASET_ROOT=/path/to/dataset
python scripts/eval_domains.py \
  --catalog reports/domains/catalog.json \
  --out reports/domains/domain_eval.json \
  --import-domain-tags data/domain_tags.example.csv
```

Real metadata CSVs stay local (git-ignored); only the example schema is tracked.
