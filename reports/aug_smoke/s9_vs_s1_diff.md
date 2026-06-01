# S9 vs S1 effective train+aug diff (P1-AUG)

**Date:** 2026-05-30  
**Metrics:** S1 test count MAE **68.9**; S9 **73.2** (+4.3 vs S1, +11.9 vs best2 @ 61.3)

## Configs

| Smoke | train JSON | aug YAML |
|-------|------------|----------|
| **S1** | `configs/experiments/train_aug_s1_close3_smoke.json` | `configs/aug/robustness_smoke_close3.yaml` → `robustness_smoke_base.yaml` |
| **S9** | `configs/experiments/train_aug_s9_no_aug_yaml_smoke.json` | `aug_config: null` |

Summaries: [`s1_summary.json`](s1_summary.json), [`s9_summary.json`](s9_summary.json).

## Merge path (runtime)

`load_train_config_json` → `scripts/train._merge_train_config(_BASELINE_DEFAULTS)` → `harchoc.train_config.effective_train_aug_merged`

S9 has no `aug_config`; aug kwargs come from `_BASELINE_DEFAULTS` in `scripts/train.py` plus whatever Ultralytics applies when keys are not forwarded. Recipe fingerprint (`effective_train_recipe_fingerprint`) uses `effective_train_aug_merged` only — S9 is a distinct recipe, not in the S0≡S1≡S13 class.

## Shared schedule / model

Both extend `train_smoke_rank_15ep.json` (15 ep, patience 12, `yolov8m.pt`, AdamW, lr0=2e-4, imgsz 1280, batch 1, seed 0, max_det 3000).

## Field-by-field aug diff (runtime effective)

| Key | S1 (close3 + smoke_base) | S9 (`_BASELINE_DEFAULTS` only) | Notes |
|-----|--------------------------|--------------------------------|-------|
| **`close_mosaic`** | **3** | absent → **YOLO default 10** | Main schedule delta @ 15 ep |
| `mosaic` | 0.1 | 0.1 | Same |
| `translate` / `scale` | 0.05 / 0.15 | 0.05 / 0.15 | Same |
| `hsv_h` | 0.02 | 0.02 | Same |
| **`hsv_s` / `hsv_v`** | **0.35** / **0.35** | **0.30** / **0.30** | Smoke_base vs code defaults |
| **`mixup` / `cutmix`** | **0.0** (explicit) | absent | YOLO default 0 |
| **`fliplr` / `flipud`** | **0.5** / **0.0** (explicit) | absent | ~YOLO defaults |
| **`erasing`** | **0.2** (explicit) | absent | YOLO default ~0.4 when unset |
| `degrees` / `shear` / `perspective` | 0.0 each (explicit) | absent | YOLO default 0 |

### Mosaic tail @ 15 epochs (`mosaic=0.1`)

| Arm | `close_mosaic` | Mosaic-active epochs | Mosaic-off tail |
|-----|----------------|----------------------|-----------------|
| **S1** | 3 | 0–11 | 12–14 (3 ep) |
| **S9** | 10 (implicit) | 0–4 | 5–14 (10 ep) |

At `mosaic=0.1` only ~10% of batches use mosaic; the tail shift is modest but real.

## Why +4.3 MAE, not +78 like S2

| Factor | Effect |
|--------|--------|
| **Shared core** | Same 15-ep schedule, model, optimizer; S9 inherits counting-first defaults (`mosaic=0.1`, `translate=0.05`, `scale=0.15`) from `_BASELINE_DEFAULTS` |
| **`close_mosaic` 3 vs 10** | Shorter mosaic-active window, longer off tail — explains small S9 gap, not collapse |
| **HSV / erasing pins** | YAML explicitly pins smoke_base photometrics; S9 uses slightly lower HSV and may get higher default erasing |
| **Contrast S2** | S2 sets `mosaic=0`, `close_mosaic=0` → **147.4 MAE** (+78.5 vs S1). S9 keeps low mosaic; gap is schedule/pinning, not “no aug” |

**Takeaway:** Committed aug YAML is documentation + explicit pinning of values already near `_BASELINE_DEFAULTS`. S9 validates that the merge path is not broken; +4.3 is consistent with `close_mosaic` tail and minor photometric deltas.

## Reproduce

```bash
PYTHONPATH=. python -c "
from pathlib import Path
import json
from scripts.train import _merge_train_config
from harchoc.train_config import effective_train_aug_merged, load_train_config_json
repo = Path('.').resolve()
for label, name in [('S1','train_aug_s1_close3_smoke.json'), ('S9','train_aug_s9_no_aug_yaml_smoke.json')]:
    resolved = load_train_config_json(repo / 'configs/experiments' / name, repo_root=repo)
    eff = effective_train_aug_merged(_merge_train_config(resolved), repo_root=repo)
    aug = {k: eff.get(k) for k in ('mosaic','close_mosaic','hsv_s','hsv_v','translate','scale','erasing','fliplr','mixup') if k in eff or True}
    print(label, json.dumps(aug, indent=2))
"
```

See also [`training_tech_scan_2026_augmentation.md`](../../docs/research/training_tech_scan_2026_augmentation.md) (S9 row) and [`leaderboard.md`](leaderboard.md).
