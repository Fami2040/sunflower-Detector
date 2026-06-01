# Archived GPU queue manifests

**Read-only history.** Do not add new jobs here. Do not point active runbooks at these manifests unless reproducing a closed program.

| File | Program |
|------|---------|
| `gpu_queue_aug_pending.json` | S0–S14 smokes (done 2026-05-30) |
| `gpu_queue_aug_confirm.json` | 100-ep aug winner confirm |
| `gpu_queue_aug_close_phase_a.json` | Tier-2 close 15 ep |
| `gpu_queue_aug_close_100ep.json` | Tier-2 close 100 ep |
| `gpu_queue_full.json` | Historical mega-queue (RT-DETR skipped, aug tails, P0-5, CV deferred) |

**Policy:** On 8 GiB GPUs, do **not** revive `gpu_queue_full.json` as the default path for RT-DETR train @ 1280 (OOM). Use active manifests below + [`docs/EXPERIMENTS.md`](../../../docs/EXPERIMENTS.md#gpu-sequential-queue).

**Active manifests** (parent directory):

- [`gpu_queue_zoo_p0_5.json`](../gpu_queue_zoo_p0_5.json) — P0-5 zoo only
- [`gpu_queue_post_zoo.json`](../gpu_queue_post_zoo.json) — domain audit + finetune
- [`gpu_queue_post_zoo_smoke.json`](../gpu_queue_post_zoo_smoke.json) — 1-ep wiring smoke

**Runner:** [`scripts/run_gpu_queue.sh`](../../../scripts/run_gpu_queue.sh) or [`scripts/run_gpu_queue.py`](../../../scripts/run_gpu_queue.py).

## Unused train JSON (`unused_train/`)

Legacy smokes with no repo references (superseded by `train_smoke_rank_15ep.json` + `aug_smoke_index.json`).
