# Weak-tray fine-tune playbook

Operational guide for **P1-FINETUNE-TRAY** / study arm C in [backlog.md](../backlog.md#data-acquisition). Adapts **`models/best2.pt`** to high-error trays without training on canonical `test.txt`.

## Related docs

| Doc | Role |
|-----|------|
| [backlog.md § Data acquisition](../backlog.md#data-acquisition) | LOFO + finetune study arms |
| [EXPERIMENTS.md § Transfer fine-tune](EXPERIMENTS.md#transfer-fine-tune-tray-domain) | CLI flags |
| [domain_shift_transfer_literature.md §5–6](research/domain_shift_transfer_literature.md) | Literature (TFA, Gandhi freeze, LOFO) |
| [architecture_recommendations.md](manuscript/architecture_recommendations.md) | TFA / staged freeze backlog row |
| [reviewer gap §12](manuscript/reviewer_comments_backlog_gap.md#12-manuscript-draft--domain-adaptation-plan-discussion) | MS-DOMAIN-ADAPT narrative |
| [HSP_BASELINE_MODELS.md](HSP_BASELINE_MODELS.md) | `best2.pt` vs `classifier.pt` (gate is deploy-only) |

## Metrics (do not mix)

| Metric | Use |
|--------|-----|
| **Tray test count MAE** | Success on holdout `data/domains/test_{tray_key}.txt` |
| **Canonical test MAE** | Gate: must stay within ~**+10%** of global **61.3** ([backlog](backlog.md)) |
| **TIDE buckets** | Diagnose *error type* on weak trays before GPU (loc/bg vs miss) — [`error_test_report.json`](../reports/hsp/error_test_report.json) |

`classifier.pt` is **not** used in tray fine-tune (benchtop heads only).

## Phase 0 — Audit (CPU)

```bash
export DATASET_ROOT=/path/to/dataset

python scripts/eval_domains.py --catalog reports/domains/catalog.json \
  --write-domain-splits --domains-dir data/domains

mamba run -n harchoc python scripts/eval_domains.py --merge-tray-count-mae \
  --device cpu --locked-conf-from reports/hsp/threshold_val.json \
  --catalog reports/domains/catalog.json --out reports/domains/domain_eval.json

python scripts/experiment.py domain-tray-audit --out reports/domains/weak_tray_plan.json
```

Read `recommended_tray_keys` in [`reports/domains/weak_tray_plan.json`](../reports/domains/weak_tray_plan.json).

## Phase 1 — Train modes

| Mode | Train / val source | When |
|------|-------------------|------|
| **`tray_adapt`** (default with `--tray-key`) | `data/domains/train_{tray}` + `val_{tray}` | Adapt to one weak tray |
| **`lofo_pool`** | Canonical train/val **minus** holdout tray images | Pool model before tray-specific pass |
| **`canonical`** | Full `data/splits` (legacy) | Ablation only |

Implementation: `harchoc/finetune_tray_splits.py` — asserts **no path** in train/val appears in `data/splits/test.txt`.

## Phase 2 — Staged fine-tune (GPU)

```bash
# Stage 1 — frozen backbone (25 ep)
mamba run -n harchoc python scripts/finetune.py --stage 1 \
  --tray-key 349-10-2 --train-mode tray_adapt \
  --dataset-root "$DATASET_ROOT" \
  --out reports/transfer/finetune_349-10-2_s1.json

# Stage 2 — full unfreeze from stage-1 best.pt
mamba run -n harchoc python scripts/finetune.py --stage 2 \
  --base-weights runs/transfer/finetune_tray_s1/weights/best.pt \
  --tray-key 349-10-2 --train-mode tray_adapt \
  --dataset-root "$DATASET_ROOT" \
  --out reports/transfer/finetune_349-10-2_s2.json
```

Or via experiment router:

```bash
mamba run -n harchoc python scripts/experiment.py finetune-tray --stage 1 \
  --tray-key 349-10-2 --dataset-root "$DATASET_ROOT"
```

Dry-run (CI-safe):

```bash
python scripts/finetune.py --dry-run --tray-key 349-10-2 \
  --out reports/transfer/finetune_dry.json
```

`finetune_run.v1` records `split_plan`, `tray_eval_before` / `tray_eval_after`, and `train_split_file` paths.

## Phase 3 — Optional tray TIDE

Global TIDE is in [`tide_bucket_summary.json`](../reports/hsp/tide_bucket_summary.json). For tray-specific FP mix, export preds on `test_{tray}.txt` and run `error_analysis.py` (same HSP export conf **0.001**). High **Miss** → recall/`max_det`; high **loc+bg** → `tray_adapt` is appropriate.

## Config example

[`configs/transfer/finetune_tray_holdout.example.yaml`](../configs/transfer/finetune_tray_holdout.example.yaml) documents YAML knobs (`train_mode`, `tray_key`).

## Aug policy

Keep [`configs/aug/robustness_minimal.yaml`](../configs/aug/robustness_minimal.yaml) unchanged ([backlog § Aug closed](../backlog.md#aug-closed)).
