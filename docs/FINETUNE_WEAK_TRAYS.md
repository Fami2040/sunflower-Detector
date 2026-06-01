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

## Methods paragraph (manuscript / reviewer paste)

**Status:** Protocol **defined**; GPU runs **queued** ([`gpu_queue_post_zoo.json`](../configs/experiments/gpu_queue_post_zoo.json)) — treat finetune numbers as **planned** until `reports/transfer/finetune_*_s2.json` exist.

For trays with poor held-out counting performance, we adapt the production detector (`models/best2.pt`) using images from tray-specific train/validation lists only (`data/domains/train_{tray_key}`, `val_{tray_key}`), with automated checks that no path in those lists appears in the frozen canonical test split (`data/splits/test.txt`). Weak trays are prioritized from per-tray count MAE at the validation-locked confidence already used for headline metrics (global reference **61.3** seeds/image on test; top candidates **`3a5-9`**, **`350`**, **`200-3-1`** in [`weak_tray_plan.json`](../reports/domains/weak_tray_plan.json)). Global error decomposition ([`tide_bucket_summary.json`](../reports/hsp/tide_bucket_summary.json)) guides whether tray adaptation targets localization/background false positives versus deferring finetune when missed detections dominate. Optimization uses a two-stage schedule (frozen backbone, then full-network fine-tune at reduced learning rate) aligned with staged transfer-learning practice in the literature. A finetuned checkpoint is accepted for manuscript reporting only if canonical test count MAE remains within **10%** of **61.3** while tray-holdout MAE improves.

**Evidence:** [`docs/manuscript/reviewer_comments_backlog_gap.md` §20](manuscript/reviewer_comments_backlog_gap.md#20-manuscript-draft--tray-finetuning-methods--discussion) · [`reports/hsp/p0_summary.md` § Finetuning](../reports/hsp/p0_summary.md#finetuning--domain-adaptation-planned-gpu).

---

## Metrics (do not mix)

| Metric | Use |
|--------|-----|
| **Tray test count MAE** | Success on holdout `data/domains/test_{tray_key}.txt` |
| **Canonical test MAE** | Gate: must stay within ~**+10%** of global **61.3** ([backlog](backlog.md)) |
| **TIDE buckets** | Diagnose *error type* on weak trays before GPU (loc/bg vs miss) — [`error_test_report.json`](../reports/hsp/error_test_report.json) |

`classifier.pt` is **not** used in tray fine-tune (benchtop heads only).

## Wired in `scripts/finetune.py` (P1-FINETUNE-TRAY)

| Input | Flag / module |
|-------|----------------|
| Weak-tray ranking | `--from-weak-plan`, `--weak-plan`, `--audit-trays` → `harchoc/finetune_tray_audit.py` |
| TIDE train-mode hint | `--tide-summary` → `harchoc/finetune_pipeline.finetune_tide_guidance` |
| HSP counting MAE | `--hsp-counting` (default on): export + `error_analysis` @ `--locked-conf-from` |
| Canonical gate | `--global-mae-ref`, `--canonical-gate-pct` → `finetune_outcome.canonical_gate` |
| Before/after tray MAE | `--tray-eval` → `finetune_outcome.tray_holdout` (delta per tray) |
| Debug smoke | `--debug` (2 epochs) |

Run metadata schema: `finetune_run.v1` with `weak_tray_plan`, `tide_guidance`, `finetune_outcome`.

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

## GPU queue (post-zoo)

Run **after** P0-5 **`zoo_matrix_p0_5`** completes (`zoo_yolo_only` → [`matrix_train.json`](../reports/hsp/matrix_train.json)). One GPU, sequential — same runner as aug/zoo queues.

| Step | Action |
|------|--------|
| 0 | `./scripts/kill_stray_gpu_jobs.sh` |
| 1 | P0-5 in [`gpu_queue_zoo_p0_5.json`](../configs/experiments/gpu_queue_zoo_p0_5.json) (`zoo_matrix_p0_5`; legacy: [`archive/gpu_queue_full.json`](../configs/experiments/archive/gpu_queue_full.json)) |
| 2 | Post-zoo manifest below |
| 3 | Keep **`models/best2.pt`** unless a zoo or finetune run beats **61.3** canonical test MAE |

**Manifest:** [`configs/experiments/gpu_queue_post_zoo.json`](../configs/experiments/gpu_queue_post_zoo.json) (repro/preflight → domain audit → finetune weak trays). Job IDs are defined in that file; typical pattern:

| Job id (example) | Purpose |
|------------------|---------|
| `repro_post_zoo` | HSP / reviewer-2 regen (`experiment.py repro` stages) |
| `manuscript_preflight` | Figures, tables, docx chain |
| `domain_tray_audit_refresh` | Refresh [`weak_tray_plan.json`](../reports/domains/weak_tray_plan.json) (`eval_domains` splits + merge count MAE + `domain-tray-audit`; CPU except merge step uses mamba) |
| `finetune_weak_tray_{1,2,3}` | Stage-1 tray adapt (`finetune_tray_stage1.json` + `finetune_stage1.yaml`: 25 ep, lr 0.001, freeze backbone) for plan trays **`3a5-9`**, **`350`**, **`200-3-1`** |
| `finetune_weak_tray_{1,2,3}_s2` | Stage-2 full unfreeze (`finetune_tray_stage2.json` + `finetune_stage2.yaml`: 25 ep, lr 0.0005); `--base-weights` = `runs/transfer/finetune_<tray>_s1/weights/best.pt` after matching s1 job |

```bash
export DATASET_ROOT="$(pwd)/data/raw/extracted/dataset"
export HARCHOC_EXPORT_DEVICE=cpu

./scripts/kill_stray_gpu_jobs.sh
GPU_QUEUE_MANIFEST=configs/experiments/gpu_queue_post_zoo.json \
  ./scripts/run_gpu_queue.sh dry-run

GPU_QUEUE_MANIFEST=configs/experiments/gpu_queue_post_zoo.json \
  ./scripts/run_gpu_queue.sh run
# or: ./scripts/run_gpu_queue.sh resume --job finetune_3a5-9_s1
```

Manual finetune (outside queue) uses the same env:

```bash
mamba run -n harchoc python scripts/finetune.py --stage 1 \
  --tray-key 3a5-9 --train-mode tray_adapt \
  --dataset-root "$DATASET_ROOT" \
  --out reports/transfer/finetune_3a5-9_s1.json
```

State: `reports/gpu_queue/run_state.json` · logs: `reports/gpu_queue/logs/{job_id}/`. Ops: [EXPERIMENTS § GPU queue](EXPERIMENTS.md#gpu-sequential-queue) · [backlog § GPU runbook](../backlog.md#gpu-runbook).

**Post-zoo queue kinds** ([`gpu_queue_post_zoo.json`](../configs/experiments/gpu_queue_post_zoo.json)): `domain_tray_audit_refresh` (CPU: `eval_domains.py --write-domain-splits` → `--merge-tray-count-mae` → `experiment.py domain-tray-audit`); `finetune_tray` with `skip_if_missing_plan` (skip when `weak_tray_plan` missing/empty); `head_roi_eval` (eval-only smoke at locked conf via `eval.py --dry-run`).

## Aug policy

Keep [`configs/aug/robustness_minimal.yaml`](../configs/aug/robustness_minimal.yaml) unchanged ([backlog § Aug closed](../backlog.md#aug-closed)).
