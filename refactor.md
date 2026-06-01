# DRY / sprawl audit

**Branch:** `pr/backlog-ci-dataset` · **Updated:** 2026-06-01

Consolidated modules live in `harchoc/*` (`deploy_filters`, `json_io`, `experiment_argv`, `hsp_eval_chain`, `gpu_queue_*`, …). Extend [`scripts/experiment.py`](scripts/experiment.py) before adding new top-level scripts ([`.cursor/rules/extend-before-add-script.mdc`](.cursor/rules/extend-before-add-script.mdc)). Historical detail: [`docs/plans/dry-refactor-plan.md`](docs/plans/dry-refactor-plan.md) and git history.

## Open items

| ID | Notes |
|----|--------|
| — | Phase 5 archive/CLI DRY done (2026-06); see [dry-refactor-plan Phase 5](docs/plans/dry-refactor-plan.md#phase-5-2026-06--archive--cli-dry). Science/GPU: [backlog Now](backlog.md#now) |

## Science vs production

Manuscript/HSP: train → eval → threshold lock → error analysis → dual metric (`experiment.py repro`).  
Production: `telegram_bot.py` / `run_infer_once.py` via `sahi_infer` + `deploy_filters` (`deploy-parity` for checks).

**Remaining GPU backlog:** P0-5 zoo, finetune trays, CV train — [backlog.md](backlog.md).

## Do not chase

- Third detect class for uncertainty (graded trust on two-class preds).
- Custom training loops replacing Ultralytics.
- Splitting `telegram_bot.py` monolith (shared logic already extracted).

*Consolidations completed 2026-05 — `unittest` green on `pr/backlog-ci-dataset`.*
