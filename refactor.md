# DRY / sprawl / reinvention audit

**Purpose:** Living map of duplication, script sprawl, and library reinvention in this repo.  
**Audited:** 2026-05-29 · branch `pr/backlog-ci-dataset`  
**Last consolidated:** 2026-05-29 (reconcile: §4 GPU gaps, Partial→defer links, Verified footer)  
**Related:** [backlog.md](backlog.md), [extend-before-add-script](.cursor/rules/extend-before-add-script.mdc), [silent_failure_audit_scripts.md](docs/debug/silent_failure_audit_scripts.md)

---

## Already centralized (do not re-litigate)

| Area | Location |
|------|----------|
| GT/pred matching, counting | `harchoc/detection_match.py` → `threshold_sweep.py`, `error_analysis.py` |
| Bench / zoo | `harchoc/bench_config.py`, `model_zoo.py`, `data_yaml.py` |
| Matrix orchestration | `scripts/benchmark_matrix.py` calls `train.main` / `eval.main` (not inline Ultralytics loops) |
| Matrix row metadata | `harchoc/bench_config.bench_matrix_metadata()` → dry-run / train JSON `matrix_metadata` |
| CLI / dataset / JSON write | `scripts/_common_cli.py`, `harchoc/experiment_config.py`, `scripts/experiment.py` |
| JSON load | `harchoc/json_io.py` → `_common_cli.read_json` / `read_json_dict`; wrappers removed in sweep/error/dual-metric (2026-05-29 batch) |
| Deploy post-filter | `harchoc/deploy_filters.py` → `run_infer_once.py`, `telegram_bot.py` (`DeployFilterConfig.resolve()`) |
| SAHI slice infer | `harchoc/sahi_infer.py` → `run_infer_once.py`, `telegram_bot.py` (`SahiSliceConfig`, `run_sliced_prediction`) |
| Post-train eval policy | `harchoc/post_train_eval.py` → `train.py`, `benchmark_matrix.py`, `rtdetr_smoke.py` |
| Experiment argv | `harchoc/experiment_argv.py` → all subcommands (splits, describe, eval, cv-eval, benchmark, train, map-cpu, dual-metric, repro, gradcam, deploy-parity, tune-sahi); `experiment.py` dispatch-only (**DRY-EXPERIMENT-ARGV** Done) |
| Deploy vs HSP parity | `harchoc/deploy_hsp_parity.py` → `experiment.py deploy-parity` → `reports/hsp/deploy_hsp_parity.json` (**R-SCI-2** Done) |
| Domain eval | `harchoc/domain_eval.py` + `domain_eval_loop.py` → `eval_domains.py` (`--write-domain-splits`, `--run-tray-eval`, `--run-all-trays` CPU + dry-run plan) | **Done** — [**P1-DOMAIN-EVAL**](backlog.md) (`--run-all-trays` mAP + `--merge-tray-count-mae`); **MS-GEN** Partial (§4 repo draft **Done**; **P1-DOMAIN-TAGS** scaffold) |
| GPU probe | `harchoc/gpu_probe.py` + `scripts/check_gpu.py` (`sanity` / `smoke-ultralytics` subcommands) |
| Threshold lock, dual-metric | `harchoc/threshold_lock.py`, `dual_metric_report.py` |
| Strict ML surfacing | `harchoc/strict_ml.py`, `scripts/strict_ml_smoke.py` |
| Pre-train gate | `scripts/pre_train_gate.py` (manifest + unittest; `--full` + `HARCHOC_STRICT_ML=1`) |
| Agent batch verify | `scripts/update_agent_batch_verify.py` → `reports/hsp/agent_batch_verify.json` |

**Rule:** Prefer extending `experiment.py` / existing scripts before adding `scripts/*.py` (see cursor rule).

---

## 1. Hard duplication (copy-paste)

| Hotspot | Files | Issue | Status |
|--------|--------|--------|--------|
| `_path` bootstrap | Every `scripts/*.py` | 29× identical try/import | **Done** — `harchoc/script_entry.py` (`bootstrap_repo_imports`); all `scripts/*.py` entrypoints migrated except `_path.py`, `bootstrap_env.py`, `_common_cli.py` |

**Resolved (2026-05-29):** deploy post-filter → [`harchoc/deploy_filters.py`](harchoc/deploy_filters.py) (**DRY-DEPLOY-FILTER**); `_load_json` → [`harchoc/json_io.py`](harchoc/json_io.py) (**DRY-JSON-IO**). See [§6](#6-recommended-consolidation-order) and [backlog DRY table](backlog.md).

---

## 2. Sprawl — entrypoints and scaffolds

**Scale:** 29 executable files under `scripts/`, plus root `run_infer_once.py`, `telegram_bot.py`.

| Pattern | Examples | Status | Notes |
|--------|----------|--------|-------|
| Scaffold / CV eval | `cv_eval.py` | **Done** (**DRY-SPRAWL-CV**) | Routed via `experiment.py cv-eval` → `argv_for_cv_eval`; `experiments.v1` `run.kind: cv_eval`; implementation in `scripts/cv_eval.py`. Per-fold GPU train → **backlog P1-CV-TRAIN** |
| Finetune tray eval | `finetune.py` → `train.main` + `harchoc/finetune_tray_eval.py` | **Done** (**P1-FINETUNE-LOOP**) | Staged unfreeze `--stage 1|2` + `finetune_stage{1,2}.yaml`; `eval_domains --write-domain-splits` → 133 lists; tray eval path verified (dry-run + CPU `--run-tray-eval` on `349-10-2`); 2-ep GPU smoke `--no-tray-eval` → `finetune_smoke_s1/best.pt`. Discussion draft **Done** (**MS-DOMAIN-ADAPT**); full 25+25 ep GPU metrics open |
| GPU probe proliferation | `check_gpu.py`, `gpu_sanity.py`, `gpu_smoke_ultralytics.py`, `strict_ml_smoke.py`, `rtdetr_smoke.py` | **Done** (**DRY-GPU-SMOKE**) | Runbook: `check_gpu sanity` / `smoke-ultralytics`; legacy shims **removed** |
| `experiment.py` argv layer | `harchoc/experiment_argv.py` | **Done** (**DRY-EXPERIMENT-ARGV**) | All subcommands including `deploy-parity`, `gradcam`, `cv-eval`; `experiment.py` dispatch-only |
| Legacy parallel tree | `yolo-sunflower-seed-detector/training.py` | **Done** (**DRY-LEGACY-TRAIN**) | One-line shim → `scripts/train.py` |
| Monolith bot | `telegram_bot.py` (~1.7k lines) | **Done** (parity tooling) | `deploy_filters` + `sahi_infer` **Done**; conf + per-image SAHI vs full-frame counts via `deploy_hsp_parity.py` + `experiment.py deploy-parity --sample-images N` (**R-SCI-2** Done); optional `run_infer_once.py --fullframe-export`; bot file not split (**Defer** §7) |
| Matrix Ultralytics parse | `benchmark_matrix._ultralytics_eval_one` | **Done** (**DRY-MATRIX-RESULTS**) | `results_dict_error` on attr/parse fail (no silent `None`) |

---

## 3. Reinventing the wheel vs env / domain libs

| Area | Repo | Ecosystem alternative | Notes |
|------|------|----------------------|--------|
| Isotonic calibration | `harchoc/isotonic.py` (PAVA) | `sklearn.isotonic.IsotonicRegression` | Used from `platt.py`; sklearn path when available (**DRY-CALIB** Done) |
| Platt scaling | `harchoc/platt.py` | `sklearn.linear_model.LogisticRegression` / `scipy.optimize` | sklearn sigmoid when available; scipy NLL fallback (**DRY-CALIB** Done) |
| TIDE ΔAP | `harchoc/tide_summary.py` (count-share proxy + COCO export) | [`tidecv`](https://github.com/dbolya/tide) | **DRY-TIDE** **Done** — `--tidecv` + `export_coco_*_for_tide` + `tidecv_compare.v1`; official GPU ΔAP manuscript table → **P1-TIDE** |
| Grad-CAM | `harchoc/gradcam_panel.py` | [pytorch-grad-cam](https://github.com/jacobgil/pytorch-grad-cam) | **Done** — custom YOLO train-graph on FP crops justified; canonical `experiment.py gradcam` → `make_figures` (`argv_for_gradcam`); routing doc [`docs/manuscript/gradcam_routing.md`](docs/manuscript/gradcam_routing.md) |
| Bootstrap CI | `harchoc/stats_ci.py` | scipy optional | **Done** — bootstrap when scipy available, percentile fallback otherwise; no change needed |
| KS drift | `split_drift.py` | `scipy.stats.ks_2samp` when available | **Done** — `--with-ks` requires scipy; dual path; no change needed |

**Correct reuse (do not reimplement):** Ultralytics YOLO train/val, optional SuperGradients — sprawl is in wrappers, not the detector API.

---

## 4. Science pipeline vs production (wiring gap)

```text
Manuscript/HSP:  train → post_train_eval (CPU default) → eval → threshold_sweep
                         → error_analysis → dual_metric → make_figures
                         ↑ locked conf (--locked-conf-from), export JSON, metric_roles
                         ↑ experiment.py map-cpu (SCI-MAP-CPU): test mAP on CPU + dual_metric row

Production:      telegram_bot / run_infer_once → harchoc/sahi_infer + deploy_filters
                         ↑ DeployFilterConfig.resolve() (HARCHOC_LOCKED_CONF*)
                         ↑ experiment.py deploy-parity --sample-images N: per-image SAHI vs full-frame @ locked conf
```

**Remaining gap (GPU backlog only):** full zoo matrix train [**P0-5**](backlog.md); per-fold CV GPU training [**P1-CV-TRAIN**](backlog.md); full finetune stage1→stage2 + before/after tray metrics (GPU follow-up; **MS-DOMAIN-ADAPT** repo draft **Done**). Science partial deferred in [backlog](backlog.md): **MS-GEN** repo draft **Done** (**P1-DOMAIN-TAGS** tags TBD); **P1-FINETUNE-LOOP** wiring **Done**; deploy parity tooling **R-SCI-2** Done.

---

## 5. Config / artifact sprawl (operational)

- **Triple config:** `configs/bench/*.yaml` + runtime `train_bench` (`bench_config` + `matrix_rows`) + aug YAML — aug smokes use `aug_smoke_index.json` + `train_smoke_rank_15ep.json` (**DRY-AUG-SMOKE-CONFIG**); committed exceptions S9–S13 + sweeps only.
- **Root sprawl:** deploy SAHI grid — **Done** via `experiment.py tune-sahi` dry-run argv only (removed `tune_sahi_params.py`).
- **Reports:** canonical `reports/hsp/` on disk; tracked git content is `README.md` only — generated `reports/hsp/*.md`, `reports/aug_smoke/*.{md,json}` removed (**DRY-TRACKED-REPORTS**).
- **Docs:** `docs/research/*` + `docs/manuscript/*` — intentional synthesis, not code DRY.

---

## 6. Recommended consolidation order

DRY consolidations here support [backlog.md § Model improvement stack](backlog.md#model-improvement-stack-test-count-mae) **steps 1–3**: shared JSON I/O (`DRY-JSON-IO`), deploy post-filter parity (`DRY-DEPLOY-FILTER`, `DRY-BOT-LOCKED-CONF`), and manuscript↔deploy conf bridge (`DRY-EVAL-BRIDGE`). Count MAE at val-locked conf remains the stack success metric — not val mAP alone.

| Pri | ID | Status | Evidence |
|-----|-----|--------|----------|
| P0 | `DRY-DEPLOY-FILTER` | **Done** | [`harchoc/deploy_filters.py`](harchoc/deploy_filters.py) · [backlog](backlog.md) |
| P1 | `DRY-JSON-IO` | **Done** | [`harchoc/json_io.py`](harchoc/json_io.py) · [backlog](backlog.md) |
| P1 | `DRY-MATRIX-RESULTS` | **Done** | `results_dict_error` in `benchmark_matrix._ultralytics_eval_one` · [backlog](backlog.md) |
| P1 | `DRY-GPU-SMOKE` | **Done** | `check_gpu.py sanity` / `smoke-ultralytics` · [backlog](backlog.md) |
| P1 | `DRY-FINETUNE` | **Done** | [`scripts/finetune.py`](scripts/finetune.py) → `train.main` · [backlog](backlog.md) |
| P2 | `DRY-CALIB` | **Done** | `platt.py` sklearn + PAVA · [backlog](backlog.md) |
| P2 | `DRY-TIDE` | **Done** | COCO export adapter + `tidecv_compare.v1` (`build_tidecv_compare`, structured skip); official ΔAP GPU manuscript table **P1-TIDE** · [backlog](backlog.md) |
| P2 | `DRY-EVAL-BRIDGE` | **Done** | `eval.py --locked-conf-from` · [backlog](backlog.md) |
| P2 | `DRY-EXPERIMENT-ARGV` | **Done** | [`harchoc/experiment_argv.py`](harchoc/experiment_argv.py): all subcommands; shared `_argv_dataset` internal; `experiment.py` dispatch-only · [backlog](backlog.md) |
| P2 | `DRY-BOT-LOCKED-CONF` | **Done** | `DeployFilterConfig.resolve()` · [backlog](backlog.md) |
| P2 | `DRY-LEGACY-TRAIN` | **Done** | [`yolo-sunflower-seed-detector/training.py`](yolo-sunflower-seed-detector/training.py) shim · [backlog](backlog.md) |
| P2 | `DRY-SPRAWL-CV` | **Done** | `experiment.py cv-eval` → `argv_for_cv_eval`; `experiments.v1` `run.kind: cv_eval` · per-fold GPU train remains **P1-CV-TRAIN** · [backlog](backlog.md) |
| P1 | `DRY-EXPORT-PROTOCOL` | **Done** | [`harchoc/hsp_export_protocol.py`](harchoc/hsp_export_protocol.py) · shared export conf/IoU/split/device |
| P1 | `DRY-MAMBA-RUN` | **Done** | `run_repo_python` / `repo_python_cmd` in [`harchoc/ml_env.py`](harchoc/ml_env.py) |
| P0 | `DRY-HSP-CHAIN` | **Done** | [`harchoc/hsp_eval_chain.py`](harchoc/hsp_eval_chain.py) · aug smoke / SG / external |
| P0 | `DRY-ERROR-CORE` | **Done** | [`harchoc/error_analysis_core.py`](harchoc/error_analysis_core.py) · thin `scripts/error_analysis.py` |
| P2 | `DRY-TRACKED-REPORTS` | **Done** | `.gitignore` · `reports/README.md` — metrics local/CI only |
| P1 | `DRY-PREDS-DEDUP` | **Done** | [`harchoc/equivalence_index.py`](harchoc/equivalence_index.py) · queue + leaderboard |
| P1 | `DRY-TRAIN-BENCH-RUNTIME` | **Done** | [`harchoc/bench_config.py`](harchoc/bench_config.py) · matrix_rows + `train_bench_base.json` |
| P1 | `DRY-AUG-SMOKE-CONFIG` | **Done** | [`harchoc/aug_smoke_train.py`](harchoc/aug_smoke_train.py) · index + `train_smoke_rank_15ep.json`; S9–S13 committed |
| P1 | `DRY-GPU-QUEUE-SPLIT` | **Done** | [`gpu_queue.py`](harchoc/gpu_queue.py) facade · [`gpu_queue_manifest.py`](harchoc/gpu_queue_manifest.py) · [`gpu_queue_dedup.py`](harchoc/gpu_queue_dedup.py) · [`gpu_queue_stages.py`](harchoc/gpu_queue_stages.py) · [`gpu_queue_runner.py`](harchoc/gpu_queue_runner.py) · [`gpu_queue_skip.py`](harchoc/gpu_queue_skip.py) |

---

## 7. What not to chase

- Third YOLO class for “uncertainty” — **MS-FUZZY-BOUND** uses graded trust on 2-class preds (locked conf + score band + FP taxonomy); not a new detect class.
- New top-level scripts for workflows that fit `experiment.py` subcommands.
- Replacing Ultralytics with custom training loops.
- **`telegram_bot.py` monolith split** — **Defer**; shared deploy logic lives in `deploy_filters` / `sahi_infer` / `deploy_hsp_parity`; bot wiring unchanged.
- **pytorch-grad-cam** — **Deferred** (not in CI/env). Custom `harchoc/gradcam_panel.py` covers FP-crop mosaic; see [`docs/manuscript/gradcam_routing.md`](docs/manuscript/gradcam_routing.md).

---

*Update this file when a consolidation lands; link commit or backlog Done row in the table.*

Verified: `compileall` OK; `unittest discover` **530+** tests OK — 2026-05-30 (DRY phase-2: aug runtime, gpu_queue split, shim removal).
