# Training tech scan (2026): DETR-family, dense seeds, query caps

> **Backlog alignment:** [Model improvement stack](../../backlog.md#model-improvement-stack-test-count-mae) steps **1** (eval `max_det=3000`, P0-1 **Done**), **5** (**P0-4**→**P0-5**, **P1-ZOO-PARITY**), and **7** (**P1-RTDETR-Q** — **matrix zoo row only**; does **not** replace `models/best2.pt` YOLO path).

Literature and product scan (May 2026) against **harchoc** sunflower-seed training: Ultralytics zoo @ `imgsz=1280`, bench matrix, RT-DETR-L, and dense-tray query limits. Read-only code review; no train-script changes implied here.

**Scope:** `scripts/train.py`, `scripts/benchmark_matrix.py`, `configs/experiments/train_bench_*.json`, `configs/bench/*.yaml`, `configs/aug/robustness_minimal.yaml`, `harchoc/rtdetr_limits.py`, `harchoc/train_kwargs.py`.

---

## Current setup summary

### Training entrypoints and config layering

| Layer | Role |
|-------|------|
| `configs/experiments/train_bench_base.json` | Shared recipe: 100 epochs, `imgsz=1280`, AdamW (`lr0=2e-4`, `lrf=0.01`), `conf=0.05`, `iou=0.3`, `max_det=3000`, `patience=50`, `aug_config` → `robustness_minimal.yaml` |
| `configs/experiments/train_bench_<model>.json` | Per-model overlay (`extends` base): `model`, `batch`, notes; RT-DETR adds query-cap metadata |
| `configs/bench/*.yaml` | Matrix zoo: backend, `groups`, `infer.imgsz`, `infer.max_det`, epochs/patience; pairs with committed `train_bench_*.json` |
| `scripts/train.py` | Merges JSON + aug YAML → `YOLO(model).train(**ultralytics_train_kwargs)` |
| `scripts/benchmark_matrix.py` | Plan / train / eval chain; forwards `imgsz`, eval `max_det` from bench + train `eval` block |

**Zoo (Ultralytics + one SG):** `yolov8n/s/m/l`, `yolov10s`, `yolo11s`, `rtdetr-l.pt`, `yolo_nas_s` (SuperGradients). RT-DETR is the only DETR-family member; it sits in bench group `sota_2026` (`configs/bench/rtdetr_l_default.yaml`).

### Dense seed detection (@ 1280)

- **Task:** 2-class YOLO boxes (developed / aborted), very dense trays; counting-first eval on frozen `data/splits/`.
- **Resolution:** All bench recipes lock **`imgsz=1280`** (train and `infer.imgsz` in YAML). This is appropriate for 5–15 px seeds vs COCO-default 640 DETR pretraining.
- **Post-processing (train):** `conf=0.05`, `iou=0.3`, **`max_det=3000`** in `train_bench_base.json` — aligned with counting literature in `docs/research/augmentation_robustness_literature.md`.
- **Post-processing (matrix eval):** `eval.max_det: 3000` in base JSON and **`infer.max_det: 3000`** in bench YAML (P0-1 **Done**). RT-DETR rows still capped by **`num_queries=300`** decoder slots — separate from YOLO `max_det` parity.
- **Tiling:** Bench YAML sets `infer.tiling: none` (reserved field). **SAHI sliced inference exists only on the deploy path** (`run_infer_once.py`, `telegram_bot.py`), not in matrix train/eval.

### Augmentation (committed)

`configs/aug/robustness_minimal.yaml` (single aug recipe in repo):

- `mosaic: 0.1`, `close_mosaic: 15`, `mixup/cutmix: 0`
- Photometric-heavy: `hsv_*`, mild `translate`/`scale`, `fliplr=0.5`, `erasing=0.2`
- Schedule guard: `epochs - patience >= close_mosaic` (100 − 50 ≥ 15)

Conservative for **count integrity**; no DEIM-style dense matching or dynamic aug schedules.

### RT-DETR / query cap (set-prediction limit)

| Item | Value | Source (2026-05-29) |
|------|--------|---------------------|
| Model | `rtdetr-l.pt` via Ultralytics (**RT-DETR v1**; not RT-DETRv2/v4 weights in zoo) | `train_bench_rtdetr-l.json` |
| `num_queries` | **300** | `ULTRALYTICS_RTDETR_DEFAULT_NUM_QUERIES` in `harchoc/rtdetr_limits.py`; committed JSON |
| Documented peak GT / image | **1015** | `SUNFLOWER_DOCUMENTED_PEAK_GT_BOXES_PER_IMAGE` in `harchoc/rtdetr_limits.py`; optional JSON / `HARCHOC_RTDETR_PEAK_GT_BOXES_PER_IMAGE` |
| Policy (committed) | `accept_rtdetr_query_truncation: true` | `train_bench_rtdetr-l.json`; `validate_rtdetr_query_cap()` warns instead of exiting when accepted |
| Train kwargs | `num_queries` whitelisted in `harchoc/train_kwargs.py` | Documents intent; does **not** retune frozen `rtdetr-l.pt` decoder width |

**Backlog status (`backlog.md`, 2026-05-29):** RT-DETR query-cap **policy + train guard + smoke harness** = **Done**. **Next (P1-RTDETR-Q, stack step 7):** custom Ultralytics YAML with `num_queries` **≥ 1024** (target ≥ peak **1015**) + 15-ep GPU smoke — **zoo/matrix ablation only**; production detection + HSP counting stays on **`best2.pt`** (YOLO). Align eval `max_det` with raised queries (**P1-RTDETR-MAXDET**).

#### Policy (now) vs ablation (later)

| Track | Intent | Repo state |
|-------|--------|------------|
| **Policy — accept truncation** | Run matrix/zoo with honest ceiling: 300 decoder slots vs peak GT **1015** (~70% of tray GT structurally unreachable). | **Done:** keep `num_queries=300`, `accept_rtdetr_query_truncation=true`, guard in `scripts/train.py` / matrix dry-run; plan rows expose `matrix_metadata` (`nms_free`, query-cap fields) via `harchoc/bench_config.bench_matrix_metadata()`. See `docs/training_budget.md`. |
| **Ablation — raise query capacity** | Test whether DETR is viable for counting after capacity matches trays. | **Next (P1-RTDETR-Q):** fork `rtdetr-l.yaml` (or equivalent), set decoder e.g. `RTDETRDecoder, [nc, 1024, …]` per Ultralytics thread, **full retrain** @ `imgsz=1280`; smoke 1024 then 1536 if VRAM allows; set eval/export `max_det` ≥ `num_queries`. **Zoo row only** — does not replace `best2.pt`. Do **not** block zoo on this. |

**Implication:** `conf` / eval `max_det` cannot exceed the decoder’s query slots. [Ultralytics #20688](https://github.com/ultralytics/ultralytics/issues/20688) (May–Jun 2025, closed stale): changing only `num_queries` in train kwargs or a partial YAML edit left output at **300**; maintainer fix is decoder args **`[nc, num_queries, max_det]`** (e.g. `[nc, 256, 5000]`) **plus** `max_det` at predict — not `RTDETRDecoder, [nc, 500]` alone. Stock `.pt` weights stay 300-wide until retrained on the new YAML.

### What we do **not** use (DETR / training stack)

- RT-DETRv2 bag-of-freebies (selective multi-scale sampling, discrete sampling op, scale-adaptive aug/HPs)
- RT-DETRv4 VFM distillation (DINOv3 teacher, DSI/GAM) — Nov 2025 release, separate PyTorch repo
- D-FINE (FDR + GO-LSD), DEIM / DEIMv2 (dense O2O matching, MAL; CVPR 2025)
- LW-DETR / RF-DETR (ViT encoder, DINOv2/3 teachers; Roboflow stack)
- Dynamic / density-conditioned queries (DQ-DETR, Dome-DETR, UHR-DETR)
- Ultralytics RT-DETR inference knobs in recipes: `eval_idx` (early decoder exit), `num_queries` sweeps (documented in backlog, not in configs)
- `amp` / `grad_clip` in [`train_bench_rtdetr-l.json`](../../configs/experiments/train_bench_rtdetr-l.json) — allowed bench parity keys (`BENCH_PARITY_ALLOWED_DIFF_KEYS`; Ultralytics [#7594](https://github.com/ultralytics/ultralytics/issues/7594))
- SAHI or slice-assisted **training** in matrix
- Pretrain on Objects365 / large-scale DETR schedules (RT-DETR hub tables show +2–3 AP from O365; not in our manifest workflow)

---

## Gap analysis

### 1. Transformer / RT-DETR generation gap

We benchmark **Ultralytics RT-DETR-L (v1)** while the open-source line has moved through **RT-DETRv2** (Jul 2024, same-speed AP gains), **D-FINE** (Oct 2024, localization refinement on v2), **DEIM** (Dec 2024 / CVPR 2025, ~50% faster convergence, +AP especially on **small objects** when paired with D-FINE), and **RT-DETRv4** (Oct–Nov 2025, VFM distillation at **no inference cost**). None of these are in `configs/bench/` or `train_bench_*.json`.

**Practical gap:** Staying on `rtdetr-l.pt` leaves ~1–4+ COCO AP (and unknown count-MAE delta) on the table unless we port weights/recipes or add a second training backend.

### 2. Query cap vs dense trays (dominant for sunflowers)

| Mechanism | Our setup | Literature / product |
|-----------|-----------|----------------------|
| Hard cap | 300 queries | Peak GT **1015**; CrowdHuman / AI-TOD-V2 discussions use **500–1500+** for dense scenes |
| Dynamic queries | None | DQ-DETR (300/500/900/1500 by density); Dome-DETR PAQI |
| Eval `max_det` | 3000 (YOLO/matrix) | Train/export 3000; RT-DETR still limited by **300 decoder queries** |
| NMS-free set prediction | RT-DETR | YOLO zoo uses NMS; different failure mode under density |

This is the **largest structural mismatch** for RT-DETR on sunflower trays. The repo **accepts** that mismatch for committed bench runs (`accept_rtdetr_query_truncation`); the **1024+ custom-YAML ablation** is the planned fix path (`backlog.md` P1), not a blocker for policy/smoke work.

### 3. Dense small-object detection

| Technique | In repo? | Gap |
|-----------|----------|-----|
| High `imgsz` (1280) | Yes | Good; RT-DETR still COCO-640-pretrained — finetune @1280 is right but under-ablated (`imgsz` 640 vs 1280 open in backlog) |
| Low mosaic / no mixup | Yes | Aligned with counting aug review |
| SAHI / tiling | Deploy only | Matrix `tiling: none`; no train-time slice aug (RT-DETR repo PR #468; Ultralytics SAHI guide 2025) |
| DEIM small-object gains | No | DEIM-D-FINE-X reports **+1.5 AP on small** vs D-FINE-X (COCO); relevant if we adopt that stack |
| End-to-end UHR transformers | No | UHR-DETR (Apr 2026) targets UHR RS imagery — analogous problem (memory vs small objects) but new codebase |

### 4. Training recipe / convergence

- **DEIM:** Dense O2O + MAL — halves training time to strong AP on RT-DETRv2/D-FINE; we run **100-epoch** fixed schedules with **no** hybrid matching.
- **RT-DETRv2 training strategy:** Dynamic aug + scale-adaptive hyperparameters — we use static `robustness_minimal` only.
- **D-FINE:** Better box regression under IoU metrics; may or may not move **count MAE** (needs empirical test).
- **Eval vs train postprocess (YOLO):** Matrix eval `max_det=3000` matches train (P0-1 **Done**). RT-DETR comparisons still confounded by **300 query slots** until P1-RTDETR-Q ablation.

### 5. Integration / ops (already strong)

Strengths to preserve: `train_bench` + bench YAML parity tests, `HARCHOC_MAX_*` budgets, RT-DETR query-cap guard, aug YAML merge, domain-aware splits tooling. Any new detector family should plug into **`benchmark_matrix.py` + committed JSON**, not ad-hoc scripts (per repo rules).

---

## Top 5 recommendations (effort / impact)

| Rank | Recommendation | Impact | Effort | Notes |
|:--:|----------------|--------|--------|-------|
| **1** | **Ablation: custom YAML `num_queries` ≥ 1024 (≥ peak 1015), full retrain** | **Very high** for RT-DETR count/recall | **Medium** (GPU + YAML + smoke) | **Next** stack step **7** / **P1-RTDETR-Q** (zoo only; `best2.pt` unchanged). Decoder `RTDETRDecoder, [nc, num_queries, max_det]` per [#20688](https://github.com/ultralytics/ultralytics/issues/20688); 15-ep smoke; eval `max_det` ≥ queries. [Ultralytics RT-DETR docs](https://docs.ultralytics.com/models/rtdetr). |
| **2** | **Maintain eval/export `max_det=3000` for YOLO matrix rows** | **High** for fair zoo + count metrics | **Done** (P0-1) | Bench YAML + `train_bench_base.json` use 3000; S14 documents truncation at 300. RT-DETR eval cap follows `num_queries` (**P1-RTDETR-MAXDET** after P1-RTDETR-Q). |
| **3** | **Add matrix eval path for SAHI (or RT-DETR upstream slice infer) @ 1280 — optional group `deploy_parity`** | **High** for deploy-aligned science | **Medium** | Bridges train (full frame) vs `run_infer_once.py` (slice 500, overlap 0.35). Ultralytics [SAHI guide](https://docs.ultralytics.com/guides/sahi-tiled-inference/); consider slice-aware fine-tuning later (ASAHI, arXiv:2604.19233). |
| **4** | **Parallel track: RT-DETRv2-L or D-FINE-L + DEIM training recipe (official repos), 15–30 ep smoke vs Ultralytics RT-DETR** | **High AP / convergence**; count MAE TBD | **High** | Not drop-in for `YOLO('rtdetr-l.pt')`. DEIM reports +0.7 AP / −30% train cost on D-FINE; strong **small-object** story. Only pursue if P0 query cap experiment shows DETR worth keeping. |
| **5** | **RT-DETRv4-style VFM distillation OR RT-DETRv2 training freebies (dynamic aug / scale-adaptive HPs) on chosen DETR baseline** | **Medium–high** AP; **zero** inference cost (v4) | **High** (v4); **Medium** (v2 aug only) | v4: [arXiv:2510.25257](https://arxiv.org/abs/2510.25257), [RT-DETRv4 repo](https://github.com/RT-DETRs/RT-DETRv4). v2: [arXiv:2407.17140](https://arxiv.org/abs/2407.17140). Defer until query cap and eval caps are sane. |

**Honorable mention (low effort):** Ultralytics `eval_idx` / reduced `num_queries` **latency** sweeps after count baseline — deploy tuning, not training, but documented for RT-DETR-L T4 ([Ultralytics RT-DETR](https://docs.ultralytics.com/models/rtdetr)).

---

## Links

### RT-DETR family

- RT-DETR (v1): [arXiv:2304.08069](https://arxiv.org/abs/2304.08069) · [GitHub lyuwenyu/RT-DETR](https://github.com/lyuwenyu/RT-DETR)
- RT-DETRv2: [arXiv:2407.17140](https://arxiv.org/abs/2407.17140)
- RT-DETRv4: [arXiv:2510.25257](https://arxiv.org/abs/2510.25257) · [GitHub RT-DETRs/RT-DETRv4](https://github.com/RT-DETRs/RT-DETRv4)
- Ultralytics RT-DETR: [docs](https://docs.ultralytics.com/models/rtdetr) · [predict API](https://docs.ultralytics.com/reference/models/rtdetr/predict/)

### D-FINE / DEIM / LW-DETR / RF-DETR

- D-FINE: [arXiv:2410.13842](https://arxiv.org/abs/2410.13842)
- DEIM (CVPR 2025): [arXiv:2412.04234](https://arxiv.org/abs/2412.04234) · [CVPR poster](https://cvpr.thecvf.com/virtual/2025/poster/32773)
- LW-DETR: [arXiv:2406.08356](https://arxiv.org/abs/2406.08356) (LW-DETR paper; cited in D-FINE)
- RF-DETR / Roboflow crowd tuning: [issue #674](https://github.com/roboflow/rf-detr/issues/674)

### Query cap / dense & tiny objects

- Ultralytics `num_queries` / `max_det` (decoder `[nc, num_queries, max_det]`): [issue #20688](https://github.com/ultralytics/ultralytics/issues/20688)
- DQ-DETR (dynamic queries): [arXiv:2404.03507](https://arxiv.org/abs/2404.03507)
- Dome-DETR: density-oriented queries (2025; see ResearchGate / arXiv listings)
- UHR-DETR: [arXiv:2604.21435](https://arxiv.org/abs/2604.21435) (Apr 2026)

### Slicing / SAHI

- SAHI: [GitHub obss/sahi](https://github.com/obss/sahi) · [Ultralytics SAHI guide](https://docs.ultralytics.com/guides/sahi-tiled-inference/)
- ASAHI: [arXiv:2604.19233](https://arxiv.org/abs/2604.19233) (Apr 2026)

### Internal

- `configs/experiments/train_bench_base.json`, `train_bench_rtdetr-l.json`
- `configs/bench/rtdetr_l_default.yaml`, `configs/aug/robustness_minimal.yaml`
- `harchoc/rtdetr_limits.py`, `docs/research/augmentation_robustness_literature.md`
- `backlog.md` (P0-4/5 zoo, P1-RTDETR-Q step 7, SAHI deploy split)

---

## What NOT to adopt (hype / poor fit)

| Item | Why skip or defer |
|------|-------------------|
| **RF-DETR / DEIMv2 + huge ViT teachers** | Different product stack, latency and VRAM; 300-query limit still applies unless retrained. Fine for a side experiment, not a drop-in zoo row. |
| **Full UHR-DETR / Dome-DETR in production matrix** | New training stacks, RS-specific; high maintenance vs fixing Ultralytics query cap + YOLO baselines. |
| **RT-DETRv4 before query-cap fix** | Distillation improves AP but **does not remove** set-prediction slot limit; pretty demos ≠ tray counting. |
| **Default Ultralytics mosaic=1 / mixup>0** | Conflicts with counting-first aug review; DEIM “dense positives” ≠ YOLO mixup concat. |
| **Lowering `num_queries` for speed** | Ultralytics suggests 100 queries for latency; **opposite** of our density need. |
| **Objects365 mega-pretrain pivot** | Expensive data pipeline; prioritize cap + eval consistency first. |
| **ASAHI / adaptive slicing in v1** | Adds complexity before matrix eval even runs **fixed** SAHI at deploy parity. |
| **Chasing COCO AP leaderboard without test count MAE** | Sunflower success metric is **count error on test**, not mAP on val. |

---

## Summary judgment

Our training setup is **mature for YOLO-scale zoo benchmarking @ 1280** (aug, budgets, bench parity, eval `max_det=3000`) but **underuses 2025–2026 DETR advances** and **structurally under-serves dense trays on RT-DETR** via `num_queries=300` vs peak GT **1015** — **documented and accepted** for committed bench, with a **planned 1024+ YAML ablation (stack step 7, zoo only)** before treating RT-DETR as count-competitive. Highest leverage: **query-cap ablation (P1-RTDETR-Q)**, then optional **RT-DETRv2 / D-FINE+DEIM / v4** side tracks if DETR stays in the zoo. **`best2.pt` remains the YOLO production path.**

---

## Validated 2026-05-29

Cross-check of repo config, `backlog.md`, and external references (read-only).

| Claim | Verified against | Result |
|-------|------------------|--------|
| Peak GT **1015** / image | `harchoc/rtdetr_limits.py` (`SUNFLOWER_DOCUMENTED_PEAK_GT_BOXES_PER_IMAGE`) | Match |
| Default / committed `num_queries` **300** | `ULTRALYTICS_RTDETR_DEFAULT_NUM_QUERIES`, `train_bench_rtdetr-l.json` | Match |
| `accept_rtdetr_query_truncation: true` | `train_bench_rtdetr-l.json`, `rtdetr_fields_from_train_json()` | Match |
| Matrix plan/train `matrix_metadata` | `bench_matrix_metadata()` in `harchoc/bench_config.py`; dry-run `matrix.json` rows | Match |
| Query-cap policy **Done**; custom YAML ablation **Next** | `backlog.md` (lines ~64–76, ~133–136) | Match |
| Ultralytics zoo = RT-DETR **v1** (`rtdetr-l.pt`) | No RT-DETRv2/v4 weights in `configs/bench/` | Match |
| Raise queries needs **custom YAML + retrain** | [Ultralytics #20688](https://github.com/ultralytics/ultralytics/issues/20688) (`RTDETRDecoder` `[nc, num_queries, max_det]`); `docs/training_budget.md` | Match |
| **RT-DETRv2** (Jul 2024 report) | [arXiv:2407.17140](https://arxiv.org/abs/2407.17140) — bag-of-freebies, dynamic aug, same-speed AP ↑ vs v1 | Link OK |
| **RT-DETRv4** (Oct 2025) | [arXiv:2510.25257](https://arxiv.org/abs/2510.25257) — VFM distillation (DSI/GAM), no inference overhead | Link OK |
| **DEIM** (CVPR 2025) | [arXiv:2412.04234](https://arxiv.org/abs/2412.04234) — Dense O2O + MAL; ~50% train time cut with RT-DETRv2/D-FINE in paper | Link OK |


**P1-RTDETR-Q (2026-05-29):** `configs/models/rtdetr-l_nq1024.yaml` (decoder `nq=1024`) + `train_rtdetr_queries_smoke_15ep.json`; `test_train_config` + `train.py --dry-run` pass; 15-ep GPU smoke deferred — [`reports/hsp/rtdetr_queries_smoke_notes.json`](../../reports/hsp/rtdetr_queries_smoke_notes.json).

*Revisit when Ultralytics ships RT-DETRv2 weights, **P1-RTDETR-Q** query-cap ablation completes, or matrix zoo (**P0-5**) lands.*

**Manuscript SOTA:** **MS-SOTA** / **P0-5** zoo — not YOLOv8-only. **Deploy two-stage analogy:** `alshehri2025_uav` ([literature_validated.json](../manuscript/literature_validated.json)) vs `classifier.pt` + `best2.pt` (**MS-DEPLOY-2STG** Done; [gap §14](../manuscript/reviewer_comments_backlog_gap.md#14-manuscript-draft--two-stage-deploy-discussion)).

**Validated literature registry:** [`docs/manuscript/literature_validated.md`](../manuscript/literature_validated.md) · **Related Work (MS-LIT):** [`related_work_outline.md`](../manuscript/related_work_outline.md)
