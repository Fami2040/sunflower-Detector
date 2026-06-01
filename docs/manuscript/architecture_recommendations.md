# Architecture recommendations (literature-backed)

**Sources:** [`literature_validated.json`](literature_validated.json) · **Gap map:** [`reviewer_comments_backlog_gap.md`](reviewer_comments_backlog_gap.md) · **Val vs test mAP:** [`val_test_map_gap.md`](val_test_map_gap.md) · **Improvement stack:** [`backlog.md`](../backlog.md#model-improvement-stack-test-count-mae) (test count MAE @ val-locked conf)

---

## Adopt now (already in repo)

| Practice | Evidence | Backlog / artifact |
|----------|----------|-------------------|
| `max_det: 3000` on dense trays | S14 truncation study | **Done** P0-1, [`s14_maxdet_truncation.json`](../../reports/hsp/s14_maxdet_truncation.json) |
| Val tune → lock conf → test reporting | HSP protocol, threshold literature | P0-3 chain, [`dual_metric.json`](../../reports/hsp/dual_metric.json) |
| 8-model zoo @ imgsz 1280 | Detector SOTA expectation | **P0-5**, [`matrix.json`](../../reports/benchmarks/matrix.json) |
| Grad-CAM on FP crops | LWCD-YOLO, Yang/Ren XAI narrative | **Done** P2-FIG-CAM, `experiment.py gradcam` → [`gradcam_routing.md`](gradcam_routing.md) |
| Two-weight deploy gate | Alshehri analogy; production path | [`HSP_BASELINE_MODELS.md`](../HSP_BASELINE_MODELS.md), [gap §14](reviewer_comments_backlog_gap.md#14-manuscript-draft--two-stage-deploy-discussion) (**MS-DEPLOY-2STG** Done) |
| Manuscript repro one-command | Reviewer reproducibility | **Done** MS-REPRO, `experiment.py repro` |
| Label val mAP as non-generalization | Reviewer §2.2 / §9 | **Done** MS-VAL-MAP-CAVEAT, [`val_test_map_gap.md`](val_test_map_gap.md) |

---

## Adopt next (lit-backed; backlog)

Stack step numbers follow [`backlog.md` § Model improvement stack](../backlog.md#model-improvement-stack-test-count-mae).

| Recommendation | Literature | Stack | Backlog ID |
|----------------|------------|-------|------------|
| Full zoo matrix + count MAE columns | GrainNet, MS-SOTA | **5** | **P0-5**, **P1-ZOO-PARITY**, **P2-SEED-MAE** |
| Test mAP on CPU export | Eval VRAM policy | — | **SCI-MAP-CPU** (unblocks **R-SCI-1**, **MS-SOTA**) |
| Mosaic=0 ablation vs S0 baseline | `lwcd_yolo2025` | **4** | **ARCH-MOSAIC0-AB** (aug S2) |
| TFA / staged YOLO freeze on tray holdout | `tfa2020`, `gandhi2025_yolov8_freeze` | **6** | **P1-FINETUNE-LOOP**, **MS-DOMAIN-ADAPT** |
| Per-tray domain eval + limitations prose | Domain shift lit, Yang analogy | **6** | **P1-DOMAIN-EVAL**, **MS-GEN** |
| Count-first threshold (`min_count_mae`) | Eval calibration scan | **3** | **P1-FP-BUDGET** Partial — Methods draft in [p0_summary § Operating point](../../reports/hsp/p0_summary.md#methods-draft--threshold--operating-point) |
| TIDE buckets + localization narrative | GWHD, fp_taxonomy | **3**† | **P1-TIDE**, **MS-FP-LOC-NARR** |
| Graded trust on detections (locked conf + score band) | `yao2025_hfuzzy` | **3** | **MS-FUZZY-BOUND**, **P1-UNCERT-FP**, **P2-FIG** ambiguous panel |
| Maintain validated registry | — | — | **LIT-VALIDATE** |

† FP taxonomy supports **MS-FUZZY-BOUND** (background/localization ≫ cls) at stack step 3.

---

## Defer / do not chase

Low MAE ROI items also listed in [`backlog.md` § Defer](../backlog.md#model-improvement-stack-test-count-mae).

| Topic | Why |
|-------|-----|
| Full unsupervised domain adaptation (Hu/Ilyas farm UDA) | No unlabeled target trays; labels required |
| UAV-only sunflower head pipelines | Out of benchtop HSP scope unless new data |
| Third detect class for “uncertainty” | **Defer** — **MS-FUZZY-BOUND** = **graded trust on detections** (locked conf + low-conf score band + `ambiguous_summary` + FP taxonomy); Yao 2025 = fuzzy **regression** analogy only, **not** a third YOLO class or relabel protocol |
| YOLO26 mosaic≈1 / mixup spikes | Low test-MAE ROI vs S0–S14 aug grid |
| GrainNet-style EMA in YOLO head | Custom YOLOv7 graph; not Ultralytics-configurable; DAER-YOLO solo-EMA precision risk — [arch_ema_bg_spike](../research/arch_ema_bg_spike_literature.md) (**ARCH-EMA-BG-SPIKE** Done, impl. cancelled) |
| SAHI-as-train-metric before full-frame stable | Deploy parity track only after locked-conf baseline |
| RT-DETRv4 before query-cap ablation | **P1-RTDETR-Q** first |
| Ren physics-hybrid branch | No material descriptors for seeds at train time |
| Reporting peak val mAP as generalization | See [`val_test_map_gap.md`](val_test_map_gap.md) |

---

## Manuscript-only (no new architecture)

| Reviewer theme | Backlog |
|----------------|---------|
| IMRaD abstract | **MS-ABS** |
| Contribution positioning | **MS-ORIG** Done — [`originality_contribution_peers.md`](originality_contribution_peers.md) |
| Manual ~1 person-hour baseline | **MS-MANUAL-BASE** |
| n=50 validation protocol | **MS-MANUAL-N50** |
| Figure journal style | **MS-FIG-NORM** |
