# Manuscript markdown freshness

**Verified:** 2026-06-01 (branch `pr/backlog-ci-dataset`). Re-check after GPU jobs finish.

| Artifact | Status | Notes |
|----------|--------|-------|
| Headline test MAE **61.3** | Stable | `reports/hsp/dual_metric.json` |
| Test mAP50 **0.18** | Stable | `reports/hsp/eval_test_map.json` |
| Aug grid (S0–S14 + 100 ep) | **Closed** | None beat anchor; confirm **64.1** |
| Zoo `zoo_yolo_only` | **3/4 + 1 in flight** | v8m **111.9**, 11m **119.6**, 26m **95.3** done; **v10m** 100 ep training (fits 8 GiB; not an OOM deferral) |
| Ultralytics RT-DETR @ 1280 | **Blocked (VRAM)** | Measured 1-ep probe OOM on 8 GiB |
| External DETR (D-FINE / DEIM) | **Partial / integration** | D-FINE trained on same GPU; DEIM/rtdetrv2 need port/import fixes |
| Tray finetune (top-3 weak trays) | **Queued** | `gpu_queue_post_zoo.json`; not in headline table yet |

**Regenerate tables after zoo/finetune:**

```bash
mamba run -n harchoc python scripts/experiment.py manuscript-preflight
```

**Do not cite without refresh:** zoo row for `yolov10m` until `runs/hsp_zoo/yolov10m_e100_s0/weights/best.pt` exists.
