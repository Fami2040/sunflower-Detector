# HSP baseline models (pointer)

Canonical documentation: [`docs/HSP_BASELINE_MODELS.md`](../../docs/HSP_BASELINE_MODELS.md).

Checksums and class semantics: [`baseline_models_manifest.json`](baseline_models_manifest.json).

Eval spec: [`configs/experiments/eval_hsp_baseline.json`](../../configs/experiments/eval_hsp_baseline.json).

| File | Role |
|------|------|
| `models/best2.pt` | Detection: class **0** developed, **1** aborted (`imgsz=1280`, `max_det=3000` for HSP) |
| `models/classifier.pt` | Classify: **0** other, **1** sunflower (Telegram gate only; not used in `eval.py`) |
