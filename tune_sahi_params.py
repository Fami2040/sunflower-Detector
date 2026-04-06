"""
Small SAHI parameter sweep vs manual ground truth (same pipeline as telegram_bot.py).
Usage: python tune_sahi_params.py [path/to/image.png]

Set TUNE_COMBOS=quick for 8 presets only (default on Windows to reduce memory spikes).
"""
from __future__ import annotations

import gc
import os
import sys
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(ROOT)

GT_F = 417
GT_U = 420

try:
    import torch

    FORCE_DEVICE = os.getenv("FORCE_DEVICE", "").lower()
    if FORCE_DEVICE == "cuda" and torch.cuda.is_available():
        DEVICE = "cuda"
    elif FORCE_DEVICE == "cpu":
        DEVICE = "cpu"
    else:
        DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
except Exception:
    DEVICE = "cpu"

MODEL_PATH = os.path.join(ROOT, "models", "best2.pt")

from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction


def compute_counts(result, conf_thr: float) -> tuple[int, int]:
    count = {0: 0, 1: 0}
    for p in result.object_prediction_list:
        cls_id = int(p.category.id)
        score = p.score.value
        if score < conf_thr:
            continue
        count[cls_id] += 1
    return count[0], count[1]


def loss(f: int, u: int) -> float:
    return abs(f - GT_F) + abs(u - GT_U)


def run_once(detection_model, image: str, sl: int, ov: float, cf: float, nms: float) -> tuple[int, int, float]:
    t1 = time.time()
    result = get_sliced_prediction(
        image=image,
        detection_model=detection_model,
        slice_height=sl,
        slice_width=sl,
        overlap_height_ratio=ov,
        overlap_width_ratio=ov,
        postprocess_type="NMS",
        postprocess_match_threshold=nms,
    )
    f, u = compute_counts(result, cf)
    elapsed = time.time() - t1
    del result
    gc.collect()
    if DEVICE == "cuda":
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
    return f, u, elapsed


def preset_combos() -> list[tuple[int, float, float, float, str]]:
    """Hand-picked grid for dense sunflower heads (small seeds, heavy overlap)."""
    return [
        # Recall-focused (model was under-counting vs manual labels)
        (512, 0.32, 0.05, 0.60, "tiny tiles, low conf"),
        (512, 0.35, 0.06, 0.58, "tiny tiles + overlap"),
        (560, 0.30, 0.06, 0.55, "small tiles low conf"),
        (640, 0.30, 0.05, 0.50, "640 low conf"),
        (640, 0.28, 0.08, 0.52, "640 med-low conf"),
        (720, 0.28, 0.07, 0.48, "720 low conf"),
        (800, 0.28, 0.06, 0.45, "800 low conf"),
        (640, 0.26, 0.05, 0.35, "low NMS merge"),
    ]


def main() -> None:
    image = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "test_sunflower_tune.png")
    if not os.path.isfile(image):
        print(f"Image not found: {image}")
        sys.exit(1)
    if not os.path.isfile(MODEL_PATH):
        print(f"Model not found: {MODEL_PATH}")
        sys.exit(1)

    model_conf = float(os.getenv("TUNE_MODEL_CONF", "0.01"))
    mode = os.getenv("TUNE_COMBOS", "quick").lower()

    print(f"Device: {DEVICE} | image: {image}")
    print(f"Ground truth: Fertilized={GT_F}, Unfertilized={GT_U}, total={GT_F + GT_U}")
    print(f"Loading model (internal conf={model_conf})...")
    t0 = time.time()
    detection_model = AutoDetectionModel.from_pretrained(
        model_type="ultralytics",
        model_path=MODEL_PATH,
        confidence_threshold=model_conf,
        device=DEVICE,
    )
    print(f"Model loaded in {time.time() - t0:.1f}s\n")

    if mode == "quick":
        combos = preset_combos()
    else:
        from itertools import product

        combos = [
            (sl, ov, cf, nms, f"grid {sl},{ov},{cf},{nms}")
            for sl, ov, cf, nms in product(
                [640, 800, 960],
                [0.24, 0.30],
                [0.10, 0.12, 0.14],
                [0.45, 0.55],
            )
        ]

    results: list[tuple[float, int, int, int, float, float, float, float, str]] = []
    for sl, ov, cf, nms, label in combos:
        f, u, elapsed = run_once(detection_model, image, sl, ov, cf, nms)
        L = loss(f, u)
        results.append((L, f, u, sl, ov, cf, nms, elapsed, label))
        print(
            f"L={L:.0f}  F={f} U={u}  | SLICE={sl} OV={ov} CONF={cf} NMS={nms}  ({elapsed:.1f}s)  [{label}]"
        )

    results.sort(key=lambda x: x[0])
    L, f, u, sl, ov, cf, nms, _, label = results[0]

    print("\n=== BEST (min |F-417| + |U-420|) ===")
    print(f"  loss = {L:.0f}  [{label}]")
    print(f"  Fertilized={f} (target {GT_F}), Unfertilized={u} (target {GT_U})")
    print(f"  SLICE_SIZE = {sl}")
    print(f"  OVERLAP = {ov}")
    print(f"  CONF_THR = {cf}")
    print(f"  NMS_IOU = {nms}")
    print("\nSuggested defaults for telegram_bot.py:")
    print(f'  SLICE_SIZE = int(os.getenv("SLICE_SIZE", "{sl}"))')
    print(f'  OVERLAP = float(os.getenv("OVERLAP", "{ov}"))')
    print(f'  CONF_THR = float(os.getenv("CONF_THR", "{cf}"))')
    print(f'  NMS_IOU = float(os.getenv("NMS_IOU", "{nms}"))')


if __name__ == "__main__":
    main()
