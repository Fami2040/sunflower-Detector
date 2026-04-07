"""One-shot: same PREPROCESS_NORMALIZE + SAHI as telegram_bot.py (no Telegram)."""
from __future__ import annotations

import os
import sys
import tempfile
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(ROOT)

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

import cv2
import numpy as np

try:
    import torch

    fd = os.getenv("FORCE_DEVICE", "").lower()
    if fd == "cuda" and torch.cuda.is_available():
        DEVICE = "cuda"
    elif fd == "cpu":
        DEVICE = "cpu"
    else:
        DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
except Exception:
    DEVICE = "cpu"

SLICE_SIZE = int(os.getenv("SLICE_SIZE", "500"))
OVERLAP = float(os.getenv("OVERLAP", "0.35"))
_conf_legacy = os.getenv("CONF_THR", "").strip()
CONF_THR_FERTILIZED = float(
    os.getenv("CONF_THR_FERTILIZED", _conf_legacy if _conf_legacy else "0.06")
)
CONF_THR_UNFERTILIZED = float(
    os.getenv(
        "CONF_THR_UNFERTILIZED",
        _conf_legacy if _conf_legacy else "0.04",
    )
)
CONF_THR_MODEL_MIN = min(CONF_THR_FERTILIZED, CONF_THR_UNFERTILIZED)
NMS_IOU = float(os.getenv("NMS_IOU", "0.50"))
OUTPUT_JPEG_QUALITY = int(os.getenv("OUTPUT_JPEG_QUALITY", "85"))
UNFERT_DEDUP = os.getenv("UNFERT_DEDUP", "true").lower() == "true"
UNFERT_DEDUP_CENTER_RATIO = float(os.getenv("UNFERT_DEDUP_CENTER_RATIO", "1.0"))
UNFERT_DEDUP_MIN_PIX = float(os.getenv("UNFERT_DEDUP_MIN_PIX", "2.0"))

PREPROCESS_NORMALIZE = os.getenv("PREPROCESS_NORMALIZE", "false").lower() == "true"
PP_BRIGHTNESS = float(os.getenv("PP_BRIGHTNESS", "-26"))
PP_EXPOSURE = float(os.getenv("PP_EXPOSURE", "100"))
PP_CONTRAST = float(os.getenv("PP_CONTRAST", "100"))
PP_SHADOWS = float(os.getenv("PP_SHADOWS", "-100"))
PP_WARMTH = float(os.getenv("PP_WARMTH", "-100"))
PP_TINT = float(os.getenv("PP_TINT", "100"))
PP_SHARPNESS = float(os.getenv("PP_SHARPNESS", "100"))

_rel = os.getenv("DETECTION_MODEL", "models/best2.pt")
MODEL_PATH = _rel if os.path.isabs(_rel) else os.path.join(ROOT, _rel.replace("/", os.sep))

from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction


def apply_normalize_look(src_path: str, dst_path: str) -> bool:
    try:
        img = cv2.imread(src_path)
        if img is None:
            print(f"normalize: could not read {src_path}")
            return False

        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        L, A, Bch = cv2.split(lab)
        Lf = L.astype(np.float32)
        lo = L.astype(np.float32)

        exp_gain = 1.0 + (PP_EXPOSURE / 100.0) * 1.5
        Lf = Lf * exp_gain

        shadow_w = np.clip((118.0 - lo) / 118.0, 0.0, 1.0) ** 1.15
        shadow_factor = 1.0 + (PP_SHADOWS / 100.0) * 0.55
        Lf = Lf * (1.0 - shadow_w) + Lf * shadow_w * shadow_factor

        Lf = Lf + (PP_BRIGHTNESS / 100.0) * 52.0

        c = 1.0 + (PP_CONTRAST / 100.0) * 1.2
        mid = 128.0
        Lf = (Lf - mid) * c + mid
        Lf = np.clip(Lf, 0, 255)

        Af = A.astype(np.float32)
        Bf = Bch.astype(np.float32)

        w = PP_WARMTH / 100.0
        Bf = Bf + w * 20.0
        Af = Af - w * 5.0

        Bf = Bf + (PP_TINT / 100.0) * 4.0
        Af = Af + (PP_TINT / 100.0) * 22.0

        L2 = Lf.astype(np.uint8)
        A2 = np.clip(Af, 0, 255).astype(np.uint8)
        B2 = np.clip(Bf, 0, 255).astype(np.uint8)
        lab2 = cv2.merge((L2, A2, B2))
        bgr = cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)

        s = PP_SHARPNESS / 100.0
        if s > 0.02:
            u8 = bgr
            sigma = 1.0 + s * 1.8
            blur = cv2.GaussianBlur(u8, (0, 0), sigmaX=sigma)
            amount = 0.75 + s * 1.35
            bgr = cv2.addWeighted(u8, 1.0 + amount, blur, -amount, 0)

        ok = cv2.imwrite(dst_path, bgr, [cv2.IMWRITE_JPEG_QUALITY, OUTPUT_JPEG_QUALITY])
        if not ok:
            print("normalize: cv2.imwrite failed")
            return False
        print(
            f"Normalized -> {dst_path} (PP: brightness={PP_BRIGHTNESS} exposure={PP_EXPOSURE} "
            f"contrast={PP_CONTRAST} shadows={PP_SHADOWS} warmth={PP_WARMTH} tint={PP_TINT} sharpness={PP_SHARPNESS})"
        )
        return True
    except Exception as e:
        print(f"apply_normalize_look failed: {e}")
        return False


def filter_predictions(preds):
    kept = []
    unfert_candidates = []
    for p in preds:
        cls_id = int(p.category.id)
        score = p.score.value
        thr = CONF_THR_FERTILIZED if cls_id == 0 else CONF_THR_UNFERTILIZED
        if score < thr:
            continue
        if cls_id == 1 and UNFERT_DEDUP:
            unfert_candidates.append(p)
        else:
            kept.append(p)

    if not UNFERT_DEDUP or not unfert_candidates:
        return kept + unfert_candidates

    def _center_and_size(pred):
        b = pred.bbox
        x1, y1, x2, y2 = float(b.minx), float(b.miny), float(b.maxx), float(b.maxy)
        w = max(1.0, x2 - x1)
        h = max(1.0, y2 - y1)
        cx = (x1 + x2) * 0.5
        cy = (y1 + y2) * 0.5
        return cx, cy, w, h

    deduped = []
    for p in sorted(unfert_candidates, key=lambda x: x.score.value, reverse=True):
        cx, cy, w, h = _center_and_size(p)
        is_dup = False
        for k in deduped:
            kx, ky, kw, kh = _center_and_size(k)
            scale = min(w, h, kw, kh)
            radius = max(UNFERT_DEDUP_MIN_PIX, UNFERT_DEDUP_CENTER_RATIO * scale)
            if ((cx - kx) ** 2 + (cy - ky) ** 2) ** 0.5 <= radius:
                is_dup = True
                break
        if not is_dup:
            deduped.append(p)

    return kept + deduped


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python run_infer_once.py <image>")
        sys.exit(1)
    image = os.path.abspath(sys.argv[1])
    if not os.path.isfile(image):
        print(f"Not found: {image}")
        sys.exit(1)
    if not os.path.isfile(MODEL_PATH):
        print(f"Model not found: {MODEL_PATH}")
        sys.exit(1)

    work_image = image
    pre_path = None
    if PREPROCESS_NORMALIZE:
        fd, pre_path = tempfile.mkstemp(suffix=".jpg", prefix="normalized_")
        os.close(fd)
        if apply_normalize_look(image, pre_path):
            work_image = pre_path
        else:
            try:
                os.remove(pre_path)
            except OSError:
                pass
            pre_path = None
            print("Normalization failed; using original.")
    else:
        print("PREPROCESS_NORMALIZE=false")

    print(f"Device: {DEVICE} | Model: {MODEL_PATH}")
    print(f"Inference image: {work_image}")
    print(
        f"SLICE={SLICE_SIZE} OVERLAP={OVERLAP} CONF_FERT={CONF_THR_FERTILIZED} "
        f"CONF_UNFERT={CONF_THR_UNFERTILIZED} model_min={CONF_THR_MODEL_MIN} NMS_IOU={NMS_IOU}\n"
    )

    t0 = time.time()
    detection_model = AutoDetectionModel.from_pretrained(
        model_type="ultralytics",
        model_path=MODEL_PATH,
        confidence_threshold=CONF_THR_MODEL_MIN,
        device=DEVICE,
    )
    print(f"Model loaded in {time.time() - t0:.1f}s")

    t1 = time.time()
    result = get_sliced_prediction(
        image=work_image,
        detection_model=detection_model,
        slice_height=SLICE_SIZE,
        slice_width=SLICE_SIZE,
        overlap_height_ratio=OVERLAP,
        overlap_width_ratio=OVERLAP,
        postprocess_type="NMS",
        postprocess_match_threshold=NMS_IOU,
    )

    fert = unf = 0
    filtered_preds = filter_predictions(result.object_prediction_list)
    for p in filtered_preds:
        cls_id = int(p.category.id)
        if cls_id == 0:
            fert += 1
        else:
            unf += 1

    print(f"Inference: {time.time() - t1:.1f}s\n")
    print(f"Fertilized (class 0): {fert}")
    print(f"Unfertilized (class 1): {unf}")
    print(f"Total: {fert + unf}")

    if pre_path and os.path.exists(pre_path):
        try:
            os.remove(pre_path)
        except OSError:
            pass


if __name__ == "__main__":
    main()
