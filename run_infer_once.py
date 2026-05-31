"""One-shot: same PREPROCESS_NORMALIZE + SAHI as telegram_bot.py (no Telegram)."""
from __future__ import annotations

import os
import sys
import tempfile
import time
from pathlib import Path

ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(ROOT)

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

from harchoc.hsp_weights import resolve_detection_weights

def _detect_device() -> str:
    try:
        import torch

        fd = os.getenv("FORCE_DEVICE", "").lower()
        if fd == "cuda" and torch.cuda.is_available():
            return "cuda"
        if fd == "cpu":
            return "cpu"
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


DEVICE = _detect_device()

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
UNFERT_DEDUP_CENTER_RATIO = float(os.getenv("UNFERT_DEDUP_CENTER_RATIO", "1.4"))
UNFERT_DEDUP_MIN_PIX = float(os.getenv("UNFERT_DEDUP_MIN_PIX", "2.0"))
UNFERT_VS_FERT_SUPPRESS = os.getenv("UNFERT_VS_FERT_SUPPRESS", "true").lower() == "true"
UNFERT_VS_FERT_IOU = float(os.getenv("UNFERT_VS_FERT_IOU", "0.99"))
UNFERT_TIP_ON_SEED_SUPPRESS = (
    os.getenv("UNFERT_TIP_ON_SEED_SUPPRESS", "true").lower() == "true"
)
UNFERT_FERT_AREA_RATIO_MIN = float(os.getenv("UNFERT_FERT_AREA_RATIO_MIN", "1.35"))
UNFERT_FERT_EXPAND_PX = float(os.getenv("UNFERT_FERT_EXPAND_PX", "4"))

PREPROCESS_NORMALIZE = os.getenv("PREPROCESS_NORMALIZE", "false").lower() == "true"
PP_BRIGHTNESS = float(os.getenv("PP_BRIGHTNESS", "-26"))
PP_EXPOSURE = float(os.getenv("PP_EXPOSURE", "100"))
PP_CONTRAST = float(os.getenv("PP_CONTRAST", "100"))
PP_SHADOWS = float(os.getenv("PP_SHADOWS", "-100"))
PP_WARMTH = float(os.getenv("PP_WARMTH", "-100"))
PP_TINT = float(os.getenv("PP_TINT", "100"))
PP_SHARPNESS = float(os.getenv("PP_SHARPNESS", "100"))

_wpath = resolve_detection_weights()
MODEL_PATH = str(_wpath if _wpath.is_absolute() else Path(ROOT) / _wpath)


def apply_normalize_look(src_path: str, dst_path: str) -> bool:
    try:
        import cv2
        import numpy as np

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
    from harchoc.deploy_filters import DeployFilterConfig, filter_object_predictions

    return filter_object_predictions(preds, DeployFilterConfig.resolve())


def _run_fullframe_export(
    image: str,
    *,
    locked_conf_from: str,
    dataset_root: str | None,
) -> int:
    """Full-frame HSP-style export at locked conf (parity debug; not SAHI)."""
    import json
    from pathlib import Path

    root = Path(dataset_root or os.environ.get("DATASET_ROOT", ROOT)).resolve()
    if not root.is_dir():
        print(f"Dataset root not found: {root}")
        return 1
    locked_path = Path(locked_conf_from)
    if not locked_path.is_file():
        locked_path = Path(ROOT) / locked_conf_from
    from harchoc.threshold_lock import load_locked_conf

    conf = float(load_locked_conf(locked_path))
    rel = os.path.relpath(image, root) if image.startswith(str(root)) else None
    if rel is None:
        for cand in ("images/test", "images/val", "images/train"):
            p = root / cand / os.path.basename(image)
            if p.is_file():
                rel = f"{cand}/{os.path.basename(image)}"
                break
    if rel is None:
        print("Could not resolve image path relative to DATASET_ROOT")
        return 1

    import tempfile

    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as tf:
        tf.write(rel.replace("\\", "/") + "\n")
        split_path = Path(tf.name)

    out_dir = Path(ROOT) / "reports" / "deploy_parity"
    out_dir.mkdir(parents=True, exist_ok=True)
    preds_out = out_dir / "fullframe_preds.json"
    gt_out = out_dir / "fullframe_gt.json"

    from harchoc.eval_export import export_gt_preds_json

    summary = export_gt_preds_json(
        split_file=split_path,
        dataset_root=root,
        weights=Path(MODEL_PATH),
        gt_out=gt_out,
        preds_out=preds_out,
        conf=conf,
        device=DEVICE,
    )
    split_path.unlink(missing_ok=True)
    print(json.dumps(summary, indent=2))
    return 0


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="One-shot SAHI infer (telegram_bot parity).")
    ap.add_argument("image", nargs="?", help="Input image path")
    ap.add_argument(
        "--fullframe-export",
        action="store_true",
        help="Also run full-frame eval_export at locked conf (HSP parity debug).",
    )
    ap.add_argument(
        "--locked-conf-from",
        default="reports/hsp/threshold_val.json",
        help="Threshold JSON for --fullframe-export.",
    )
    ap.add_argument("--dataset-root", default="", help="Dataset root for fullframe export.")
    ns = ap.parse_args()

    if not ns.image:
        print("Usage: python run_infer_once.py <image> [--fullframe-export]")
        sys.exit(1)
    image = os.path.abspath(ns.image)
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

    from harchoc.sahi_infer import (
        SahiSliceConfig,
        load_ultralytics_detection_model,
        run_sliced_prediction,
    )

    t0 = time.time()
    detection_model = load_ultralytics_detection_model(
        MODEL_PATH,
        device=DEVICE,
        confidence_threshold=CONF_THR_MODEL_MIN,
    )
    print(f"Model loaded in {time.time() - t0:.1f}s")

    t1 = time.time()
    result = run_sliced_prediction(
        work_image,
        detection_model,
        SahiSliceConfig(slice_size=SLICE_SIZE, overlap=OVERLAP, nms_iou=NMS_IOU),
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
    print(f"Developed (class 0): {fert}")
    print(f"Aborted (class 1): {unf}")
    print(f"Total: {fert + unf}")

    if bool(ns.fullframe_export):
        rc_ff = _run_fullframe_export(
            image,
            locked_conf_from=str(ns.locked_conf_from),
            dataset_root=str(ns.dataset_root) or None,
        )
        if rc_ff != 0:
            sys.exit(rc_ff)

    if pre_path and os.path.exists(pre_path):
        try:
            os.remove(pre_path)
        except OSError:
            pass


if __name__ == "__main__":
    main()
