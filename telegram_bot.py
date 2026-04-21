# =====================================================
# Telegram Bot | Fertilized / Unfertilized Sunflower Seeds
# =====================================================

import cv2
import os
import tempfile
import numpy as np
import asyncio
from pathlib import Path
from io import BytesIO
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from telegram.error import TimedOut, NetworkError, Conflict, InvalidToken
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
import logging
from dotenv import load_dotenv
from ultralytics import YOLO
from aiohttp import web

# Load environment variables
load_dotenv()

# ================= CONFIG =================
MODEL_PATH = r"models/best2.pt"
CLASSIFIER_PATH = r"models/classifier.pt"
# Device selection: prefer CUDA for maximum speed
# Set FORCE_DEVICE="cuda" to force CUDA (will fail if not available)
# Set FORCE_DEVICE="cpu" to force CPU
try:
    import torch
    FORCE_DEVICE = os.getenv("FORCE_DEVICE", "").lower()
    
    if FORCE_DEVICE == "cuda":
        if torch.cuda.is_available():
            DEVICE = "cuda"
            print(f"✅ FORCE_DEVICE=cuda: Using CUDA (GPU) - {torch.cuda.get_device_name(0)}")
        else:
            print("❌ ERROR: FORCE_DEVICE=cuda but CUDA is not available!")
            print("   CUDA is not available on this system. Falling back to CPU.")
            print("   To use CPU, set FORCE_DEVICE=cpu or remove FORCE_DEVICE")
            DEVICE = "cpu"
    elif FORCE_DEVICE == "cpu":
        DEVICE = "cpu"
        print("ℹ️ FORCE_DEVICE=cpu: Using CPU (forced)")
    else:
        # Auto-detect: ALWAYS prefer CUDA if available for maximum speed
        if torch.cuda.is_available():
            DEVICE = "cuda"
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"✅ CUDA detected: Using GPU - {gpu_name} ({gpu_memory:.2f} GB)")
            print(f"🚀 GPU mode: Processing will be 10-50x faster than CPU!")
        else:
            DEVICE = "cpu"
            print("⚠️ CUDA not available: Using CPU (10-50x slower than GPU)")
            print("   For maximum speed, use a GPU-enabled server or local machine with CUDA")
            print("   Railway free tier only provides CPU. Consider Railway Pro or other GPU hosting.")
except Exception as e:
    DEVICE = "cpu"
    print(f"⚠️ Error detecting device: {e}, defaulting to CPU")

# ---- SAHI slicing (VERY IMPORTANT) ----
# Defaults tuned on sample heads: 500 / 0.35 / 0.50 improves center (unfertilized) recall vs 560/0.3/0.55
# Override with SLICE_SIZE, OVERLAP, NMS_IOU env vars
SLICE_SIZE = int(os.getenv("SLICE_SIZE", "500"))
OVERLAP = float(os.getenv("OVERLAP", "0.35"))

# ---- Thresholds (LOW to reduce FN) ----
# Per-class: lower CONF_THR_UNFERTILIZED helps sterile/center detections without loosening fertilized.
# Model uses min() so weak class-1 boxes are not dropped before post-filter.
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
CONF_THR = CONF_THR_FERTILIZED  # backward compat for logs / external refs
NMS_IOU = float(os.getenv("NMS_IOU", "0.50"))
# Extra de-dup for class 1 (unfertilized) to prevent dense red double-counts.
UNFERT_DEDUP = os.getenv("UNFERT_DEDUP", "true").lower() == "true"
UNFERT_DEDUP_CENTER_RATIO = float(os.getenv("UNFERT_DEDUP_CENTER_RATIO", "1.4"))
UNFERT_DEDUP_MIN_PIX = float(os.getenv("UNFERT_DEDUP_MIN_PIX", "2.0"))
# Drop unfertilized when it overlaps the same physical seed as fertilized (red on green).
UNFERT_VS_FERT_SUPPRESS = os.getenv("UNFERT_VS_FERT_SUPPRESS", "true").lower() == "true"
# High default: IoU alone mis-fires on benchmark heads; use tip-on-seed rule below. Lower (e.g. 0.15) to enable strict IoU.
UNFERT_VS_FERT_IOU = float(os.getenv("UNFERT_VS_FERT_IOU", "0.99"))
# Small red box centered on a larger green box (yellow tip on dark seed); avoids counting both.
UNFERT_TIP_ON_SEED_SUPPRESS = (
    os.getenv("UNFERT_TIP_ON_SEED_SUPPRESS", "true").lower() == "true"
)
UNFERT_FERT_AREA_RATIO_MIN = float(os.getenv("UNFERT_FERT_AREA_RATIO_MIN", "1.35"))
# Expand fertilized boxes before center-in test (red tips often sit just outside green box coordinates).
UNFERT_FERT_EXPAND_PX = float(os.getenv("UNFERT_FERT_EXPAND_PX", "4"))

# ---- Telegram / performance ----
OUTPUT_JPEG_QUALITY = int(os.getenv("OUTPUT_JPEG_QUALITY", "85"))  # smaller file uploads faster
TG_RETRY_ATTEMPTS = int(os.getenv("TG_RETRY_ATTEMPTS", "3"))

# ---- Performance optimizations ----
# Set SKIP_CLASSIFIER="true" to skip classifier for speed (saves 20+ seconds)
SKIP_CLASSIFIER = os.getenv("SKIP_CLASSIFIER", "false").lower() == "true"  # Set to "true" to skip classifier

# ---- Normalize incoming images (approx. phone-editor sliders, -100..100 scale) ----
# Defaults match: brightness -26, exposure 100, contrast 100, shadows -100, warmth -100, tint 100, sharpness 100
PREPROCESS_NORMALIZE = os.getenv("PREPROCESS_NORMALIZE", "false").lower() == "true"
PP_BRIGHTNESS = float(os.getenv("PP_BRIGHTNESS", "-26"))
PP_EXPOSURE = float(os.getenv("PP_EXPOSURE", "100"))
PP_CONTRAST = float(os.getenv("PP_CONTRAST", "100"))
PP_SHADOWS = float(os.getenv("PP_SHADOWS", "-100"))
PP_WARMTH = float(os.getenv("PP_WARMTH", "-100"))
PP_TINT = float(os.getenv("PP_TINT", "100"))
PP_SHARPNESS = float(os.getenv("PP_SHARPNESS", "100"))

# ---- Classes ----
CLASSES = ["Fertilized", "Unfertilized"]

# Telegram Bot Token (set via environment variable or .env file)
# IMPORTANT: Never commit your bot token to GitHub!
# Create a .env file with: BOT_TOKEN=your_token_here
BOT_TOKEN = os.getenv("BOT_TOKEN", "")

# Setup logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

def _is_timeout_error(e: Exception) -> bool:
    msg = str(e).lower()
    return "timed out" in msg or "timeout" in msg

async def _retry_tg(op_name: str, fn, attempts: int = TG_RETRY_ATTEMPTS):
    last_err = None
    for i in range(attempts):
        try:
            return await fn()
        except (TimedOut, NetworkError, asyncio.TimeoutError, TimeoutError) as e:
            last_err = e
            delay = 1.0 * (2 ** i)
            logger.warning(f"{op_name} timed out (attempt {i+1}/{attempts}): {e}. Retrying in {delay:.1f}s")
            await asyncio.sleep(delay)
    raise last_err if last_err else TimedOut(f"{op_name} timed out")

# ================= LOAD MODELS =================
print(f"🔄 Loading detection model on {DEVICE.upper()}...")
if DEVICE == "cuda":
    try:
        import torch
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"🚀 GPU: {gpu_name} ({gpu_memory:.2f} GB)")
    except:
        pass
print(f"📁 Detection model path: {MODEL_PATH}")
if not os.path.exists(MODEL_PATH):
    print(f"❌ ERROR: Detection model file not found at {MODEL_PATH}")
    raise FileNotFoundError(f"Detection model file not found: {MODEL_PATH}")

detection_model = AutoDetectionModel.from_pretrained(
    model_type="ultralytics",
    model_path=MODEL_PATH,
    confidence_threshold=CONF_THR_MODEL_MIN,
    device=DEVICE
)
print("✅ Detection model loaded successfully")
print(
    f"   Confidence: fert>={CONF_THR_FERTILIZED}, unfert>={CONF_THR_UNFERTILIZED}, "
    f"model_min={CONF_THR_MODEL_MIN}"
)

print(f"🔄 Loading classifier model on {DEVICE.upper()}...")
print(f"📁 Classifier model path: {CLASSIFIER_PATH}")
if not os.path.exists(CLASSIFIER_PATH):
    print(f"⚠️ WARNING: Classifier model file not found at {CLASSIFIER_PATH}")
    print("⚠️ Continuing without classifier validation...")
    classifier_model = None
else:
    try:
        classifier_model = YOLO(CLASSIFIER_PATH)
        classifier_model.to(DEVICE)
        print("✅ Classifier model loaded successfully")
    except Exception as e:
        print(f"⚠️ WARNING: Failed to load classifier model: {e}")
        print("⚠️ Continuing without classifier validation...")
        classifier_model = None

# ================= TELEGRAM BOT HANDLERS developed and aborted seeds.=================

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Send a message when the command /start is issued."""
    logger.info(f"Received /start command from user {update.effective_user.id} (@{update.effective_user.username})")
    welcome_message = (
        "🌻 **Sunflower Seed Counter Bot**\n\n"
        "Send me a sunflower image and I'll count:\n"
        "• developed seeds\n"
        "• aborted seeds\n\n"
        "Just send any image file to get started!"
    )
    try:
        await update.message.reply_text(welcome_message, parse_mode='Markdown')
        logger.info(f"Successfully sent welcome message to user {update.effective_user.id}")
    except Exception as e:
        logger.error(f"Error sending welcome message: {e}", exc_info=True)
        raise

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Send a message when the command /help is issued."""
    help_text = (
        "**How to use:**\n"
        "1. Send me a sunflower image (JPG, PNG, etc.)\n"
        "2. Wait for processing...\n"
        "3. Receive counts and annotated image\n\n"
        "**Commands:**\n"
        "/start - Start the bot\n"
        "/help - Show this help message\n\n"
        "**Note:** The bot will automatically check if your image is a sunflower before processing."
    )
    await update.message.reply_text(help_text, parse_mode='Markdown')

async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle text messages that are not commands."""
    logger.info(f"Received text message: {update.message.text}")
    welcome_text = (
        "👋 **Welcome to Sunflower Seed Counter Bot!**\n\n"
        "📸 Just send me a sunflower image and I'll automatically analyze it!\n\n"
        "I'll count:\n"
        "• ✅ Developed seeds\n"
        "• 🌱 aborted seeds\n"
        "• 📊 Total seeds\n"
        "• 📈 Fertilization percentage\n\n"
        "**No /start needed** - just send an image anytime! 🚀\n\n"
        "Use /help for more information."
    )
    await update.message.reply_text(welcome_text, parse_mode='Markdown')

def _bbox_iou(pred_a, pred_b) -> float:
    a, b = pred_a.bbox, pred_b.bbox
    ax1, ay1, ax2, ay2 = float(a.minx), float(a.miny), float(a.maxx), float(a.maxy)
    bx1, by1, bx2, by2 = float(b.minx), float(b.miny), float(b.maxx), float(b.maxy)
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    aa = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    bb = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = aa + bb - inter
    return inter / union if union > 0 else 0.0


def _bbox_area(pred) -> float:
    b = pred.bbox
    return max(
        1e-6,
        (float(b.maxx) - float(b.minx)) * (float(b.maxy) - float(b.miny)),
    )


def _unfert_center_inside_fert(unfert_pred, fert_pred, expand_px: float = 0.0) -> bool:
    """True if center of unfertilized box lies inside fertilized box (tip-on-seed case)."""
    u, f = unfert_pred.bbox, fert_pred.bbox
    cx = (float(u.minx) + float(u.maxx)) * 0.5
    cy = (float(u.miny) + float(u.maxy)) * 0.5
    ex = float(expand_px)
    return (float(f.minx) - ex) <= cx <= (float(f.maxx) + ex) and (float(f.miny) - ex) <= cy <= (
        float(f.maxy) + ex
    )


def _suppress_unfert_vs_fert(unfert_pred, fert_pred) -> bool:
    """True -> drop unfertilized (prefer fertilized for same seed)."""
    if _bbox_iou(unfert_pred, fert_pred) >= UNFERT_VS_FERT_IOU:
        return True
    if UNFERT_TIP_ON_SEED_SUPPRESS:
        af = _bbox_area(fert_pred)
        au = _bbox_area(unfert_pred)
        if af >= au * UNFERT_FERT_AREA_RATIO_MIN and _unfert_center_inside_fert(
            unfert_pred, fert_pred, UNFERT_FERT_EXPAND_PX
        ):
            return True
    return False


def _filter_predictions(result):
    """Per-class threshold, suppress unfert vs fert overlap, then unfert de-dup."""
    fert_list = []
    unfert_candidates = []

    for p in result.object_prediction_list:
        cls_id = int(p.category.id)
        score = p.score.value
        thr = CONF_THR_FERTILIZED if cls_id == 0 else CONF_THR_UNFERTILIZED
        if score < thr:
            continue
        if cls_id == 0:
            fert_list.append(p)
        else:
            unfert_candidates.append(p)

    if UNFERT_VS_FERT_SUPPRESS and fert_list and unfert_candidates:
        kept_unfert = []
        for u in unfert_candidates:
            drop = False
            for f in fert_list:
                if _suppress_unfert_vs_fert(u, f):
                    drop = True
                    break
            if not drop:
                kept_unfert.append(u)
        unfert_candidates = kept_unfert

    kept = fert_list
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


def compute_seed_counts(result):
    """
    Post-process SAHI detection results to count seeds.
    Per-class confidence: class 0 = CONF_THR_FERTILIZED, class 1 = CONF_THR_UNFERTILIZED.
    """
    count = {0: 0, 1: 0}
    filtered_preds = _filter_predictions(result)
    for p in filtered_preds:
        cls_id = int(p.category.id)
        count[cls_id] += 1

    total_seeds = count[0] + count[1]
    fertilized_seeds = count[0]

    return total_seeds, fertilized_seeds


def apply_normalize_look(src_path: str, dst_path: str) -> bool:
    """
    Map typical editor sliders (-100..100) to OpenCV LAB + unsharp mask.
    Order: exposure -> shadows -> brightness -> contrast -> warmth/tint (LAB A/B) -> sharpness.
    """
    try:
        img = cv2.imread(src_path)
        if img is None:
            logger.error(f"normalize: could not read {src_path}")
            return False

        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        L, A, Bch = cv2.split(lab)
        Lf = L.astype(np.float32)
        lo = L.astype(np.float32)

        # Exposure: multiply luminance (100 ~= strong lift)
        exp_gain = 1.0 + (PP_EXPOSURE / 100.0) * 1.5
        Lf = Lf * exp_gain

        # Shadows: negative = darken shadow regions
        shadow_w = np.clip((118.0 - lo) / 118.0, 0.0, 1.0) ** 1.15
        shadow_factor = 1.0 + (PP_SHADOWS / 100.0) * 0.55
        Lf = Lf * (1.0 - shadow_w) + Lf * shadow_w * shadow_factor

        # Brightness: offset on L
        Lf = Lf + (PP_BRIGHTNESS / 100.0) * 52.0

        # Contrast around mid gray
        c = 1.0 + (PP_CONTRAST / 100.0) * 1.2
        mid = 128.0
        Lf = (Lf - mid) * c + mid
        Lf = np.clip(Lf, 0, 255)

        Af = A.astype(np.float32)
        Bf = Bch.astype(np.float32)

        # Warmth: negative = cool (lower LAB b toward blue in OpenCV)
        w = PP_WARMTH / 100.0
        Bf = Bf + w * 20.0
        Af = Af - w * 5.0

        # Tint: shift green-magenta on A (higher A ~ more magenta/red)
        Bf = Bf + (PP_TINT / 100.0) * 4.0
        Af = Af + (PP_TINT / 100.0) * 22.0

        L2 = Lf.astype(np.uint8)
        A2 = np.clip(Af, 0, 255).astype(np.uint8)
        B2 = np.clip(Bf, 0, 255).astype(np.uint8)
        lab2 = cv2.merge((L2, A2, B2))
        bgr = cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)

        # Sharpness: unsharp mask
        s = PP_SHARPNESS / 100.0
        if s > 0.02:
            u8 = bgr
            sigma = 1.0 + s * 1.8
            blur = cv2.GaussianBlur(u8, (0, 0), sigmaX=sigma)
            amount = 0.75 + s * 1.35
            bgr = cv2.addWeighted(u8, 1.0 + amount, blur, -amount, 0)

        ok = cv2.imwrite(dst_path, bgr, [cv2.IMWRITE_JPEG_QUALITY, OUTPUT_JPEG_QUALITY])
        if not ok:
            logger.error("normalize: cv2.imwrite failed")
            return False
        logger.info(
            "Applied PREPROCESS_NORMALIZE (editor-style): "
            f"brightness={PP_BRIGHTNESS} exposure={PP_EXPOSURE} contrast={PP_CONTRAST} "
            f"shadows={PP_SHADOWS} warmth={PP_WARMTH} tint={PP_TINT} sharpness={PP_SHARPNESS}"
        )
        return True
    except Exception as e:
        logger.error(f"apply_normalize_look failed: {e}", exc_info=True)
        return False


def prepare_pipeline_image(temp_dir: str, input_path: str) -> str:
    """Classifier + detection + boxes use this path (normalized JPEG or original)."""
    if not PREPROCESS_NORMALIZE:
        return input_path
    dst = os.path.join(temp_dir, "normalized.jpg")
    if apply_normalize_look(input_path, dst):
        return dst
    logger.warning("Normalization failed; using original download")
    return input_path


def cleanup_temp_images(input_path: str, work_path=None) -> None:
    paths = {input_path}
    if work_path:
        paths.add(work_path)
    for p in paths:
        try:
            if p and os.path.exists(p):
                os.remove(p)
        except OSError as e:
            logger.warning(f"Could not remove temp file {p}: {e}")


def remove_background(image_path: str, output_path: str) -> bool:
    """
    Remove background from image using rembg or cv2 fallback.
    
    Args:
        image_path: Path to input image
        output_path: Path to save output image with removed background (PNG format)
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Try using rembg library (best quality)
        try:
            from rembg import remove
            with open(image_path, 'rb') as input_file:
                input_data = input_file.read()
                output_data = remove(input_data)
            with open(output_path, 'wb') as output_file:
                output_file.write(output_data)
            logger.info("Background removed using rembg")
            return True
        except ImportError:
            logger.warning("rembg not available, using cv2 fallback")
        except Exception as e:
            logger.warning(f"rembg failed: {e}, using cv2 fallback")
        
        # Fallback: Use cv2 with grabcut algorithm
        img = cv2.imread(image_path)
        if img is None:
            logger.error(f"Failed to read image: {image_path}")
            return False
        
        # Convert to RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        height, width = img_rgb.shape[:2]
        
        # Create mask using grabcut (simple approach: assume center region is foreground)
        mask = np.zeros((height, width), np.uint8)
        bgd_model = np.zeros((1, 65), np.float64)
        fgd_model = np.zeros((1, 65), np.float64)
        
        # Define rectangle (slightly smaller than image to avoid edges)
        rect = (int(width * 0.1), int(height * 0.1), int(width * 0.8), int(height * 0.8))
        
        # Apply grabcut
        cv2.grabCut(img_rgb, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)
        
        # Create mask where sure and likely foreground
        mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
        
        # Create 4-channel image (RGBA) with transparency
        img_rgba = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2RGBA)
        img_rgba[:, :, 3] = mask2 * 255
        
        # Save as PNG to preserve transparency
        cv2.imwrite(output_path, cv2.cvtColor(img_rgba, cv2.COLOR_RGBA2BGRA))
        logger.info("Background removed using cv2 grabcut")
        return True
        
    except Exception as e:
        logger.error(f"Error removing background: {e}", exc_info=True)
        # If background removal fails, just copy original image
        import shutil
        shutil.copy2(image_path, output_path)
        return False

def draw_bounding_boxes_no_text(image_path: str, result, output_path: str):
    """
    Draw bounding boxes on image with different colors for fertilized and unfertilized seeds.
    No text labels are drawn, only colored bounding boxes.
    
    Args:
        image_path: Path to input image (can be PNG with transparency or JPG)
        result: SAHI prediction result object
        output_path: Path to save annotated image
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Read image (handles PNG with alpha channel)
        img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            logger.error(f"Failed to read image: {image_path}")
            return False
        
        # Handle images with alpha channel (PNG with transparency)
        if img.shape[2] == 4:
            # Convert RGBA to RGB by compositing on white background
            alpha = img[:, :, 3] / 255.0
            img_rgb = img[:, :, :3].astype(np.float32)
            # Composite on white background
            img_rgb = (img_rgb * alpha[:, :, np.newaxis] + 255.0 * (1 - alpha[:, :, np.newaxis])).astype(np.uint8)
            img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB)
        elif img.shape[2] == 3:
            # Regular BGR image
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            logger.error(f"Unsupported image format: {img.shape}")
            return False
        
        # Define colors (RGB format)
        # Fertilized (class 0): Green
        # Unfertilized (class 1): Red
        COLOR_FERTILIZED = (0, 255, 0)      # Green
        COLOR_UNFERTILIZED = (255, 0, 0)    # Red
        
        # Box thickness (adaptive based on image size)
        box_thickness = max(2, int(min(img_rgb.shape[0], img_rgb.shape[1]) / 400))
        
        # Draw bounding boxes after same filtering used for counts.
        for prediction in _filter_predictions(result):
            cls_id = int(prediction.category.id)

            # Get bounding box coordinates
            bbox = prediction.bbox
            x1, y1 = int(bbox.minx), int(bbox.miny)
            x2, y2 = int(bbox.maxx), int(bbox.maxy)

            # Choose color based on class
            if cls_id == 0:  # Fertilized
                color = COLOR_FERTILIZED
            else:  # Unfertilized (class 1)
                color = COLOR_UNFERTILIZED
            
            # Draw rectangle (no text)
            cv2.rectangle(img_rgb, (x1, y1), (x2, y2), color, box_thickness)
        
        # Convert RGB back to BGR for saving
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        
        # Save image
        cv2.imwrite(output_path, img_bgr, [cv2.IMWRITE_JPEG_QUALITY, OUTPUT_JPEG_QUALITY])
        logger.info(f"Annotated image saved to {output_path}")
        return True
        
    except Exception as e:
        logger.error(f"Error drawing bounding boxes: {e}", exc_info=True)
        return False

def calculate_fertilization_percentage(fertilized_seeds: int, total_seeds: int) -> float:
    """
    Calculate fertilization percentage: (F/T) × 100
    
    Args:
        fertilized_seeds (F): number of fertilized seeds
        total_seeds (T): total number of detected seeds
        
    Returns:
        float: fertilization percentage, rounded to 2 decimal places
               Returns 0.0 if total_seeds == 0 (edge case)
    """
    if total_seeds == 0:
        return 0.0
    
    percentage = (fertilized_seeds / total_seeds) * 100.0
    return round(percentage, 2)

def format_results(total_seeds: int, fertilized_seeds: int, fertilization_percentage: float) -> str:
    """
    Format the Telegram response with seed analysis results.
    
    Args:
        total_seeds (T): total number of detected seeds
        fertilized_seeds (F): number of fertilized seeds
        fertilization_percentage: fertilization percentage (0.0-100.0)
        
    Returns:
        str: Formatted message for Telegram
    """
    if total_seeds == 0:
        return (
            "🌻 **Sunflower Seed Analysis Results**\n\n"
            "📊 Total seeds detected: 0\n"
            "✅ Fertilized seeds: 0\n"
            "📈 Fertilization rate: 0.00%\n\n"
            "⚠️ **No seeds were detected in this image.**\n"
            "Please ensure the image contains a clear view of sunflower seeds."
        )
    
    # Choose emoji based on fertilization rate
    if fertilization_percentage >= 80:
        rate_emoji = "✅"  # Excellent
    elif fertilization_percentage >= 60:
        rate_emoji = "✅"  # Good
    elif fertilization_percentage >= 40:
        rate_emoji = "✅"  # Moderate
    else:
        rate_emoji = "✅"  # Low
    
    return (
        "🌻 **Sunflower Seed Analysis Results**\n\n"
        f"📊 Total seeds detected: {total_seeds}\n"
        f"✅ Developed seeds: {fertilized_seeds}\n"
        f"📈 Fertilization rate: {fertilization_percentage:.2f}% {rate_emoji}"
    )

def is_sunflower_image(image_path, threshold=0.5):
    """Check if the image is a sunflower using classifier model.
    
    Classifier model classes:
    - Class 0: "other" (not sunflower)
    - Class 1: "sunflower"
    
    Returns True if the image is classified as a sunflower, False otherwise.
    If classifier is not available or fails, returns True to allow processing.
    """
    if classifier_model is None:
        # If classifier is not available, skip check and allow processing
        logger.info("Classifier model not available, skipping validation")
        return True
    
    try:
        # Run classification
        logger.info(f"Running classifier on image: {image_path}")
        results = classifier_model(image_path, verbose=False)
        
        # Get the first result (single image)
        result = results[0]
        
        # Check if it's a classification result (not detection)
        if hasattr(result, 'probs'):
            probs = result.probs
            
            # Get class names from the model
            class_names = result.names if hasattr(result, 'names') else {0: 'other', 1: 'sunflower'}
            logger.info(f"Classifier class names: {class_names}")
            
            # Get top prediction
            top1_idx = int(probs.top1) if hasattr(probs, 'top1') else int(np.argmax(probs.data.cpu().numpy()))
            top1_conf = float(probs.top1conf) if hasattr(probs, 'top1conf') else float(probs.data.cpu().numpy().max())
            top1_name = class_names.get(top1_idx, str(top1_idx))
            
            logger.info(f"🔍 Classifier result: class={top1_idx} ({top1_name}), confidence={top1_conf:.3f}")
            
            # Check if it's classified as sunflower (class 1)
            # The classifier has: 0='other', 1='sunflower'
            is_sunflower = (top1_idx == 1 and top1_conf >= threshold)
            
            # Also check by name in case class indices are different
            if not is_sunflower and "sunflower" in top1_name.lower():
                is_sunflower = top1_conf >= threshold
            
            # If top class is "other", also check sunflower probability directly
            if not is_sunflower:
                # Get probability for sunflower class (class 1)
                prob_data = probs.data.cpu().numpy() if hasattr(probs.data, 'cpu') else np.array(probs.data)
                if len(prob_data) >= 2:
                    sunflower_prob = float(prob_data[1])
                    logger.info(f"Sunflower class probability: {sunflower_prob:.3f}")
                    is_sunflower = sunflower_prob >= threshold
            
            if is_sunflower:
                logger.info(f"✅ Image ACCEPTED as sunflower (class={top1_idx}, name={top1_name}, conf={top1_conf:.3f})")
            else:
                logger.info(f"❌ Image REJECTED: Not sunflower (class={top1_idx}, name={top1_name}, conf={top1_conf:.3f})")
            
            return is_sunflower
        else:
            # Not a classification model, allow processing
            logger.warning("Classifier model does not appear to be a classification model (no probs attribute)")
            return True
            
    except Exception as e:
        logger.error(f"Error in classifier check: {e}", exc_info=True)
        # On error, allow processing (don't block on classifier failure)
        return True

async def process_image(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Process sunflower image and return counts."""
    temp_dir = None
    status_msg = None
    try:
        # Send processing message
        status_msg = await _retry_tg(
            "reply_text(processing)",
            lambda: update.message.reply_text("🔄 Processing image... Please wait.")
        )
        
        # Download image with timeout handling
        try:
            photo_file = await _retry_tg("get_file(photo)", lambda: update.message.photo[-1].get_file())
        except (TimedOut, NetworkError) as e:
            logger.error(f"Error getting photo file: {e}")
            await _retry_tg("edit_text(download_failed)", lambda: status_msg.edit_text("❌ Error downloading image. Please try again."))
            return
        
        # Create temporary directory
        temp_dir = tempfile.mkdtemp()
        input_path = os.path.join(temp_dir, "input_image.jpg")
        
        # Download image to temp file with retry
        try:
            await _retry_tg("download_to_drive(photo)", lambda: photo_file.download_to_drive(input_path))
        except (TimedOut, NetworkError, asyncio.TimeoutError, TimeoutError) as e:
            logger.error(f"Error downloading image to drive: {e}")
            await _retry_tg("edit_text(download_failed2)", lambda: status_msg.edit_text("❌ Error downloading image. Please try again."))
            try:
                os.rmdir(temp_dir)
            except:
                pass
            return
        except Exception as e:
            logger.error(f"Error downloading image: {e}")
            await _retry_tg("edit_text(download_failed3)", lambda: status_msg.edit_text("❌ Error downloading image. Please try again."))
            try:
                os.rmdir(temp_dir)
            except:
                pass
            return
            
        logger.info(f"Image downloaded to {input_path}")

        work_path = prepare_pipeline_image(temp_dir, input_path)
        if work_path != input_path:
            logger.info(f"Using normalized image for pipeline: {work_path}")

        # ================= CHECK IF SUNFLOWER =================
        # Skip classifier if SKIP_CLASSIFIER is set for faster processing (saves 20+ seconds!)
        if SKIP_CLASSIFIER:
            logger.info("⚡ Skipping classifier check (SKIP_CLASSIFIER enabled) - saving 20+ seconds!")
            is_sunflower = True
        else:
            try:
                await _retry_tg("edit_text(check_sunflower)", lambda: status_msg.edit_text("🔍 Checking if image is a sunflower..."))
            except:
                pass
                
            try:
                is_sunflower = is_sunflower_image(work_path)
            except Exception as e:
                logger.error(f"Error in classifier check: {e}")
                is_sunflower = True  # On error, allow processing
            
        if not is_sunflower:
            try:
                await _retry_tg("delete(status)", lambda: status_msg.delete())
            except:
                pass
            await _retry_tg(
                "reply_text(not_sunflower)",
                lambda: update.message.reply_text(
                    "❌ **This image doesn't appear to be a sunflower.**\n\n"
                    "Please send a sunflower image to count seeds.\n"
                    "The bot only processes sunflower images.",
                    parse_mode='Markdown'
                )
            )
            # Clean up temp files
            try:
                cleanup_temp_images(input_path, work_path)
                os.rmdir(temp_dir)
            except:
                pass
            return
        
        # ================= SAHI INFERENCE =================
        logger.info("✅ Classifier accepted image, starting SAHI inference...")
        logger.info(
            f"   SAHI config: SLICE_SIZE={SLICE_SIZE}, OVERLAP={OVERLAP}, "
            f"CONF_FERT={CONF_THR_FERTILIZED} CONF_UNFERT={CONF_THR_UNFERTILIZED} model_min={CONF_THR_MODEL_MIN}, DEVICE={DEVICE}"
        )
        
        # Update status message to show SAHI is starting
        try:
            await _retry_tg("edit_text(sahi_start)", lambda: status_msg.edit_text("🔄 Running detection... This may take 30-100 seconds on CPU."))
        except:
            pass
        
        import time
        start_time = time.time()
        logger.info(f"⏱️ SAHI inference started at {time.strftime('%H:%M:%S')}")
        
        # Run SAHI inference in executor to avoid blocking
        async def run_sahi_inference():
            """Run SAHI inference in thread pool to avoid blocking."""
            loop = asyncio.get_event_loop()
            wp = work_path
            return await loop.run_in_executor(
                None,
                lambda: get_sliced_prediction(
                    image=wp,
                    detection_model=detection_model,
                    slice_height=SLICE_SIZE,
                    slice_width=SLICE_SIZE,
                    overlap_height_ratio=OVERLAP,
                    overlap_width_ratio=OVERLAP,
                    postprocess_type="NMS",                 # merge duplicates
                    postprocess_match_threshold=NMS_IOU
                )
            )
        
        # Send periodic updates during long processing
        async def send_progress_updates():
            """Send periodic status updates during processing."""
            elapsed = 0
            while True:
                await asyncio.sleep(15)  # Update every 15 seconds
                elapsed += 15
                try:
                    if DEVICE == "cpu":
                        await _retry_tg(
                            "edit_text(progress)",
                            lambda: status_msg.edit_text(f"🔄 Still processing... ({elapsed}s elapsed) This may take 30-100 seconds on CPU.")
                        )
                    else:
                        await _retry_tg(
                            "edit_text(progress)",
                            lambda: status_msg.edit_text(f"🔄 Still processing... ({elapsed}s elapsed)")
                        )
                except:
                    pass
        
        try:
            logger.info(f"📥 Calling get_sliced_prediction with image={work_path}")
            # Start progress updates task
            progress_task = asyncio.create_task(send_progress_updates())
            
            # Run inference with timeout (5 minutes max)
            try:
                result = await asyncio.wait_for(run_sahi_inference(), timeout=300.0)
            except asyncio.TimeoutError:
                progress_task.cancel()
                error_msg = "❌ Processing timed out. The image might be too large. Please try again with a smaller image."
                try:
                    await status_msg.edit_text(error_msg)
                except:
                    await update.message.reply_text(error_msg)
                logger.error("SAHI inference timed out after 5 minutes")
                # Clean up
                try:
                    cleanup_temp_images(input_path, work_path)
                    os.rmdir(temp_dir)
                except:
                    pass
                return
            
            # Cancel progress updates
            progress_task.cancel()
            try:
                await progress_task
            except asyncio.CancelledError:
                pass
            
            elapsed_time = time.time() - start_time
            logger.info(f"✅ SAHI inference completed in {elapsed_time:.2f} seconds (device: {DEVICE}, slice_size: {SLICE_SIZE})")
        except Exception as e:
            logger.error(f"Error in SAHI inference: {e}", exc_info=True)
            error_msg = "❌ Error processing image. The image might be too large or corrupted. Please try again with a smaller image."
            try:
                await status_msg.edit_text(error_msg)
            except:
                await update.message.reply_text(error_msg)
            # Clean up
            try:
                cleanup_temp_images(input_path, work_path)
                os.rmdir(temp_dir)
            except:
                pass
            return
        
        logger.info(f"Total raw detections collected: {len(result.object_prediction_list)}")
        
        # Debug: Log detection details
        if len(result.object_prediction_list) > 0:
            logger.info(f"🔍 Sample detections (first 5):")
            for i, p in enumerate(result.object_prediction_list[:5]):
                logger.info(f"  Detection {i}: class={p.category.id}, score={p.score.value:.3f}, bbox={p.bbox}")
        else:
            logger.warning("⚠️ WARNING: No detections found by SAHI! Model may not be detecting seeds.")
            logger.warning(f"   Image path: {work_path}")
            logger.warning(
                f"   Using CONF_FERT={CONF_THR_FERTILIZED} CONF_UNFERT={CONF_THR_UNFERTILIZED}, SLICE_SIZE={SLICE_SIZE}, DEVICE={DEVICE}"
            )
        
        # ================= COUNT SEEDS =================
        await _retry_tg("edit_text(counting)", lambda: status_msg.edit_text("🔢 Counting seeds..."))
        
        # Compute seed counts using modular function
        total_seeds, fertilized_seeds = compute_seed_counts(result)
        logger.info(f"📊 Counted seeds: Total={total_seeds}, Fertilized={fertilized_seeds}, Unfertilized={total_seeds - fertilized_seeds}")
        
        # Calculate fertilization percentage
        fertilization_percentage = calculate_fertilization_percentage(fertilized_seeds, total_seeds)
        
        # Format results
        result_text = format_results(total_seeds, fertilized_seeds, fertilization_percentage)
        
        # ================= DRAW BOUNDING BOXES =================
        await _retry_tg("edit_text(annotating)", lambda: status_msg.edit_text("🎨 Drawing bounding boxes..."))
        
        # Draw on same pixels the detector saw (normalized if enabled)
        annotated_path = os.path.join(temp_dir, "annotated.jpg")
        logger.info(f"Drawing bounding boxes on image: {work_path}")
        boxes_drawn = draw_bounding_boxes_no_text(work_path, result, annotated_path)
        logger.info(f"Bounding boxes drawn: {boxes_drawn}, output path: {annotated_path}, exists: {os.path.exists(annotated_path) if boxes_drawn else False}")
        
        # Delete status message
        try:
            await _retry_tg("delete(status2)", lambda: status_msg.delete())
        except:
            pass
        
        # Send annotated image with results
        if boxes_drawn and os.path.exists(annotated_path):
            try:
                logger.info(f"Sending annotated image: {annotated_path}")
                caption = result_text + "\n\n🟢 Green boxes = Fertilized\n🔴 Red boxes = Unfertilized"
                # Use file path directly (telegram library handles file opening/closing)
                await _retry_tg(
                    "reply_photo(result)",
                    lambda: update.message.reply_photo(
                        photo=annotated_path,
                        caption=caption,
                        parse_mode='Markdown'
                    )
                )
                logger.info("✅ Annotated image sent successfully")
            except Exception as e:
                logger.error(f"Error sending annotated image: {e}", exc_info=True)
                # Fallback: send text only if image fails
                await _retry_tg(
                    "reply_text(result)",
                    lambda: update.message.reply_text(
                        result_text,
                        parse_mode='Markdown'
                    )
                )
        else:
            # If annotation failed, just send text results
            logger.warning("Failed to create annotated image, sending text only")
            await _retry_tg(
                "reply_text(result)",
                lambda: update.message.reply_text(
                    result_text,
                    parse_mode='Markdown'
                )
            )
        
        logger.info(f"Processed image: Fertilized={fertilized_seeds}, Total={total_seeds}, Percentage={fertilization_percentage:.2f}%")
        
        # Clean up temp files
        try:
            cleanup_temp_images(input_path, work_path)
            if os.path.exists(annotated_path):
                os.remove(annotated_path)
            if os.path.exists(temp_dir):
                os.rmdir(temp_dir)
        except Exception as e:
            logger.warning(f"Failed to cleanup temp files: {e}")
            
    except (TimedOut, NetworkError, asyncio.TimeoutError, TimeoutError) as e:
        error_msg = "❌ Request timed out. Please try again with a smaller image or check your connection."
        logger.error(f"Timeout error: {e}", exc_info=True)
        try:
            if status_msg:
                await _retry_tg("edit_text(timeout)", lambda: status_msg.edit_text(error_msg))
            else:
                await _retry_tg("reply_text(timeout)", lambda: update.message.reply_text(error_msg))
        except:
            try:
                await update.message.reply_text(error_msg)
            except:
                pass
    except Exception as e:
        if _is_timeout_error(e):
            error_msg = "❌ Request timed out. Please try again with a smaller image or check your connection."
        else:
            error_msg = f"❌ Error processing image: {str(e)}"
        logger.error(error_msg, exc_info=True)
        try:
            if status_msg:
                await _retry_tg("edit_text(error)", lambda: status_msg.edit_text(error_msg))
            else:
                await _retry_tg("reply_text(error)", lambda: update.message.reply_text(error_msg))
        except:
            try:
                await update.message.reply_text(error_msg)
            except:
                pass
    finally:
        if temp_dir and os.path.exists(temp_dir):
            try:
                if "input_path" in locals():
                    cleanup_temp_images(input_path, locals().get("work_path"))
                ann = os.path.join(temp_dir, "annotated.jpg")
                if os.path.exists(ann):
                    os.remove(ann)
                os.rmdir(temp_dir)
            except Exception as e:
                logger.warning(f"Failed to cleanup temp files: {e}")

async def handle_document(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle document (image file) uploads."""
    temp_dir = None
    status_msg = None
    try:
        document = update.message.document
        
        # Check if it's an image
        if document.mime_type and document.mime_type.startswith('image/'):
            # Send processing message
            status_msg = await _retry_tg(
                "reply_text(processing_doc)",
                lambda: update.message.reply_text("🔄 Processing image... Please wait.")
            )
            
            # Download document with timeout handling
            try:
                file = await _retry_tg("get_file(document)", lambda: document.get_file())
            except (TimedOut, NetworkError) as e:
                logger.error(f"Error getting file: {e}")
                await _retry_tg("edit_text(doc_download_failed)", lambda: status_msg.edit_text("❌ Error downloading file. Please try again."))
                return
            
            # Create temporary directory
            temp_dir = tempfile.mkdtemp()
            input_path = os.path.join(temp_dir, "input_image.jpg")
            
            # Download file to temp location with retry
            try:
                await _retry_tg("download_to_drive(doc)", lambda: file.download_to_drive(input_path))
            except (TimedOut, NetworkError, asyncio.TimeoutError, TimeoutError) as e:
                logger.error(f"Error downloading file to drive: {e}")
                await _retry_tg("edit_text(doc_download_failed2)", lambda: status_msg.edit_text("❌ Error downloading file. Please try again."))
                try:
                    os.rmdir(temp_dir)
                except:
                    pass
                return
            except Exception as e:
                logger.error(f"Error downloading file: {e}")
                await _retry_tg("edit_text(doc_download_failed3)", lambda: status_msg.edit_text("❌ Error downloading file. Please try again."))
                try:
                    os.rmdir(temp_dir)
                except:
                    pass
                return
                
            logger.info(f"Document downloaded to {input_path}")

            work_path = prepare_pipeline_image(temp_dir, input_path)
            if work_path != input_path:
                logger.info(f"Using normalized image for pipeline: {work_path}")

            # ================= CHECK IF SUNFLOWER =================
            try:
                await _retry_tg("edit_text(check_sunflower_doc)", lambda: status_msg.edit_text("🔍 Checking if image is a sunflower..."))
            except:
                pass
                
            try:
                is_sunflower = is_sunflower_image(work_path)
            except Exception as e:
                logger.error(f"Error in classifier check: {e}")
                is_sunflower = True  # On error, allow processing
                
            if not is_sunflower:
                try:
                    await _retry_tg("delete(status_doc)", lambda: status_msg.delete())
                except:
                    pass
                await _retry_tg(
                    "reply_text(not_sunflower_doc)",
                    lambda: update.message.reply_text(
                        "❌ **This image doesn't appear to be a sunflower.**\n\n"
                        "Please send a sunflower image to count seeds.\n"
                        "The bot only processes sunflower images.",
                        parse_mode='Markdown'
                    )
                )
                # Clean up temp files
                try:
                    cleanup_temp_images(input_path, work_path)
                    os.rmdir(temp_dir)
                except:
                    pass
                return
            
            # ================= SAHI INFERENCE =================
            logger.info("✅ Classifier accepted image, starting SAHI inference...")
            logger.info(
            f"   SAHI config: SLICE_SIZE={SLICE_SIZE}, OVERLAP={OVERLAP}, "
            f"CONF_FERT={CONF_THR_FERTILIZED} CONF_UNFERT={CONF_THR_UNFERTILIZED} model_min={CONF_THR_MODEL_MIN}, DEVICE={DEVICE}"
        )
            
            # Update status message to show SAHI is starting
            try:
                await _retry_tg("edit_text(sahi_start_doc)", lambda: status_msg.edit_text("🔄 Running detection... This may take 30-100 seconds on CPU."))
            except:
                pass
            
            import time
            start_time = time.time()
            logger.info(f"⏱️ SAHI inference started at {time.strftime('%H:%M:%S')}")
            
            # Run SAHI inference in executor to avoid blocking
            async def run_sahi_inference_doc():
                """Run SAHI inference in thread pool to avoid blocking."""
                loop = asyncio.get_event_loop()
                wp = work_path
                return await loop.run_in_executor(
                    None,
                    lambda: get_sliced_prediction(
                        image=wp,
                        detection_model=detection_model,
                        slice_height=SLICE_SIZE,
                        slice_width=SLICE_SIZE,
                        overlap_height_ratio=OVERLAP,
                        overlap_width_ratio=OVERLAP,
                        postprocess_type="NMS",                 # merge duplicates
                        postprocess_match_threshold=NMS_IOU
                    )
                )
            
            # Send periodic updates during long processing
            async def send_progress_updates_doc():
                """Send periodic status updates during processing."""
                elapsed = 0
                while True:
                    await asyncio.sleep(15)  # Update every 15 seconds
                    elapsed += 15
                    try:
                        if DEVICE == "cpu":
                            await _retry_tg(
                                "edit_text(progress_doc)",
                                lambda: status_msg.edit_text(f"🔄 Still processing... ({elapsed}s elapsed) This may take 30-100 seconds on CPU.")
                            )
                        else:
                            await _retry_tg(
                                "edit_text(progress_doc)",
                                lambda: status_msg.edit_text(f"🔄 Still processing... ({elapsed}s elapsed)")
                            )
                    except:
                        pass
            
            try:
                logger.info(f"📥 Calling get_sliced_prediction with image={work_path}")
                # Start progress updates task
                progress_task = asyncio.create_task(send_progress_updates_doc())
                
                # Run inference with timeout (5 minutes max)
                try:
                    result = await asyncio.wait_for(run_sahi_inference_doc(), timeout=300.0)
                except asyncio.TimeoutError:
                    progress_task.cancel()
                    error_msg = "❌ Processing timed out. The image might be too large. Please try again with a smaller image."
                    try:
                        await status_msg.edit_text(error_msg)
                    except:
                        await update.message.reply_text(error_msg)
                    logger.error("SAHI inference timed out after 5 minutes")
                    # Clean up
                    try:
                        cleanup_temp_images(input_path, work_path)
                        os.rmdir(temp_dir)
                    except:
                        pass
                    return
                
                # Cancel progress updates
                progress_task.cancel()
                try:
                    await progress_task
                except asyncio.CancelledError:
                    pass
                
                elapsed_time = time.time() - start_time
                logger.info(f"✅ SAHI inference completed in {elapsed_time:.2f} seconds (device: {DEVICE}, slice_size: {SLICE_SIZE})")
                logger.info(f"⏱️ SAHI inference finished at {time.strftime('%H:%M:%S')}")
            except Exception as e:
                logger.error(f"Error in SAHI inference: {e}", exc_info=True)
                error_msg = "❌ Error processing image. The image might be too large or corrupted. Please try again with a smaller image."
                try:
                    await status_msg.edit_text(error_msg)
                except:
                    await update.message.reply_text(error_msg)
                # Clean up
                try:
                    cleanup_temp_images(input_path, work_path)
                    os.rmdir(temp_dir)
                except:
                    pass
                return
            
            logger.info(f"Total raw detections collected: {len(result.object_prediction_list)}")
            
            # Debug: Log detection details
            if len(result.object_prediction_list) > 0:
                logger.info(f"🔍 Sample detections (first 5):")
                for i, p in enumerate(result.object_prediction_list[:5]):
                    logger.info(f"  Detection {i}: class={p.category.id}, score={p.score.value:.3f}, bbox={p.bbox}")
            else:
                logger.warning("⚠️ WARNING: No detections found by SAHI! Model may not be detecting seeds.")
                logger.warning(f"   Image path: {work_path}")
                logger.warning(
                f"   Using CONF_FERT={CONF_THR_FERTILIZED} CONF_UNFERT={CONF_THR_UNFERTILIZED}, SLICE_SIZE={SLICE_SIZE}, DEVICE={DEVICE}"
            )
            
            # ================= COUNT SEEDS =================
            await _retry_tg("edit_text(counting_doc)", lambda: status_msg.edit_text("🔢 Counting seeds..."))
            
            # Compute seed counts using modular function
            total_seeds, fertilized_seeds = compute_seed_counts(result)
            logger.info(f"📊 Counted seeds: Total={total_seeds}, Fertilized={fertilized_seeds}, Unfertilized={total_seeds - fertilized_seeds}")
            
            # Calculate fertilization percentage
            fertilization_percentage = calculate_fertilization_percentage(fertilized_seeds, total_seeds)
            
            # Format results
            result_text = format_results(total_seeds, fertilized_seeds, fertilization_percentage)
            
            # ================= DRAW BOUNDING BOXES =================
            await _retry_tg("edit_text(annotating_doc)", lambda: status_msg.edit_text("🎨 Drawing bounding boxes..."))
            
            annotated_path = os.path.join(temp_dir, "annotated.jpg")
            logger.info(f"Drawing bounding boxes on image: {work_path}")
            boxes_drawn = draw_bounding_boxes_no_text(work_path, result, annotated_path)
            logger.info(f"Bounding boxes drawn: {boxes_drawn}, output path: {annotated_path}, exists: {os.path.exists(annotated_path) if boxes_drawn else False}")
            
            try:
                await _retry_tg("delete(status_doc2)", lambda: status_msg.delete())
            except:
                pass
            
            # Send annotated image with results
            if boxes_drawn and os.path.exists(annotated_path):
                try:
                    logger.info(f"Sending annotated image: {annotated_path}")
                    caption = result_text + "\n\n🟢 Green boxes = Fertilized\n🔴 Red boxes = Unfertilized"
                    # Use file path directly (telegram library handles file opening/closing)
                    await _retry_tg(
                        "reply_photo(result_doc)",
                        lambda: update.message.reply_photo(
                            photo=annotated_path,
                            caption=caption,
                            parse_mode='Markdown'
                        )
                    )
                    logger.info("✅ Annotated image sent successfully")
                except Exception as e:
                    logger.error(f"Error sending annotated image: {e}", exc_info=True)
                    # Fallback: send text only if image fails
                    await _retry_tg(
                        "reply_text(result_doc)",
                        lambda: update.message.reply_text(
                            result_text,
                            parse_mode='Markdown'
                        )
                    )
            else:
                # If annotation failed, just send text results
                logger.warning("Failed to create annotated image, sending text only")
                await _retry_tg(
                    "reply_text(result_doc)",
                    lambda: update.message.reply_text(
                        result_text,
                        parse_mode='Markdown'
                    )
                )
            
            logger.info(f"Processed document: Fertilized={fertilized_seeds}, Total={total_seeds}, Percentage={fertilization_percentage:.2f}%")

            try:
                cleanup_temp_images(input_path, work_path)
                if os.path.exists(annotated_path):
                    os.remove(annotated_path)
                os.rmdir(temp_dir)
            except Exception as e:
                logger.warning(f"Failed to cleanup temp files: {e}")
            
        else:
            await update.message.reply_text("❌ Please send an image file (JPG, PNG, etc.)")
            
    except (TimedOut, NetworkError, asyncio.TimeoutError, TimeoutError) as e:
        error_msg = "❌ Request timed out. Please try again with a smaller image or check your connection."
        logger.error(f"Timeout error: {e}", exc_info=True)
        try:
            if status_msg:
                await _retry_tg("edit_text(timeout_doc)", lambda: status_msg.edit_text(error_msg))
            else:
                await _retry_tg("reply_text(timeout_doc)", lambda: update.message.reply_text(error_msg))
        except:
            try:
                await update.message.reply_text(error_msg)
            except:
                pass
    except Exception as e:
        if _is_timeout_error(e):
            error_msg = "❌ Request timed out. Please try again with a smaller image or check your connection."
        else:
            error_msg = f"❌ Error processing document: {str(e)}"
        logger.error(error_msg, exc_info=True)
        try:
            if status_msg:
                await _retry_tg("edit_text(error_doc)", lambda: status_msg.edit_text(error_msg))
            else:
                await _retry_tg("reply_text(error_doc)", lambda: update.message.reply_text(error_msg))
        except:
            try:
                await update.message.reply_text(error_msg)
            except:
                pass
    finally:
        if temp_dir and os.path.exists(temp_dir):
            try:
                if "input_path" in locals():
                    cleanup_temp_images(input_path, locals().get("work_path"))
                ann = os.path.join(temp_dir, "annotated.jpg")
                if os.path.exists(ann):
                    os.remove(ann)
                os.rmdir(temp_dir)
            except Exception as e:
                logger.warning(f"Failed to cleanup temp files: {e}")

async def log_update(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Log all incoming updates for debugging."""
    if update.message:
        logger.info(f"📨 Received update: message_id={update.message.message_id}, "
                   f"chat_id={update.message.chat.id}, "
                   f"user_id={update.effective_user.id}, "
                   f"text={update.message.text}, "
                   f"has_photo={update.message.photo is not None}, "
                   f"has_document={update.message.document is not None}")
    elif update.callback_query:
        logger.info(f"📨 Received callback_query: {update.callback_query.data}")
    else:
        logger.info(f"📨 Received update: {update}")

async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle errors."""
    logger.error(f"Exception while handling an update: {context.error}", exc_info=context.error)
    
    # Try to send error message to user if possible
    if update and hasattr(update, 'effective_chat'):
        try:
            await context.bot.send_message(
                chat_id=update.effective_chat.id,
                text="❌ An error occurred while processing your request. Please try again."
            )
        except:
            pass
    
    # Handle conflict errors - log but don't exit (let polling retry automatically)
    if isinstance(context.error, Conflict) or (isinstance(context.error, Exception) and "Conflict" in str(context.error)):
        logger.warning("Bot conflict detected - another instance may be running")
        logger.warning("This bot will wait and retry automatically. Please stop the other instance.")
        logger.warning("Telegram only allows ONE bot instance to poll at a time.")
        # Don't exit - let the polling retry mechanism handle it
        # The bot will automatically start working once the conflict is resolved

async def health_check_handler(request):
    """Simple health check endpoint for Railway."""
    return web.Response(text="OK", status=200)

def run_health_server():
    """Start a simple HTTP server for Railway health checks (runs in background thread)."""
    async def start_server():
        try:
            app = web.Application()
            app.router.add_get('/health', health_check_handler)
            app.router.add_get('/', health_check_handler)  # Also respond to root
            
            # Get port from Railway environment variable, default to 8080
            port = int(os.getenv('PORT', '8080'))
            runner = web.AppRunner(app)
            await runner.setup()
            site = web.TCPSite(runner, '0.0.0.0', port)
            await site.start()
            logger.info(f"✅ Health check server started on port {port}")
            print(f"✅ Health check server running on port {port} (for Railway)")
            
            # Keep running
            while True:
                await asyncio.sleep(3600)  # Sleep for 1 hour, then check again
        except Exception as e:
            logger.warning(f"Could not start health check server: {e}")
            print(f"⚠️ Health check server failed (non-critical): {e}")
    
    # Run in new event loop in background thread
    def run_in_thread():
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(start_server())
        except Exception as e:
            logger.warning(f"Health server thread error: {e}")
    
    import threading
    thread = threading.Thread(target=run_in_thread, daemon=True)
    thread.start()
    return thread


def main():
    """Start the bot."""
    logger.info("=" * 60)
    logger.info("Starting Sunflower Seed Counter Telegram Bot")
    logger.info("=" * 60)
    
    if not BOT_TOKEN:
        error_msg = "❌ Error: BOT_TOKEN not found! Please set BOT_TOKEN environment variable or add it to .env file"
        logger.error(error_msg)
        print(error_msg)
        print("\n" + "=" * 60)
        print("📋 HOW TO FIX ON RAILWAY:")
        print("=" * 60)
        print("1. Go to Railway Dashboard → Your Project → Your Service")
        print("2. Click on 'Variables' tab (or 'Environment' / 'Config')")
        print("3. Click '+ New Variable' button")
        print("4. Variable Name: BOT_TOKEN")
        print("5. Variable Value: <paste the token from @BotFather — never commit it>")
        print("6. Click 'Save' or 'Add'")
        print("7. Go to 'Deployments' tab → Click 'Redeploy'")
        print("8. Wait 2-3 minutes and check 'Logs' tab")
        print("=" * 60)
        
        # Debug: Check all environment variables (for debugging, but don't print token)
        logger.debug("Available environment variables:")
        env_vars = [k for k in os.environ.keys() if 'BOT' in k.upper() or 'TOKEN' in k.upper()]
        if env_vars:
            logger.debug(f"Found BOT/TOKEN related env vars: {env_vars}")
        else:
            logger.debug("No BOT_TOKEN or similar variables found in environment")
        
        return
    
    logger.info(f"BOT_TOKEN found: {BOT_TOKEN[:10]}...{BOT_TOKEN[-5:]}")
    print("🤖 Bot is starting...")
    
    try:
        # Create application with timeout settings
        application = Application.builder().token(BOT_TOKEN).build()
        logger.info("Application created successfully")
        
        # Quick connection test using a simple synchronous approach
        # We'll let run_polling handle the actual connection, but verify token format
        if not BOT_TOKEN or len(BOT_TOKEN) < 20:
            logger.error("Invalid BOT_TOKEN format")
            print("❌ Error: Invalid BOT_TOKEN format. Token should be longer than 20 characters.")
            return
        
        logger.info("BOT_TOKEN format validated")
        
        # Configure request timeout settings
        try:
            if hasattr(application.bot, 'request'):
                application.bot.request.timeout = 120  # 2 minutes for requests
                application.bot.request.connect_timeout = 30  # 30 seconds for connection
                logger.info("Bot timeout settings configured: 120s request, 30s connect")
        except Exception as e:
            logger.warning(f"Could not configure bot timeout settings: {e}")
        
        # Add pre-processing handler to log all updates
        application.add_handler(MessageHandler(filters.ALL, log_update), group=-1)
        
        # Register handlers (order matters - more specific first)
        application.add_handler(CommandHandler("start", start))
        application.add_handler(CommandHandler("help", help_command))
        application.add_handler(MessageHandler(filters.PHOTO, process_image))
        application.add_handler(MessageHandler(filters.Document.IMAGE, handle_document))
        application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))
        logger.info("Handlers registered successfully")
        
        # Log handler count
        logger.info(f"Total handlers registered: {len(application.handlers[0])}")
        
        # Add error handler
        application.add_error_handler(error_handler)
        
        # Start health check server for Railway (runs in background thread)
        health_thread = None
        try:
            health_thread = run_health_server()
            import time
            time.sleep(1)  # Give health server a moment to start
        except Exception as e:
            logger.warning(f"Could not start health check server: {e}")
        
        # Start bot
        # Note: We skip pre-flight conflict checks to avoid event loop conflicts
        # Conflict detection will happen during polling via the error handler
        logger.info("Starting bot polling...")
        print("✅ Bot is ready and polling for messages...")
        print("📸 Send any image to analyze (no /start needed!)")
        print("📱 Use /help for commands")
        print("Press Ctrl+C to stop")
        
        # Run polling - this will handle connection verification internally
        # Use retry with exponential backoff for conflicts - bot will stay active and auto-resume
        print("🔄 Bot will automatically retry if conflicts are detected...")
        conflict_retries = 0
        max_conflict_retries = 10
        import time
        
        while conflict_retries < max_conflict_retries:
            try:
                application.run_polling(
                    allowed_updates=Update.ALL_TYPES,
                    drop_pending_updates=True,  # Drop pending updates to avoid conflicts
                    bootstrap_retries=3  # Retry bootstrap (includes conflict handling)
                )
                # If we get here, polling stopped normally (not a conflict)
                break
            except InvalidToken as e:
                logger.error(f"Invalid token detected: {e}")
                print("\n" + "=" * 60)
                print("❌ INVALID BOT TOKEN!")
                print("=" * 60)
                print("The bot token is invalid/expired/revoked.")
                print("You need to get a new token from @BotFather on Telegram.")
                print("=" * 60)
                raise  # Re-raise to be caught by outer handler
            except Conflict as e:
                conflict_retries += 1
                wait_time = min(5 * conflict_retries, 30)  # Exponential backoff, max 30s
                logger.warning(f"Conflict detected (attempt {conflict_retries}/{max_conflict_retries}): {e}")
                print(f"\n⚠️ Bot conflict detected - waiting {wait_time}s before retry...")
                print("💡 Please stop the other bot instance (Railway or local).")
                print("✅ This bot will automatically start working once the conflict is resolved.")
                if conflict_retries < max_conflict_retries:
                    time.sleep(wait_time)
                    logger.info(f"Retrying after {wait_time}s wait (attempt {conflict_retries + 1}/{max_conflict_retries})...")
                    # Rebuild application for retry (needed after run_polling() stops)
                    application = Application.builder().token(BOT_TOKEN).build()
                    if hasattr(application.bot, 'request'):
                        application.bot.request.timeout = 120
                        application.bot.request.connect_timeout = 30
                    application.add_handler(MessageHandler(filters.ALL, log_update), group=-1)
                    application.add_handler(CommandHandler("start", start))
                    application.add_handler(CommandHandler("help", help_command))
                    application.add_handler(MessageHandler(filters.PHOTO, process_image))
                    application.add_handler(MessageHandler(filters.Document.IMAGE, handle_document))
                    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))
                    application.add_error_handler(error_handler)
                    print("🔄 Retrying bot startup...")
                else:
                    logger.error("Max conflict retries reached. Please resolve the conflict manually.")
                    print("\n❌ Max retries reached. Please stop the other bot instance and restart this one.")
                    print("💡 To stop Railway bot: Go to Railway Dashboard → Settings → Delete/Pause service")
                    print("💡 To stop local bot: Press Ctrl+C in the terminal")
                    return
        
    except KeyboardInterrupt:
        logger.info("Bot stopped by user")
        print("\n🛑 Bot stopped by user")
    except InvalidToken as e:
        logger.error(f"Invalid token error: {e}")
        print("\n" + "=" * 60)
        print("❌ INVALID BOT TOKEN ERROR!")
        print("=" * 60)
        print(f"The token `{BOT_TOKEN[:10]}...{BOT_TOKEN[-5:]}` was rejected by Telegram.")
        print("")
        print("📋 HOW TO FIX:")
        print("=" * 60)
        print("1. Open Telegram and search for @BotFather")
        print("2. Send /start to BotFather")
        print("3. Send /newbot to create a new bot (or /token to get existing bot token)")
        print("4. Follow the instructions to get your bot token")
        print("5. Copy the token (format: 123456789:ABCdefGHIjklMNOpqrsTUVwxyz)")
        print("")
        print("6. In Railway Dashboard:")
        print("   - Go to Your Service → Variables tab")
        print("   - Update BOT_TOKEN with your new token")
        print("   - Click 'Save'")
        print("   - Go to Deployments → Click 'Redeploy'")
        print("")
        print("⚠️  The current token is invalid/expired/revoked.")
        print("    You MUST get a new token from @BotFather!")
        print("=" * 60)
    except Conflict as e:
        logger.error(f"Conflict error: {e}")
        print("\n" + "=" * 60)
        print("⚠️ CONFLICT ERROR: Another bot instance is already running!")
        print("=" * 60)
        print("Please stop the other instance before starting this one.")
        print("=" * 60)
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        print(f"❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Health server runs as daemon thread, will exit automatically
        pass

if __name__ == '__main__':
    main()

