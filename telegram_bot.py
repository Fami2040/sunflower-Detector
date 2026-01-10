# =====================================================
# SAHI — Slice, Predict ALL, Merge, and COUNT
# Telegram Bot | Fertilized / Unfertilized Sunflower Seeds
# =====================================================

import cv2
import os
import tempfile
import numpy as np
import asyncio
from pathlib import Path
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from telegram.error import TimedOut, NetworkError
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
import logging
from dotenv import load_dotenv
from ultralytics import YOLO

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
SLICE_SIZE = int(os.getenv("SLICE_SIZE", "800"))       # smaller → more slices → more recall
OVERLAP = float(os.getenv("OVERLAP", "0.25"))          # overlap avoids border misses

# ---- Thresholds (LOW to reduce FN) ----
CONF_THR = float(os.getenv("CONF_THR", "0.05"))        # allow almost everything
NMS_IOU = float(os.getenv("NMS_IOU", "0.3"))           # reasonable merge

# ---- Telegram / performance ----
# Optimized for CPU: smaller max size = faster processing
MAX_IMAGE_SIDE = int(os.getenv("MAX_IMAGE_SIDE", "1200"))  # reduced to 1200 for MUCH faster CPU processing
OUTPUT_JPEG_QUALITY = int(os.getenv("OUTPUT_JPEG_QUALITY", "85"))  # smaller file uploads faster
TG_RETRY_ATTEMPTS = int(os.getenv("TG_RETRY_ATTEMPTS", "3"))

# ---- Performance optimizations ----
# Set SKIP_CLASSIFIER="true" to skip classifier for speed (saves 20+ seconds)
SKIP_CLASSIFIER = os.getenv("SKIP_CLASSIFIER", "false").lower() == "true"  # Set to "true" to skip classifier

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

def downscale_image_inplace(image_path: str, max_side: int = MAX_IMAGE_SIDE) -> None:
    """Downscale big images to reduce processing time and Telegram upload timeouts."""
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Failed to read image (cv2.imread returned None)")
    h, w = img.shape[:2]
    m = max(h, w)
    if m <= max_side:
        return
    scale = max_side / float(m)
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    # Save back as JPEG with decent quality
    cv2.imwrite(image_path, resized, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    logger.info(f"Downscaled image from {w}x{h} -> {new_w}x{new_h} (scale: {scale:.3f}, saved {100*(1-scale**2):.1f}% pixels)")

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
    confidence_threshold=CONF_THR,
    device=DEVICE
)
print("✅ Detection model loaded successfully")

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

# ================= TELEGRAM BOT HANDLERS =================

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Send a message when the command /start is issued."""
    logger.info(f"Received /start command from user {update.effective_user.id} (@{update.effective_user.username})")
    welcome_message = (
        "🌻 **Sunflower Seed Counter Bot**\n\n"
        "Send me a sunflower image and I'll count:\n"
        "• Fertilized seeds\n"
        "• Unfertilized seeds\n\n"
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
    await update.message.reply_text(
        "👋 Please send me a sunflower image to analyze!\n\n"
        "Use /help for more information."
    )

def compute_seed_counts(result):
    """
    Post-process SAHI detection results to count seeds.
    Matches the original script style.
    """
    count = {0: 0, 1: 0}
    
    for p in result.object_prediction_list:
        cls_id = int(p.category.id)
        score = p.score.value
        
        # Final safety filter (VERY LOW threshold)
        if score < CONF_THR:
            continue
        
        count[cls_id] += 1
    
    total_seeds = count[0] + count[1]
    fertilized_seeds = count[0]
    
    return total_seeds, fertilized_seeds

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
        f"✅ Fertilized seeds: {fertilized_seeds}\n"
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

        # Downscale big images early (helps classifier + SAHI + upload)
        try:
            downscale_image_inplace(input_path)
        except Exception as e:
            logger.warning(f"Downscale skipped/failed: {e}")
        
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
                is_sunflower = is_sunflower_image(input_path)
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
                os.remove(input_path)
                os.rmdir(temp_dir)
            except:
                pass
            return
        
        # ================= SAHI INFERENCE =================
        logger.info("✅ Classifier accepted image, starting SAHI inference...")
        logger.info(f"   SAHI config: SLICE_SIZE={SLICE_SIZE}, OVERLAP={OVERLAP}, CONF_THR={CONF_THR}, DEVICE={DEVICE}")
        
        # Update status message to show SAHI is starting
        try:
            await _retry_tg("edit_text(sahi_start)", lambda: status_msg.edit_text("🔄 Running detection... This may take 30-100 seconds."))
        except:
            pass
        
        import time
        start_time = time.time()
        logger.info(f"⏱️ SAHI inference started at {time.strftime('%H:%M:%S')}")
            
        try:
            logger.info(f"📥 Calling get_sliced_prediction with image={input_path}")
            result = get_sliced_prediction(
                image=input_path,
                detection_model=detection_model,
                slice_height=SLICE_SIZE,
                slice_width=SLICE_SIZE,
                overlap_height_ratio=OVERLAP,
                overlap_width_ratio=OVERLAP,
                postprocess_type="NMS",                 # merge duplicates
                postprocess_match_threshold=NMS_IOU
            )
            elapsed_time = time.time() - start_time
            logger.info(f"SAHI inference completed in {elapsed_time:.2f} seconds (device: {DEVICE}, slice_size: {SLICE_SIZE})")
        except Exception as e:
            logger.error(f"Error in SAHI inference: {e}", exc_info=True)
            error_msg = "❌ Error processing image. The image might be too large or corrupted. Please try again with a smaller image."
            try:
                await status_msg.edit_text(error_msg)
            except:
                await update.message.reply_text(error_msg)
            # Clean up
            try:
                os.remove(input_path)
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
            logger.warning(f"   Image path: {input_path}")
            logger.warning(f"   Using CONF_THR={CONF_THR}, SLICE_SIZE={SLICE_SIZE}, DEVICE={DEVICE}")
        
        # ================= COUNT SEEDS =================
        await _retry_tg("edit_text(counting)", lambda: status_msg.edit_text("🔢 Counting seeds..."))
        
        # Compute seed counts using modular function
        total_seeds, fertilized_seeds = compute_seed_counts(result)
        logger.info(f"📊 Counted seeds: Total={total_seeds}, Fertilized={fertilized_seeds}, Unfertilized={total_seeds - fertilized_seeds}")
        
        # Calculate fertilization percentage
        fertilization_percentage = calculate_fertilization_percentage(fertilized_seeds, total_seeds)
        
        # Format results
        result_text = format_results(total_seeds, fertilized_seeds, fertilization_percentage)
        
        # Delete status message and send results
        try:
            await _retry_tg("delete(status2)", lambda: status_msg.delete())
        except:
            pass
        
        # Send text results only (no image)
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
            if os.path.exists(input_path):
                os.remove(input_path)
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
        # Clean up temp files
        if temp_dir and os.path.exists(temp_dir):
            try:
                if 'input_path' in locals() and os.path.exists(input_path):
                    os.remove(input_path)
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

            # Downscale big images early (helps classifier + SAHI + upload)
            try:
                downscale_image_inplace(input_path)
            except Exception as e:
                logger.warning(f"Downscale skipped/failed: {e}")
            
            # ================= CHECK IF SUNFLOWER =================
            try:
                await _retry_tg("edit_text(check_sunflower_doc)", lambda: status_msg.edit_text("🔍 Checking if image is a sunflower..."))
            except:
                pass
                
            try:
                is_sunflower = is_sunflower_image(input_path)
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
                    os.remove(input_path)
                    os.rmdir(temp_dir)
                except:
                    pass
                return
            
            # ================= SAHI INFERENCE =================
            logger.info("✅ Classifier accepted image, starting SAHI inference...")
            logger.info(f"   SAHI config: SLICE_SIZE={SLICE_SIZE}, OVERLAP={OVERLAP}, CONF_THR={CONF_THR}, DEVICE={DEVICE}")
            
            # Update status message to show SAHI is starting
            try:
                await _retry_tg("edit_text(sahi_start_doc)", lambda: status_msg.edit_text("🔄 Running SAHI detection... This may take 30-90 seconds on CPU."))
            except:
                pass
            
            import time
            start_time = time.time()
            logger.info(f"⏱️ SAHI inference started at {time.strftime('%H:%M:%S')}")
            logger.info(f"📥 Calling get_sliced_prediction with image={input_path}")
                
            try:
                result = get_sliced_prediction(
                    image=input_path,
                    detection_model=detection_model,
                    slice_height=SLICE_SIZE,
                    slice_width=SLICE_SIZE,
                    overlap_height_ratio=OVERLAP,
                    overlap_width_ratio=OVERLAP,
                    postprocess_type="NMS",                 # merge duplicates
                    postprocess_match_threshold=NMS_IOU
                )
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
                    os.remove(input_path)
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
                logger.warning(f"   Image path: {input_path}")
                logger.warning(f"   Using CONF_THR={CONF_THR}, SLICE_SIZE={SLICE_SIZE}, DEVICE={DEVICE}")
            
            # ================= COUNT SEEDS =================
            await _retry_tg("edit_text(counting_doc)", lambda: status_msg.edit_text("🔢 Counting seeds..."))
            
            # Compute seed counts using modular function
            total_seeds, fertilized_seeds = compute_seed_counts(result)
            logger.info(f"📊 Counted seeds: Total={total_seeds}, Fertilized={fertilized_seeds}, Unfertilized={total_seeds - fertilized_seeds}")
            
            # Calculate fertilization percentage
            fertilization_percentage = calculate_fertilization_percentage(fertilized_seeds, total_seeds)
            
            # Format results
            result_text = format_results(total_seeds, fertilized_seeds, fertilization_percentage)
            
            try:
                await _retry_tg("delete(status_doc2)", lambda: status_msg.delete())
            except:
                pass
            
            # Send text results only (no image)
            await _retry_tg(
                "reply_text(result_doc)",
                lambda: update.message.reply_text(
                    result_text,
                    parse_mode='Markdown'
                )
            )
            
            logger.info(f"Processed document: Fertilized={fertilized_seeds}, Total={total_seeds}, Percentage={fertilization_percentage:.2f}%")
            
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
        # Clean up temp files
        if temp_dir and os.path.exists(temp_dir):
            try:
                if 'input_path' in locals() and os.path.exists(input_path):
                    os.remove(input_path)
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
    
    # Handle conflict errors
    if isinstance(context.error, Exception) and "Conflict" in str(context.error):
        logger.error("Bot conflict detected - another instance may be running")
        print("⚠️ WARNING: Bot conflict detected!")
        print("Make sure only one bot instance is running.")
        print("Stopping this instance...")
        return

async def verify_bot_connection(bot):
    """Verify bot can connect to Telegram API."""
    try:
        bot_info = await bot.get_me()
        logger.info(f"✅ Bot connected successfully: @{bot_info.username} ({bot_info.first_name})")
        print(f"✅ Bot connected: @{bot_info.username}")
        return True
    except Exception as e:
        logger.error(f"❌ Failed to connect to Telegram API: {e}", exc_info=True)
        print(f"❌ Failed to connect to Telegram API: {e}")
        return False

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
        print("5. Variable Value: 8527984904:AAEZSOQ25RMpyRcsYEy1TWxiYeEbZfzDqHY")
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
        
        # Start bot
        logger.info("Starting bot polling...")
        print("✅ Bot is ready and polling for messages...")
        print("📱 Send /start to test the bot")
        print("Press Ctrl+C to stop")
        
        # Run polling - this will handle connection verification internally
        application.run_polling(
            allowed_updates=Update.ALL_TYPES,
            drop_pending_updates=True  # Drop pending updates to avoid conflicts
        )
        
    except KeyboardInterrupt:
        logger.info("Bot stopped by user")
        print("\n🛑 Bot stopped by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        print(f"❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()

