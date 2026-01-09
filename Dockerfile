# Use Python 3.11 slim image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies for OpenCV (headless)
# Note: libgl1-mesa-glx replaced by libgl1 in newer Debian
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Upgrade pip first
RUN pip install --upgrade pip setuptools wheel

# Install PyTorch CPU version first (smaller, faster)
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Copy and install other requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt || true

# Reinstall packages that might conflict
RUN pip install --no-cache-dir \
    python-telegram-bot>=22.5 \
    sahi>=0.11.18 \
    opencv-python-headless>=4.8.0 \
    ultralytics>=8.0.0 \
    numpy>=1.24.0 \
    pillow>=10.0.0 \
    python-dotenv>=1.0.0

# Copy application code
COPY . .

# Verify setup
RUN python -c "import torch; import cv2; import telegram; print('✅ All imports successful')" || echo "⚠️ Some imports failed"

# Run the bot
CMD ["python", "telegram_bot.py"]

