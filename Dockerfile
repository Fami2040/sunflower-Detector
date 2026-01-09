# Use Python 3.11 slim image for smaller size
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies for OpenCV and other libraries
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
# Use CPU-only PyTorch for Railway (smaller, faster)
COPY requirements-railway.txt requirements.txt .

# Upgrade pip and install Python dependencies
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements-railway.txt || \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Verify models exist
RUN ls -lh models/ || echo "Warning: Models directory check"

# Run the bot
CMD ["python", "telegram_bot.py"]

