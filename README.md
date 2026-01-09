# 🌻 Sunflower Seed Counter Telegram Bot

A Telegram bot that uses SAHI (Slice, Predict ALL, Merge, and COUNT) to analyze sunflower images and count fertilized and unfertilized seeds.

## Features

- **SAHI Processing**: Slices images into smaller pieces for better detection accuracy
- **Automatic Counting**: Counts fertilized and unfertilized sunflower seeds
- **Image Classification**: Validates that uploaded images are sunflowers before processing
- **Real-time Status Updates**: Shows processing progress with status messages
- **Text-only Results**: Returns clean count statistics (no annotated images)
- **Telegram Integration**: Easy-to-use Telegram bot interface

## Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (optional, but recommended for faster processing)
- Telegram Bot Token (obtain from [@BotFather](https://t.me/BotFather))

## Installation

1. **Clone or navigate to the project directory**

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables:**
   - Copy `.env.example` to `.env`
   - Edit `.env` and add your Telegram bot token:
     ```
     BOT_TOKEN=your_telegram_bot_token_here
     ```

4. **Ensure model files are in place:**
   - The bot expects `models/best.pt` (or `models/best2.pt`) for seed detection
   - The bot expects `models/classifier.pt` for sunflower validation
   - Place your trained YOLO models in the `models/` directory

## Configuration

You can modify these settings in `telegram_bot.py`:

```python
# SAHI slicing parameters
SLICE_SIZE = 800       # smaller → more slices → more recall
OVERLAP = 0.25         # overlap avoids border misses

# Detection thresholds
CONF_THR = 0.15        # confidence threshold (lower = more detections)
NMS_IOU = 0.3          # non-maximum suppression IOU threshold

# Device
DEVICE = "cuda"        # use "cpu" if CUDA not available
```

## Usage

1. **Start the bot:**
   ```bash
   python telegram_bot.py
   ```

2. **In Telegram:**
   - Start a conversation with your bot
   - Send `/start` to see welcome message
   - Send a sunflower image (as photo or document)
   - Watch the processing status messages:
     - 🔍 Checking if image is a sunflower...
     - 🔄 Processing sunflower image...
     - 🔢 Counting seeds...
   - Receive text results with seed counts

## Bot Commands

- `/start` - Start the bot and see welcome message
- `/help` - Show help information

## Output

The bot returns text-only results with count statistics:
- **Fertilized seeds**: Count of class 0
- **Unfertilized seeds**: Count of class 1
- **Total seeds**: Sum of both classes

If a non-sunflower image is uploaded, the bot will reject it with a clear message.

## Model Information

- **Detection Model**: YOLO (Ultralytics) - `models/best.pt` or `models/best2.pt`
  - Class 0: Fertilized seeds
  - Class 1: Unfertilized seeds
- **Classifier Model**: YOLO Classification - `models/classifier.pt`
  - Class 0: Other (not sunflower)
  - Class 1: Sunflower

## Troubleshooting

1. **Model not found error:**
   - Ensure `models/best.pt` exists in the project directory
   - Check that the model file is a valid YOLO model

2. **CUDA errors:**
   - Change `DEVICE = "cpu"` in `telegram_bot.py` if you don't have CUDA

3. **Import errors:**
   - Make sure all dependencies are installed: `pip install -r requirements.txt`

4. **Bot token error:**
   - Verify your bot token is correct in `.env` file
   - Make sure `.env` file exists (copy from `.env.example`)

## License

This project is provided as-is for educational and research purposes.



