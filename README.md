# 🌻 Sunflower Seed Counter Telegram Bot

A Telegram bot that uses SAHI (Slice, Predict ALL, Merge, and COUNT) to analyze sunflower images and count fertilized and unfertilized seeds.

## Features

- **SAHI Processing**: Slices images into smaller pieces for better detection accuracy
- **Automatic Counting**: Counts fertilized and unfertilized sunflower seeds
- **Annotated Output**: Returns images with bounding boxes and labels
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
   - The bot expects `models/best.pt` to exist
   - Make sure your YOLO model is located at `models/best.pt`

## Configuration

You can modify these settings in `telegram_bot.py`:

```python
# SAHI slicing parameters
SLICE_SIZE = 850       # smaller → more slices → more recall
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
   - Wait for processing
   - Receive annotated image with seed counts

## Bot Commands

- `/start` - Start the bot and see welcome message
- `/help` - Show help information

## Output

The bot returns:
- An annotated image with bounding boxes for detected seeds
- Count statistics:
  - **Fertilized seeds**: Count of class 0
  - **Unfertilized seeds**: Count of class 1
  - **Total seeds**: Sum of both classes

## Model Information

- **Model Type**: YOLO (Ultralytics)
- **Model Path**: `models/best.pt`
- **Classes**:
  - Class 0: Fertilized seeds (red boxes)
  - Class 1: Unfertilized seeds (yellow boxes)

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


