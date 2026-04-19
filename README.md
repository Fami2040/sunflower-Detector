🌻 Sunflower Seed Counter (Telegram Bot)

A deep learning–powered system for automatic analysis of sunflower head images, designed to count fertilized and unfertilized seeds with high accuracy.

👉 Try the system directly via Telegram:
https://t.me/sunflower_detector1_bot

👉 Annotated dataset (CVAT, 2500 images):
https://izba-memes.ru/share/y9xGFqCW

🚀 Overview

This project provides an end-to-end pipeline for digital phenotyping of sunflower heads, combining:

Deep learning–based seed detection (YOLO)
Image validation (sunflower vs non-sunflower)
Telegram bot interface for easy access

The system is designed for research and breeding applications, enabling fast and objective seed counting.

✨ Features
🌱 Seed Counting – Detects and counts fertilized & unfertilized seeds
🧠 Image Validation – Filters non-sunflower images automatically
⚡ Fast Processing – Optimized with slicing (SAHI) for high-resolution images
📱 Telegram Bot Interface – No setup needed for end users
📊 Clean Output – Returns structured count statistics (text-only)
🧠 Model Details
Detection Model (YOLO)
Class 0 → Fertilized seeds
Class 1 → Unfertilized seeds
Classifier Model (YOLO Classification)
Class 0 → Non-sunflower
Class 1 → Sunflower
⚙️ Installation
git clone <your-repo-url>
cd <your-project>
pip install -r requirements.txt
Environment setup
cp .env.example .env

Add your Telegram bot token:

BOT_TOKEN=your_telegram_bot_token_here
▶️ Usage

Run the bot:

python telegram_bot.py

Then in Telegram:

Open your bot
Send /start
Upload a sunflower image
Receive seed counts instantly




The bot returns:

Fertilized seeds
Total seeds

Non-sunflower images are automatically rejected.

📦 Project Structure
models/
  ├── best.pt
  ├── best2.pt
  └── classifier.pt

telegram_bot.py
requirements.txt
🧪 Dataset
Annotated using CVAT
~2500 labeled sunflower images
Includes seed-level annotations for detection tasks

📎 Download:
https://izba-memes.ru/share/y9xGFqCW

⚠️ Notes
GPU (CUDA) is recommended for faster inference
Ensure model files are placed in the models/ directory
Designed for research and prototyping purposes
📄 License

This project is provided for research and educational use.
