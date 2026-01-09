# 🚀 Try Render.com Instead - It's More Reliable!

Railway keeps failing. **Render.com is often more reliable** for Python bots with ML models.

## Quick Deploy to Render:

### Option 1: Use render.yaml (Automatic)
1. Go to https://render.com
2. Sign up/login (free)
3. Click "New" → "Blueprint"
4. Connect your GitHub repo: `Fami2040/sunflower-Detector`
5. Render will auto-detect `render.yaml`
6. Set environment variable: `BOT_TOKEN=8527984904:AAEZSOQ25RMpyRcsYEy1TWxiYeEbZfzDqHY`
7. Click "Apply" → Deploy!

### Option 2: Manual Setup
1. Go to https://render.com
2. Click "New" → "Background Worker"
3. Connect GitHub repo
4. Settings:
   - **Name**: `sunflower-bot`
   - **Environment**: `Python 3`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `python telegram_bot.py`
   - **Plan**: Free (or paid for better performance)
5. Add Environment Variable:
   - Key: `BOT_TOKEN`
   - Value: `8527984904:AAEZSOQ25RMpyRcsYEy1TWxiYeEbZfzDqHY`
6. Click "Create Background Worker"

## Why Render is Better:

✅ **More reliable** for Python projects  
✅ **Better error messages** in logs  
✅ **Free tier available**  
✅ **Auto-deploys** from GitHub  
✅ **Better documentation**  

## Current render.yaml is Ready!

Your `render.yaml` is already configured. Just connect the repo and deploy!

