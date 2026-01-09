# 🚀 Deployment Guide - Free Hosting Options

This guide covers free hosting options for your Telegram bot.

## Option 1: Railway (Recommended) ⭐

**Free Tier**: $5 credit/month (enough for small bots)

### Steps:
1. Go to [railway.app](https://railway.app) and sign up with GitHub
2. Click "New Project" → "Deploy from GitHub repo"
3. Select your `sunflower-Detector` repository
4. Railway will auto-detect Python and install dependencies
5. Add environment variable:
   - Go to Variables tab
   - Add: `BOT_TOKEN` = `your_telegram_bot_token`
6. Deploy! The bot will start automatically

### Advantages:
- ✅ Easy GitHub integration
- ✅ Automatic deployments
- ✅ Free $5 credit/month
- ✅ Good documentation

---

## Option 2: Render

**Free Tier**: 750 hours/month (enough for 24/7)

### Steps:
1. Go to [render.com](https://render.com) and sign up
2. Click "New" → "Background Worker"
3. Connect your GitHub repository
4. Configure:
   - **Name**: `sunflower-bot`
   - **Environment**: `Python 3`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `python telegram_bot.py`
5. Add environment variable:
   - **Key**: `BOT_TOKEN`
   - **Value**: `your_telegram_bot_token`
6. Click "Create Background Worker"

### Advantages:
- ✅ Truly free tier (750 hours/month)
- ✅ Easy setup
- ✅ Auto-deploy from GitHub

### Note:
- Free tier spins down after 15 minutes of inactivity
- First request after spin-down takes ~30 seconds

---

## Option 3: Fly.io

**Free Tier**: 3 shared VMs, 3GB storage

### Steps:
1. Install Fly CLI: `iwr https://fly.io/install.ps1 -useb | iex`
2. Sign up: `fly auth signup`
3. Create app: `fly launch` (in project directory)
4. Add secrets: `fly secrets set BOT_TOKEN=your_token`
5. Deploy: `fly deploy`

### Advantages:
- ✅ Generous free tier
- ✅ Global edge network
- ✅ Good performance

---

## Option 4: PythonAnywhere

**Free Tier**: Limited, but works for bots

### Steps:
1. Sign up at [pythonanywhere.com](https://www.pythonanywhere.com)
2. Upload your files via Files tab
3. Create a new task in Tasks tab:
   - Command: `python3.10 telegram_bot.py`
   - Hourly schedule
4. Add environment variable in Files → `.env`

### Advantages:
- ✅ Simple interface
- ✅ Free tier available

### Limitations:
- ⚠️ Free tier has restrictions
- ⚠️ May need to upgrade for 24/7 operation

---

## Option 5: Replit

**Free Tier**: Available, but with limitations

### Steps:
1. Go to [replit.com](https://replit.com)
2. Import from GitHub
3. Add `.env` file with `BOT_TOKEN`
4. Click "Run" button

### Advantages:
- ✅ Very easy to use
- ✅ Built-in code editor

### Limitations:
- ⚠️ Free tier may have uptime limits
- ⚠️ May need "Always On" upgrade

---

## Environment Variables

All platforms require you to set:
```
BOT_TOKEN=your_telegram_bot_token_here
```

**Never commit your bot token to GitHub!**

---

## Recommended: Railway or Render

For a production bot, I recommend:
1. **Railway** - Best balance of free credits and ease of use
2. **Render** - Truly free, but may spin down after inactivity

Both integrate easily with GitHub and auto-deploy on push!

---

## Troubleshooting

### Bot stops working:
- Check logs in your hosting platform
- Verify `BOT_TOKEN` is set correctly
- Ensure all dependencies are in `requirements.txt`

### Timeout errors:
- Check if platform has request timeout limits
- Consider upgrading to paid tier for better performance

### Model loading issues:
- Ensure model files are in the repository
- Check file paths in code match deployment structure




