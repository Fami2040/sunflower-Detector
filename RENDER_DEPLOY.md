# 🚀 Render.com Deployment - Step by Step Guide

Follow these steps to deploy your bot to Render.com:

## Step 1: Sign Up / Login
1. Go to [render.com](https://render.com)
2. Sign up with your GitHub account (recommended) or email

## Step 2: Create New Background Worker
1. Click the **"New +"** button in the top right
2. Select **"Background Worker"**

## Step 3: Connect Repository
1. If you signed up with GitHub, you'll see your repositories
2. Select **"sunflower-Detector"** repository
3. Click **"Connect"**

## Step 4: Configure the Service
Fill in the following settings:

### Basic Settings:
- **Name**: `sunflower-bot` (or any name you prefer)
- **Region**: Choose closest to you (e.g., `Oregon (US West)`)
- **Branch**: `main` (or `master` if that's your default branch)

### Build & Deploy:
- **Environment**: `Python 3`
- **Build Command**: `pip install -r requirements.txt`
- **Start Command**: `python telegram_bot.py`

### Environment Variables:
Click **"Add Environment Variable"** and add:
- **Key**: `BOT_TOKEN`
- **Value**: `8527984904:AAEZSOQ25RMpyRcsYEy1TWxiYeEbZfzDqHY` (your actual token)

⚠️ **IMPORTANT**: Replace with your actual bot token!

### Advanced Settings (Optional):
- **Auto-Deploy**: `Yes` (deploys automatically on git push)
- **Health Check Path**: Leave empty (not needed for background workers)

## Step 5: Deploy
1. Click **"Create Background Worker"** at the bottom
2. Render will start building and deploying your bot
3. Watch the logs - it will show:
   - Installing dependencies
   - Loading models
   - Bot starting up

## Step 6: Verify It's Working
1. Once deployed, check the logs for: `🤖 Bot is starting...`
2. Go to Telegram and send `/start` to your bot
3. If it responds, deployment is successful! ✅

## Troubleshooting

### Bot not responding:
- Check the logs in Render dashboard
- Verify `BOT_TOKEN` is set correctly
- Make sure all model files are in the repository

### Build fails:
- Check logs for missing dependencies
- Verify `requirements.txt` is correct
- Check Python version compatibility

### Bot stops after inactivity:
- Free tier spins down after 15 minutes of no activity
- First request after spin-down takes ~30 seconds to wake up
- This is normal for free tier

## Monitoring

- **Logs**: View real-time logs in Render dashboard
- **Metrics**: Check CPU/Memory usage
- **Restarts**: Bot auto-restarts on failure

## Auto-Deploy

Once set up, every time you push to GitHub:
1. Render detects the change
2. Automatically rebuilds and redeploys
3. Your bot updates without manual intervention

---

**Your bot will be live at**: Render will keep it running 24/7 (with free tier limitations)

Need help? Check the logs in Render dashboard for detailed error messages.




