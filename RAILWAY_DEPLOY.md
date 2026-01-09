# 🚂 Railway.app Deployment - Step by Step Guide

Railway is actually **EASIER** than Render and has a better free tier!

## Why Railway?
- ✅ **Easier setup** - Auto-detects everything
- ✅ **$5 free credit/month** - Enough for small bots
- ✅ **Better performance** - No spin-down delays
- ✅ **One-click GitHub deploy**

## Step 1: Sign Up
1. Go to [railway.app](https://railway.app)
2. Click **"Start a New Project"**
3. Sign up with **GitHub** (recommended - one click!)

## Step 2: Deploy from GitHub
1. After signing up, you'll see **"New Project"**
2. Click **"Deploy from GitHub repo"**
3. Select your **"sunflower-Detector"** repository
4. Railway will automatically:
   - Detect it's Python
   - Read `requirements.txt`
   - Set up the build

## Step 3: Add Environment Variable
1. Click on your deployed service
2. Go to **"Variables"** tab
3. Click **"New Variable"**
4. Add:
   - **Key**: `BOT_TOKEN`
   - **Value**: `8527984904:AAEZSOQ25RMpyRcsYEy1TWxiYeEbZfzDqHY`
5. Click **"Add"**

## Step 4: Configure Start Command (if needed)
1. Go to **"Settings"** tab
2. Under **"Deploy"** section, set:
   - **Start Command**: `python telegram_bot.py`
3. Railway should auto-detect this, but verify it's set

## Step 5: Deploy!
1. Railway will automatically start building
2. Watch the **"Deployments"** tab
3. You'll see logs showing:
   - Installing dependencies
   - Loading models
   - "🤖 Bot is starting..."

## Step 6: Test
1. Once deployed, check logs for: `Bot is starting...`
2. Go to Telegram
3. Send `/start` to your bot
4. If it responds, you're done! ✅

## That's It!

Railway is much simpler - it auto-detects everything from your GitHub repo!

## Monitoring

- **Logs**: Real-time logs in Railway dashboard
- **Metrics**: CPU, Memory, Network usage
- **Deployments**: See all deployments and rollback if needed

## Auto-Deploy

Every time you push to GitHub:
- Railway detects the change
- Automatically rebuilds and redeploys
- Your bot updates instantly!

## Free Tier Details

- **$5 credit/month** (free)
- **500 hours runtime** (enough for 24/7)
- **Auto-scaling**
- **No spin-down delays** (unlike Render free tier)

---

**Your bot will be live at**: Railway keeps it running 24/7 on free tier!

Need help? Check the logs in Railway dashboard.




