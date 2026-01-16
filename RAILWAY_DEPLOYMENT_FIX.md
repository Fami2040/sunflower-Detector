# 🔧 Railway Deployment Fix - Bot Not Online

## Problem
Railway deployment shows "completed" but bot is not online/working.

## Root Causes
1. **No Health Check**: Railway couldn't verify the service was alive
2. **Missing Dependency**: `aiohttp` needed for health check server
3. **Service Type**: Railway needs to know this is a worker service

## ✅ Fixes Applied

### 1. Added Health Check Server
- Added simple HTTP server on `/health` endpoint
- Railway can now ping this to verify service is alive
- Runs in background thread, doesn't interfere with bot

### 2. Updated Requirements
- Added `aiohttp>=3.9.0` to `requirements.txt`

### 3. Updated Railway Configuration
- Added health check path to `railway.json`
- Railway will now monitor the service properly

## 📋 What You Need to Do

### Step 1: Verify Environment Variables
1. Go to Railway Dashboard → Your Project → Your Service
2. Click "Variables" tab
3. Make sure `BOT_TOKEN` is set:
   - Name: `BOT_TOKEN`
   - Value: `8527984904:AAEZSOQ25RMpyRcsYEy1TWxiYeEbZfzDqHY`

### Step 2: Redeploy
1. Go to Railway Dashboard → Your Service
2. Click "Deployments" tab
3. Click "Redeploy" on the latest deployment
4. Wait 2-3 minutes for build to complete

### Step 3: Check Logs
1. After deployment completes, go to "Logs" tab
2. Look for these messages:
   - ✅ `Health check server running on port 8080 (for Railway)`
   - ✅ `Bot is ready and polling for messages...`
   - ✅ `Starting bot polling...`

### Step 4: Test the Bot
1. Open Telegram
2. Find your bot
3. Send `/start` command
4. If bot responds, it's working! ✅

## 🔍 Troubleshooting

### If Bot Still Not Working:

1. **Check Logs for Errors**:
   - Look for `❌ ERROR` messages
   - Common errors:
     - `BOT_TOKEN not found` → Set environment variable
     - `Model file not found` → Models should be in repo
     - `ImportError` → Check requirements.txt

2. **Check Service Status**:
   - Railway Dashboard → Service → Should show "Active"
   - If "Crashed", check logs for crash reason

3. **Verify Model Files**:
   - Models should be in `models/` directory:
     - `models/best2.pt`
     - `models/classifier.pt`
   - These should be committed to Git

4. **Check Health Endpoint**:
   - Railway provides a public URL
   - Try accessing: `https://your-service.railway.app/health`
   - Should return "OK"

## 📝 Files Changed

1. `telegram_bot.py` - Added health check server
2. `requirements.txt` - Added aiohttp
3. `railway.json` - Added health check configuration

## 🎯 Expected Behavior

After redeploy:
- ✅ Build completes successfully
- ✅ Service shows "Active" status
- ✅ Logs show health server started
- ✅ Logs show bot polling started
- ✅ Bot responds to Telegram messages

## 💡 Additional Notes

- The health check server runs on port 8080 (or Railway's PORT env var)
- It doesn't interfere with the Telegram bot
- If health server fails, bot will still work (non-critical)
- Railway will automatically restart the service if it crashes

---

**If you still have issues after following these steps, check the Railway logs and share the error messages!**

