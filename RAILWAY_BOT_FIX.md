# Railway Bot Fix - Troubleshooting Guide

## Changes Made

1. **Updated `railway.json`**: Changed from NIXPACKS to DOCKERFILE builder for more reliable builds
2. **Enhanced Error Handling**: Added connection verification and better logging
3. **Improved Startup**: Bot now verifies Telegram connection before starting polling

## What to Check on Railway

### 1. Verify BOT_TOKEN is Set
- Go to your Railway project
- Click on **Variables** tab
- Ensure `BOT_TOKEN` is set with your bot token: `8527984904:AAEZSOQ25RMpyRcsYEy1TWxiYeEbZfzDqHY`
- If missing, click **+ New Variable** and add it

### 2. Check Deployment Logs
- Go to **Deployments** tab
- Click on the latest deployment
- Check the logs for:
  - ✅ `Bot connected: @your_bot_username`
  - ✅ `Bot is ready and polling for messages...`
  - ❌ Any error messages

### 3. Common Issues

**Issue: "BOT_TOKEN not found!"**
- **Solution**: Add BOT_TOKEN in Railway Variables

**Issue: "Failed to connect to Telegram API"**
- **Solution**: Check if BOT_TOKEN is correct and bot is not banned

**Issue: Bot deployed but not responding**
- **Solution**: 
  1. Check logs for errors
  2. Verify bot is running (should see "Bot is ready and polling")
  3. Try sending `/start` to the bot
  4. Check if there are multiple bot instances running

### 4. Manual Redeploy
If needed, trigger a manual redeploy:
- Go to **Settings** → **Deployments**
- Click **Redeploy** on the latest deployment

## Testing the Bot

1. Open Telegram
2. Find your bot (search for the bot username)
3. Send `/start` - should receive welcome message
4. Send a sunflower image - should process and return results

## Next Steps

The changes have been pushed to GitHub. Railway should auto-deploy. Monitor the deployment logs to ensure:
- ✅ Build succeeds
- ✅ Bot starts successfully
- ✅ Connection to Telegram is verified
- ✅ Bot is polling for messages

If issues persist, check the Railway logs and share the error messages.

