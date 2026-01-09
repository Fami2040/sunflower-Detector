# Railway Bot Debugging Guide

## Changes Made

I've added comprehensive logging and debugging features to help identify why the bot isn't responding:

1. **Update Logging**: All incoming updates are now logged with details
2. **Text Message Handler**: Bot now responds to text messages (not just commands)
3. **Enhanced Error Handling**: Better error messages and logging
4. **Connection Verification**: Bot verifies Telegram connection before starting

## How to Debug on Railway

### Step 1: Check Railway Logs

1. Go to your Railway project dashboard
2. Click on **Deployments** tab
3. Click on the latest deployment
4. Click **View Logs** or check the **Logs** tab

### Step 2: Look for These Key Messages

**✅ Good Signs:**
```
✅ Bot connected: @your_bot_username
✅ Bot is ready and polling for messages...
Total handlers registered: 5
```

**❌ Problem Signs:**
```
❌ Error: BOT_TOKEN not found!
❌ Failed to connect to Telegram API
❌ Fatal error: ...
```

### Step 3: Test the Bot

1. Open Telegram
2. Find your bot
3. Send `/start`
4. Check Railway logs - you should see:
   ```
   📨 Received update: message_id=..., chat_id=..., user_id=..., text=/start
   Received /start command from user ...
   Successfully sent welcome message to user ...
   ```

### Step 4: Common Issues & Solutions

#### Issue 1: "BOT_TOKEN not found!"
**Solution:**
- Go to Railway → **Variables** tab
- Add `BOT_TOKEN` with value: `8527984904:AAEZSOQ25RMpyRcsYEy1TWxiYeEbZfzDqHY`
- Redeploy the service

#### Issue 2: "Failed to connect to Telegram API"
**Possible causes:**
- Invalid BOT_TOKEN
- Network issues
- Bot token revoked

**Solution:**
- Verify BOT_TOKEN is correct
- Check if bot is still active in @BotFather
- Try regenerating the token

#### Issue 3: Bot starts but no logs when sending messages
**Possible causes:**
- Bot is not receiving updates
- Webhook is set (conflicts with polling)
- Multiple bot instances running

**Solution:**
- Check if webhook is set: Go to Railway logs and look for webhook-related errors
- Ensure only one deployment is running
- Check Railway service status

#### Issue 4: Bot receives updates but doesn't respond
**Check logs for:**
- Error messages in the logs
- Handler registration messages
- Exception traces

**Solution:**
- Share the error logs
- Check if models are loading correctly
- Verify all handlers are registered

### Step 5: Manual Testing Checklist

- [ ] Bot starts without errors
- [ ] Connection to Telegram verified
- [ ] `/start` command works
- [ ] `/help` command works
- [ ] Text messages are handled
- [ ] Image uploads are processed

## What the Logs Should Show

When you send `/start` to the bot, you should see in Railway logs:

```
📨 Received update: message_id=123, chat_id=456, user_id=789, text=/start, has_photo=False, has_document=False
Received /start command from user 789 (@username)
Successfully sent welcome message to user 789
```

If you don't see the "📨 Received update" message, the bot is not receiving updates from Telegram.

## Next Steps

1. **Check Railway logs** after the new deployment
2. **Send `/start`** to your bot
3. **Share the logs** if the bot still doesn't respond
4. **Check Railway Variables** to ensure BOT_TOKEN is set

## Quick Commands

To check if bot is running:
- Railway Dashboard → Service → Logs

To redeploy:
- Railway Dashboard → Deployments → Click "Redeploy"

To check variables:
- Railway Dashboard → Variables

---

**Note:** The bot should auto-deploy from GitHub. Wait 2-3 minutes after the push, then check the logs.

