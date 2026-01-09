# Railway Setup - Complete Guide

## What You Need to Add

### BOT_TOKEN Environment Variable

**Your Bot Token:** `8527984904:AAEZSOQ25RMpyRcsYEy1TWxiYeEbZfzDqHY`

## Step-by-Step: Add BOT_TOKEN to Railway

1. **Go to Railway Dashboard**
   - Visit: https://railway.app/dashboard
   - Login with your account

2. **Select Your Project**
   - Find and click on your project (likely "sunflower-Detector")

3. **Open Your Service**
   - Click on the service that's running the bot

4. **Go to Variables Tab**
   - Click on **Variables** (or **Environment** or **Config** tab)
   - Look for existing variables or an empty list

5. **Add New Variable**
   - Click **+ New Variable** or **+ Add Variable** button
   - **Name**: `BOT_TOKEN` (exactly as shown, uppercase)
   - **Value**: `8527984904:AAEZSOQ25RMpyRcsYEy1TWxiYeEbZfzDqHY`
   - Click **Add** or **Save**

6. **Redeploy**
   - Go to **Deployments** tab
   - Find the latest deployment
   - Click **Redeploy** button (or it may auto-redeploy)
   - Wait 2-3 minutes for deployment to complete

7. **Verify**
   - Click on **Logs** tab
   - You should see:
     ```
     ✅ Bot connected: @your_bot_username
     ✅ Bot is ready and polling for messages...
     ```
   - NOT this:
     ```
     ❌ Error: BOT_TOKEN not found!
     ```

## Test Your Bot

After deployment completes:

1. Open Telegram
2. Find your bot (search for the bot username)
3. Send `/start`
4. You should receive a welcome message

## Quick Troubleshooting

**Problem: Can't find Variables tab**
- Make sure you're looking at the correct service
- Try clicking on "Settings" or "Configuration"

**Problem: Variable added but still getting "BOT_TOKEN not found"**
- Make sure you clicked "Redeploy" after adding the variable
- Check that variable name is exactly `BOT_TOKEN` (no spaces)
- Verify the value is the complete token

**Problem: Bot starts but doesn't respond**
- Check logs for errors
- Verify bot is receiving updates (should see "📨 Received update" in logs)
- Make sure only one deployment is active

## Visual Guide

```
Railway Dashboard
  └── Your Project
      └── Your Service
          ├── Variables Tab ← ADD BOT_TOKEN HERE
          ├── Deployments Tab ← REDEPLOY AFTER ADDING
          └── Logs Tab ← CHECK FOR ERRORS
```

---

**I cannot directly access your Railway account**, but these instructions should help you add the token yourself. If you need more help, let me know what specific step you're stuck on!

