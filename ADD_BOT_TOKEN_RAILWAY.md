# How to Add BOT_TOKEN to Railway

## Step-by-Step Instructions

### Option 1: Using Railway Web Dashboard (Easiest)

1. **Go to Railway Dashboard**
   - Visit: https://railway.app/dashboard
   - Login with your account

2. **Select Your Project**
   - Click on your "sunflower-Detector" project (or the project name you used)

3. **Go to Variables Tab**
   - Click on your service/deployment
   - Click on the **Variables** tab (or **Environment** tab)

4. **Add BOT_TOKEN**
   - Click **+ New Variable** or **+ Add Variable**
   - **Variable Name**: `BOT_TOKEN`
   - **Variable Value**: `8527984904:AAEZSOQ25RMpyRcsYEy1TWxiYeEbZfzDqHY`
   - Click **Add** or **Save**

5. **Redeploy**
   - Go to **Deployments** tab
   - Click **Redeploy** on the latest deployment
   - Wait for deployment to complete (2-3 minutes)

6. **Verify**
   - Check **Logs** tab
   - You should see: `✅ Bot connected: @your_bot_username`
   - Not: `❌ Error: BOT_TOKEN not found!`

---

### Option 2: Using Railway CLI (If Installed)

1. **Install Railway CLI** (if not installed):
   ```bash
   npm i -g @railway/cli
   ```

2. **Login**:
   ```bash
   railway login
   ```

3. **Link Project**:
   ```bash
   railway link
   ```

4. **Add Variable**:
   ```bash
   railway variables set BOT_TOKEN=8527984904:AAEZSOQ25RMpyRcsYEy1TWxiYeEbZfzDqHY
   ```

5. **Redeploy**:
   ```bash
   railway up
   ```

---

## Quick Verification

After adding the token, check the logs. You should see:

**✅ Success:**
```
BOT_TOKEN found: 8527984904...DqHY
🤖 Bot is starting...
✅ Bot connected: @your_bot_username
✅ Bot is ready and polling for messages...
```

**❌ Failure:**
```
❌ Error: BOT_TOKEN not found!
Please set BOT_TOKEN environment variable or add it to .env file
```

---

## Troubleshooting

**If you don't see the Variables tab:**
- Make sure you're looking at the correct service
- Some Railway UI layouts may call it "Environment Variables" or "Config"

**If the variable doesn't save:**
- Make sure there are no extra spaces
- The variable name must be exactly: `BOT_TOKEN` (uppercase)
- The value should start with numbers and contain the full token

**If deployment fails:**
- Check that you clicked "Redeploy" after adding the variable
- Check the build logs for errors
- Ensure the variable is set in the correct service/environment

---

## Security Note

⚠️ **Important**: Never commit your BOT_TOKEN to GitHub!
- The token is already in `.gitignore`
- Only set it in Railway Variables (not in code)
- If your token is ever exposed, regenerate it in @BotFather

---

## Need Help?

If you're still having issues:
1. Screenshot your Railway Variables page
2. Share the Railway deployment logs
3. Describe what happens when you try to add the variable

