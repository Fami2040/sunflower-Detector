# 🔄 Update Bot Token in Railway

## ✅ Token Updated in Code
Your new token has been updated in:
- `telegram_bot.py`
- `run_bot.bat`
- `start_bot_with_token.ps1`
- `set_bot_photo.py`
- `render.yaml`

## 🚀 Update Railway (Choose One Method)

### Method 1: Railway Dashboard (Easiest)

1. **Go to Railway Dashboard**
   - Open https://railway.app
   - Login to your account

2. **Navigate to Your Service**
   - Click on your project
   - Click on your service (the one running the bot)

3. **Update Environment Variable**
   - Click on **"Variables"** tab (or "Environment" / "Config")
   - Find `BOT_TOKEN` in the list
   - Click on it to edit
   - **Update the value to:** `8490011366:AAGsDtjayDyWhf_wXFAqWVDkg5X3kOmx81w`
   - Click **"Save"** or **"Update"**

4. **Redeploy**
   - Go to **"Deployments"** tab
   - Click **"Redeploy"** on the latest deployment
   - Wait 2-3 minutes for deployment to complete

5. **Verify**
   - Go to **"Logs"** tab
   - You should see:
     - ✅ `BOT_TOKEN found: 8490011366...Omx81w`
     - ✅ `Bot is ready and polling for messages...`
     - ✅ `Health check server running on port 8080`

### Method 2: Railway CLI (If Installed)

If you have Railway CLI installed:

```powershell
# Update token
railway variables set BOT_TOKEN=8490011366:AAGsDtjayDyWhf_wXFAqWVDkg5X3kOmx81w

# Redeploy
railway up
```

Or run the script:
```powershell
powershell -ExecutionPolicy Bypass -File update_railway_token.ps1
```

## 🧪 Test Your Bot

After updating and redeploying:

1. **Open Telegram**
2. **Find your bot** (search for the username)
3. **Send `/start`**
4. **Bot should respond!** ✅

## 📝 New Token

**Your new bot token:**
```
8490011366:AAGsDtjayDyWhf_wXFAqWVDkg5X3kOmx81w
```

⚠️ **Keep this token secret!** Never share it publicly.

---

**After updating Railway, your bot should work!** 🎉

