# 🔑 Set BOT_TOKEN in Railway

## Quick Steps:

1. **Go to Railway Dashboard**
   - https://railway.app
   - Click on your project/service

2. **Click "Variables" tab** (or "Environment" tab)

3. **Add Environment Variable:**
   - Click **"New Variable"** or **"+"** button
   - **Key**: `BOT_TOKEN`
   - **Value**: `8527984904:AAEZSOQ25RMpyRcsYEy1TWxiYeEbZfzDqHY`
   - Click **"Add"** or **"Save"**

4. **Redeploy:**
   - Railway will automatically redeploy when you add the variable
   - OR click **"Redeploy"** button manually

5. **Wait for deployment** (~1-2 minutes)

6. **Check logs** - you should see:
   ```
   ✅ Detection model loaded successfully
   ✅ Classifier model loaded successfully
   🤖 Bot is starting...
   ```

## Visual Guide:

```
Railway Dashboard
  └─ Your Service
      └─ Variables Tab
          └─ New Variable
              ├─ Key: BOT_TOKEN
              ├─ Value: 8527984904:AAEZSOQ25RMpyRcsYEy1TWxiYeEbZfzDqHY
              └─ Add/Save
```

## After Setting:

- Bot will automatically restart
- Check logs to confirm it's running
- Test in Telegram with `/start`

---

**That's it! Your bot will be live in ~1 minute!** 🚀




