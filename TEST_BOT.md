# 🧪 How to Test Your Telegram Bot

## ✅ Step 1: Verify Bot is Running

1. **Go to Railway Dashboard**
2. **Click on your service**
3. **Check "Logs" tab**
4. **Look for these messages:**
   ```
   🔄 Loading detection model on cpu...
   ✅ Detection model loaded successfully
   🔄 Loading classifier model on cpu...
   ✅ Classifier model loaded successfully
   🤖 Bot is starting...
   ```

If you see these, your bot is **running**! ✅

## 🧪 Step 2: Test in Telegram

### Test 1: Start Command
1. **Open Telegram**
2. **Find your bot** (search for the bot name)
3. **Send**: `/start`
4. **Expected response**: Welcome message about sunflower seed counting

### Test 2: Help Command
1. **Send**: `/help`
2. **Expected response**: Help instructions

### Test 3: Send a Sunflower Image
1. **Send a sunflower image** (photo or file)
2. **Expected behavior**:
   - Bot responds: "🔄 Processing image... Please wait."
   - Then: "🔍 Checking if image is a sunflower..."
   - Then: "🔍 Running detection..."
   - Finally: Results with counts:
     ```
     📊 FINAL COUNTS
     🌻 Fertilized : X
     🌱 Unfertilized : Y
     📦 TOTAL SEEDS : Z
     ```

### Test 4: Send Non-Sunflower Image
1. **Send a random image** (not a sunflower)
2. **Expected response**: 
   ```
   ❌ This image doesn't appear to be a sunflower.
   Please send a sunflower image to count seeds.
   ```

## 🔍 Step 3: Check for Issues

### If Bot Doesn't Respond:
- Check Railway logs for errors
- Verify `BOT_TOKEN` is set correctly in Railway
- Check if bot is still running (not crashed)

### If Processing Fails:
- Check Railway logs for specific error
- Verify model files are present
- Check memory/CPU usage in Railway

### If Timeout Errors:
- This is normal for very large images
- Bot will retry automatically
- Try with smaller images

## 📊 Monitoring

**Railway Dashboard:**
- **Logs**: Real-time bot activity
- **Metrics**: CPU/Memory usage
- **Deployments**: Build history

## ✅ Success Indicators

Your bot is working if:
- ✅ Responds to `/start` and `/help`
- ✅ Processes sunflower images
- ✅ Rejects non-sunflower images
- ✅ Returns seed counts
- ✅ Shows processing status messages

---

**Your bot is live! Test it now in Telegram!** 🎉




