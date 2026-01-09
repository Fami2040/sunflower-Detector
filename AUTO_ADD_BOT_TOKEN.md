# 🤖 Auto-Add BOT_TOKEN to Railway

## Option 1: Using Railway CLI (Recommended - Fastest)

### Step 1: Install Railway CLI
```bash
npm i -g @railway/cli
```

### Step 2: Login to Railway
```bash
railway login
```

### Step 3: Link Your Project
```bash
railway link
```
(Select your project when prompted)

### Step 4: Run the Auto-Add Script

**On Windows (PowerShell):**
```powershell
.\add_bot_token_railway.ps1
```

**On Mac/Linux (Bash):**
```bash
chmod +x add_bot_token_railway.sh
./add_bot_token_railway.sh
```

**OR manually run these commands:**

```bash
railway variables set BOT_TOKEN=8527984904:AAEZSOQ25RMpyRcsYEy1TWxiYeEbZfzDqHY
railway up
```

That's it! The script will:
1. ✅ Check if Railway CLI is installed
2. ✅ Check if you're logged in
3. ✅ Link your project (if needed)
4. ✅ Set BOT_TOKEN variable
5. ✅ Redeploy your service

---

## Option 2: Manual Method (If CLI doesn't work)

### Quick Steps:
1. Go to: https://railway.app/dashboard
2. Click your project → Click your service
3. Click "Variables" tab
4. Click "+ New Variable"
5. Name: `BOT_TOKEN`
6. Value: `8527984904:AAEZSOQ25RMpyRcsYEy1TWxiYeEbZfzDqHY`
7. Click "Save"
8. Go to "Deployments" tab → Click "Redeploy"

---

## Option 3: Using Railway API (Advanced)

If you have Railway API token:

```bash
# Set your Railway API token
export RAILWAY_TOKEN="your_railway_api_token"

# Get your project ID from Railway dashboard
export PROJECT_ID="your_project_id"

# Add BOT_TOKEN
curl -X POST "https://api.railway.app/v1/variables" \
  -H "Authorization: Bearer $RAILWAY_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "projectId": "'$PROJECT_ID'",
    "name": "BOT_TOKEN",
    "value": "8527984904:AAEZSOQ25RMpyRcsYEy1TWxiYeEbZfzDqHY"
  }'
```

---

## Troubleshooting

### "railway: command not found"
**Solution:** Install Railway CLI:
```bash
npm i -g @railway/cli
```

### "Not logged in"
**Solution:** Login first:
```bash
railway login
```

### "Project not found"
**Solution:** Link your project:
```bash
railway link
```
Then select your project from the list.

### Script fails
**Solution:** Use Option 2 (Manual Method) instead - it's very quick!

---

## Quick One-Liner (If you have Railway CLI installed)

```bash
railway link && railway variables set BOT_TOKEN=8527984904:AAEZSOQ25RMpyRcsYEy1TWxiYeEbZfzDqHY && railway up
```

---

**Try Option 1 first - it's the fastest!** 🚀

