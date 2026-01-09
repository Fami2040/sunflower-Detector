# ⚡ Quick Setup - Add BOT_TOKEN to Railway

## Option 1: Using Railway CLI (Fastest - 30 seconds)

### Step 1: Install Railway CLI (one-time)
```bash
npm install -g @railway/cli
```

### Step 2: Login to Railway
```bash
railway login
```
(This will open a browser for authentication)

### Step 3: Link to Your Project
```bash
railway link
```
(Select your project from the list)

### Step 4: Add BOT_TOKEN
```bash
railway variables set BOT_TOKEN=8527984904:AAEZSOQ25RMpyRcsYEy1TWxiYeEbZfzDqHY
```

### Step 5: Redeploy
```bash
railway up
```

**Done!** Wait 2-3 minutes and check Railway logs.

---

## Option 2: Using Helper Script (Windows)

1. Open PowerShell in this directory
2. Run:
   ```powershell
   python add_bot_token_railway.py
   ```
   OR
   ```cmd
   add_token_railway_cli.bat
   ```

---

## Option 3: Manual (If CLI doesn't work)

### Quick Steps:
1. **Railway Dashboard** → Your Project → Your Service
2. **Variables** tab → **+ New Variable**
3. **Name**: `BOT_TOKEN`
4. **Value**: `8527984904:AAEZSOQ25RMpyRcsYEy1TWxiYeEbZfzDqHY`
5. **Save** → **Deployments** tab → **Redeploy**
6. Wait 2-3 minutes

---

## Verify It Worked

Check Railway logs - you should see:
```
✅ BOT_TOKEN found: 8527984904...DqHY
✅ Bot is ready and polling for messages...
```

**NOT:**
```
❌ Error: BOT_TOKEN not found!
```

---

## Troubleshooting

### "railway: command not found"
- Install Railway CLI: `npm install -g @railway/cli`
- Or use manual method (Option 3)

### "Not logged in"
- Run: `railway login`

### "Not linked to project"
- Run: `railway link`
- Select your project

---

**The fastest way is Option 1 (Railway CLI)!** ⚡

