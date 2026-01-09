# Quick Railway Fix Guide

## If Dockerfile Build Fails, Try Nixpacks:

Railway is now configured to use **Nixpacks** (automatic detection) instead of Dockerfile.

## Steps to Fix:

1. **In Railway Dashboard:**
   - Go to your project settings
   - Under "Build" section, make sure it's set to "Nixpacks" or "Auto-detect"
   - If you see "Dockerfile" selected, change it to "Nixpacks"

2. **Clear Build Cache:**
   - In Railway, go to your service
   - Click "Settings" → "Clear Build Cache"
   - Then redeploy

3. **Check Environment Variables:**
   - Make sure `BOT_TOKEN` is set in Railway environment variables
   - Go to: Variables tab → Add `BOT_TOKEN=8527984904:AAEZSOQ25RMpyRcsYEy1TWxiYeEbZfzDqHY`

4. **Redeploy:**
   - Click "Redeploy" button
   - Or push a new commit to trigger auto-deploy

## Alternative: Use Dockerfile Manually

If Nixpacks fails, you can switch back to Dockerfile:

1. In Railway dashboard → Settings → Build
2. Select "Dockerfile" as builder
3. Set Dockerfile path to: `Dockerfile`
4. Redeploy

## Common Issues:

- **Build timeout**: PyTorch installation takes time (5-10 minutes)
- **Memory error**: Railway free tier has limits, models might be too large
- **Missing files**: Make sure models are committed to Git

## Check Logs:

1. Go to Railway dashboard
2. Click on your service
3. Click "Deployments" tab
4. Click on the failed deployment
5. Check "Build Logs" for specific error




