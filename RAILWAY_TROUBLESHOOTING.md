# Railway Build Troubleshooting Guide

## Current Configuration

We're using **Dockerfile** as the build method (more reliable than Nixpacks for complex dependencies).

## Common Build Failures & Solutions

### 1. Build Timeout
**Symptom**: Build fails after 10-15 minutes
**Solution**: 
- Model files might be too large
- PyTorch installation takes time
- **Fix**: Already using `--no-cache-dir` to speed up builds

### 2. Missing System Dependencies
**Symptom**: `ImportError: libGL.so.1` or similar
**Solution**: Dockerfile includes all required system libraries

### 3. Memory Issues
**Symptom**: Build killed due to OOM
**Solution**: Railway free tier has limited memory
- Consider upgrading plan
- Or use lighter model versions

### 4. Model Files Not Found
**Symptom**: `FileNotFoundError: Detection model file not found`
**Solution**: 
- Ensure models are committed to Git
- Check Railway logs for file listing

## Switching Build Methods

### Option 1: Dockerfile (Current - Recommended)
```json
{
  "build": {
    "builder": "DOCKERFILE",
    "dockerfilePath": "Dockerfile"
  }
}
```

### Option 2: Nixpacks (Alternative)
```json
{
  "build": {
    "builder": "NIXPACKS"
  }
}
```

## Environment Variables Required

Make sure these are set in Railway:
- `BOT_TOKEN=8527984904:AAEZSOQ25RMpyRcsYEy1TWxiYeEbZfzDqHY`
- `PYTHONUNBUFFERED=1` (optional, for better logs)

## Checking Build Logs

1. Go to Railway dashboard
2. Click on your service
3. Go to "Deployments" tab
4. Click on failed deployment
5. Check "Build Logs" for specific error

## Quick Fixes

1. **Clear build cache**: Redeploy from Railway dashboard
2. **Check model sizes**: Large files (>100MB) might cause issues
3. **Verify requirements.txt**: All packages compatible with Python 3.11
4. **Check Python version**: Railway should use Python 3.11




