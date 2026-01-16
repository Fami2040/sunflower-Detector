# PowerShell script to update BOT_TOKEN in Railway
# This uses Railway CLI - make sure you have it installed: npm install -g @railway/cli

Write-Host "🚀 Updating BOT_TOKEN in Railway..." -ForegroundColor Green
Write-Host ""

$NEW_TOKEN = "8490011366:AAGsDtjayDyWhf_wXFAqWVDkg5X3kOmx81w"

# Check if Railway CLI is installed
$railwayInstalled = Get-Command railway -ErrorAction SilentlyContinue

if (-not $railwayInstalled) {
    Write-Host "⚠️  Railway CLI not found!" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "📋 MANUAL STEPS TO UPDATE IN RAILWAY:" -ForegroundColor Cyan
    Write-Host "=" * 60
    Write-Host "1. Go to: https://railway.app" -ForegroundColor White
    Write-Host "2. Open your project → Your service" -ForegroundColor White
    Write-Host "3. Click 'Variables' tab" -ForegroundColor White
    Write-Host "4. Find 'BOT_TOKEN' and click to edit" -ForegroundColor White
    Write-Host "5. Update value to: $NEW_TOKEN" -ForegroundColor White
    Write-Host "6. Click 'Save'" -ForegroundColor White
    Write-Host "7. Go to 'Deployments' → Click 'Redeploy'" -ForegroundColor White
    Write-Host ""
    Write-Host "💡 To install Railway CLI: npm install -g @railway/cli" -ForegroundColor Yellow
    Write-Host "   Then run: railway login" -ForegroundColor Yellow
    Write-Host ""
} else {
    Write-Host "✅ Railway CLI found!" -ForegroundColor Green
    Write-Host ""
    Write-Host "Updating BOT_TOKEN..." -ForegroundColor Yellow
    
    # Try to update via CLI
    railway variables set BOT_TOKEN=$NEW_TOKEN
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host ""
        Write-Host "✅ Token updated successfully!" -ForegroundColor Green
        Write-Host ""
        Write-Host "🔄 Triggering redeploy..." -ForegroundColor Yellow
        railway up
        Write-Host ""
        Write-Host "✅ Done! Check Railway dashboard for deployment status." -ForegroundColor Green
    } else {
        Write-Host ""
        Write-Host "❌ Failed to update via CLI. Please update manually:" -ForegroundColor Red
        Write-Host "   1. Go to Railway Dashboard → Variables" -ForegroundColor White
        Write-Host "   2. Update BOT_TOKEN to: $NEW_TOKEN" -ForegroundColor White
        Write-Host "   3. Redeploy" -ForegroundColor White
    }
}

Write-Host ""
pause

