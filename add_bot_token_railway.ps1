# PowerShell script to add BOT_TOKEN to Railway using Railway CLI

Write-Host "🚀 Adding BOT_TOKEN to Railway..." -ForegroundColor Green
Write-Host ""

# Check if Railway CLI is installed
try {
    $null = railway --version 2>&1
    Write-Host "✅ Railway CLI found" -ForegroundColor Green
} catch {
    Write-Host "❌ Railway CLI is not installed!" -ForegroundColor Red
    Write-Host ""
    Write-Host "📥 Install Railway CLI first:" -ForegroundColor Yellow
    Write-Host "   npm i -g @railway/cli" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Then run this script again."
    exit 1
}

Write-Host ""

# Check if logged in
try {
    $null = railway whoami 2>&1
    Write-Host "✅ Logged in to Railway" -ForegroundColor Green
} catch {
    Write-Host "🔐 Please login to Railway first:" -ForegroundColor Yellow
    Write-Host "   railway login" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Then run this script again."
    exit 1
}

Write-Host ""

# Link project if not already linked
Write-Host "🔗 Linking to Railway project..." -ForegroundColor Cyan
railway link 2>&1 | Out-Null
if ($LASTEXITCODE -eq 0) {
    Write-Host "✅ Project linked" -ForegroundColor Green
} else {
    Write-Host "⚠️ Project already linked or manual link needed" -ForegroundColor Yellow
}

Write-Host ""

# Set BOT_TOKEN
Write-Host "📝 Setting BOT_TOKEN variable..." -ForegroundColor Cyan
$BOT_TOKEN = "8490011366:AAGsDtjayDyWhf_wXFAqWVDkg5X3kOmx81w"

railway variables set "BOT_TOKEN=$BOT_TOKEN"

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "✅ BOT_TOKEN successfully added to Railway!" -ForegroundColor Green
    Write-Host ""
    Write-Host "🔄 Redeploying service..." -ForegroundColor Cyan
    railway up
    Write-Host ""
    Write-Host "✅ Done! Check Railway dashboard logs to verify." -ForegroundColor Green
} else {
    Write-Host ""
    Write-Host "❌ Failed to set BOT_TOKEN" -ForegroundColor Red
    Write-Host "Please try setting it manually in Railway dashboard." -ForegroundColor Yellow
    exit 1
}

