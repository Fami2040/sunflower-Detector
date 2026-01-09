# PowerShell script to start the Telegram Bot with token
Write-Host "🌻 Starting Sunflower Seed Counter Telegram Bot..." -ForegroundColor Green
Write-Host ""

# Set bot token
$env:BOT_TOKEN = "8527984904:AAEZSOQ25RMpyRcsYEy1TWxiYeEbZfzDqHY"

# Use venv_gpu Python (has all packages and CUDA support)
$pythonPath = "c:/Cursor Project/venv_gpu/Scripts/python.exe"

if (Test-Path $pythonPath) {
    Write-Host "✅ Using venv_gpu Python" -ForegroundColor Green
    Write-Host "✅ BOT_TOKEN set" -ForegroundColor Green
    Write-Host ""
    & $pythonPath telegram_bot.py
} else {
    Write-Host "❌ venv_gpu not found, trying Python 3.13..." -ForegroundColor Yellow
    $pythonPath = "C:\Program Files\Python313\python.exe"
    if (Test-Path $pythonPath) {
        & $pythonPath telegram_bot.py
    } else {
        Write-Host "❌ Python not found!" -ForegroundColor Red
        python telegram_bot.py
    }
}

