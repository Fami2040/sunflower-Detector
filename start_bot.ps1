# PowerShell script to start the Telegram Bot
Write-Host "🌻 Starting Sunflower Seed Counter Telegram Bot..." -ForegroundColor Green
Write-Host ""

# Use venv_gpu Python (has all packages and CUDA support)
$pythonPath = "c:/Cursor Project/venv_gpu/Scripts/python.exe"

if (Test-Path $pythonPath) {
    Write-Host "✅ Using venv_gpu Python" -ForegroundColor Green
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


