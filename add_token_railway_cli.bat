@echo off
REM Quick script to add BOT_TOKEN to Railway using Railway CLI (Windows)

set BOT_TOKEN=8490011366:AAGsDtjayDyWhf_wXFAqWVDkg5X3kOmx81w

echo ============================================================
echo 🚀 Adding BOT_TOKEN to Railway
echo ============================================================

REM Check if Railway CLI is installed
where railway >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo ❌ Railway CLI is not installed!
    echo.
    echo Install it with:
    echo   npm install -g @railway/cli
    exit /b 1
)

echo ✅ Railway CLI found

REM Check if logged in
railway whoami >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo ⚠️  Not logged in to Railway
    echo.
    echo Please login first:
    echo   railway login
    exit /b 1
)

echo ✅ Logged in to Railway

REM Check if linked to project
railway status >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo ⚠️  Not linked to a Railway project
    echo.
    echo Please link to your project first:
    echo   railway link
    exit /b 1
)

echo ✅ Linked to Railway project

REM Add the variable
echo.
echo 🔄 Adding BOT_TOKEN variable...
railway variables set BOT_TOKEN=%BOT_TOKEN%

if %ERRORLEVEL% EQU 0 (
    echo ✅ BOT_TOKEN added successfully!
    echo.
    echo 🔄 Triggering redeployment...
    railway up
    echo.
    echo ✅ Done! Wait 2-3 minutes and check Railway logs.
) else (
    echo ❌ Failed to add BOT_TOKEN
    exit /b 1
)

