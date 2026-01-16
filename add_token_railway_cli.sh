#!/bin/bash
# Quick script to add BOT_TOKEN to Railway using Railway CLI

BOT_TOKEN="8490011366:AAGsDtjayDyWhf_wXFAqWVDkg5X3kOmx81w"

echo "============================================================"
echo "🚀 Adding BOT_TOKEN to Railway"
echo "============================================================"

# Check if Railway CLI is installed
if ! command -v railway &> /dev/null; then
    echo "❌ Railway CLI is not installed!"
    echo ""
    echo "Install it with:"
    echo "  npm install -g @railway/cli"
    echo "  OR"
    echo "  brew install railway"
    exit 1
fi

echo "✅ Railway CLI found"

# Check if logged in
if ! railway whoami &> /dev/null; then
    echo "⚠️  Not logged in to Railway"
    echo ""
    echo "Please login first:"
    echo "  railway login"
    exit 1
fi

echo "✅ Logged in to Railway"

# Check if linked to project
if ! railway status &> /dev/null; then
    echo "⚠️  Not linked to a Railway project"
    echo ""
    echo "Please link to your project first:"
    echo "  railway link"
    exit 1
fi

echo "✅ Linked to Railway project"

# Add the variable
echo ""
echo "🔄 Adding BOT_TOKEN variable..."
railway variables set BOT_TOKEN="$BOT_TOKEN"

if [ $? -eq 0 ]; then
    echo "✅ BOT_TOKEN added successfully!"
    echo ""
    echo "🔄 Triggering redeployment..."
    railway up
    echo ""
    echo "✅ Done! Wait 2-3 minutes and check Railway logs."
else
    echo "❌ Failed to add BOT_TOKEN"
    exit 1
fi

