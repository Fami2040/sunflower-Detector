#!/bin/bash
# Script to add BOT_TOKEN to Railway using Railway CLI

echo "🚀 Adding BOT_TOKEN to Railway..."
echo ""

# Check if Railway CLI is installed
if ! command -v railway &> /dev/null; then
    echo "❌ Railway CLI is not installed!"
    echo ""
    echo "📥 Install Railway CLI first:"
    echo "   npm i -g @railway/cli"
    echo ""
    echo "Then run this script again."
    exit 1
fi

echo "✅ Railway CLI found"
echo ""

# Check if logged in
if ! railway whoami &> /dev/null; then
    echo "🔐 Please login to Railway first:"
    echo "   railway login"
    echo ""
    echo "Then run this script again."
    exit 1
fi

echo "✅ Logged in to Railway"
echo ""

# Link project if not already linked
echo "🔗 Linking to Railway project..."
railway link || echo "⚠️ Project already linked or manual link needed"
echo ""

# Set BOT_TOKEN
echo "📝 Setting BOT_TOKEN variable..."
BOT_TOKEN="8527984904:AAEZSOQ25RMpyRcsYEy1TWxiYeEbZfzDqHY"

railway variables set BOT_TOKEN="$BOT_TOKEN"

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ BOT_TOKEN successfully added to Railway!"
    echo ""
    echo "🔄 Redeploying service..."
    railway up
    echo ""
    echo "✅ Done! Check Railway dashboard logs to verify."
else
    echo ""
    echo "❌ Failed to set BOT_TOKEN"
    echo "Please try setting it manually in Railway dashboard."
    exit 1
fi

