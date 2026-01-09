#!/usr/bin/env python3
"""
Quick script to test if BOT_TOKEN is accessible and valid.
Run this locally or on Railway to verify the token setup.
"""

import os
from dotenv import load_dotenv
import asyncio

# Load environment variables
load_dotenv()

BOT_TOKEN = os.getenv("BOT_TOKEN", "")

async def test_token():
    """Test if BOT_TOKEN is valid by connecting to Telegram API."""
    if not BOT_TOKEN:
        print("❌ ERROR: BOT_TOKEN not found!")
        print("\n💡 To fix:")
        print("   1. Set BOT_TOKEN environment variable")
        print("   2. Or create a .env file with: BOT_TOKEN=your_token_here")
        print("   3. For Railway: Add BOT_TOKEN in Variables tab")
        return False
    
    print(f"✅ BOT_TOKEN found: {BOT_TOKEN[:10]}...{BOT_TOKEN[-5:]}")
    print("\n🔄 Testing connection to Telegram API...")
    
    try:
        from telegram import Bot
        bot = Bot(token=BOT_TOKEN)
        bot_info = await bot.get_me()
        
        print(f"\n✅ SUCCESS! Bot connected successfully!")
        print(f"   Bot Username: @{bot_info.username}")
        print(f"   Bot Name: {bot_info.first_name}")
        print(f"   Bot ID: {bot_info.id}")
        print(f"\n🎉 Your bot token is valid and working!")
        return True
        
    except Exception as e:
        print(f"\n❌ FAILED to connect to Telegram API!")
        print(f"   Error: {e}")
        print(f"\n💡 Possible issues:")
        print(f"   1. Invalid BOT_TOKEN - check if token is correct")
        print(f"   2. Token revoked - generate new token from @BotFather")
        print(f"   3. Network issue - check internet connection")
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("Telegram Bot Token Tester")
    print("=" * 60)
    print()
    
    result = asyncio.run(test_token())
    
    print("\n" + "=" * 60)
    if result:
        print("✅ Test passed! Bot is ready to use.")
    else:
        print("❌ Test failed! Fix the issues above.")
    print("=" * 60)

