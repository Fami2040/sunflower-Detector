#!/usr/bin/env python3
"""
Script to set Telegram bot profile picture.
Note: This requires the bot to be added to a channel/group as admin,
or you can use BotFather instead (easier method).
"""

import os
import requests
from dotenv import load_dotenv

load_dotenv()

BOT_TOKEN = os.getenv("BOT_TOKEN", "8527984904:AAEZSOQ25RMpyRcsYEy1TWxiYeEbZfzDqHY")

def set_bot_photo_via_channel(photo_path: str, channel_username: str):
    """
    Set bot profile picture by adding bot to channel as admin and using setChatPhoto.
    This is a workaround since Telegram Bot API doesn't directly support setting bot profile picture.
    """
    # First, add bot to channel as admin (manual step)
    # Then use this API
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/setChatPhoto"
    
    with open(photo_path, 'rb') as photo:
        files = {'photo': photo}
        data = {'chat_id': f"@{channel_username}"}
        response = requests.post(url, files=files, data=data)
    
    return response.json()

def get_bot_info():
    """Get current bot information."""
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/getMe"
    response = requests.get(url)
    return response.json()

if __name__ == "__main__":
    print("🤖 Telegram Bot Profile Picture Setter")
    print("=" * 50)
    
    # Get bot info
    bot_info = get_bot_info()
    if bot_info.get("ok"):
        bot = bot_info["result"]
        print(f"✅ Bot: @{bot.get('username', 'N/A')}")
        print(f"   Name: {bot.get('first_name', 'N/A')}")
    else:
        print(f"❌ Error: {bot_info}")
        exit(1)
    
    print("\n⚠️  IMPORTANT:")
    print("Telegram Bot API doesn't directly support setting bot profile pictures.")
    print("You have two options:\n")
    print("OPTION 1: Use BotFather (EASIEST)")
    print("1. Open Telegram and message @BotFather")
    print("2. Send: /mybots")
    print("3. Select your bot")
    print("4. Click 'Edit Bot' → 'Edit Botpic'")
    print("5. Send your sunflower image")
    print("\nOPTION 2: Programmatic (Complex)")
    print("1. Create a channel")
    print("2. Add your bot as admin")
    print("3. Use setChatPhoto API on the channel")
    print("4. This sets the channel photo, not bot photo")
    
    print("\n💡 RECOMMENDATION: Use BotFather (Option 1) - it's the official way!")




