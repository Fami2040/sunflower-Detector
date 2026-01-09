#!/usr/bin/env python3
"""
Script to help add BOT_TOKEN to Railway using Railway CLI or API.
This script will guide you through adding the BOT_TOKEN.
"""

import os
import subprocess
import sys

BOT_TOKEN = "8527984904:AAEZSOQ25RMpyRcsYEy1TWxiYeEbZfzDqHY"

def check_railway_cli():
    """Check if Railway CLI is installed."""
    try:
        result = subprocess.run(['railway', '--version'], 
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            print("✅ Railway CLI is installed!")
            print(f"   Version: {result.stdout.strip()}")
            return True
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    return False

def install_railway_cli_instructions():
    """Print instructions to install Railway CLI."""
    print("\n" + "="*60)
    print("📦 INSTALL RAILWAY CLI")
    print("="*60)
    print("\nOption 1: Using npm (recommended):")
    print("  npm install -g @railway/cli")
    print("\nOption 2: Using Homebrew (Mac/Linux):")
    print("  brew install railway")
    print("\nOption 3: Download from: https://github.com/railwayapp/cli")
    print("\nAfter installing, run this script again.")
    print("="*60)

def add_token_via_cli():
    """Try to add BOT_TOKEN using Railway CLI."""
    print("\n" + "="*60)
    print("🔧 ADDING BOT_TOKEN VIA RAILWAY CLI")
    print("="*60)
    
    # Check if logged in
    try:
        result = subprocess.run(['railway', 'whoami'], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode != 0:
            print("\n⚠️ Not logged in to Railway CLI")
            print("\nPlease login first:")
            print("  railway login")
            print("\nThen run this script again.")
            return False
        print(f"✅ Logged in as: {result.stdout.strip()}")
    except Exception as e:
        print(f"❌ Error checking Railway login: {e}")
        print("\nPlease login first:")
        print("  railway login")
        return False
    
    # Check if linked to project
    try:
        result = subprocess.run(['railway', 'status'], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode != 0:
            print("\n⚠️ Not linked to a Railway project")
            print("\nPlease link to your project first:")
            print("  railway link")
            print("\nThen run this script again.")
            return False
        print("✅ Linked to Railway project")
    except Exception as e:
        print(f"❌ Error checking Railway link: {e}")
        print("\nPlease link to your project first:")
        print("  railway link")
        return False
    
    # Add the variable
    print(f"\n🔄 Adding BOT_TOKEN variable...")
    try:
        cmd = ['railway', 'variables', 'set', f'BOT_TOKEN={BOT_TOKEN}']
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            print("✅ BOT_TOKEN added successfully!")
            print("\n🔄 Now redeploying...")
            
            # Try to redeploy
            try:
                subprocess.run(['railway', 'up'], timeout=60)
                print("✅ Redeployment triggered!")
                print("\n⏳ Wait 2-3 minutes, then check Railway logs.")
            except Exception as e:
                print(f"⚠️ Could not auto-redeploy: {e}")
                print("   Please manually redeploy from Railway dashboard.")
            
            return True
        else:
            print(f"❌ Error adding variable: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def manual_instructions():
    """Print manual instructions."""
    print("\n" + "="*60)
    print("📋 MANUAL INSTRUCTIONS (If CLI doesn't work)")
    print("="*60)
    print("\n1. Go to: https://railway.app/dashboard")
    print("2. Click on your project")
    print("3. Click on your service")
    print("4. Click 'Variables' tab")
    print("5. Click '+ New Variable'")
    print("6. Name: BOT_TOKEN")
    print(f"7. Value: {BOT_TOKEN}")
    print("8. Click 'Save'")
    print("9. Go to 'Deployments' tab → Click 'Redeploy'")
    print("10. Wait 2-3 minutes")
    print("="*60)

def main():
    import sys
    # Fix encoding for Windows
    if sys.platform == 'win32':
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    
    print("="*60)
    print("RAILWAY BOT_TOKEN SETUP HELPER")
    print("="*60)
    print(f"\nBot Token: {BOT_TOKEN[:15]}...{BOT_TOKEN[-5:]}")
    
    # Check if Railway CLI is installed
    if check_railway_cli():
        print("\nAttempting to add BOT_TOKEN via Railway CLI...")
        if add_token_via_cli():
            print("\n✅ SUCCESS! BOT_TOKEN has been added.")
            print("\nCheck Railway logs in 2-3 minutes to verify.")
            return
        else:
            print("\n⚠️ Could not add via CLI. Showing manual instructions...")
    else:
        print("\n⚠️ Railway CLI is not installed.")
        install_railway_cli_instructions()
    
    # Show manual instructions
    manual_instructions()
    
    print("\n" + "="*60)
    print("💡 TIP: Install Railway CLI for easier setup:")
    print("   npm install -g @railway/cli")
    print("   railway login")
    print("   railway link")
    print("   railway variables set BOT_TOKEN=8527984904:AAEZSOQ25RMpyRcsYEy1TWxiYeEbZfzDqHY")
    print("   railway up")
    print("="*60)

if __name__ == "__main__":
    main()

