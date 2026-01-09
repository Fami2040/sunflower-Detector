#!/usr/bin/env python3
"""
Quick test script to verify all dependencies can be imported.
Run this locally to check if requirements are correct.
"""
import sys

def test_imports():
    """Test all required imports."""
    errors = []
    
    print("Testing imports...")
    
    try:
        import torch
        print(f"✅ torch {torch.__version__}")
        print(f"   CUDA available: {torch.cuda.is_available()}")
    except Exception as e:
        errors.append(f"❌ torch: {e}")
    
    try:
        import torchvision
        print(f"✅ torchvision {torchvision.__version__}")
    except Exception as e:
        errors.append(f"❌ torchvision: {e}")
    
    try:
        import cv2
        print(f"✅ opencv-python {cv2.__version__}")
    except Exception as e:
        errors.append(f"❌ opencv-python: {e}")
    
    try:
        import telegram
        print(f"✅ python-telegram-bot {telegram.__version__}")
    except Exception as e:
        errors.append(f"❌ python-telegram-bot: {e}")
    
    try:
        import sahi
        print(f"✅ sahi")
    except Exception as e:
        errors.append(f"❌ sahi: {e}")
    
    try:
        from ultralytics import YOLO
        print(f"✅ ultralytics")
    except Exception as e:
        errors.append(f"❌ ultralytics: {e}")
    
    try:
        import numpy
        print(f"✅ numpy {numpy.__version__}")
    except Exception as e:
        errors.append(f"❌ numpy: {e}")
    
    try:
        from PIL import Image
        print(f"✅ pillow")
    except Exception as e:
        errors.append(f"❌ pillow: {e}")
    
    try:
        from dotenv import load_dotenv
        print(f"✅ python-dotenv")
    except Exception as e:
        errors.append(f"❌ python-dotenv: {e}")
    
    if errors:
        print("\n❌ Import errors found:")
        for error in errors:
            print(f"   {error}")
        return False
    else:
        print("\n✅ All imports successful!")
        return True

if __name__ == "__main__":
    success = test_imports()
    sys.exit(0 if success else 1)

