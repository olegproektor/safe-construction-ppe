#!/usr/bin/env python3
"""
Lightweight Camera Test - No OpenCV Required
Tests basic camera functionality without heavy dependencies
"""

import sys
from pathlib import Path

def check_opencv_installation():
    """Check if OpenCV is properly installed"""
    print("🔍 Checking OpenCV installation...")
    
    try:
        import cv2
        print(f"✅ OpenCV version: {cv2.__version__}")
        return True
    except ImportError as e:
        print(f"❌ OpenCV import failed: {e}")
        return False
    except Exception as e:
        print(f"❌ OpenCV DLL error: {e}")
        print("This usually means:")
        print("  - Virtual memory (page file) is too small")
        print("  - OpenCV installation is corrupted")
        print("  - Missing Visual C++ redistributables")
        return False

def suggest_opencv_fixes():
    """Suggest fixes for OpenCV issues"""
    print("\n🔧 Suggested fixes for OpenCV issues:")
    print("1. Increase virtual memory (page file):")
    print("   - Open System Properties > Advanced > Performance Settings")
    print("   - Virtual Memory > Change > Custom size: 4096-8192 MB")
    print("   - Restart computer")
    print()
    print("2. Reinstall OpenCV:")
    print("   pip uninstall opencv-python")
    print("   pip install opencv-python")
    print()
    print("3. Install Visual C++ Redistributable:")
    print("   Download from Microsoft website")
    print()
    print("4. Try OpenCV-headless (no GUI):")
    print("   pip install opencv-python-headless")

def test_web_camera_alternative():
    """Test web-based camera functionality"""
    print("\n🌐 Web Camera Alternative:")
    print("Instead of desktop camera, you can use the web interface:")
    print(f"1. Start server: start_server.bat")
    print(f"2. Open browser: http://localhost:8080")
    print(f"3. Click 'Start Camera' to use browser webcam")
    print(f"4. This doesn't require OpenCV!")

def main():
    print("🏗️ Safe Construction - Lightweight Camera Test")
    print("=" * 60)
    
    # Check current directory
    current_dir = Path.cwd()
    print(f"📁 Current directory: {current_dir}")
    
    # Check if we're in the right directory
    if not (current_dir / "app" / "main.py").exists():
        print("❌ Not in project directory!")
        print("Please run this from the Safeconstruction folder")
        return
    
    print("✅ In correct project directory")
    
    # Test OpenCV
    opencv_works = check_opencv_installation()
    
    if not opencv_works:
        suggest_opencv_fixes()
        test_web_camera_alternative()
        
        print("\n💡 Quick Solution:")
        print("Use the web interface instead of desktop camera!")
        print("1. Double-click: start_server.bat")
        print("2. Open: http://localhost:8080")
        print("3. Use browser camera (no OpenCV needed)")
    else:
        print("\n✅ OpenCV is working! You can use desktop camera:")
        print("python app/online_camera.py")
    
    print("\n" + "=" * 60)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ Error: {e}")
    finally:
        print("\nPress Enter to exit...")
        input()