#!/usr/bin/env python3
"""
Simple Camera Test Script
Tests if camera is accessible from Python/OpenCV
"""

import cv2
import sys

def test_camera():
    """Test camera access using OpenCV"""
    print("🔍 Testing camera access...")
    
    # Try to open camera
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ Error: Cannot open camera")
        print("Possible causes:")
        print("- Camera is being used by another application")
        print("- Camera drivers are not installed")
        print("- Camera permissions are denied")
        return False
    
    # Try to read a frame
    ret, frame = cap.read()
    
    if ret:
        print("✅ Camera access successful!")
        print(f"📏 Frame size: {frame.shape[1]}x{frame.shape[0]}")
        
        # Show camera info
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        
        print(f"📊 Camera properties:")
        print(f"   - Resolution: {int(width)}x{int(height)}")
        print(f"   - FPS: {fps}")
        
        # Test for 5 seconds
        print("🎥 Testing camera for 5 seconds...")
        print("Press 'q' to quit early or wait...")
        
        import time
        start_time = time.time()
        
        while time.time() - start_time < 5:
            ret, frame = cap.read()
            if not ret:
                print("❌ Error reading frame")
                break
                
            cv2.imshow('Camera Test - Press Q to quit', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cv2.destroyAllWindows()
        print("✅ Camera test completed successfully")
        
    else:
        print("❌ Error: Cannot read from camera")
        print("Camera opened but no frames available")
    
    cap.release()
    return ret

if __name__ == "__main__":
    print("🏗️ Safe Construction - Camera Test Utility")
    print("=" * 50)
    
    try:
        success = test_camera()
        if success:
            print("\n✅ Camera is working! The issue might be browser permissions.")
            print("💡 Try the following:")
            print("   1. Refresh your browser")
            print("   2. Check browser camera permissions")
            print("   3. Try a different browser")
        else:
            print("\n❌ Camera hardware/driver issue detected")
            print("💡 Try the following:")
            print("   1. Check Windows Camera app")
            print("   2. Update camera drivers")
            print("   3. Check Windows Privacy settings")
            print("   4. Run fix_camera_permissions.bat as Administrator")
            
    except Exception as e:
        print(f"\n❌ Error during camera test: {e}")
        print("💡 This might indicate:")
        print("   - OpenCV not installed properly")
        print("   - Camera driver issues")
        print("   - System permission problems")
    
    print("\nPress Enter to exit...")
    input()