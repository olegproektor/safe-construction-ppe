# PPE Detection Demo Troubleshooting Guide

## Problem: RTSP Stream Timeout

**Error message:**
```
Stream timeout triggered after 30109.156000 ms
```

**Solutions:**

### 1. ✅ Use Local Webcam (Recommended)
```bash
# Simple command - uses your computer's webcam
python app/online_camera.py
```
**OR double-click:** `run_local_webcam_demo.bat`

### 2. ✅ Try HTTP Video Stream
```bash
python app/online_camera.py --stream_url "https://commondatastorage.googleapis.com/gtv-videos-bucket/sample/BigBuckBunny.mp4"
```

### 3. ✅ Use Your Own Video File
```bash
python app/online_camera.py --stream_url "path/to/your/video.mp4"
```

## Two Different Camera Modes

### 🌐 Web Browser Mode
- **URL:** http://localhost:8080
- **Usage:** Click "Start Camera" on main page
- **Camera:** Uses browser webcam
- **Display:** Shows in web browser

### 🖥️ Desktop Application Mode  
- **Usage:** Run commands in terminal
- **Camera:** External cameras, video files, streams
- **Display:** Opens separate OpenCV window (not in browser!)
- **Control:** Press 'q' to quit

## Common Issues

### Issue 1: PowerShell vs Command Prompt
**Problem:** PowerShell may have syntax issues
**Solution:** Use Command Prompt instead
1. Press `Win + R`
2. Type `cmd` and press Enter
3. Navigate to project folder
4. Run the command

### Issue 2: "Nothing happens in browser"
**This is normal!** Desktop application mode doesn't use the browser.
- It opens a separate OpenCV window
- Look for a new window on your taskbar
- The window title will be "PPE Detection - Live Stream"

### Issue 3: Camera Permission Denied
```bash
# Test if camera works
python test_camera.py
```

### Issue 4: Dependencies Missing
```bash
# Install requirements
pip install -r requirements.txt
```

## Quick Test Commands

### Test 1: Local Webcam
```bash
cd "C:\Users\lespo\Downloads\Safeconstruction"
python app/online_camera.py
```

### Test 2: Sample Video
```bash
python app/online_camera.py --stream_url "https://commondatastorage.googleapis.com/gtv-videos-bucket/sample/BigBuckBunny.mp4"
```

### Test 3: Check Camera Hardware
```bash
python test_camera.py
```

## Expected Behavior

When working correctly, you should see:
1. Console output showing camera initialization
2. A new OpenCV window opens (separate from browser)
3. Real-time video with PPE detection boxes
4. Console logging of detection results
5. Press 'q' in the OpenCV window to quit

## Need Help?

1. **Check camera hardware:** Run `python test_camera.py`
2. **Try local webcam first:** Simplest and most reliable
3. **Use Command Prompt:** Not PowerShell
4. **Look for OpenCV window:** It's separate from browser
5. **Check firewall:** May block video downloads

## Demo Page Access

- **Main app:** http://localhost:8080
- **Demo page:** http://localhost:8080/demo
- **API docs:** http://localhost:8080/docs