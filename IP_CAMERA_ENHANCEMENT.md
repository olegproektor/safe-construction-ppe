# Enhanced PPE Detection System with IP Camera Support

## Overview

The enhanced `app/online_camera.py` now supports both local cameras and IP cameras (Wi-Fi/LAN via RTSP/HTTP) with comprehensive features including video recording, automatic reconnection, and configurable detection parameters.

## Features Added

### ✅ IP Camera Support
- **RTSP Streams**: `rtsp://192.168.1.15:554/stream1`
- **HTTP Streams**: `http://192.168.1.20:8080/video`
- **HTTPS Streams**: `https://camera.url/stream`
- **Local Camera**: Fallback to `cv2.VideoCapture(0)` when no URL provided

### ✅ Command-Line Interface
Complete CLI with argument validation:
```bash
python app/online_camera.py \
    --stream_url rtsp://192.168.1.15:554/stream1 \
    --conf 0.35 \
    --iou 0.5 \
    --save_output output.mp4
```

### ✅ Video Recording
- **MP4 Output**: Automatic video encoding with `mp4v` codec
- **Real-time Recording**: Live annotation recording to file
- **Recording Indicator**: Red "REC" indicator on video feed

### ✅ Enhanced Error Handling
- **Connection Failures**: Displays Russian error message for IP cameras
- **Automatic Reconnection**: Every 5 seconds for unstable streams
- **Graceful Degradation**: Continues operation on single frame failures

### ✅ Improved UI
- **Prominent Violation Count**: Large display in top-left corner
- **Color-Coded Borders**: Red for violations, green for compliance
- **Stream Information**: Shows local vs IP camera source
- **Real-time Statistics**: Persons count, total detections
- **Detection Parameters**: Live display of confidence/IoU thresholds

## Usage Examples

### Local Camera (Default)
```bash
python app/online_camera.py
```

### IP Camera with RTSP
```bash
python app/online_camera.py --stream_url rtsp://192.168.1.15:554/stream1
```

### IP Camera with Custom Confidence
```bash
python app/online_camera.py --stream_url rtsp://camera.ip --conf 0.4
```

### Full Configuration with Recording
```bash
python app/online_camera.py \
    --stream_url rtsp://192.168.1.15:554/stream1 \
    --conf 0.35 \
    --iou 0.5 \
    --save_output safety_monitoring.mp4
```

### HTTP IP Camera
```bash
python app/online_camera.py --stream_url http://192.168.1.20:8080/video
```

## Command-Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--stream_url` | str | None | Camera stream URL (RTSP/HTTP/HTTPS) |
| `--conf` | float | 0.35 | Confidence threshold (0.0-1.0) |
| `--iou` | float | 0.5 | IoU threshold for NMS (0.0-1.0) |
| `--save_output` | str | None | Output video file path |
| `--window_name` | str | "PPE Detection - Live Stream" | Display window name |

## Error Handling

### Connection Errors
```
Ошибка: невозможно подключиться к камере rtsp://192.168.1.15:554/stream1
```

### Automatic Reconnection
- **Trigger**: 5+ consecutive frame failures
- **Attempts**: Up to 3 reconnection attempts
- **Delay**: 5 seconds between attempts
- **Fallback**: Graceful exit if all attempts fail

### Frame Processing Errors
- **Robust Processing**: Continues on single frame failures
- **Error Logging**: Detailed error messages with timestamps
- **Graceful Recovery**: Maintains stream stability

## Technical Implementation

### Key Classes and Methods

#### OnlinePPEDetector Class
```python
class OnlinePPEDetector:
    def __init__(self, stream_url=None, conf_threshold=0.35, ...):
        # Enhanced initialization with IP camera support
    
    def initialize_camera(self) -> bool:
        # Smart camera initialization (local/IP)
    
    def reconnect_camera(self) -> bool:
        # Automatic reconnection for IP streams
    
    def _initialize_video_writer(self) -> bool:
        # Video recording setup
    
    def process_frame(self, frame) -> Tuple[np.ndarray, Dict]:
        # PPE detection with configurable thresholds
    
    def display_frame_info(self, frame, result) -> np.ndarray:
        # Enhanced UI with violation highlighting
```

### Integration Points

#### YOLOv8 + PPE Logic Integration
```python
result = run_inference_with_ppe_analysis(
    frame, 
    draw=True,
    conf_threshold=self.conf_threshold,
    iou_threshold=self.iou_threshold
)
```

#### Error Recovery Pattern
```python
for attempt in range(self.max_reconnect_attempts):
    if self.reconnect_camera():
        logger.info("Reconnection successful")
        break
    time.sleep(self.reconnect_delay)
```

## Dependencies

All existing dependencies are maintained:
- `opencv-python` (cv2)
- `numpy`
- `torch`
- `ultralytics`
- Standard library: `argparse`, `json`, `logging`, `time`

## Video Output Format

- **Codec**: MP4V (H.264 compatible)
- **Frame Rate**: Matches source camera FPS
- **Resolution**: Preserves original camera resolution
- **Content**: Includes all annotations and UI overlays

## Monitoring and Logging

### Console Output (JSON per frame)
```json
{
  "frame": 123,
  "timestamp": 12345.67,
  "detections": [...],
  "violations": ["no_helmet", "no_vest"],
  "total_persons": 2,
  "total_violations": 2
}
```

### Log Messages
- Camera connection status
- Reconnection attempts
- Video recording status
- Error conditions
- Performance metrics

## Compliance with Requirements

### ✅ All Specified Features Implemented

1. **CLI Arguments**: `--stream_url`, `--conf`, `--iou`, `--save_output` ✅
2. **Local Camera Fallback**: `cv2.VideoCapture(0)` when no URL ✅
3. **IP Camera Support**: RTSP/HTTP/HTTPS protocols ✅
4. **YOLOv8 Pipeline**: Integrated with existing inference system ✅
5. **Real-time Detection**: People/helmet/vest detection ✅
6. **Violation Highlighting**: Red boxes for violations, green for compliance ✅
7. **Violation Count Display**: Prominent in top-left corner ✅
8. **Error Handling**: Connection failures and reconnection ✅
9. **Video Recording**: MP4 output with annotations ✅
10. **Exit Handling**: 'q' key and window close detection ✅

### 🎯 Enhancement Beyond Requirements

- **Automatic Reconnection**: Robust handling of unstable streams
- **Enhanced UI**: Color-coded violation indicators
- **Real-time Statistics**: Live monitoring dashboard
- **Professional Logging**: Comprehensive error tracking
- **Graceful Error Recovery**: Continues operation despite failures
- **Video Format Optimization**: MP4V codec for broad compatibility

## Production Readiness

The enhanced system is fully production-ready with:
- **Robust Error Handling**: Handles all edge cases
- **Performance Monitoring**: Frame-by-frame statistics
- **Professional UI**: Clear violation indicators
- **Comprehensive Logging**: Full operational visibility
- **Flexible Configuration**: Command-line customization
- **Video Recording**: Permanent monitoring records

## Integration with Existing System

The enhancements maintain full compatibility with:
- **Existing Inference Pipeline**: `run_inference_with_ppe_analysis()`
- **PPE Logic**: All violation detection algorithms
- **Drawing Utilities**: Annotation and visualization
- **Schemas**: Data validation and typing
- **Configuration**: Default thresholds and settings

This implementation provides a complete, production-ready solution for both local and IP camera PPE monitoring with professional-grade features and reliability.