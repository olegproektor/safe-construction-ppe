# 🚧 Safe Construction - PPE Detection System

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)](https://fastapi.tiangolo.com/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-orange.svg)](https://github.com/ultralytics/ultralytics)

An advanced computer vision system for monitoring Personal Protective Equipment (PPE) compliance on construction sites. Detects people without **helmets** or **safety vests** in real-time using YOLOv8 and provides comprehensive violation tracking.

## 🌟 Features

### 🔍 **Core Detection Capabilities**
- **Object Detection**: Person, helmet, and safety vest detection using YOLOv8
- **PPE Compliance Analysis**: Determines `has_helmet` and `has_vest` for each person
- **Violation Detection**: Identifies `no_helmet` and `no_vest` violations
- **Confidence Scoring**: Configurable confidence and IoU thresholds
- **Visual Annotations**: Highlighted bounding boxes with color-coded compliance status

### 🌐 **Web Interface**
- **Real-time Detection**: Live webcam PPE monitoring
- **Interactive Dashboard**: Statistics, compliance rates, and violation summaries  
- **Settings Panel**: Adjustable detection parameters
- **Auto-save Violations**: Smart session-based violation recording
- **Multi-tab Interface**: Camera, Documentation, and About sections

### 📡 **REST API**
- **FastAPI Framework**: High-performance async API
- **Image Detection**: Upload and analyze images via `/detect/image`
- **Health Monitoring**: System status and uptime via `/health`
- **Model Information**: Detection parameters via `/model/info`
- **Swagger Documentation**: Interactive API docs at `/docs`

### 📊 **Violation Management**
- **Smart Session Tracking**: Groups continuous violations into sessions
- **Rate Limiting**: Prevents spam with configurable intervals
- **Comprehensive Records**: JSON files with metadata and statistics
- **Folder Management**: Download, clear, and browse violation records
- **Export Capabilities**: ZIP archive downloads

### 🎥 **Camera Support**
- **Web Browser Camera**: Real-time detection in browser
- **Desktop Application**: `online_camera.py` for local camera processing
- **IP Camera Support**: RTSP and HTTP camera streams
- **Video Processing**: Batch processing with `video_infer.py`

## 🏗️ System Architecture

```
Input (Image/Video/Camera)
         ↓
    YOLOv8 Detection
         ↓
Object Classification (person/helmet/vest)
         ↓
    PPE Matching (IoU-based)
         ↓
   Violation Analysis
         ↓
JSON Response + Annotated Image
         ↓
   Web Interface / API
```

## 🚀 Quick Start

### 1. **Clone Repository**
```bash
git clone https://github.com/your-repo/safeconstruction.git
cd safeconstruction
```

### 2. **Install Dependencies**
```bash
pip install -r requirements.txt
```

### 3. **Start Server**
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8080 --reload
```

### 4. **Access Interface**
- **Web Interface**: http://localhost:8080
- **API Documentation**: http://localhost:8080/docs
- **Health Check**: http://localhost:8080/health

## 📱 Usage Examples

### **Web Interface**
1. Open http://localhost:8080
2. Click "📷 Start Camera" and allow camera access
3. Click "🔍 Start Detection" to begin monitoring
4. Enable "Auto-save Violations" in settings for automatic recording

### **API Usage**

#### Health Check
```bash
curl http://localhost:8080/health
```

#### Image Detection
```bash
curl -X POST "http://localhost:8080/detect/image" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@test_image.jpg" \
  -F "draw=true" \
  -F "return_image=false"
```

#### Get Annotated Image (Base64)
```bash
curl -X POST "http://localhost:8080/detect/image" \
  -F "file=@test_image.jpg" \
  -F "draw=true" \
  -F "return_image=true"
```

### **Desktop Camera Application**
```bash
# Local camera
python app/online_camera.py

# IP camera
python app/online_camera.py --stream_url rtsp://camera-ip:554/stream
```

### **Video Processing**
```bash
python scripts/video_infer.py --input data/construction_site.mp4 --output results/
```

## 📋 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Web interface |
| `GET` | `/health` | System health check |
| `GET` | `/status` | Detailed system status |
| `GET` | `/model/info` | Model information |
| `POST` | `/detect/image` | Image PPE detection |
| `GET` | `/violations/stats` | Violation statistics |
| `POST` | `/violations/save` | Save violation record |
| `POST` | `/violations/clear` | Clear violations folder |
| `GET` | `/violations/download-archive` | Download violations ZIP |
| `POST` | `/violations/open-folder` | Open violations folder |

## 🎛️ Configuration

### **Environment Variables**
```bash
# Model configuration
MODEL_WEIGHTS=yolov8n.pt
CONFIDENCE_THRESHOLD=0.5
IOU_THRESHOLD=0.4

# Server configuration
HOST=0.0.0.0
PORT=8080

# PPE detection thresholds
PPE_OVERLAP_THRESHOLD=0.3
```

### **Detection Parameters**
- **Confidence Threshold**: 0.0-1.0 (default: 0.5)
- **IoU Threshold**: 0.0-1.0 (default: 0.4)
- **PPE Overlap Threshold**: 0.0-1.0 (default: 0.3)

### **Violation Session Settings**
- **Save Interval**: 10 seconds between saves
- **Minimum Duration**: 3 seconds per session
- **Minimum Frames**: 15 violation frames required

## 📊 Response Format

### **Detection Response**
```json
{
  "image_id": "img_123",
  "width": 1280,
  "height": 720,
  "detections": [
    {
      "id": "det_1",
      "category": "person",
      "confidence": 0.91,
      "bbox": [100, 200, 180, 400]
    },
    {
      "id": "det_2",
      "category": "helmet",
      "confidence": 0.88,
      "bbox": [120, 180, 60, 40],
      "person_id": "pers_1",
      "matched": true
    }
  ],
  "persons": [
    {
      "person_id": "pers_1",
      "bbox": [100, 200, 180, 400],
      "has_helmet": true,
      "has_vest": true,
      "violations": []
    }
  ],
  "violations_summary": {
    "total_persons": 1,
    "compliant_persons": 1,
    "violations_count": {
      "no_helmet": 0,
      "no_vest": 0
    },
    "compliance_rate": 1.0
  },
  "timings": {
    "preprocessing": 5.1,
    "inference": 22.7,
    "postprocessing": 3.4
  }
}
```

### **Violation Record**
```json
{
  "saved_at": "2025-08-28T12:34:56.789",
  "source": "web_interface",
  "session_id": "abc12345",
  "session_duration_seconds": 15,
  "violation_frames_count": 45,
  "violations_summary": {
    "total_persons": 2,
    "compliant_persons": 1,
    "violations_count": {
      "no_helmet": 1,
      "no_vest": 0
    },
    "compliance_rate": 0.5
  },
  "persons": [
    {
      "person_id": "person_1",
      "has_helmet": false,
      "has_vest": true,
      "violations": ["no_helmet"]
    }
  ]
}
```

## 🐳 Docker Deployment

### **Build Image**
```bash
docker build -t safeconstruction .
```

### **Run Container**
```bash
docker run -p 8080:8080 safeconstruction
```

### **Docker Compose**
```bash
docker-compose up -d
```

## 🧪 Testing

### **Camera Testing**
```bash
python test_camera.py
```

### **API Testing**
```bash
python test_lite.py
```

### **Health Check**
```bash
curl http://localhost:8080/health
```

## 📁 Project Structure

```
safeconstruction/
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI application
│   ├── inference.py         # YOLOv8 inference logic
│   ├── ppe_logic.py         # PPE matching algorithms
│   ├── schemas.py           # Pydantic data models
│   ├── online_camera.py     # Desktop camera application
│   ├── static/              # Web interface assets
│   │   ├── index.html
│   │   ├── script.js
│   │   └── style.css
│   └── utils/               # Utility functions
├── scripts/
│   └── video_infer.py       # Video processing script
├── configs/
│   ├── default.yaml         # Default configuration
│   └── production.yaml      # Production settings
├── tests/                   # Unit tests
├── violations/              # Violation records storage
├── data/                    # Test data
├── requirements.txt         # Python dependencies
├── Dockerfile              # Docker configuration
├── docker-compose.yml      # Docker Compose setup
└── README.md               # This file
```

## 🛠️ Troubleshooting

### **Camera Issues**
- Check `CAMERA_TROUBLESHOOTING.md` for detailed camera setup guide
- Ensure camera permissions are enabled in browser
- Try different browsers or use `127.0.0.1` instead of `localhost`

### **Model Loading**
- YOLOv8 model is downloaded automatically on first run
- For custom models, place `.pt` files in project root
- Check model compatibility with Ultralytics YOLOv8

### **Performance Optimization**
- Use GPU inference for better performance
- Adjust confidence thresholds for speed vs accuracy
- Consider model size: `yolov8n.pt` (fastest) vs `yolov8x.pt` (most accurate)

## 📚 Additional Documentation

- **Setup Guide**: `SETUP_GUIDE.md` - Detailed installation and configuration
- **Camera Troubleshooting**: `CAMERA_TROUBLESHOOTING.md` - Camera setup and issues
- **IP Camera Enhancement**: `IP_CAMERA_ENHANCEMENT.md` - Advanced camera features

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) - Object detection model
- [FastAPI](https://fastapi.tiangolo.com/) - Modern web framework
- [OpenCV](https://opencv.org/) - Computer vision library

## 📞 Support

For support and questions:
- Create an issue on GitHub
- Check existing documentation
- Review troubleshooting guides

---

**Safe Construction** - Enhancing workplace safety through AI-powered PPE detection 🚧