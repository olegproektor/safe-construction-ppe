# PPE Detection System - Complete Setup and Deployment Guide

## 📁 Project Structure Verification

### ✅ Current Project Structure

```
Safe construction/
├── app/                     # 🔥 Main application code
│   ├── __init__.py         # Package initialization
│   ├── main.py             # 🚀 FastAPI application (20.8KB)
│   ├── inference.py        # 🧠 YOLOv8 inference (24.2KB)
│   ├── ppe_logic.py        # ⚡ PPE matching logic (18.6KB)
│   ├── schemas.py          # 📋 Pydantic schemas (17.0KB)
│   ├── online_camera.py    # 📹 Camera support (24.3KB) - USB/IP
│   ├── utils/
│   │   ├── __init__.py     # Package initialization
│   │   └── draw.py         # 🎨 Visualization (19.5KB)
│   ├── static/             # 🌐 Web interface assets
│   │   ├── index.html      # Web UI
│   │   ├── script.js       # Frontend logic
│   │   └── style.css       # Styling
│   └── templates/          # 📄 Jinja2 templates
│
├── scripts/                # 🔧 Utility scripts
│   ├── __init__.py        # Package initialization
│   └── video_infer.py     # 🎬 Video processing (NEW - 547 lines)
│
├── tests/                  # 🧪 Test suite
│   ├── __init__.py        # Package initialization
│   ├── conftest.py        # Test fixtures
│   ├── test_api.py        # 🔍 API tests (23.8KB)
│   └── test_inference.py  # Inference tests
│
├── configs/                # ⚙️ Configuration files
│   ├── default.yaml       # 📝 Default config (NEW - 137 lines)
│   └── production.yaml    # 🏭 Production config (NEW - 143 lines)
│
├── data/                   # 📊 Data storage
│   ├── images/            # Input images
│   └── videos/            # Input/output videos
│
├── requirements.txt        # 📦 Python dependencies
├── Dockerfile             # 🐳 Container configuration
├── docker-compose.yml     # 🏗️ Multi-container setup
├── .env.example           # 🔐 Environment variables template
├── .gitignore             # 🚫 Git ignore rules
└── README.md              # 📖 Documentation
```

### 🎯 All Required Files Status: ✅ COMPLETE

| Component | Status | Size | Description |
|-----------|--------|------|-------------|
| `app/main.py` | ✅ | 20.8KB | FastAPI application |
| `app/inference.py` | ✅ | 24.2KB | YOLOv8 inference |
| `app/ppe_logic.py` | ✅ | 18.6KB | PPE matching logic |
| `app/schemas.py` | ✅ | 17.0KB | Pydantic schemas |
| `app/utils/draw.py` | ✅ | 19.5KB | Visualization |
| `app/online_camera.py` | ✅ | 24.3KB | **Enhanced with IP camera support** |
| `scripts/video_infer.py` | ✅ | **NEW** | **Complete video processing script** |
| `tests/test_api.py` | ✅ | 23.8KB | Comprehensive API tests |
| `configs/default.yaml` | ✅ | **NEW** | **Complete configuration** |
| `requirements.txt` | ✅ | 0.6KB | Dependencies |
| `Dockerfile` | ✅ | 3.5KB | Container setup |

---

## 🚀 Installation and Setup

### 1. Environment Setup

#### 🐍 Local Installation (Windows/Linux/macOS)

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# Linux/macOS:
source venv/bin/activate
# Windows:
venv\\Scripts\\activate

# Install dependencies
pip install -r requirements.txt
```

#### 📦 Dependencies Overview

```txt
# Core ML/CV dependencies
ultralytics>=8.0.0      # YOLOv8
torch>=1.13.0           # PyTorch
opencv-python-headless>=4.5.0  # Computer Vision
numpy>=1.21.0           # Numerical computing

# Web framework
fastapi>=0.68.0         # API framework
uvicorn[standard]>=0.15.0  # ASGI server
pydantic>=1.8.0         # Data validation
python-multipart>=0.0.5 # File uploads

# Testing
pytest>=6.0.0           # Testing framework
httpx>=0.24.0           # HTTP client for tests

# Additional utilities
Pillow>=8.0.0           # Image processing
python-dotenv>=0.19.0   # Environment variables
```

### 2. Model Setup

#### 🧠 Default Model (Automatic Download)

```bash
# The system will automatically download yolov8n.pt on first run
# No manual setup required!
```

#### 🎯 Custom Model (Optional)

```bash
# Create weights directory
mkdir -p weights

# Copy your custom model
cp your_ppe_model.pt weights/

# Update configuration
# Edit configs/default.yaml:
model:
  weights: \"weights/your_ppe_model.pt\"
```

---

## 🔥 Running the System

### 1. 🌐 API Server

#### Development Mode
```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8080
```

#### Production Mode
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8080 --workers 4
```

#### 🔗 Access Points
- **Swagger UI**: http://localhost:8080/docs
- **ReDoc**: http://localhost:8080/redoc
- **Web Interface**: http://localhost:8080/
- **Health Check**: http://localhost:8080/health

### 2. 📹 Camera Applications

#### Local Camera (USB Webcam)
```bash
python app/online_camera.py
```

#### IP Camera (RTSP Stream)
```bash
python app/online_camera.py --stream_url rtsp://192.168.1.15:554/stream1
```

#### With Video Recording
```bash
python app/online_camera.py \\n    --stream_url rtsp://192.168.1.15:554/stream1 \\n    --conf 0.35 \\n    --iou 0.5 \\n    --save_output monitoring_$(date +%Y%m%d_%H%M%S).mp4
```

#### HTTP/HTTPS IP Cameras
```bash
# HTTP camera
python app/online_camera.py --stream_url http://192.168.1.20:8080/video

# HTTPS camera  
python app/online_camera.py --stream_url https://camera.example.com/stream
```

### 3. 🎬 Video Processing

#### Process Video File
```bash
python scripts/video_infer.py --input input_video.mp4 --output processed_video.mp4
```

#### Advanced Video Processing
```bash
python scripts/video_infer.py \\n    --input construction_site.mp4 \\n    --output safety_analysis.mp4 \\n    --conf 0.4 \\n    --iou 0.6 \\n    --save_detections \\n    --skip_frames 2
```

#### Process RTSP Stream to Video
```bash
python scripts/video_infer.py \\n    --input rtsp://192.168.1.15:554/stream1 \\n    --output stream_recording.mp4 \\n    --save_detections
```

---

## 🧪 Testing and Validation

### 1. Run Test Suite
```bash
# Run all tests
pytest tests/

# Run with verbose output
pytest tests/ -v

# Run specific test file
pytest tests/test_api.py -v

# Run with coverage
pytest tests/ --cov=app --cov-report=html
```

### 2. Health Check
```bash
# Check API health
curl http://localhost:8080/health

# Expected response:
{
  \"status\": \"healthy\",
  \"timestamp\": \"2025-08-26T...\",
  \"version\": \"1.0.0\",
  \"model_loaded\": true,
  \"uptime_seconds\": 123.45
}
```

### 3. Test Image Detection
```bash
# Test with sample image
curl -X POST \"http://localhost:8080/detect/image\" \\n  -F \"file=@test_image.jpg\" \\n  -F \"conf=0.35\" \\n  -F \"iou=0.5\" \\n  -F \"draw=true\"
```

---

## 🐳 Docker Deployment

### 1. Build Container
```bash
docker build -t safe-construction .
```

### 2. Run Container

#### CPU Version
```bash
docker run -it --rm -p 8080:8080 safe-construction
```

#### GPU Version (NVIDIA)
```bash
docker run -it --rm --gpus all -p 8080:8080 safe-construction
```

#### With Volume Mounts
```bash
docker run -it --rm \\n  -p 8080:8080 \\n  -v $(pwd)/data:/app/data \\n  -v $(pwd)/weights:/app/weights \\n  safe-construction
```

### 3. Docker Compose
```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

---

## ⚙️ Configuration

### 1. Environment Variables

Create `.env` file from template:
```bash
cp .env.example .env
```

Edit `.env`:
```env
# Model configuration
MODEL_WEIGHTS=yolov8n.pt
DEVICE=auto
CONFIDENCE_THRESHOLD=0.35
IOU_THRESHOLD=0.5

# API configuration
API_HOST=0.0.0.0
API_PORT=8000
API_WORKERS=4

# Logging
LOG_LEVEL=INFO
LOG_FILE=logs/app.log
```

### 2. YAML Configuration

#### Development
Use `configs/default.yaml` for development settings.

#### Production
Use `configs/production.yaml` for production deployment:
- Higher accuracy model (yolov8s.pt)
- GPU acceleration
- Multiple workers
- Enhanced logging
- Performance monitoring

---

## 📊 Usage Examples

### 1. Factory Safety Monitoring
```bash
# Monitor factory floor with IP camera
python app/online_camera.py \\n    --stream_url rtsp://factory-cam-01.local:554/main \\n    --conf 0.4 \\n    --save_output factory_safety_$(date +%Y%m%d).mp4
```

### 2. Construction Site Analysis
```bash
# Process construction site video
python scripts/video_infer.py \\n    --input site_footage.mp4 \\n    --output compliance_report.mp4 \\n    --save_detections \\n    --conf 0.35
```

### 3. Real-time Compliance Dashboard
```bash
# Start API server for web dashboard
uvicorn app.main:app --host 0.0.0.0 --port 8080

# Access web interface at http://localhost:8080
# Use browser camera for real-time detection
```

### 4. Batch Video Processing
```bash
# Process multiple videos
for video in data/videos/*.mp4; do
    output=\"data/output/processed_$(basename $video)\"
    python scripts/video_infer.py --input \"$video\" --output \"$output\" --save_detections
done
```

---

## 🔧 Troubleshooting

### Common Issues

#### 1. Model Download Issues
```bash
# Manual model download
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt
```

#### 2. Camera Connection Issues
```bash
# Check available cameras
python -c \"import cv2; print([cv2.VideoCapture(i).isOpened() for i in range(10)])\"

# Test RTSP stream
ffplay rtsp://192.168.1.15:554/stream1
```

#### 3. GPU Issues
```bash
# Check CUDA availability
python -c \"import torch; print(torch.cuda.is_available())\"

# Install CUDA-enabled PyTorch
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

#### 4. Port Already in Use
```bash
# Use different port
uvicorn app.main:app --host 0.0.0.0 --port 8001

# Or kill existing process
lsof -ti:8080 | xargs kill -9
```

---

## 🎯 Performance Optimization

### 1. Model Selection
- **Fast**: `yolov8n.pt` (6.2MB) - Real-time applications
- **Balanced**: `yolov8s.pt` (21.5MB) - Production use
- **Accurate**: `yolov8m.pt` (49.7MB) - High accuracy needs

### 2. Hardware Acceleration
```bash
# NVIDIA GPU
export CUDA_VISIBLE_DEVICES=0

# Apple Silicon (M1/M2)
export PYTORCH_ENABLE_MPS_FALLBACK=1
```

### 3. Inference Optimization
```yaml
# configs/production.yaml
model:
  image_size: 640      # Balance speed vs accuracy
  max_batch_size: 8    # GPU memory dependent
  
performance:
  use_tensorrt: true   # NVIDIA GPU optimization
  use_multiprocessing: true
  max_workers: 8
```

---

## 🔐 Security Considerations

### 1. API Security
```yaml
# configs/production.yaml
security:
  enable_api_key: true
  api_key_header: \"X-API-Key\"
  enable_rate_limiting: true
  rate_limit_requests: 100
  rate_limit_period: 60
```

### 2. Network Security
- Use HTTPS in production
- Implement proper CORS settings
- Secure RTSP streams with authentication

### 3. Data Privacy
- Configure automatic cleanup of temporary files
- Implement data retention policies
- Secure storage of recorded videos

---

## 📈 Monitoring and Logging

### 1. Application Logs
```bash
# View real-time logs
tail -f logs/app.log

# Search for errors
grep ERROR logs/app.log
```

### 2. Performance Metrics
- Processing time per frame
- Memory usage monitoring
- GPU utilization tracking
- Detection accuracy metrics

### 3. Health Monitoring
```bash
# Automated health checks
while true; do
    curl -f http://localhost:8080/health || echo \"Service down!\"
    sleep 30
done
```

---

## 🎉 System Ready!

Your PPE Detection System is now completely set up and ready for production use with:

✅ **Complete File Structure** - All components in place  
✅ **Enhanced IP Camera Support** - RTSP/HTTP/HTTPS streams  
✅ **Video Processing** - Batch and real-time processing  
✅ **Web Interface** - Browser-based real-time detection  
✅ **Comprehensive Configuration** - Development and production  
✅ **Docker Support** - Containerized deployment  
✅ **Testing Suite** - Automated quality assurance  
✅ **Professional Documentation** - Complete setup guide  

**Next Steps:**
1. Start with local camera: `python app/online_camera.py`
2. Test API: `uvicorn app.main:app --reload`
3. Try video processing: `python scripts/video_infer.py --input test.mp4 --output result.mp4`
4. Deploy with Docker: `docker build -t safe-construction . && docker run -p 8080:8080 safe-construction`

**Support & Documentation:**
- 📖 Full documentation in `README.md`
- 🔧 Configuration examples in `configs/`
- 🧪 Test examples in `tests/`
- 🎬 Video processing guide in `scripts/video_infer.py`

**Happy PPE Monitoring! 🦺⛑️🏗️**