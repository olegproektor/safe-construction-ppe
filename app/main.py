"""
FastAPI Main Application for PPE Detection System with Web Interface

This module provides both REST API endpoints and a web interface for the PPE detection system.
It includes endpoints for health checks, model information, image detection, and serves
the real-time web interface for webcam-based PPE monitoring.

Endpoints:
- GET /: Web interface for real-time detection
- GET /health: Health check
- GET /model/info: Model information and default parameters
- POST /detect/image: Image detection with PPE analysis
- Static files serving for web interface assets

Author: PPE Detection System
Date: 2025-08-26
"""

import base64
import io
import logging
import time
import traceback
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Any, List

import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from PIL import Image
import uvicorn

# Import application modules
from .inference import (
    load_model, run_inference, get_cached_model, 
    clear_model_cache, get_detections_by_category,
    DEFAULT_CONF_THRESHOLD, DEFAULT_IOU_THRESHOLD
)
from .ppe_logic import (
    match_ppe, calculate_violations_summary, 
    get_default_thresholds, validate_thresholds
)
from .schemas import (
    DetectionResponse, ErrorResponse, HealthCheckResponse,
    Detection, PersonStatus, ViolationsSummary, ProcessingTimings
)
from .utils.draw import draw_detections

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Application configuration
APP_VERSION = "1.0.0"
APP_TITLE = "PPE Detection API with Web Interface"
APP_DESCRIPTION = "Personal Protective Equipment Detection System using YOLOv8 with Real-time Web Interface"
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/jpg", "image/png", "image/bmp"}
DEFAULT_MODEL_PATH = "yolov8n.pt"

# Global variables
app_start_time = time.time()
global_model = None
model_info = {}
last_violation_save_time = 0  # Track last violation save for rate limiting
VIOLATION_SAVE_INTERVAL = 10  # Minimum seconds between violation saves (reduced for testing)
active_violation_sessions = {}  # Track active violation sessions to prevent duplicates
MIN_SESSION_DURATION = 3  # Minimum 3 seconds for a violation session (reduced for testing)
MIN_VIOLATION_FRAMES = 15  # Minimum 15 frames with violations before saving (reduced for testing)

# Get current directory for static files
current_dir = Path(__file__).parent
static_dir = current_dir / "static"
templates_dir = current_dir / "templates"

# Create FastAPI application
app = FastAPI(
    title=APP_TITLE,
    description=APP_DESCRIPTION,
    version=APP_VERSION,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

# Mount static files
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")
    logger.info(f"Static files mounted from: {static_dir}")
else:
    logger.warning(f"Static directory not found: {static_dir}")

# Setup templates (if templates directory exists and has files)
if templates_dir.exists():
    # Check if there are any template files in the directory
    template_files = list(templates_dir.glob('*.html'))
    if template_files:
        templates = Jinja2Templates(directory=str(templates_dir))
        logger.info(f"Templates directory found with {len(template_files)} template(s): {templates_dir}")
    else:
        templates = None
        logger.info(f"Templates directory found but empty: {templates_dir}")
else:
    templates = None
    logger.info("No templates directory found, using static HTML")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Exception handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    """
    Handle HTTP exceptions with consistent error response format.
    
    Args:
        request (Request): FastAPI request object
        exc (HTTPException): HTTP exception
        
    Returns:
        JSONResponse: Formatted error response
    """
    error_response = ErrorResponse(
        error=exc.detail,
        error_code=f"HTTP_{exc.status_code}",
        details={"status_code": exc.status_code}
    )
    
    logger.error(f"HTTP {exc.status_code}: {exc.detail}")
    
    return JSONResponse(
        status_code=exc.status_code,
        content=error_response.dict()
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """
    Handle general exceptions with error logging.
    
    Args:
        request (Request): FastAPI request object
        exc (Exception): General exception
        
    Returns:
        JSONResponse: Formatted error response
    """
    error_id = str(uuid.uuid4())[:8]
    
    logger.error(f"Error ID {error_id}: {str(exc)}")
    logger.error(f"Traceback: {traceback.format_exc()}")
    
    error_response = ErrorResponse(
        error="Internal server error occurred",
        error_code="INTERNAL_SERVER_ERROR",
        details={"error_id": error_id}
    )
    
    return JSONResponse(
        status_code=500,
        content=error_response.dict()
    )


# Startup and shutdown events
@app.on_event("startup")
async def startup_event() -> None:
    """
    Initialize application resources on startup.
    """
    global global_model, model_info
    
    logger.info(f"Starting {APP_TITLE} v{APP_VERSION}")
    
    try:
        # Load default model
        logger.info(f"Loading model: {DEFAULT_MODEL_PATH}")
        global_model = get_cached_model(DEFAULT_MODEL_PATH, "auto")
        
        # Store model information
        model_info = {
            "model_path": DEFAULT_MODEL_PATH,
            "classes": list(global_model.model.names.values()) if hasattr(global_model, 'model') else [],
            "conf_default": DEFAULT_CONF_THRESHOLD,
            "iou_default": DEFAULT_IOU_THRESHOLD,
            "device": str(global_model.device) if hasattr(global_model, 'device') else "unknown",
            "loaded_at": datetime.now().isoformat()
        }
        
        logger.info(f"Model loaded successfully on device: {model_info['device']}")
        logger.info(f"Available classes: {model_info['classes']}")
        
    except Exception as e:
        logger.error(f"Failed to load model during startup: {e}")
        # Continue startup but mark model as unavailable
        model_info = {
            "model_path": None,
            "classes": [],
            "conf_default": DEFAULT_CONF_THRESHOLD,
            "iou_default": DEFAULT_IOU_THRESHOLD,
            "device": "unavailable",
            "error": str(e)
        }


@app.on_event("shutdown")
async def shutdown_event() -> None:
    """
    Clean up resources on shutdown.
    """
    logger.info("Shutting down PPE Detection API")
    
    # Clear model cache to free memory
    clear_model_cache()
    
    logger.info("Shutdown complete")


# Utility functions
def validate_image_file(file: UploadFile) -> None:
    """
    Validate uploaded image file.
    
    Args:
        file (UploadFile): Uploaded file object
        
    Raises:
        HTTPException: If file validation fails
    """
    # Check file size
    if hasattr(file, 'size') and file.size and file.size > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum size is {MAX_FILE_SIZE // (1024*1024)}MB"
        )
    
    # Check content type
    if file.content_type not in ALLOWED_IMAGE_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Allowed types: {', '.join(ALLOWED_IMAGE_TYPES)}"
        )


def process_uploaded_image(file: UploadFile) -> np.ndarray:
    """
    Process uploaded image file into numpy array.
    
    Args:
        file (UploadFile): Uploaded image file
        
    Returns:
        np.ndarray: Image as numpy array in BGR format
        
    Raises:
        HTTPException: If image processing fails
    """
    try:
        # Read file content
        image_data = file.file.read()
        
        # Convert to PIL Image
        pil_image = Image.open(io.BytesIO(image_data))
        
        # Convert to RGB if necessary
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')
        
        # Convert to numpy array
        image_array = np.array(pil_image)
        
        # Convert RGB to BGR for OpenCV
        image_bgr = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
        
        return image_bgr
        
    except Exception as e:
        logger.error(f"Error processing image: {e}")
        raise HTTPException(
            status_code=400,
            detail=f"Failed to process image: {str(e)}"
        )


def encode_image_to_base64(image: np.ndarray, format: str = "jpg") -> str:
    """
    Encode numpy image array to base64 string.
    
    Args:
        image (np.ndarray): Image array in BGR format
        format (str): Output format (jpg, png)
        
    Returns:
        str: Base64 encoded image string
    """
    try:
        # Encode image
        if format.lower() in ['jpg', 'jpeg']:
            _, buffer = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 85])
        else:
            _, buffer = cv2.imencode('.png', image)
        
        # Convert to base64
        image_base64 = base64.b64encode(buffer).decode('utf-8')
        
        return image_base64
        
    except Exception as e:
        logger.error(f"Error encoding image to base64: {e}")
        return ""


# Web Interface Routes

@app.get(
    "/",
    response_class=HTMLResponse,
    summary="Web Interface",
    description="Real-time PPE detection web interface with webcam support"
)
async def web_interface(request: Request) -> HTMLResponse:
    """
    Serve the main web interface for real-time PPE detection.
    
    Returns:
        HTMLResponse: HTML page with webcam interface
    """
    # If templates are available, use them
    if templates:
        return templates.TemplateResponse(
            "index.html", 
            {"request": request, "title": "Safe Construction - PPE Detection"}
        )
    
    # Otherwise, serve static HTML directly
    static_html_path = static_dir / "index.html"
    if static_html_path.exists():
        with open(static_html_path, 'r', encoding='utf-8') as f:
            html_content = f.read()
        return HTMLResponse(content=html_content)
    
    # Fallback: basic HTML
    return HTMLResponse(
        content="""
        <!DOCTYPE html>
        <html><head><title>PPE Detection</title></head>
        <body>
            <h1>PPE Detection System</h1>
            <p>Web interface files not found. Please check static files setup.</p>
            <p><a href="/docs">API Documentation</a></p>
        </body></html>
        """
    )


# API Endpoints

@app.get(
    "/health",
    response_model=HealthCheckResponse,
    summary="Health Check",
    description="Check the health status of the PPE detection service"
)
async def health_check() -> HealthCheckResponse:
    """
    Health check endpoint.
    
    Returns service status, uptime, and model availability information.
    
    Returns:
        HealthCheckResponse: Service health information
    """
    uptime_seconds = time.time() - app_start_time
    model_loaded = global_model is not None
    
    # Determine service status
    if model_loaded:
        status = "healthy"
    else:
        status = "degraded"
    
    return HealthCheckResponse(
        status=status,
        version=APP_VERSION,
        model_loaded=model_loaded,
        uptime_seconds=uptime_seconds
    )


@app.get(
    "/model/info",
    response_model=Dict[str, Any],
    summary="Model Information",
    description="Get information about the loaded model and default parameters"
)
async def get_model_info() -> Dict[str, Any]:
    """
    Get model information and default parameters.
    
    Returns model classes, default thresholds, device information,
    and other configuration details.
    
    Returns:
        Dict[str, Any]: Model information and parameters
    """
    if not global_model:
        raise HTTPException(
            status_code=503,
            detail="Model not available. Service may be starting up or experiencing issues."
        )
    
    # Get PPE matching thresholds
    ppe_thresholds = get_default_thresholds()
    
    response = {
        "classes": model_info.get("classes", []),
        "conf_default": model_info.get("conf_default", DEFAULT_CONF_THRESHOLD),
        "iou_default": model_info.get("iou_default", DEFAULT_IOU_THRESHOLD),
        "device": model_info.get("device", "unknown"),
        "model_path": model_info.get("model_path"),
        "ppe_thresholds": ppe_thresholds,
        "max_file_size_mb": MAX_FILE_SIZE // (1024 * 1024),
        "allowed_image_types": list(ALLOWED_IMAGE_TYPES),
        "loaded_at": model_info.get("loaded_at")
    }
    
    return response


@app.post(
    "/detect/image",
    response_model=DetectionResponse,
    summary="PPE Detection on Image",
    description="Perform PPE detection on an uploaded image with configurable parameters"
)
async def detect_image(
    file: UploadFile = File(..., description="Image file for PPE detection"),
    conf: float = Form(DEFAULT_CONF_THRESHOLD, description="Confidence threshold (0.0-1.0)"),
    iou: float = Form(DEFAULT_IOU_THRESHOLD, description="IoU threshold for NMS (0.0-1.0)"),
    ppe_overlap: float = Form(0.3, description="PPE overlap threshold (0.0-1.0)"),
    draw: bool = Form(True, description="Whether to draw annotations on image"),
    return_image: bool = Form(False, description="Whether to return annotated image as base64")
) -> DetectionResponse:
    """
    Perform PPE detection on an uploaded image.
    
    This endpoint accepts an image file and performs comprehensive PPE detection
    including object detection, person-PPE matching, and violation analysis.
    
    Args:
        file (UploadFile): Image file (JPEG, PNG, BMP)
        conf (float): Confidence threshold for detections (0.0-1.0)
        iou (float): IoU threshold for Non-Maximum Suppression (0.0-1.0)
        ppe_overlap (float): Minimum overlap threshold for PPE matching (0.0-1.0)
        draw (bool): Whether to draw bounding boxes and annotations
        return_image (bool): Whether to include base64 encoded result image
        
    Returns:
        DetectionResponse: Complete detection results with person status and violations
        
    Raises:
        HTTPException: For validation errors, processing failures, or service unavailability
    """
    # Validate model availability
    if not global_model:
        raise HTTPException(
            status_code=503,
            detail="Model not available. Please try again later."
        )
    
    # Validate parameters
    if not (0.0 <= conf <= 1.0):
        raise HTTPException(status_code=400, detail="Confidence threshold must be between 0.0 and 1.0")
    
    if not (0.0 <= iou <= 1.0):
        raise HTTPException(status_code=400, detail="IoU threshold must be between 0.0 and 1.0")
    
    if not (0.0 <= ppe_overlap <= 1.0):
        raise HTTPException(status_code=400, detail="PPE overlap threshold must be between 0.0 and 1.0")
    
    # Validate and process image
    validate_image_file(file)
    
    try:
        # Start timing
        start_time = time.time()
        
        # Process uploaded image
        preprocessing_start = time.time()
        image = process_uploaded_image(file)
        preprocessing_time = time.time() - preprocessing_start
        
        # Get image dimensions
        height, width = image.shape[:2]
        
        # Generate unique image ID
        image_id = f"img_{int(time.time())}_{uuid.uuid4().hex[:8]}"
        
        logger.info(f"Processing image {image_id}: {width}x{height}, conf={conf}, iou={iou}")
        
        # Run inference
        inference_start = time.time()
        detections_raw = run_inference(global_model, image, conf, iou)
        inference_time = time.time() - inference_start
        
        # Convert detections to schema format
        detections = [
            Detection(
                id=f"det_{i}_{uuid.uuid4().hex[:6]}",
                category=det['category'],
                confidence=det['confidence'],
                bbox=det['bbox'],
                person_id=None,
                matched=False
            )
            for i, det in enumerate(detections_raw)
        ]
        
        # Separate detections by category
        categorized = get_detections_by_category(detections_raw)
        person_boxes = categorized['person']
        helmet_boxes = categorized['helmet']
        vest_boxes = categorized['vest']
        
        # Perform PPE matching
        postprocessing_start = time.time()
        
        # Prepare thresholds
        thresholds = {
            'ppe_overlap_threshold': ppe_overlap,
            'min_ppe_area_ratio': 0.01,
            'max_ppe_area_ratio': 0.5,
            'min_iou_threshold': 0.1
        }
        thresholds = validate_thresholds(thresholds)
        
        # Match PPE to persons
        persons = match_ppe(person_boxes, helmet_boxes, vest_boxes, thresholds)
        
        # Calculate violations summary
        violations_summary = calculate_violations_summary(persons)
        
        postprocessing_time = time.time() - postprocessing_start
        
        # Draw annotations if requested
        annotated_image = None
        if draw:
            annotated_image = draw_detections(image, detections_raw, persons)
        
        # Prepare timing information
        total_time = time.time() - start_time
        timings = ProcessingTimings(
            inference_time=inference_time,
            preprocessing_time=preprocessing_time,
            postprocessing_time=postprocessing_time,
            total_time=total_time
        )
        
        # Prepare response
        response = DetectionResponse(
            image_id=image_id,
            width=width,
            height=height,
            detections=detections,
            persons=persons,
            violations_summary=violations_summary,
            timings=timings,
            model_version="YOLOv8"
        )
        
        # Add base64 image if requested
        if return_image and annotated_image is not None:
            image_base64 = encode_image_to_base64(annotated_image)
            # Add to response (extend the schema or use additional field)
            response_dict = response.dict()
            response_dict['annotated_image_base64'] = image_base64
            
            logger.info(f"Image {image_id} processed successfully in {total_time:.3f}s")
            total_violations = sum(violations_summary.violations_count.values())
            logger.info(f"Results: {len(detections)} detections, {len(persons)} persons, "
                       f"{total_violations} violations")
            
            return JSONResponse(content=response_dict)
        
        logger.info(f"Image {image_id} processed successfully in {total_time:.3f}s")
        total_violations = sum(violations_summary.violations_count.values())
        logger.info(f"Results: {len(detections)} detections, {len(persons)} persons, "
                   f"{total_violations} violations")
        
        return response
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Error processing image: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process image: {str(e)}"
        )


# Additional utility endpoints

@app.get(
    "/status",
    summary="Service Status",
    description="Get detailed service status and statistics"
)
async def get_status() -> Dict[str, Any]:
    """
    Get detailed service status and statistics.
    
    Returns:
        Dict[str, Any]: Detailed status information
    """
    uptime_seconds = time.time() - app_start_time
    
    status = {
        "service": APP_TITLE,
        "version": APP_VERSION,
        "uptime_seconds": uptime_seconds,
        "uptime_human": f"{uptime_seconds / 3600:.1f} hours",
        "model_loaded": global_model is not None,
        "model_info": model_info,
        "timestamp": datetime.now().isoformat(),
        "endpoints": {
            "web_interface": "/",
            "health": "/health",
            "model_info": "/model/info",
            "detect_image": "/detect/image",
            "documentation": "/docs"
        }
    }
    
    # Convert any datetime objects in model_info to strings
    if "loaded_at" in status["model_info"] and isinstance(status["model_info"]["loaded_at"], datetime):
        status["model_info"]["loaded_at"] = status["model_info"]["loaded_at"].isoformat()
    
    return status


# Violations Folder Management Endpoints

@app.get(
    "/violations/stats",
    summary="Get Violations Folder Statistics",
    description="Get statistics about violations folder contents"
)
async def get_violations_stats() -> Dict[str, Any]:
    """
    Get statistics about the violations folder.
    
    Returns:
        Dict[str, Any]: Folder statistics including file counts and sizes
    """
    import os
    from pathlib import Path
    
    try:
        # Define violations folder path
        violations_folder = Path.cwd() / "violations"
        
        if not violations_folder.exists():
            violations_folder.mkdir(exist_ok=True)
        
        # Count different file types
        video_files = list(violations_folder.glob("*.mp4")) + list(violations_folder.glob("*.avi"))
        json_files = list(violations_folder.glob("*.json"))
        csv_files = list(violations_folder.glob("*.csv"))
        
        # Calculate total size
        total_size = 0
        for file_path in violations_folder.iterdir():
            if file_path.is_file():
                total_size += file_path.stat().st_size
        
        # Format size
        if total_size < 1024 * 1024:
            size_str = f"{total_size / 1024:.1f} KB"
        elif total_size < 1024 * 1024 * 1024:
            size_str = f"{total_size / (1024 * 1024):.1f} MB"
        else:
            size_str = f"{total_size / (1024 * 1024 * 1024):.1f} GB"
        
        # Find last violation timestamp
        last_violation = "Никогда"
        if json_files:
            latest_file = max(json_files, key=lambda x: x.stat().st_mtime)
            from datetime import datetime
            last_violation = datetime.fromtimestamp(latest_file.stat().st_mtime).strftime("%d.%m.%Y %H:%M")
        
        return {
            "video_files": len(video_files),
            "json_files": len(json_files),
            "csv_files": len(csv_files),
            "total_size": size_str,
            "last_violation": last_violation,
            "folder_path": str(violations_folder)
        }
        
    except Exception as e:
        logger.error(f"Error getting violations stats: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get violations statistics: {str(e)}"
        )


@app.post(
    "/violations/open-folder",
    summary="Open Violations Folder",
    description="Open the violations folder in the system file explorer"
)
async def open_violations_folder() -> Dict[str, str]:
    """
    Open the violations folder in the system file explorer.
    
    Returns:
        Dict[str, str]: Success message
    """
    import os
    import platform
    import subprocess
    from pathlib import Path
    
    try:
        # Define violations folder path
        violations_folder = Path.cwd() / "violations"
        
        # Create folder if it doesn't exist
        if not violations_folder.exists():
            violations_folder.mkdir(exist_ok=True)
        
        # Open folder based on operating system
        system = platform.system()
        
        if system == "Windows":
            os.startfile(str(violations_folder))
        elif system == "Darwin":  # macOS
            subprocess.run(["open", str(violations_folder)])
        else:  # Linux and others
            subprocess.run(["xdg-open", str(violations_folder)])
        
        logger.info(f"Opened violations folder: {violations_folder}")
        return {"message": f"Папка нарушений открыта: {violations_folder}"}
        
    except Exception as e:
        logger.error(f"Error opening violations folder: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to open violations folder: {str(e)}"
        )


@app.get(
    "/violations/download-archive",
    summary="Download Violations Archive",
    description="Download a ZIP archive of all violation files"
)
async def download_violations_archive():
    """
    Create and download a ZIP archive of all violation files.
    
    Returns:
        Response: ZIP file download
    """
    import zipfile
    import tempfile
    from pathlib import Path
    from fastapi.responses import FileResponse
    
    try:
        # Define violations folder path
        violations_folder = Path.cwd() / "violations"
        
        if not violations_folder.exists() or not any(violations_folder.iterdir()):
            raise HTTPException(
                status_code=404,
                detail="No violation files found to archive"
            )
        
        # Create temporary ZIP file
        temp_dir = Path(tempfile.gettempdir())
        zip_filename = f"violations_archive_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"
        zip_path = temp_dir / zip_filename
        
        # Create ZIP archive
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for file_path in violations_folder.rglob('*'):
                if file_path.is_file():
                    # Add file to ZIP with relative path
                    arcname = file_path.relative_to(violations_folder)
                    zipf.write(file_path, arcname)
        
        logger.info(f"Created violations archive: {zip_path}")
        
        # Return file for download
        return FileResponse(
            path=str(zip_path),
            filename=zip_filename,
            media_type='application/zip'
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating violations archive: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to create violations archive: {str(e)}"
        )


@app.post(
    "/violations/save",
    summary="Save Violation Record",
    description="Save a violation record to the violations folder"
)
async def save_violation_record(violation_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Save violation record to the violations folder as JSON with enhanced rate limiting and session management.
    
    Args:
        violation_data (Dict[str, Any]): Violation data including timestamp, detections, etc.
        
    Returns:
        Dict[str, str]: Success message with file path
    """
    import json
    from datetime import datetime
    from pathlib import Path
    import hashlib
    
    try:
        global last_violation_save_time, active_violation_sessions
        current_time = time.time()
        
        # Enhanced rate limiting - much stricter
        time_since_last_save = current_time - last_violation_save_time
        if time_since_last_save < VIOLATION_SAVE_INTERVAL:
            return {
                "message": f"⏸️ Сохранение ограничено: {VIOLATION_SAVE_INTERVAL - time_since_last_save:.1f}с до следующего",
                "file_path": "rate_limited",
                "rate_limited": True
            }
        
        # Validate session quality
        session_duration = violation_data.get('session_duration_seconds', 0)
        violation_frames = violation_data.get('violation_frames_count', 0)
        
        if session_duration < MIN_SESSION_DURATION:
            return {
                "message": f"⏸️ Сессия слишком короткая: {session_duration}с (мин. {MIN_SESSION_DURATION}с)",
                "file_path": "session_too_short",
                "rate_limited": True
            }
        
        if violation_frames < MIN_VIOLATION_FRAMES:
            return {
                "message": f"⏸️ Недостаточно кадров с нарушениями: {violation_frames} (мин. {MIN_VIOLATION_FRAMES})",
                "file_path": "insufficient_frames",
                "rate_limited": True
            }
        
        # Create session fingerprint to prevent duplicates
        violations_summary = violation_data.get('violations_summary', {})
        session_key = hashlib.md5(
            f"{violations_summary.get('total_persons', 0)}_{violations_summary.get('violations_count', {})}".encode()
        ).hexdigest()[:8]
        
        # Check if similar session was recently saved
        if session_key in active_violation_sessions:
            last_session_time = active_violation_sessions[session_key]
            if current_time - last_session_time < VIOLATION_SAVE_INTERVAL * 2:  # Double interval for similar sessions
                return {
                    "message": f"⏸️ Похожая сессия недавно сохранена (ID: {session_key})",
                    "file_path": "duplicate_session",
                    "rate_limited": True
                }
        
        # Define violations folder path
        violations_folder = Path.cwd() / "violations"
        
        # Create folder if it doesn't exist
        if not violations_folder.exists():
            violations_folder.mkdir(exist_ok=True)
        
        # Generate filename with session info
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        filename = f"violation_session_{session_key}_{timestamp}.json"
        file_path = violations_folder / filename
        
        # Add enhanced metadata
        enhanced_data = {
            "saved_at": datetime.now().isoformat(),
            "source": "web_interface",
            "version": "2.0",
            "session_id": session_key,
            "rate_limiting_applied": True,
            "validation_passed": {
                "min_duration": session_duration >= MIN_SESSION_DURATION,
                "min_frames": violation_frames >= MIN_VIOLATION_FRAMES,
                "rate_limit_ok": time_since_last_save >= VIOLATION_SAVE_INTERVAL
            },
            **violation_data
        }
        
        # Save to file
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(enhanced_data, f, indent=2, ensure_ascii=False)
        
        # Update tracking
        last_violation_save_time = current_time
        active_violation_sessions[session_key] = current_time
        
        # Clean old session entries
        cutoff_time = current_time - VIOLATION_SAVE_INTERVAL * 3
        active_violation_sessions = {k: v for k, v in active_violation_sessions.items() if v > cutoff_time}
        
        logger.info(f"Saved enhanced violation session: {filename}")
        logger.info(f"  Session ID: {session_key}, Duration: {session_duration}s, Frames: {violation_frames}")
        
        return {
            "message": f"✅ Сессия нарушений сохранена: {session_duration}с, {violation_frames} кадров",
            "file_path": str(file_path),
            "session_id": session_key,
            "rate_limited": False
        }
        
    except Exception as e:
        logger.error(f"Error saving violation record: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to save violation record: {str(e)}"
        )


@app.post(
    "/violations/clear",
    summary="Clear Violations Folder",
    description="Delete all files in the violations folder"
)
async def clear_violations_folder() -> Dict[str, str]:
    """
    Clear all files from the violations folder.
    
    Returns:
        Dict[str, str]: Success message with count of deleted files
    """
    import shutil
    from pathlib import Path
    
    try:
        # Define violations folder path
        violations_folder = Path.cwd() / "violations"
        
        if not violations_folder.exists():
            return {"message": "Папка нарушений не существует"}
        
        # Count files before deletion
        file_count = sum(1 for f in violations_folder.rglob('*') if f.is_file())
        
        if file_count == 0:
            return {"message": "Папка нарушений уже пуста"}
        
        # Clear all files and subdirectories
        for item in violations_folder.iterdir():
            if item.is_file():
                item.unlink()
            elif item.is_dir():
                shutil.rmtree(item)
        
        logger.info(f"Cleared violations folder: {file_count} files deleted")
        return {
            "message": f"Папка нарушений очищена. Удалено файлов: {file_count}"
        }
        
    except Exception as e:
        logger.error(f"Error clearing violations folder: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to clear violations folder: {str(e)}"
        )











# Development server configuration
if __name__ == "__main__":
    # This allows running the server directly with: python app/main.py
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8080,
        reload=True,
        log_level="info"
    )