"""
YOLOv8 Inference Module for PPE Detection

This module provides functionality for loading YOLOv8 models and running
inference on images to detect Personal Protective Equipment (PPE) items
such as helmets, safety vests, and persons.

Author: PPE Detection System
Date: 2025-08-26
"""

import logging
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any

import numpy as np
import torch
from ultralytics import YOLO
from ultralytics.engine.results import Results

# Import schemas and other modules for the wrapper function
try:
    from .schemas import Detection, PersonStatus, ViolationType
    from .ppe_logic import (
        match_ppe, calculate_violations_summary, 
        get_default_thresholds
    )
except ImportError:
    # Handle case when running as standalone
    Detection = None
    PersonStatus = None
    ViolationType = None
    match_ppe = None
    calculate_violations_summary = None
    get_default_thresholds = None

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# PPE class mappings for different model types
PPE_CLASS_NAMES = {
    'person': ['person', 'people', 'human'],
    'helmet': ['helmet', 'hard_hat', 'hardhat', 'hard hat', 'safety_helmet'],
    'vest': ['vest', 'safety_vest', 'safetyvest', 'safety vest', 'hi_vis', 'hi-vis']
}

# Default confidence and IoU thresholds
DEFAULT_CONF_THRESHOLD = 0.35
DEFAULT_IOU_THRESHOLD = 0.5


def detect_device() -> str:
    """
    Automatically detect the best available device for inference.
    
    Checks for CUDA GPU availability, then MPS (Apple Silicon), 
    and falls back to CPU if neither is available.
    
    Returns:
        str: Device string ('cuda', 'mps', or 'cpu')
    """
    try:
        # Check for CUDA (NVIDIA GPU)
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            logger.info(f"CUDA available with {device_count} GPU(s)")
            return 'cuda'
        
        # Check for MPS (Apple Silicon)
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            logger.info("MPS (Apple Silicon) available")
            return 'mps'
        
        # Fallback to CPU
        logger.info("Using CPU for inference")
        return 'cpu'
        
    except Exception as e:
        logger.warning(f"Error detecting device: {e}. Falling back to CPU.")
        return 'cpu'


def validate_model_path(weights: str) -> str:
    """
    Validate and resolve model path.
    
    Checks if the model file exists or if it's a valid ultralytics model name.
    
    Args:
        weights (str): Path to model weights or model name
        
    Returns:
        str: Validated model path or name
        
    Raises:
        ValueError: If weights parameter is invalid
    """
    if not weights or not isinstance(weights, str):
        raise ValueError("Model weights must be a non-empty string")
    
    # List of standard ultralytics model names that should be allowed for automatic download
    standard_models = [
        'yolov8n.pt', 'yolov8s.pt', 'yolov8m.pt', 'yolov8l.pt', 'yolov8x.pt',
        'yolov8n-cls.pt', 'yolov8s-cls.pt', 'yolov8m-cls.pt', 'yolov8l-cls.pt', 'yolov8x-cls.pt',
        'yolov8n-seg.pt', 'yolov8s-seg.pt', 'yolov8m-seg.pt', 'yolov8l-seg.pt', 'yolov8x-seg.pt',
        'yolov8n-pose.pt', 'yolov8s-pose.pt', 'yolov8m-pose.pt', 'yolov8l-pose.pt', 'yolov8x-pose.pt',
        'yolo11n.pt', 'yolo11s.pt', 'yolo11m.pt', 'yolo11l.pt', 'yolo11x.pt'
    ]
    
    # If it's a standard ultralytics model, allow it (will be downloaded automatically)
    if weights in standard_models:
        logger.info(f"Using standard Ultralytics model: {weights} (will download if needed)")
        return weights
    
    # Check if it's a local file path
    if weights.endswith('.pt') or weights.endswith('.onnx'):
        model_path = Path(weights)
        if not model_path.exists():
            # Try relative to current directory
            current_dir = Path.cwd()
            relative_path = current_dir / weights
            if relative_path.exists():
                return str(relative_path)
            
            # Try in models directory
            models_dir = current_dir / 'models'
            models_path = models_dir / weights
            if models_path.exists():
                return str(models_path)
                
            # For custom models, raise error if file doesn't exist
            raise FileNotFoundError(f"Model file not found: {weights}")
        return str(model_path)
    
    # Assume it's a ultralytics model name (allow it through)
    return weights


def load_model(weights: str, device: str = "auto") -> YOLO:
    """
    Load YOLOv8 model with specified weights and device.
    
    Initializes a YOLOv8 model using ultralytics library with automatic
    device detection if not specified. Supports both local model files
    and pre-trained ultralytics models.
    
    Args:
        weights (str): Path to model weights file or ultralytics model name
                      Examples: 'yolov8n.pt', 'yolov8s.pt', './models/custom.pt'
        device (str, optional): Device to run inference on. 
                               Options: 'auto', 'cpu', 'cuda', 'mps'
                               Defaults to 'auto' for automatic detection.
    
    Returns:
        YOLO: Loaded YOLOv8 model ready for inference
        
    Raises:
        FileNotFoundError: If specified model file doesn't exist
        RuntimeError: If model loading fails
        ValueError: If invalid parameters provided
        
    Example:
        >>> model = load_model('yolov8n.pt', 'auto')
        >>> model = load_model('./models/ppe_model.pt', 'cuda')
    """
    try:
        # Validate inputs
        validated_weights = validate_model_path(weights)
        
        # Determine device
        if device == "auto":
            device = detect_device()
        elif device not in ['cpu', 'cuda', 'mps']:
            logger.warning(f"Unknown device '{device}', falling back to auto-detection")
            device = detect_device()
        
        logger.info(f"Loading model: {validated_weights} on device: {device}")
        
        # Load the model
        start_time = time.time()
        model = YOLO(validated_weights)
        
        # Move model to specified device
        model.to(device)
        
        load_time = time.time() - start_time
        logger.info(f"Model loaded successfully in {load_time:.2f} seconds")
        
        # Log model information
        if hasattr(model, 'model') and hasattr(model.model, 'names'):
            class_names = model.model.names
            logger.info(f"Model classes: {list(class_names.values())}")
        
        return model
        
    except FileNotFoundError as e:
        logger.error(f"Model file not found: {e}")
        raise
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise RuntimeError(f"Model loading failed: {str(e)}") from e


def normalize_class_name(class_name: str) -> str:
    """
    Normalize detected class name to standard PPE categories.
    
    Maps various class name variations to standardized categories
    for consistent processing throughout the system.
    
    Args:
        class_name (str): Raw class name from model
        
    Returns:
        str: Normalized class name ('person', 'helmet', 'vest', or original)
    """
    if not isinstance(class_name, str):
        return str(class_name).lower()
        
    normalized = class_name.lower().strip().replace('_', ' ').replace('-', ' ')
    
    # Check against known PPE categories
    for category, variations in PPE_CLASS_NAMES.items():
        if normalized in variations or any(var in normalized for var in variations):
            return category
    
    # Return original if no match found
    return normalized


def postprocess_detections(
    results: Results,
    conf_threshold: float = DEFAULT_CONF_THRESHOLD,
    iou_threshold: float = DEFAULT_IOU_THRESHOLD
) -> List[Dict[str, Union[str, float, List[float]]]]:
    """
    Post-process YOLOv8 detection results.
    
    Extracts and formats detection results from YOLOv8 Results object,
    applying confidence filtering and returning standardized detection format.
    
    Args:
        results (Results): YOLOv8 detection results
        conf_threshold (float): Minimum confidence threshold for detections
        iou_threshold (float): IoU threshold (for reference, NMS applied in model)
        
    Returns:
        List[Dict]: List of detection dictionaries with keys:
            - 'category': Object class name (normalized)
            - 'confidence': Detection confidence score
            - 'bbox': Bounding box coordinates [x1, y1, x2, y2]
    """
    detections = []
    
    try:
        # Extract detection data
        if results.boxes is None or len(results.boxes) == 0:
            logger.debug("No detections found in results")
            return detections
        
        boxes = results.boxes.xyxy.cpu().numpy()  # Bounding boxes
        confidences = results.boxes.conf.cpu().numpy()  # Confidence scores
        class_ids = results.boxes.cls.cpu().numpy()  # Class IDs
        
        # Get class names
        class_names = results.names
        
        for i, (box, conf, cls_id) in enumerate(zip(boxes, confidences, class_ids)):
            # Apply confidence threshold
            if conf < conf_threshold:
                continue
            
            # Get class name and normalize it
            class_name = class_names.get(int(cls_id), f"class_{int(cls_id)}")
            normalized_category = normalize_class_name(class_name)
            
            # Format bounding box
            x1, y1, x2, y2 = box.astype(float)
            bbox = [x1, y1, x2, y2]
            
            # Create detection dictionary
            detection = {
                'category': normalized_category,
                'confidence': float(conf),
                'bbox': bbox
            }
            
            detections.append(detection)
        
        logger.debug(f"Processed {len(detections)} detections above confidence threshold {conf_threshold}")
        
    except Exception as e:
        logger.error(f"Error post-processing detections: {e}")
        raise RuntimeError(f"Detection post-processing failed: {str(e)}") from e
    
    return detections


def run_inference(
    model: YOLO,
    image: np.ndarray,
    conf_threshold: float = DEFAULT_CONF_THRESHOLD,
    iou_threshold: float = DEFAULT_IOU_THRESHOLD
) -> List[Dict[str, Union[str, float, List[float]]]]:
    """
    Run PPE detection inference on an image.
    
    Performs object detection using the loaded YOLOv8 model to identify
    persons, helmets, and safety vests in the input image.
    
    Args:
        model (YOLO): Loaded YOLOv8 model instance
        image (np.ndarray): Input image as numpy array (BGR format)
        conf_threshold (float, optional): Minimum confidence threshold for detections.
                                         Defaults to 0.35.
        iou_threshold (float, optional): IoU threshold for Non-Maximum Suppression.
                                        Defaults to 0.5.
    
    Returns:
        List[Dict]: List of detection dictionaries, each containing:
            - 'category' (str): Detected object category ('person', 'helmet', 'vest')
            - 'confidence' (float): Detection confidence score (0.0 to 1.0)
            - 'bbox' (List[float]): Bounding box coordinates [x1, y1, x2, y2]
    
    Raises:
        ValueError: If invalid inputs provided
        RuntimeError: If inference fails
        
    Example:
        >>> model = load_model('yolov8n.pt')
        >>> image = cv2.imread('construction_site.jpg')
        >>> detections = run_inference(model, image, conf_threshold=0.5)
        >>> for det in detections:
        ...     print(f"{det['category']}: {det['confidence']:.2f}")
    """
    try:
        # Input validation
        if model is None:
            raise ValueError("Model cannot be None")
        
        if image is None or not isinstance(image, np.ndarray):
            raise ValueError("Image must be a numpy array")
        
        if image.size == 0:
            raise ValueError("Image cannot be empty")
        
        if not (0.0 <= conf_threshold <= 1.0):
            raise ValueError("Confidence threshold must be between 0.0 and 1.0")
        
        if not (0.0 <= iou_threshold <= 1.0):
            raise ValueError("IoU threshold must be between 0.0 and 1.0")
        
        logger.debug(f"Running inference on image shape: {image.shape}")
        logger.debug(f"Confidence threshold: {conf_threshold}, IoU threshold: {iou_threshold}")
        
        # Run inference
        start_time = time.time()
        
        results = model(
            image,
            conf=conf_threshold,
            iou=iou_threshold,
            verbose=False  # Suppress ultralytics output
        )
        
        inference_time = time.time() - start_time
        logger.debug(f"Inference completed in {inference_time:.3f} seconds")
        
        # Process results
        if not results or len(results) == 0:
            logger.debug("No results returned from model")
            return []
        
        # Take first result (single image inference)
        result = results[0]
        
        # Post-process detections
        detections = postprocess_detections(result, conf_threshold, iou_threshold)
        
        logger.info(f"Detected {len(detections)} objects in {inference_time:.3f}s")
        
        # Log detection summary
        if detections:
            categories = {}
            for det in detections:
                cat = det['category']
                categories[cat] = categories.get(cat, 0) + 1
            
            summary = ", ".join([f"{count} {cat}" for cat, count in categories.items()])
            logger.info(f"Detection summary: {summary}")
        
        return detections
        
    except ValueError as e:
        logger.error(f"Invalid input parameters: {e}")
        raise
    except Exception as e:
        logger.error(f"Inference failed: {e}")
        raise RuntimeError(f"Inference execution failed: {str(e)}") from e


def filter_detections_by_category(
    detections: List[Dict[str, Union[str, float, List[float]]]],
    categories: List[str]
) -> List[Dict[str, Union[str, float, List[float]]]]:
    """
    Filter detections to include only specified categories.
    
    Args:
        detections (List[Dict]): List of detection dictionaries
        categories (List[str]): List of categories to keep
        
    Returns:
        List[Dict]: Filtered detections
    """
    if not categories:
        return detections
    
    normalized_categories = [cat.lower() for cat in categories]
    filtered = [
        det for det in detections 
        if det['category'].lower() in normalized_categories
    ]
    
    logger.debug(f"Filtered {len(detections)} detections to {len(filtered)} for categories: {categories}")
    return filtered


def get_detections_by_category(
    detections: List[Dict[str, Union[str, float, List[float]]]]
) -> Dict[str, List[List[float]]]:
    """
    Group detections by category and extract bounding boxes.
    
    Convenience function to separate detections into person, helmet, and vest
    categories for use with PPE matching algorithms.
    
    Args:
        detections (List[Dict]): List of detection dictionaries
        
    Returns:
        Dict[str, List[List[float]]]: Dictionary with categories as keys and
                                     lists of bounding boxes as values
    """
    categorized = {
        'person': [],
        'helmet': [],
        'vest': []
    }
    
    for detection in detections:
        category = detection['category'].lower()
        bbox = detection['bbox']
        
        if category in categorized:
            categorized[category].append(bbox)
    
    # Log summary
    summary = {cat: len(boxes) for cat, boxes in categorized.items()}
    logger.debug(f"Categorized detections: {summary}")
    
    return categorized


def run_inference_with_ppe_analysis(
    image: np.ndarray,
    draw: bool = True,
    conf_threshold: float = DEFAULT_CONF_THRESHOLD,
    iou_threshold: float = DEFAULT_IOU_THRESHOLD,
    ppe_overlap: float = 0.3
) -> Dict[str, Any]:
    """
    High-level inference function for real-time PPE detection.
    
    This function provides a simplified interface that handles model loading,
    inference, PPE matching, and optional drawing in a single call.
    
    Args:
        image (np.ndarray): Input image as numpy array (BGR format)
        draw (bool): Whether to draw annotations on the image
        conf_threshold (float): Confidence threshold for detections
        iou_threshold (float): IoU threshold for NMS
        ppe_overlap (float): Minimum overlap threshold for PPE matching
        
    Returns:
        Dict[str, Any]: Complete analysis results containing:
            - 'detections': List of raw detections
            - 'persons': List of PersonStatus objects
            - 'violations': List of violations found
            - 'annotated_frame': Annotated image (if draw=True)
            - 'violations_summary': Summary of violations
            - 'total_persons': Count of detected persons
            - 'total_violations': Count of violations
            
    Raises:
        RuntimeError: If model is not available or inference fails
        ValueError: If invalid parameters provided
    """
    try:
        # Get the global model (loaded by main.py)
        model = get_cached_model()
        if model is None:
            # Try to load default model if not cached
            try:
                model = load_model("yolov8n.pt")
            except Exception as e:
                raise RuntimeError(f"Model not available: {str(e)}")
        
        # Validate inputs
        if image is None or not isinstance(image, np.ndarray):
            raise ValueError("Image must be a numpy array")
        
        if image.size == 0:
            raise ValueError("Image cannot be empty")
        
        # Run core inference
        detections_raw = run_inference(model, image, conf_threshold, iou_threshold)
        
        # Separate detections by category for PPE matching
        person_boxes = []
        helmet_boxes = []
        vest_boxes = []
        
        for det in detections_raw:
            category = det['category'].lower()
            bbox = det['bbox']
            
            if category == 'person':
                person_boxes.append(bbox)
            elif category in ['helmet', 'hard_hat']:
                helmet_boxes.append(bbox)
            elif category in ['vest', 'safety_vest']:
                vest_boxes.append(bbox)
        
        # Convert to Detection objects for compatibility
        detections = [
            Detection(
                category=det['category'],
                confidence=det['confidence'],
                bbox=det['bbox']
            )
            for det in detections_raw
        ]
        
        # Get PPE matching thresholds
        thresholds = get_default_thresholds()
        thresholds['ppe_overlap_threshold'] = ppe_overlap
        
        # Perform PPE matching with separated boxes
        persons = match_ppe(person_boxes, helmet_boxes, vest_boxes, thresholds)
        
        # Calculate violations summary
        violations_summary = calculate_violations_summary(persons)
        
        # Prepare annotated frame if requested
        annotated_frame = None
        if draw:
            try:
                # Import here to avoid circular imports
                from .utils.draw import draw_detections
                annotated_frame = draw_detections(image, detections_raw, persons)
            except Exception as e:
                logger.warning(f"Failed to draw annotations: {e}")
                annotated_frame = image.copy()
        
        # Collect all violations from persons
        all_violations = []
        for person in persons:
            if hasattr(person, 'violations'):
                all_violations.extend(person.violations)
            elif isinstance(person, dict) and 'violations' in person:
                all_violations.extend(person['violations'])
        
        # Prepare complete result
        result = {
            'detections': detections_raw,
            'persons': persons,
            'violations': all_violations,
            'violations_summary': violations_summary,
            'total_persons': len([d for d in detections_raw if d.get('category') == 'person']),
            'total_violations': len(all_violations),
            'model_info': {
                'conf_threshold': conf_threshold,
                'iou_threshold': iou_threshold,
                'ppe_overlap': ppe_overlap
            }
        }
        
        # Add annotated frame if available
        if annotated_frame is not None:
            result['annotated_frame'] = annotated_frame
        
        logger.debug(f"PPE analysis complete: {result['total_persons']} persons, {result['total_violations']} violations")
        
        return result
        
    except Exception as e:
        logger.error(f"PPE analysis failed: {str(e)}")
        # Return minimal error result
        return {
            'detections': [],
            'persons': [],
            'violations': [],
            'violations_summary': {},
            'total_persons': 0,
            'total_violations': 0,
            'error': str(e),
            'annotated_frame': image.copy() if image is not None else None
        }


def validate_model_performance(model: YOLO, test_image_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Validate model performance with a test inference.
    
    Runs a quick test to ensure the model is working correctly.
    
    Args:
        model (YOLO): Loaded model to test
        test_image_path (Optional[str]): Path to test image, if None uses random image
        
    Returns:
        Dict[str, Any]: Performance metrics and status
    """
    try:
        # Create or load test image
        if test_image_path and os.path.exists(test_image_path):
            import cv2
            test_image = cv2.imread(test_image_path)
        else:
            # Create a dummy test image
            test_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        
        # Run test inference
        start_time = time.time()
        detections = run_inference(model, test_image, conf_threshold=0.1)
        inference_time = time.time() - start_time
        
        return {
            'status': 'success',
            'inference_time': inference_time,
            'detections_count': len(detections),
            'model_ready': True
        }
        
    except Exception as e:
        logger.error(f"Model validation failed: {e}")
        return {
            'status': 'failed',
            'error': str(e),
            'model_ready': False
        }


# Global model instance for caching
_cached_model: Optional[YOLO] = None
_model_path: Optional[str] = None
_model_device: Optional[str] = None


def get_cached_model(
    weights: str = "yolov8n.pt",
    device: str = "auto",
    force_reload: bool = False
) -> YOLO:
    """
    Get cached model instance or load new one if needed.
    
    Provides model caching to avoid reloading the same model multiple times.
    
    Args:
        weights (str): Model weights path or name
        device (str): Device to use
        force_reload (bool): Force reload even if cached
        
    Returns:
        YOLO: Cached or newly loaded model
    """
    global _cached_model, _model_path, _model_device
    
    # Check if we need to reload
    need_reload = (
        force_reload or
        _cached_model is None or
        _model_path != weights or
        _model_device != device
    )
    
    if need_reload:
        logger.info(f"Loading model: {weights} on {device}")
        _cached_model = load_model(weights, device)
        _model_path = weights
        _model_device = device
    else:
        logger.debug("Using cached model")
    
    return _cached_model


def clear_model_cache() -> None:
    """
    Clear the cached model to free memory.
    """
    global _cached_model, _model_path, _model_device
    
    if _cached_model is not None:
        logger.info("Clearing model cache")
        del _cached_model
        _cached_model = None
        _model_path = None
        _model_device = None
        
        # Force garbage collection
        import gc
        gc.collect()
        
        # Clear CUDA cache if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()