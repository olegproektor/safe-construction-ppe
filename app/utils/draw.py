"""
Drawing and Visualization Utilities for PPE Detection

This module provides functions for drawing bounding boxes, labels, and annotations
on images to visualize PPE detection results. It uses OpenCV for rendering and
follows a color-coding scheme for different compliance states.

Color Scheme:
- Green: Compliant person (has helmet + vest)
- Red: Non-compliant person (missing PPE)
- Light Blue: Helmet detection
- Orange: Vest detection

Author: PPE Detection System
Date: 2025-08-26
"""

import math
from typing import Dict, List, Optional, Tuple, Union, Any

import cv2
import numpy as np

# Import schemas for type hints
try:
    from ..schemas import Detection, PersonStatus
except ImportError:
    # Fallback for when running as standalone
    Detection = Dict[str, Any]
    PersonStatus = Dict[str, Any]


# Color definitions (BGR format for OpenCV)
COLORS = {
    'compliant_person': (0, 255, 0),      # Green for compliant person
    'violation_person': (0, 0, 255),      # Red for person with violations
    'helmet': (255, 255, 0),              # Light blue for helmet
    'vest': (0, 165, 255),                # Orange for vest
    'person_default': (0, 255, 255),      # Yellow for unprocessed person
    'text_bg': (0, 0, 0),                 # Black background for text
    'text_fg': (255, 255, 255),           # White text
    'border': (255, 255, 255)             # White border
}

# Drawing parameters
DRAWING_PARAMS = {
    'bbox_thickness': 2,
    'text_thickness': 1,
    'text_scale': 0.6,
    'text_padding': 5,
    'confidence_decimals': 2,
    'min_text_size': 12
}


def calculate_text_size(
    text: str, 
    font_scale: float = DRAWING_PARAMS['text_scale'],
    thickness: int = DRAWING_PARAMS['text_thickness']
) -> Tuple[Tuple[int, int], int]:
    """
    Calculate text size and baseline for OpenCV text rendering.
    
    Args:
        text (str): Text to measure
        font_scale (float): Font scale factor
        thickness (int): Text thickness
        
    Returns:
        Tuple containing:
        - (width, height): Text dimensions in pixels
        - baseline: Baseline offset
    """
    (text_width, text_height), baseline = cv2.getTextSize(
        text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
    )
    return (text_width, text_height), baseline


def draw_text_with_background(
    image: np.ndarray,
    text: str,
    position: Tuple[int, int],
    font_scale: float = DRAWING_PARAMS['text_scale'],
    thickness: int = DRAWING_PARAMS['text_thickness'],
    text_color: Tuple[int, int, int] = COLORS['text_fg'],
    bg_color: Tuple[int, int, int] = COLORS['text_bg'],
    padding: int = DRAWING_PARAMS['text_padding']
) -> np.ndarray:
    """
    Draw text with background rectangle for better visibility.
    
    Args:
        image (np.ndarray): Input image to draw on
        text (str): Text to draw
        position (Tuple[int, int]): Text position (x, y)
        font_scale (float): Font scale factor
        thickness (int): Text thickness
        text_color (Tuple[int, int, int]): Text color in BGR
        bg_color (Tuple[int, int, int]): Background color in BGR
        padding (int): Padding around text
        
    Returns:
        np.ndarray: Image with text drawn
    """
    # Calculate text dimensions
    (text_width, text_height), baseline = calculate_text_size(text, font_scale, thickness)
    
    x, y = position
    
    # Calculate background rectangle coordinates
    bg_x1 = x - padding
    bg_y1 = y - text_height - padding
    bg_x2 = x + text_width + padding
    bg_y2 = y + baseline + padding
    
    # Ensure coordinates are within image bounds
    height, width = image.shape[:2]
    bg_x1 = max(0, bg_x1)
    bg_y1 = max(0, bg_y1)
    bg_x2 = min(width, bg_x2)
    bg_y2 = min(height, bg_y2)
    
    # Draw background rectangle
    cv2.rectangle(image, (bg_x1, bg_y1), (bg_x2, bg_y2), bg_color, -1)
    
    # Draw text
    cv2.putText(
        image, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
        font_scale, text_color, thickness
    )
    
    return image


def draw_bounding_box(
    image: np.ndarray,
    bbox: List[float],
    color: Tuple[int, int, int],
    thickness: int = DRAWING_PARAMS['bbox_thickness'],
    label: Optional[str] = None,
    confidence: Optional[float] = None
) -> np.ndarray:
    """
    Draw a bounding box with optional label and confidence score.
    
    Args:
        image (np.ndarray): Input image to draw on
        bbox (List[float]): Bounding box coordinates [x1, y1, x2, y2]
        color (Tuple[int, int, int]): Box color in BGR format
        thickness (int): Box line thickness
        label (Optional[str]): Object label to display
        confidence (Optional[float]): Confidence score to display
        
    Returns:
        np.ndarray: Image with bounding box drawn
    """
    # Convert bbox coordinates to integers
    x1, y1, x2, y2 = [int(coord) for coord in bbox]
    
    # Ensure coordinates are valid
    height, width = image.shape[:2]
    x1 = max(0, min(x1, width - 1))
    y1 = max(0, min(y1, height - 1))
    x2 = max(x1 + 1, min(x2, width))
    y2 = max(y1 + 1, min(y2, height))
    
    # Draw bounding box
    cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
    
    # Draw label if provided
    if label is not None:
        # Format label text
        if confidence is not None:
            conf_str = f"{confidence:.{DRAWING_PARAMS['confidence_decimals']}f}"
            text = f"{label}: {conf_str}"
        else:
            text = label
        
        # Position label above the bounding box
        label_x = x1
        label_y = y1 - 10
        
        # If label would be outside image, position it inside the box
        if label_y < 20:
            label_y = y1 + 25
        
        # Draw label with background
        draw_text_with_background(
            image, text, (label_x, label_y),
            text_color=COLORS['text_fg'],
            bg_color=color
        )
    
    return image


def get_person_color(person: Union[PersonStatus, Dict[str, Any]]) -> Tuple[int, int, int]:
    """
    Determine the color for a person's bounding box based on PPE compliance.
    
    Args:
        person (Union[PersonStatus, Dict]): Person status information
        
    Returns:
        Tuple[int, int, int]: BGR color for the person's bounding box
    """
    # Handle both PersonStatus objects and dictionaries
    if hasattr(person, 'has_helmet'):
        has_helmet = person.has_helmet
        has_vest = person.has_vest
        violations = getattr(person, 'violations', [])
    else:
        has_helmet = person.get('has_helmet', False)
        has_vest = person.get('has_vest', False)
        violations = person.get('violations', [])
    
    # Green if compliant (has both helmet and vest, no violations)
    if has_helmet and has_vest and len(violations) == 0:
        return COLORS['compliant_person']
    else:
        # Red if non-compliant
        return COLORS['violation_person']


def format_person_label(person: Union[PersonStatus, Dict[str, Any]]) -> str:
    """
    Format label text for a person based on their PPE status.
    
    Args:
        person (Union[PersonStatus, Dict]): Person status information
        
    Returns:
        str: Formatted label text
    """
    # Handle both PersonStatus objects and dictionaries
    if hasattr(person, 'has_helmet'):
        has_helmet = person.has_helmet
        has_vest = person.has_vest
        person_id = getattr(person, 'person_id', 'Person')
        violations = getattr(person, 'violations', [])
    else:
        has_helmet = person.get('has_helmet', False)
        has_vest = person.get('has_vest', False)
        person_id = person.get('person_id', 'Person')
        violations = person.get('violations', [])
    
    # Create status indicators
    helmet_status = "✓" if has_helmet else "✗"
    vest_status = "✓" if has_vest else "✗"
    
    # Format label
    label = f"{person_id}\nH:{helmet_status} V:{vest_status}"
    
    # Add violation count if any
    if violations:
        label += f"\nViolations: {len(violations)}"
    
    return label


def draw_detection_bbox(
    image: np.ndarray,
    detection: Union[Detection, Dict[str, Any]]
) -> np.ndarray:
    """
    Draw bounding box for a single detection.
    
    Args:
        image (np.ndarray): Input image
        detection (Union[Detection, Dict]): Detection information
        
    Returns:
        np.ndarray: Image with detection drawn
    """
    # Handle both Detection objects and dictionaries
    if hasattr(detection, 'category'):
        category = detection.category
        confidence = detection.confidence
        bbox = detection.bbox
    else:
        category = detection.get('category', 'unknown')
        confidence = detection.get('confidence', 0.0)
        bbox = detection.get('bbox', [0, 0, 1, 1])
    
    # Determine color based on category
    if 'helmet' in category.lower():
        color = COLORS['helmet']
    elif 'vest' in category.lower():
        color = COLORS['vest']
    else:
        color = COLORS['person_default']
    
    # Draw bounding box with label
    draw_bounding_box(
        image, bbox, color,
        label=category.capitalize(),
        confidence=confidence
    )
    
    return image


def draw_person_bbox(
    image: np.ndarray,
    person: Union[PersonStatus, Dict[str, Any]]
) -> np.ndarray:
    """
    Draw bounding box for a person with PPE compliance status.
    
    Args:
        image (np.ndarray): Input image
        person (Union[PersonStatus, Dict]): Person status information
        
    Returns:
        np.ndarray: Image with person drawn
    """
    # Handle both PersonStatus objects and dictionaries
    if hasattr(person, 'bbox'):
        bbox = person.bbox
    else:
        bbox = person.get('bbox', [0, 0, 1, 1])
    
    # Get color based on compliance
    color = get_person_color(person)
    
    # Format label
    label = format_person_label(person)
    
    # Draw bounding box
    draw_bounding_box(image, bbox, color, label=label)
    
    return image


def draw_detections(
    image: np.ndarray,
    detections: List[Union[Detection, Dict[str, Any]]],
    persons: List[Union[PersonStatus, Dict[str, Any]]]
) -> np.ndarray:
    """
    Draw PPE detection results on an image with color-coded bounding boxes.
    
    This is the main function for visualizing PPE detection results. It draws:
    - Person bboxes: Green if compliant (helmet+vest), Red if violations
    - PPE bboxes: Light blue for helmets, Orange for vests
    - Labels with class names and confidence scores
    
    Args:
        image (np.ndarray): Input image as numpy array (BGR format)
        detections (List): List of Detection objects or dictionaries with:
            - category: Object class name
            - confidence: Detection confidence score
            - bbox: Bounding box coordinates [x1, y1, x2, y2]
        persons (List): List of PersonStatus objects or dictionaries with:
            - person_id: Unique person identifier
            - bbox: Person bounding box coordinates
            - has_helmet: Boolean helmet detection status
            - has_vest: Boolean vest detection status
            - violations: List of violation strings
    
    Returns:
        np.ndarray: Annotated image with all detections and person status drawn
        
    Example:
        >>> import cv2
        >>> image = cv2.imread('construction_site.jpg')
        >>> detections = [{'category': 'helmet', 'confidence': 0.85, 'bbox': [100, 50, 150, 100]}]
        >>> persons = [{'person_id': 'person_1', 'bbox': [80, 100, 200, 300], 
        ...             'has_helmet': True, 'has_vest': False, 'violations': ['no_vest']}]
        >>> annotated = draw_detections(image, detections, persons)
        >>> cv2.imshow('PPE Detection', annotated)
    """
    if image is None or image.size == 0:
        raise ValueError("Input image cannot be None or empty")
    
    # Create a copy of the image to avoid modifying the original
    annotated_image = image.copy()
    
    try:
        # Draw PPE detections first (helmets and vests)
        if detections:
            for detection in detections:
                # Only draw PPE items, not persons (persons handled separately)
                category = detection.get('category', '') if isinstance(detection, dict) else detection.category
                
                if category.lower() in ['helmet', 'vest', 'hard_hat', 'safety_vest']:
                    draw_detection_bbox(annotated_image, detection)
        
        # Draw person bounding boxes with compliance status
        if persons:
            for person in persons:
                draw_person_bbox(annotated_image, person)
        
        # Add summary information
        if persons:
            total_persons = len(persons)
            compliant_count = 0
            
            for person in persons:
                # Handle both object and dictionary formats
                if hasattr(person, 'has_helmet'):
                    has_helmet = person.has_helmet
                    has_vest = person.has_vest
                    violations = getattr(person, 'violations', [])
                else:
                    has_helmet = person.get('has_helmet', False)
                    has_vest = person.get('has_vest', False)
                    violations = person.get('violations', [])
                
                if has_helmet and has_vest and len(violations) == 0:
                    compliant_count += 1
            
            # Draw summary text
            summary_text = f"Persons: {total_persons} | Compliant: {compliant_count} | Violations: {total_persons - compliant_count}"
            
            # Position summary at top of image
            height, width = annotated_image.shape[:2]
            summary_position = (10, 30)
            
            draw_text_with_background(
                annotated_image, summary_text, summary_position,
                font_scale=0.7, thickness=2,
                text_color=COLORS['text_fg'],
                bg_color=(0, 0, 0)  # Black background
            )
        
        return annotated_image
        
    except Exception as e:
        # Log error and return original image if drawing fails
        print(f"Error drawing detections: {e}")
        return image


def draw_detection_summary(
    image: np.ndarray,
    detection_counts: Dict[str, int],
    compliance_rate: float,
    position: Tuple[int, int] = (10, 10)
) -> np.ndarray:
    """
    Draw detection summary statistics on the image.
    
    Args:
        image (np.ndarray): Input image
        detection_counts (Dict[str, int]): Count of each detection type
        compliance_rate (float): Overall compliance rate (0.0 to 1.0)
        position (Tuple[int, int]): Position to draw summary
        
    Returns:
        np.ndarray: Image with summary drawn
    """
    lines = []
    
    # Add detection counts
    for category, count in detection_counts.items():
        lines.append(f"{category.capitalize()}: {count}")
    
    # Add compliance rate
    compliance_percent = compliance_rate * 100
    lines.append(f"Compliance: {compliance_percent:.1f}%")
    
    # Draw each line
    x, y = position
    line_height = 25
    
    for i, line in enumerate(lines):
        line_y = y + (i * line_height)
        draw_text_with_background(
            image, line, (x, line_y),
            font_scale=0.6, thickness=1,
            text_color=COLORS['text_fg'],
            bg_color=COLORS['text_bg']
        )
    
    return image


def create_legend(
    width: int = 300,
    height: int = 200
) -> np.ndarray:
    """
    Create a legend image explaining the color coding.
    
    Args:
        width (int): Legend width in pixels
        height (int): Legend height in pixels
        
    Returns:
        np.ndarray: Legend image
    """
    # Create blank image
    legend = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Legend items
    items = [
        ("Compliant Person", COLORS['compliant_person']),
        ("Non-compliant Person", COLORS['violation_person']),
        ("Helmet", COLORS['helmet']),
        ("Safety Vest", COLORS['vest'])
    ]
    
    # Draw legend items
    y_start = 30
    item_height = 35
    box_size = 20
    
    for i, (label, color) in enumerate(items):
        y = y_start + (i * item_height)
        
        # Draw color box
        cv2.rectangle(legend, (10, y - box_size//2), (10 + box_size, y + box_size//2), color, -1)
        cv2.rectangle(legend, (10, y - box_size//2), (10 + box_size, y + box_size//2), COLORS['border'], 1)
        
        # Draw label
        cv2.putText(legend, label, (40, y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS['text_fg'], 1)
    
    # Add title
    cv2.putText(legend, "PPE Detection Legend", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLORS['text_fg'], 2)
    
    return legend


def combine_image_with_legend(
    image: np.ndarray,
    show_legend: bool = True
) -> np.ndarray:
    """
    Combine the main image with a legend.
    
    Args:
        image (np.ndarray): Main annotated image
        show_legend (bool): Whether to include legend
        
    Returns:
        np.ndarray: Combined image with legend
    """
    if not show_legend:
        return image
    
    # Create legend
    legend = create_legend()
    
    # Resize legend to match image height if needed
    img_height = image.shape[0]
    legend_height = legend.shape[0]
    
    if legend_height != img_height:
        legend_width = int(legend.shape[1] * img_height / legend_height)
        legend = cv2.resize(legend, (legend_width, img_height))
    
    # Combine horizontally
    combined = np.hstack([image, legend])
    
    return combined


# Utility functions for debugging and analysis

def save_annotated_image(
    image: np.ndarray,
    output_path: str,
    quality: int = 95
) -> bool:
    """
    Save annotated image to file.
    
    Args:
        image (np.ndarray): Annotated image
        output_path (str): Output file path
        quality (int): JPEG quality (if saving as JPEG)
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        if output_path.lower().endswith('.jpg') or output_path.lower().endswith('.jpeg'):
            cv2.imwrite(output_path, image, [cv2.IMWRITE_JPEG_QUALITY, quality])
        else:
            cv2.imwrite(output_path, image)
        return True
    except Exception as e:
        print(f"Error saving image: {e}")
        return False


def get_visualization_config() -> Dict[str, Any]:
    """
    Get current visualization configuration.
    
    Returns:
        Dict[str, Any]: Configuration parameters
    """
    return {
        'colors': COLORS.copy(),
        'drawing_params': DRAWING_PARAMS.copy()
    }


def update_visualization_config(
    colors: Optional[Dict[str, Tuple[int, int, int]]] = None,
    drawing_params: Optional[Dict[str, Union[int, float]]] = None
) -> None:
    """
    Update visualization configuration.
    
    Args:
        colors (Optional[Dict]): Color overrides
        drawing_params (Optional[Dict]): Drawing parameter overrides
    """
    global COLORS, DRAWING_PARAMS
    
    if colors:
        COLORS.update(colors)
    
    if drawing_params:
        DRAWING_PARAMS.update(drawing_params)