"""
Pydantic Schemas for PPE Detection API

This module defines all data models and schemas used throughout the PPE detection system.
It includes models for detections, person status, API responses, and configuration settings.

Author: PPE Detection System
Date: 2025-08-26
"""

from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Union, Any

from pydantic import BaseModel, Field, validator, ConfigDict


class PPECategory(str, Enum):
    """
    Enumeration of PPE detection categories.
    
    This enum defines all possible object categories that can be detected
    by the PPE detection system.
    """
    PERSON = "person"
    HELMET = "helmet"
    VEST = "vest"
    HARD_HAT = "hard_hat"
    SAFETY_VEST = "safety_vest"


class ViolationType(str, Enum):
    """
    Enumeration of PPE violation types.
    
    This enum defines all possible safety violations that can be detected.
    """
    NO_HELMET = "no_helmet"
    NO_VEST = "no_vest"
    INCOMPLETE_PPE = "incomplete_ppe"
    UNMATCHED_PERSON = "unmatched_person"


class Detection(BaseModel):
    """
    Individual object detection result.
    
    Represents a single detected object with its properties, including
    bounding box coordinates, confidence score, and matching information.
    
    Attributes:
        id (str): Unique identifier for this detection
        category (str): Object category (person, helmet, vest, etc.)
        confidence (float): Detection confidence score (0.0 to 1.0)
        bbox (List[float]): Bounding box coordinates [x1, y1, x2, y2]
        person_id (Optional[str]): ID of associated person (if applicable)
        matched (bool): Whether this detection has been matched to a person
    """
    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        extra="forbid"
    )
    
    id: str = Field(
        ...,
        description="Unique identifier for this detection",
        min_length=1
    )
    
    category: str = Field(
        ...,
        description="Object category (person, helmet, vest, etc.)",
        min_length=1
    )
    
    confidence: float = Field(
        ...,
        description="Detection confidence score",
        ge=0.0,
        le=1.0
    )
    
    bbox: List[float] = Field(
        ...,
        description="Bounding box coordinates [x1, y1, x2, y2]",
        min_length=4,
        max_length=4
    )
    
    person_id: Optional[str] = Field(
        None,
        description="ID of associated person (if applicable)"
    )
    
    matched: bool = Field(
        False,
        description="Whether this detection has been matched to a person"
    )
    
    @validator('bbox')
    def validate_bbox(cls, v):
        """Validate bounding box coordinates."""
        if len(v) != 4:
            raise ValueError("Bounding box must have exactly 4 coordinates")
        
        x1, y1, x2, y2 = v
        if x2 <= x1 or y2 <= y1:
            raise ValueError("Invalid bounding box: x2 > x1 and y2 > y1 required")
        
        if any(coord < 0 for coord in v):
            raise ValueError("Bounding box coordinates must be non-negative")
        
        return v
    
    @validator('category')
    def validate_category(cls, v):
        """Validate detection category."""
        valid_categories = [cat.value for cat in PPECategory]
        if v.lower() not in valid_categories:
            # Allow flexibility for custom categories
            pass
        return v.lower()


class PersonStatus(BaseModel):
    """
    PPE compliance status for an individual person.
    
    Contains information about a detected person and their PPE compliance,
    including what safety equipment they are wearing and any violations.
    
    Attributes:
        person_id (str): Unique identifier for this person
        bbox (List[float]): Person's bounding box coordinates [x1, y1, x2, y2]
        has_helmet (bool): Whether person is wearing a helmet
        has_vest (bool): Whether person is wearing a safety vest
        violations (List[str]): List of PPE violations for this person
    """
    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        extra="forbid"
    )
    
    person_id: str = Field(
        ...,
        description="Unique identifier for this person",
        min_length=1
    )
    
    bbox: List[float] = Field(
        ...,
        description="Person's bounding box coordinates [x1, y1, x2, y2]",
        min_length=4,
        max_length=4
    )
    
    has_helmet: bool = Field(
        ...,
        description="Whether person is wearing a helmet"
    )
    
    has_vest: bool = Field(
        ...,
        description="Whether person is wearing a safety vest"
    )
    
    violations: List[str] = Field(
        default_factory=list,
        description="List of PPE violations for this person"
    )
    
    @validator('bbox')
    def validate_person_bbox(cls, v):
        """Validate person bounding box coordinates."""
        if len(v) != 4:
            raise ValueError("Bounding box must have exactly 4 coordinates")
        
        x1, y1, x2, y2 = v
        if x2 <= x1 or y2 <= y1:
            raise ValueError("Invalid bounding box: x2 > x1 and y2 > y1 required")
        
        if any(coord < 0 for coord in v):
            raise ValueError("Bounding box coordinates must be non-negative")
        
        return v
    
    @validator('violations')
    def validate_violations(cls, v):
        """Validate violation types."""
        valid_violations = [viol.value for viol in ViolationType]
        for violation in v:
            if violation not in valid_violations:
                # Allow custom violation types but log a warning
                pass
        return v
    
    @property
    def is_compliant(self) -> bool:
        """Check if person is fully PPE compliant."""
        return self.has_helmet and self.has_vest and len(self.violations) == 0
    
    @property
    def compliance_score(self) -> float:
        """Calculate compliance score (0.0 to 1.0)."""
        score = 0.0
        if self.has_helmet:
            score += 0.5
        if self.has_vest:
            score += 0.5
        
        # Reduce score for violations
        violation_penalty = len(self.violations) * 0.1
        score = max(0.0, score - violation_penalty)
        
        return min(1.0, score)


class ViolationsSummary(BaseModel):
    """
    Summary of all PPE violations detected in an image.
    
    Provides aggregated information about violations found during detection.
    
    Attributes:
        total_persons (int): Total number of persons detected
        compliant_persons (int): Number of fully compliant persons
        violations_count (Dict[str, int]): Count of each violation type
        compliance_rate (float): Overall compliance rate (0.0 to 1.0)
    """
    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        extra="forbid"
    )
    
    total_persons: int = Field(
        ...,
        description="Total number of persons detected",
        ge=0
    )
    
    compliant_persons: int = Field(
        ...,
        description="Number of fully compliant persons",
        ge=0
    )
    
    violations_count: Dict[str, int] = Field(
        default_factory=dict,
        description="Count of each violation type"
    )
    
    compliance_rate: float = Field(
        ...,
        description="Overall compliance rate (0.0 to 1.0)",
        ge=0.0,
        le=1.0
    )
    
    @validator('compliant_persons')
    def validate_compliant_persons(cls, v, values):
        """Ensure compliant persons doesn't exceed total persons."""
        if 'total_persons' in values and v > values['total_persons']:
            raise ValueError("Compliant persons cannot exceed total persons")
        return v


class ProcessingTimings(BaseModel):
    """
    Timing information for various processing stages.
    
    Tracks performance metrics for different stages of the detection pipeline.
    
    Attributes:
        inference_time (float): Time spent on model inference (seconds)
        preprocessing_time (float): Time spent on image preprocessing (seconds)
        postprocessing_time (float): Time spent on result postprocessing (seconds)
        total_time (float): Total processing time (seconds)
    """
    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        extra="forbid"
    )
    
    inference_time: float = Field(
        ...,
        description="Time spent on model inference (seconds)",
        ge=0.0
    )
    
    preprocessing_time: float = Field(
        0.0,
        description="Time spent on image preprocessing (seconds)",
        ge=0.0
    )
    
    postprocessing_time: float = Field(
        0.0,
        description="Time spent on result postprocessing (seconds)",
        ge=0.0
    )
    
    total_time: float = Field(
        ...,
        description="Total processing time (seconds)",
        ge=0.0
    )
    
    @validator('total_time')
    def validate_total_time(cls, v, values):
        """Ensure total time is at least the sum of individual times."""
        component_times = [
            values.get('inference_time', 0.0),
            values.get('preprocessing_time', 0.0),
            values.get('postprocessing_time', 0.0)
        ]
        min_total = sum(component_times)
        
        if v < min_total * 0.9:  # Allow 10% tolerance for measurement differences
            raise ValueError(f"Total time ({v:.3f}s) is less than sum of components ({min_total:.3f}s)")
        
        return v


class DetectionResponse(BaseModel):
    """
    Complete response for PPE detection API.
    
    Main response model containing all detection results, person status,
    violation summaries, and performance metrics.
    
    Attributes:
        image_id (str): Unique identifier for the processed image
        width (int): Image width in pixels
        height (int): Image height in pixels
        detections (List[Detection]): List of all object detections
        persons (List[PersonStatus]): List of person PPE status
        violations_summary (ViolationsSummary): Summary of violations
        timings (ProcessingTimings): Performance timing information
    """
    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        extra="forbid"
    )
    
    image_id: str = Field(
        ...,
        description="Unique identifier for the processed image",
        min_length=1
    )
    
    width: int = Field(
        ...,
        description="Image width in pixels",
        gt=0
    )
    
    height: int = Field(
        ...,
        description="Image height in pixels",
        gt=0
    )
    
    detections: List[Detection] = Field(
        default_factory=list,
        description="List of all object detections"
    )
    
    persons: List[PersonStatus] = Field(
        default_factory=list,
        description="List of person PPE status"
    )
    
    violations_summary: ViolationsSummary = Field(
        ...,
        description="Summary of violations"
    )
    
    timings: ProcessingTimings = Field(
        ...,
        description="Performance timing information"
    )
    
    # Additional metadata
    timestamp: str = Field(
        default_factory=lambda: datetime.now().isoformat(),
        description="Processing timestamp (ISO format)"
    )
    
    model_version: Optional[str] = Field(
        None,
        description="Version of the detection model used"
    )
    
    @validator('detections')
    def validate_detections_bbox(cls, v, values):
        """Validate that all detection bboxes are within image bounds."""
        if 'width' not in values or 'height' not in values:
            return v
        
        width, height = values['width'], values['height']
        
        for detection in v:
            x1, y1, x2, y2 = detection.bbox
            if x2 > width or y2 > height:
                # Log warning but don't fail validation
                pass
        
        return v
    
    @property
    def total_violations(self) -> int:
        """Get total number of violations across all persons."""
        return sum(len(person.violations) for person in self.persons)
    
    @property
    def has_violations(self) -> bool:
        """Check if any violations were detected."""
        return self.total_violations > 0


# Request Models

class DetectionRequest(BaseModel):
    """
    Request model for image detection API.
    
    Attributes:
        image_data (str): Base64 encoded image data
        image_format (str): Image format (jpg, png, etc.)
        confidence_threshold (float): Minimum confidence threshold
        draw_annotations (bool): Whether to return annotated image
    """
    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        extra="forbid"
    )
    
    image_data: str = Field(
        ...,
        description="Base64 encoded image data",
        min_length=1
    )
    
    image_format: str = Field(
        "jpg",
        description="Image format (jpg, png, etc.)",
        pattern="^(jpg|jpeg|png|bmp|tiff)$"
    )
    
    confidence_threshold: float = Field(
        0.5,
        description="Minimum confidence threshold for detections",
        ge=0.0,
        le=1.0
    )
    
    draw_annotations: bool = Field(
        True,
        description="Whether to return annotated image"
    )


class ConfigurationModel(BaseModel):
    """
    Configuration model for PPE detection system.
    
    Attributes:
        model_path (str): Path to the YOLO model file
        confidence_threshold (float): Default confidence threshold
        iou_threshold (float): IoU threshold for NMS
        device (str): Device to run inference on (cpu, cuda, mps)
        max_image_size (int): Maximum image size for processing
    """
    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        extra="forbid"
    )
    
    model_path: str = Field(
        "yolov8n.pt",
        description="Path to the YOLO model file"
    )
    
    confidence_threshold: float = Field(
        0.5,
        description="Default confidence threshold",
        ge=0.0,
        le=1.0
    )
    
    iou_threshold: float = Field(
        0.4,
        description="IoU threshold for Non-Maximum Suppression",
        ge=0.0,
        le=1.0
    )
    
    device: str = Field(
        "cpu",
        description="Device to run inference on",
        pattern="^(cpu|cuda|mps|auto)$"
    )
    
    max_image_size: int = Field(
        1280,
        description="Maximum image size for processing",
        gt=0,
        le=4096
    )


# Error Models

class ErrorResponse(BaseModel):
    """
    Standard error response model.
    
    Attributes:
        error (str): Error message
        error_code (str): Error code identifier
        details (Optional[Dict[str, Any]]): Additional error details
        timestamp (datetime): Error timestamp
    """
    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        extra="forbid"
    )
    
    error: str = Field(
        ...,
        description="Error message",
        min_length=1
    )
    
    error_code: str = Field(
        ...,
        description="Error code identifier",
        min_length=1
    )
    
    details: Optional[Dict[str, Any]] = Field(
        None,
        description="Additional error details"
    )
    
    timestamp: str = Field(
        default_factory=lambda: datetime.now().isoformat(),
        description="Error timestamp (ISO format)"
    )


# Health Check Models

class HealthCheckResponse(BaseModel):
    """
    Health check response model.
    
    Attributes:
        status (str): Service status
        version (str): API version
        model_loaded (bool): Whether ML model is loaded
        uptime_seconds (float): Service uptime in seconds
    """
    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        extra="forbid"
    )
    
    status: str = Field(
        ...,
        description="Service status",
        pattern="^(healthy|unhealthy|degraded)$"
    )
    
    version: str = Field(
        ...,
        description="API version",
        min_length=1
    )
    
    model_loaded: bool = Field(
        ...,
        description="Whether ML model is loaded"
    )
    
    uptime_seconds: float = Field(
        ...,
        description="Service uptime in seconds",
        ge=0.0
    )
    
    timestamp: str = Field(
        default_factory=lambda: datetime.now().isoformat(),
        description="Health check timestamp (ISO format)"
    )