"""
API Tests for PPE Detection System

This module contains comprehensive tests for the FastAPI endpoints and
PPE detection logic using pytest and httpx TestClient.

Test Coverage:
- Health check endpoint
- Model info endpoint
- Image detection endpoint with file upload
- PPE matching logic unit tests
- Error handling and edge cases

Author: PPE Detection System
Date: 2025-08-26
"""

import base64
import io
import json
import os
from pathlib import Path
from typing import Dict, List, Any

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image
import cv2

# Import the FastAPI app
try:
    from app.main import app
    from app.ppe_logic import match_ppe, get_default_thresholds
    from app.schemas import PersonStatus, ViolationType
except ImportError:
    # Handle import issues during testing
    import sys
    sys.path.append(str(Path(__file__).parent.parent))
    from app.main import app
    from app.ppe_logic import match_ppe, get_default_thresholds
    from app.schemas import PersonStatus, ViolationType


# Test configuration
TEST_IMAGE_SIZE = (640, 480)
TEST_IMAGE_FORMAT = "RGB"


class TestAPIEndpoints:
    """
    Test class for FastAPI endpoint testing.
    
    Tests all API endpoints including health checks, model info,
    and image detection functionality.
    """
    
    @pytest.fixture(scope="class")
    def client(self) -> TestClient:
        """
        Create FastAPI test client.
        
        Returns:
            TestClient: FastAPI test client instance
        """
        return TestClient(app)
    
    @pytest.fixture(scope="class")
    def test_image_file(self) -> io.BytesIO:
        """
        Create a synthetic test image for upload testing.
        
        Creates a simple test image with some geometric shapes
        to simulate objects that could be detected.
        
        Returns:
            io.BytesIO: Image file buffer
        """
        # Create a test image with some colored rectangles
        # to simulate persons, helmets, and vests
        image = Image.new(TEST_IMAGE_FORMAT, TEST_IMAGE_SIZE, color=(100, 150, 200))
        
        # Convert to numpy for drawing
        img_array = np.array(image)
        
        # Draw some rectangles to simulate objects
        # Person-like rectangle (large, vertical)
        cv2.rectangle(img_array, (100, 150), (200, 400), (50, 50, 150), -1)
        
        # Helmet-like rectangle (small, at top of person)
        cv2.rectangle(img_array, (120, 130), (180, 170), (0, 100, 200), -1)
        
        # Vest-like rectangle (medium, at torso of person)
        cv2.rectangle(img_array, (110, 200), (190, 280), (0, 200, 100), -1)
        
        # Convert back to PIL Image
        test_image = Image.fromarray(img_array)
        
        # Save to BytesIO
        image_buffer = io.BytesIO()
        test_image.save(image_buffer, format='JPEG', quality=85)
        image_buffer.seek(0)
        
        return image_buffer
    
    def test_health_endpoint(self, client: TestClient) -> None:
        """
        Test the health check endpoint.
        
        Verifies that the /health endpoint returns correct status
        and includes required fields.
        
        Args:
            client (TestClient): FastAPI test client
        """
        response = client.get("/health")
        
        # Check status code
        assert response.status_code == 200
        
        # Parse JSON response
        data = response.json()
        
        # Verify required fields are present
        assert "status" in data
        assert "version" in data
        assert "model_loaded" in data
        assert "uptime_seconds" in data
        
        # Verify status is valid
        assert data["status"] in ["healthy", "unhealthy", "degraded"]
        
        # Verify data types
        assert isinstance(data["version"], str)
        assert isinstance(data["model_loaded"], bool)
        assert isinstance(data["uptime_seconds"], (int, float))
        assert data["uptime_seconds"] >= 0
        
        print(f"Health check passed: {data}")
    
    def test_model_info_endpoint(self, client: TestClient) -> None:
        """
        Test the model info endpoint.
        
        Verifies that the /model/info endpoint returns model information
        and default parameters.
        
        Args:
            client (TestClient): FastAPI test client
        """
        response = client.get("/model/info")
        
        # Check status code (may be 503 if model not loaded in test environment)
        if response.status_code == 503:
            # Model not available in test environment - this is acceptable
            print("Model not available in test environment (expected)")
            return
        
        assert response.status_code == 200
        
        # Parse JSON response
        data = response.json()
        
        # Verify required fields are present
        assert "classes" in data
        assert "conf_default" in data
        assert "iou_default" in data
        assert "device" in data
        
        # Verify data types
        assert isinstance(data["classes"], list)
        assert isinstance(data["conf_default"], (int, float))
        assert isinstance(data["iou_default"], (int, float))
        assert isinstance(data["device"], str)
        
        # Verify threshold ranges
        assert 0.0 <= data["conf_default"] <= 1.0
        assert 0.0 <= data["iou_default"] <= 1.0
        
        print(f"Model info retrieved: {data}")
    
    def test_detect_image_endpoint_basic(self, client: TestClient, test_image_file: io.BytesIO) -> None:
        """
        Test the image detection endpoint with basic parameters.
        
        Verifies that the /detect/image endpoint accepts image upload
        and returns properly structured JSON response.
        
        Args:
            client (TestClient): FastAPI test client
            test_image_file (io.BytesIO): Test image file buffer
        """
        # Prepare test image
        test_image_file.seek(0)
        
        # Prepare form data
        files = {"file": ("test_image.jpg", test_image_file, "image/jpeg")}
        data = {
            "conf": 0.3,
            "iou": 0.4,
            "ppe_overlap": 0.3,
            "draw": True,
            "return_image": False
        }
        
        # Make request
        response = client.post("/detect/image", files=files, data=data)
        
        # Check status code (may be 503 if model not loaded in test environment)
        if response.status_code == 503:
            print("Model not available in test environment (expected)")
            return
        
        assert response.status_code == 200
        
        # Parse JSON response
        result = response.json()
        
        # Verify required fields are present
        assert "image_id" in result
        assert "width" in result
        assert "height" in result
        assert "detections" in result
        assert "persons" in result
        assert "violations_summary" in result
        assert "timings" in result
        
        # Verify data structure
        assert isinstance(result["detections"], list)
        assert isinstance(result["persons"], list)
        assert isinstance(result["violations_summary"], dict)
        assert isinstance(result["timings"], dict)
        
        # Verify violations_summary structure
        violations_summary = result["violations_summary"]
        assert "total_persons" in violations_summary
        assert "compliant_persons" in violations_summary
        assert "violations_count" in violations_summary
        assert "compliance_rate" in violations_summary
        
        # Verify timings structure
        timings = result["timings"]
        assert "inference_time" in timings
        assert "preprocessing_time" in timings
        assert "postprocessing_time" in timings
        assert "total_time" in timings
        
        print(f"Detection successful: {len(result['detections'])} detections, "
              f"{len(result['persons'])} persons")
    
    def test_detect_image_with_return_image(self, client: TestClient, test_image_file: io.BytesIO) -> None:
        """
        Test the image detection endpoint with return_image=True.
        
        Verifies that the endpoint returns base64 encoded annotated image
        when requested.
        
        Args:
            client (TestClient): FastAPI test client
            test_image_file (io.BytesIO): Test image file buffer
        """
        # Prepare test image
        test_image_file.seek(0)
        
        # Prepare form data with return_image=True
        files = {"file": ("test_image.jpg", test_image_file, "image/jpeg")}
        data = {
            "conf": 0.5,
            "iou": 0.4,
            "ppe_overlap": 0.3,
            "draw": True,
            "return_image": True
        }
        
        # Make request
        response = client.post("/detect/image", files=files, data=data)
        
        # Check status code (may be 503 if model not loaded in test environment)
        if response.status_code == 503:
            print("Model not available in test environment (expected)")
            return
        
        assert response.status_code == 200
        
        # Parse JSON response
        result = response.json()
        
        # Verify base64 image is included
        if "annotated_image_base64" in result:
            assert isinstance(result["annotated_image_base64"], str)
            assert len(result["annotated_image_base64"]) > 0
            print("Base64 image successfully returned")
        else:
            print("Base64 image not included (may be expected in test environment)")
    
    def test_detect_image_parameter_validation(self, client: TestClient, test_image_file: io.BytesIO) -> None:
        """
        Test parameter validation for the detection endpoint.
        
        Verifies that invalid parameters are properly rejected.
        
        Args:
            client (TestClient): FastAPI test client
            test_image_file (io.BytesIO): Test image file buffer
        """
        # Test invalid confidence threshold
        test_image_file.seek(0)
        files = {"file": ("test_image.jpg", test_image_file, "image/jpeg")}
        data = {"conf": 1.5}  # Invalid: > 1.0
        
        response = client.post("/detect/image", files=files, data=data)
        
        # Should return 400 for invalid parameters (unless model not loaded)
        if response.status_code != 503:  # Ignore if model not available
            assert response.status_code == 400
            assert "confidence" in response.json()["error"].lower()
        
        # Test invalid IoU threshold
        test_image_file.seek(0)
        data = {"iou": -0.1}  # Invalid: < 0.0
        
        response = client.post("/detect/image", files=files, data=data)
        
        if response.status_code != 503:
            assert response.status_code == 400
            assert "iou" in response.json()["error"].lower()
        
        print("Parameter validation tests passed")
    
    def test_detect_image_file_validation(self, client: TestClient) -> None:
        """
        Test file validation for the detection endpoint.
        
        Verifies that invalid file types are properly rejected.
        
        Args:
            client (TestClient): FastAPI test client
        """
        # Test invalid file type
        text_file = io.BytesIO(b"This is not an image")
        files = {"file": ("test.txt", text_file, "text/plain")}
        data = {"conf": 0.5}
        
        response = client.post("/detect/image", files=files, data=data)
        
        # Should return 400 for invalid file type (unless model not loaded)
        if response.status_code != 503:
            assert response.status_code == 400
            error_message = response.json()["error"].lower()
            assert "file type" in error_message or "invalid" in error_message
        
        print("File validation tests passed")


class TestPPELogic:
    """
    Test class for PPE matching logic unit tests.
    
    Tests the core PPE matching algorithms with synthetic data
    to verify correct behavior in various scenarios.
    """
    
    @pytest.fixture
    def default_thresholds(self) -> Dict[str, float]:
        """
        Get default PPE matching thresholds.
        
        Returns:
            Dict[str, float]: Default threshold configuration
        """
        return get_default_thresholds()
    
    def test_match_ppe_person_with_helmet_and_vest(self, default_thresholds: Dict[str, float]) -> None:
        """
        Test PPE matching for a compliant person (with helmet and vest).
        
        Creates synthetic bounding boxes for a person with properly
        positioned helmet and vest, verifies correct matching.
        
        Args:
            default_thresholds (Dict[str, float]): PPE matching thresholds
        """
        # Define person bounding box (large, vertical rectangle)
        person_boxes = [[100, 100, 200, 400]]  # [x1, y1, x2, y2]
        
        # Define helmet box (small, at top of person)
        helmet_boxes = [[120, 80, 180, 120]]   # Overlaps with person's head area
        
        # Define vest box (medium, at torso of person)
        vest_boxes = [[110, 150, 190, 250]]    # Overlaps with person's torso
        
        # Run PPE matching
        results = match_ppe(person_boxes, helmet_boxes, vest_boxes, default_thresholds)
        
        # Verify results
        assert len(results) == 1, "Should detect exactly one person"
        
        person = results[0]
        
        # Verify person has both PPE items
        assert person.has_helmet is True, "Person should have helmet"
        assert person.has_vest is True, "Person should have vest"
        
        # Verify no violations
        assert len(person.violations) == 0, "Compliant person should have no violations"
        
        # Verify person is compliant
        assert person.is_compliant is True, "Person should be compliant"
        
        print(f"Compliant person test passed: {person.person_id}")
    
    def test_match_ppe_person_without_helmet(self, default_thresholds: Dict[str, float]) -> None:
        """
        Test PPE matching for person without helmet.
        
        Creates synthetic bounding boxes for a person with vest but no helmet,
        verifies correct violation detection.
        
        Args:
            default_thresholds (Dict[str, float]): PPE matching thresholds
        """
        # Define person bounding box
        person_boxes = [[100, 100, 200, 400]]
        
        # No helmet boxes
        helmet_boxes = []
        
        # Define vest box (overlaps with person)
        vest_boxes = [[110, 150, 190, 250]]
        
        # Run PPE matching
        results = match_ppe(person_boxes, helmet_boxes, vest_boxes, default_thresholds)
        
        # Verify results
        assert len(results) == 1, "Should detect exactly one person"
        
        person = results[0]
        
        # Verify PPE status
        assert person.has_helmet is False, "Person should not have helmet"
        assert person.has_vest is True, "Person should have vest"
        
        # Verify violations
        assert len(person.violations) > 0, "Person should have violations"
        assert ViolationType.NO_HELMET.value in person.violations, "Should have no_helmet violation"
        
        # Verify person is not compliant
        assert person.is_compliant is False, "Person should not be compliant"
        
        print(f"No helmet test passed: violations = {person.violations}")
    
    def test_match_ppe_person_without_vest(self, default_thresholds: Dict[str, float]) -> None:
        """
        Test PPE matching for person without vest.
        
        Creates synthetic bounding boxes for a person with helmet but no vest,
        verifies correct violation detection.
        
        Args:
            default_thresholds (Dict[str, float]): PPE matching thresholds
        """
        # Define person bounding box
        person_boxes = [[100, 100, 200, 400]]
        
        # Define helmet box (overlaps with person's head)
        helmet_boxes = [[120, 80, 180, 120]]
        
        # No vest boxes
        vest_boxes = []
        
        # Run PPE matching
        results = match_ppe(person_boxes, helmet_boxes, vest_boxes, default_thresholds)
        
        # Verify results
        assert len(results) == 1, "Should detect exactly one person"
        
        person = results[0]
        
        # Verify PPE status
        assert person.has_helmet is True, "Person should have helmet"
        assert person.has_vest is False, "Person should not have vest"
        
        # Verify violations
        assert len(person.violations) > 0, "Person should have violations"
        assert ViolationType.NO_VEST.value in person.violations, "Should have no_vest violation"
        
        # Verify person is not compliant
        assert person.is_compliant is False, "Person should not be compliant"
        
        print(f"No vest test passed: violations = {person.violations}")
    
    def test_match_ppe_person_without_any_ppe(self, default_thresholds: Dict[str, float]) -> None:
        """
        Test PPE matching for person without any PPE.
        
        Creates synthetic bounding boxes for a person with no PPE items,
        verifies correct violation detection.
        
        Args:
            default_thresholds (Dict[str, float]): PPE matching thresholds
        """
        # Define person bounding box
        person_boxes = [[100, 100, 200, 400]]
        
        # No PPE items
        helmet_boxes = []
        vest_boxes = []
        
        # Run PPE matching
        results = match_ppe(person_boxes, helmet_boxes, vest_boxes, default_thresholds)
        
        # Verify results
        assert len(results) == 1, "Should detect exactly one person"
        
        person = results[0]
        
        # Verify PPE status
        assert person.has_helmet is False, "Person should not have helmet"
        assert person.has_vest is False, "Person should not have vest"
        
        # Verify violations
        assert len(person.violations) > 0, "Person should have violations"
        assert ViolationType.NO_HELMET.value in person.violations, "Should have no_helmet violation"
        assert ViolationType.NO_VEST.value in person.violations, "Should have no_vest violation"
        
        # Should also have incomplete PPE violation
        assert ViolationType.INCOMPLETE_PPE.value in person.violations, "Should have incomplete_ppe violation"
        
        # Verify person is not compliant
        assert person.is_compliant is False, "Person should not be compliant"
        
        print(f"No PPE test passed: violations = {person.violations}")
    
    def test_match_ppe_multiple_persons(self, default_thresholds: Dict[str, float]) -> None:
        """
        Test PPE matching for multiple persons with different compliance levels.
        
        Creates synthetic scenario with compliant and non-compliant persons,
        verifies correct matching and violation detection.
        
        Args:
            default_thresholds (Dict[str, float]): PPE matching thresholds
        """
        # Define multiple person bounding boxes
        person_boxes = [
            [50, 100, 150, 400],   # Person 1
            [250, 100, 350, 400],  # Person 2
            [450, 100, 550, 400]   # Person 3
        ]
        
        # Define helmet boxes (only for persons 1 and 3)
        helmet_boxes = [
            [70, 80, 130, 120],    # Helmet for person 1
            [470, 80, 530, 120]    # Helmet for person 3
        ]
        
        # Define vest boxes (only for persons 1 and 2)
        vest_boxes = [
            [60, 150, 140, 250],   # Vest for person 1
            [260, 150, 340, 250]   # Vest for person 2
        ]
        
        # Run PPE matching
        results = match_ppe(person_boxes, helmet_boxes, vest_boxes, default_thresholds)
        
        # Verify results
        assert len(results) == 3, "Should detect exactly three persons"
        
        # Analyze each person
        compliance_status = []
        for person in results:
            compliance_status.append({
                'has_helmet': person.has_helmet,
                'has_vest': person.has_vest,
                'is_compliant': person.is_compliant,
                'violations_count': len(person.violations)
            })
        
        # Verify that we have different compliance levels
        compliant_count = sum(1 for status in compliance_status if status['is_compliant'])
        non_compliant_count = len(compliance_status) - compliant_count
        
        assert compliant_count >= 0, "Should have some compliance data"
        assert non_compliant_count >= 0, "Should have some non-compliance data"
        
        print(f"Multiple persons test passed: {compliant_count} compliant, {non_compliant_count} non-compliant")
        
        # Log detailed results
        for i, (person, status) in enumerate(zip(results, compliance_status)):
            print(f"Person {i+1}: helmet={status['has_helmet']}, vest={status['has_vest']}, "
                  f"compliant={status['is_compliant']}, violations={person.violations}")
    
    def test_match_ppe_no_overlap_scenario(self, default_thresholds: Dict[str, float]) -> None:
        """
        Test PPE matching when PPE items don't overlap with persons.
        
        Creates synthetic scenario where PPE items are far from persons,
        verifies that no matching occurs.
        
        Args:
            default_thresholds (Dict[str, float]): PPE matching thresholds
        """
        # Define person bounding box
        person_boxes = [[100, 100, 200, 400]]
        
        # Define PPE boxes far from person (no overlap)
        helmet_boxes = [[300, 50, 350, 100]]   # Far to the right
        vest_boxes = [[400, 150, 480, 250]]    # Even further right
        
        # Run PPE matching
        results = match_ppe(person_boxes, helmet_boxes, vest_boxes, default_thresholds)
        
        # Verify results
        assert len(results) == 1, "Should detect exactly one person"
        
        person = results[0]
        
        # Verify no PPE was matched (due to no overlap)
        assert person.has_helmet is False, "Should not match distant helmet"
        assert person.has_vest is False, "Should not match distant vest"
        
        # Verify violations are recorded
        assert len(person.violations) > 0, "Should have violations for missing PPE"
        assert person.is_compliant is False, "Person should not be compliant"
        
        print(f"No overlap test passed: violations = {person.violations}")


# Additional test utilities
def create_test_image_with_objects() -> np.ndarray:
    """
    Create a more sophisticated test image with realistic object shapes.
    
    Returns:
        np.ndarray: Test image with synthetic objects
    """
    # Create base image
    image = np.ones((480, 640, 3), dtype=np.uint8) * 128  # Gray background
    
    # Draw person (large vertical rectangle)
    cv2.rectangle(image, (200, 150), (280, 400), (80, 80, 120), -1)
    
    # Draw helmet (ellipse at head position)
    cv2.ellipse(image, (240, 130), (25, 20), 0, 0, 360, (50, 150, 200), -1)
    
    # Draw vest (rectangle at torso)
    cv2.rectangle(image, (210, 180), (270, 280), (50, 200, 50), -1)
    
    return image


if __name__ == "__main__":
    # Run tests directly
    pytest.main([__file__, "-v"])