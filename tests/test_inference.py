"""
Inference Tests for PPE Detection ML Components

This module contains comprehensive tests for the inference pipeline,
model loading, detection processing, and PPE matching logic.

Author: PPE Detection System
Date: 2025-08-26
"""

import numpy as np
import pytest
from unittest.mock import Mock, patch, MagicMock
from typing import List, Dict, Any

# Import modules to test
try:
    from app.inference import (
        load_model, run_inference, run_inference_with_ppe_analysis,
        detect_device, validate_model_path, normalize_class_name,
        postprocess_detections, get_detections_by_category
    )
    from app.ppe_logic import (
        match_ppe, calculate_iou, calculate_overlap_ratio,
        find_best_ppe_match, calculate_violations_summary
    )
    from app.schemas import Detection, PersonStatus, ViolationType
except ImportError as e:
    pytest.skip(f"Could not import modules: {e}", allow_module_level=True)


class TestDeviceDetection:
    """Test device detection and validation."""
    
    def test_detect_device_cpu_fallback(self):
        """Test CPU fallback when no GPU available."""
        with patch('torch.cuda.is_available', return_value=False), \
             patch('torch.backends.mps.is_available', return_value=False):
            device = detect_device()
            assert device == 'cpu'
    
    @patch('torch.cuda.is_available')
    @patch('torch.cuda.device_count')
    def test_detect_device_cuda(self, mock_device_count, mock_cuda_available):
        """Test CUDA device detection."""
        mock_cuda_available.return_value = True
        mock_device_count.return_value = 1
        
        device = detect_device()
        assert device == 'cuda'
    
    def test_detect_device_mps(self):
        """Test MPS device detection for Apple Silicon."""
        with patch('torch.cuda.is_available', return_value=False):
            # Mock MPS availability
            mock_backends = Mock()
            mock_backends.mps.is_available.return_value = True
            
            with patch('torch.backends', mock_backends):
                device = detect_device()
                assert device == 'mps'


class TestModelPathValidation:
    """Test model path validation and resolution."""
    
    def test_validate_empty_path(self):
        """Test validation with empty path."""
        with pytest.raises(ValueError, match="Model weights must be a non-empty string"):
            validate_model_path("")
        
        with pytest.raises(ValueError, match="Model weights must be a non-empty string"):
            validate_model_path(None)
    
    def test_validate_ultralytics_model_name(self):
        """Test validation with standard ultralytics model names."""
        model_names = ['yolov8n.pt', 'yolov8s.pt', 'yolov8m.pt', 'yolov8l.pt']
        
        for name in model_names:
            result = validate_model_path(name)
            assert result == name
    
    @patch('pathlib.Path.exists')
    def test_validate_local_file_exists(self, mock_exists):
        """Test validation with existing local file."""
        mock_exists.return_value = True
        
        result = validate_model_path("custom_model.pt")
        assert "custom_model.pt" in result
    
    @patch('pathlib.Path.exists')
    def test_validate_local_file_not_exists(self, mock_exists):
        """Test validation with non-existing local file."""
        mock_exists.return_value = False
        
        with pytest.raises(FileNotFoundError):
            validate_model_path("nonexistent_model.pt")


class TestClassNameNormalization:
    """Test class name normalization for PPE categories."""
    
    def test_normalize_person_variations(self):
        """Test normalization of person class variations."""
        person_variations = ['person', 'people', 'human', 'Person', 'PEOPLE']
        
        for variation in person_variations:
            result = normalize_class_name(variation)
            assert result == 'person'
    
    def test_normalize_helmet_variations(self):
        """Test normalization of helmet class variations."""
        helmet_variations = [
            'helmet', 'hard_hat', 'hardhat', 'hard hat', 
            'safety_helmet', 'HELMET', 'Hard-Hat'
        ]
        
        for variation in helmet_variations:
            result = normalize_class_name(variation)
            assert result == 'helmet'
    
    def test_normalize_vest_variations(self):
        """Test normalization of vest class variations."""
        vest_variations = [
            'vest', 'safety_vest', 'safetyvest', 'safety vest',
            'hi_vis', 'hi-vis', 'VEST', 'Safety-Vest'
        ]
        
        for variation in vest_variations:
            result = normalize_class_name(variation)
            assert result == 'vest'
    
    def test_normalize_unknown_class(self):
        """Test normalization of unknown class names."""
        unknown_classes = ['car', 'bicycle', 'unknown_object']
        
        for unknown in unknown_classes:
            result = normalize_class_name(unknown)
            assert result == unknown.lower()
    
    def test_normalize_non_string_input(self):
        """Test normalization with non-string input."""
        result = normalize_class_name(123)
        assert result == '123'
        
        result = normalize_class_name(None)
        assert result == 'none'


class TestDetectionPostprocessing:
    """Test detection postprocessing and categorization."""
    
    def create_mock_results(self, detections_data: List[Dict]) -> Mock:
        """Create mock YOLOv8 results object."""
        mock_results = Mock()
        
        # Mock boxes
        mock_boxes = Mock()
        mock_boxes.xyxy = np.array([d['bbox'] for d in detections_data])
        mock_boxes.conf = np.array([d['conf'] for d in detections_data])
        mock_boxes.cls = np.array([d['cls'] for d in detections_data])
        
        mock_results.boxes = mock_boxes
        
        # Mock names
        mock_results.names = {0: 'person', 1: 'helmet', 2: 'vest'}
        
        return mock_results
    
    def test_postprocess_detections_basic(self):
        """Test basic detection postprocessing."""
        detections_data = [
            {'bbox': [100, 100, 200, 200], 'conf': 0.8, 'cls': 0},  # person
            {'bbox': [120, 100, 160, 140], 'conf': 0.7, 'cls': 1},  # helmet
            {'bbox': [110, 150, 190, 200], 'conf': 0.6, 'cls': 2},  # vest
        ]
        
        mock_results = self.create_mock_results(detections_data)
        
        detections = postprocess_detections(mock_results, conf_threshold=0.5)
        
        assert len(detections) == 3
        
        # Check first detection (person)
        person_det = detections[0]
        assert person_det['category'] == 'person'
        assert person_det['confidence'] == 0.8
        assert person_det['bbox'] == [100.0, 100.0, 200.0, 200.0]
    
    def test_postprocess_detections_confidence_filtering(self):
        """Test confidence threshold filtering."""
        detections_data = [
            {'bbox': [100, 100, 200, 200], 'conf': 0.8, 'cls': 0},  # Above threshold
            {'bbox': [120, 100, 160, 140], 'conf': 0.3, 'cls': 1},  # Below threshold
            {'bbox': [110, 150, 190, 200], 'conf': 0.6, 'cls': 2},  # Above threshold
        ]
        
        mock_results = self.create_mock_results(detections_data)
        
        detections = postprocess_detections(mock_results, conf_threshold=0.5)
        
        # Should filter out detection with conf=0.3
        assert len(detections) == 2
        assert all(det['confidence'] >= 0.5 for det in detections)
    
    def test_get_detections_by_category(self):
        """Test detection categorization."""
        detections = [
            {'category': 'person', 'bbox': [100, 100, 200, 200]},
            {'category': 'helmet', 'bbox': [120, 100, 160, 140]},
            {'category': 'vest', 'bbox': [110, 150, 190, 200]},
            {'category': 'person', 'bbox': [300, 100, 400, 200]},
        ]
        
        categorized = get_detections_by_category(detections)
        
        assert len(categorized['person']) == 2
        assert len(categorized['helmet']) == 1
        assert len(categorized['vest']) == 1
        
        # Check specific bounding boxes
        assert categorized['helmet'][0] == [120, 100, 160, 140]
        assert categorized['vest'][0] == [110, 150, 190, 200]


class TestPPELogic:
    """Test PPE matching and logic functions."""
    
    def test_calculate_iou_perfect_overlap(self):
        """Test IoU calculation with perfect overlap."""
        box1 = [100, 100, 200, 200]
        box2 = [100, 100, 200, 200]
        
        iou = calculate_iou(box1, box2)
        assert abs(iou - 1.0) < 1e-6
    
    def test_calculate_iou_no_overlap(self):
        """Test IoU calculation with no overlap."""
        box1 = [100, 100, 200, 200]
        box2 = [300, 300, 400, 400]
        
        iou = calculate_iou(box1, box2)
        assert abs(iou - 0.0) < 1e-6
    
    def test_calculate_iou_partial_overlap(self):
        """Test IoU calculation with partial overlap."""
        box1 = [0, 0, 10, 10]    # Area: 100
        box2 = [5, 5, 15, 15]    # Area: 100, Intersection: 25, Union: 175
        
        iou = calculate_iou(box1, box2)
        expected_iou = 25.0 / 175.0  # ≈ 0.143
        
        assert abs(iou - expected_iou) < 1e-3
    
    def test_calculate_overlap_ratio(self):
        """Test overlap ratio calculation."""
        person_box = [100, 100, 200, 200]  # Area: 10000
        ppe_box = [120, 120, 160, 160]     # Area: 1600, fully inside person
        
        overlap_ratio = calculate_overlap_ratio(person_box, ppe_box)
        
        # PPE box is fully inside person box
        assert abs(overlap_ratio - 1.0) < 1e-6
    
    def test_match_ppe_compliant_person(self):
        """Test PPE matching for compliant person."""
        # Person with both helmet and vest properly positioned
        person_boxes = [[100, 100, 200, 300]]  # Person
        helmet_boxes = [[120, 100, 180, 150]]  # Helmet at head
        vest_boxes = [[110, 150, 190, 220]]    # Vest at torso
        
        thresholds = {
            'ppe_overlap_threshold': 0.3,
            'min_ppe_area_ratio': 0.01,
            'max_ppe_area_ratio': 0.5,
            'min_iou_threshold': 0.1
        }
        
        persons = match_ppe(person_boxes, helmet_boxes, vest_boxes, thresholds)
        
        assert len(persons) == 1
        person = persons[0]
        
        assert person.has_helmet is True
        assert person.has_vest is True
        assert len(person.violations) == 0
    
    def test_match_ppe_no_helmet(self):
        """Test PPE matching for person without helmet."""
        person_boxes = [[100, 100, 200, 300]]
        helmet_boxes = []  # No helmet
        vest_boxes = [[110, 150, 190, 220]]  # Has vest
        
        thresholds = {
            'ppe_overlap_threshold': 0.3,
            'min_ppe_area_ratio': 0.01,
            'max_ppe_area_ratio': 0.5,
            'min_iou_threshold': 0.1
        }
        
        persons = match_ppe(person_boxes, helmet_boxes, vest_boxes, thresholds)
        
        assert len(persons) == 1
        person = persons[0]
        
        assert person.has_helmet is False
        assert person.has_vest is True
        assert ViolationType.NO_HELMET.value in person.violations
    
    def test_match_ppe_no_vest(self):
        """Test PPE matching for person without vest."""
        person_boxes = [[100, 100, 200, 300]]
        helmet_boxes = [[120, 100, 180, 150]]  # Has helmet
        vest_boxes = []  # No vest
        
        thresholds = {
            'ppe_overlap_threshold': 0.3,
            'min_ppe_area_ratio': 0.01,
            'max_ppe_area_ratio': 0.5,
            'min_iou_threshold': 0.1
        }
        
        persons = match_ppe(person_boxes, helmet_boxes, vest_boxes, thresholds)
        
        assert len(persons) == 1
        person = persons[0]
        
        assert person.has_helmet is True
        assert person.has_vest is False
        assert ViolationType.NO_VEST.value in person.violations
    
    def test_match_ppe_no_ppe_items(self):
        """Test PPE matching for person with no PPE items."""
        person_boxes = [[100, 100, 200, 300]]
        helmet_boxes = []  # No helmet
        vest_boxes = []    # No vest
        
        thresholds = {
            'ppe_overlap_threshold': 0.3,
            'min_ppe_area_ratio': 0.01,
            'max_ppe_area_ratio': 0.5,
            'min_iou_threshold': 0.1
        }
        
        persons = match_ppe(person_boxes, helmet_boxes, vest_boxes, thresholds)
        
        assert len(persons) == 1
        person = persons[0]
        
        assert person.has_helmet is False
        assert person.has_vest is False
        
        expected_violations = {
            ViolationType.NO_HELMET.value,
            ViolationType.NO_VEST.value,
            ViolationType.INCOMPLETE_PPE.value
        }
        assert set(person.violations) == expected_violations
    
    def test_calculate_violations_summary(self):
        """Test violations summary calculation."""
        # Create test persons with different violation patterns
        person1 = PersonStatus(
            person_id="person_1",
            bbox=[100, 100, 200, 300],
            has_helmet=True,
            has_vest=True,
            violations=[]
        )
        
        person2 = PersonStatus(
            person_id="person_2",
            bbox=[300, 100, 400, 300],
            has_helmet=False,
            has_vest=True,
            violations=[ViolationType.NO_HELMET.value]
        )
        
        person3 = PersonStatus(
            person_id="person_3",
            bbox=[500, 100, 600, 300],
            has_helmet=False,
            has_vest=False,
            violations=[
                ViolationType.NO_HELMET.value,
                ViolationType.NO_VEST.value,
                ViolationType.INCOMPLETE_PPE.value
            ]
        )
        
        persons = [person1, person2, person3]
        summary = calculate_violations_summary(persons)
        
        assert summary.total_persons == 3
        assert summary.compliant_persons == 1
        assert summary.violation_persons == 2
        assert summary.compliance_rate == pytest.approx(1/3, rel=1e-3)
        
        # Check violation type counts
        violation_counts = summary.violation_types
        assert violation_counts[ViolationType.NO_HELMET.value] == 2
        assert violation_counts[ViolationType.NO_VEST.value] == 1
        assert violation_counts[ViolationType.INCOMPLETE_PPE.value] == 1


class TestInferenceIntegration:
    """Test high-level inference integration."""
    
    @patch('app.inference.get_cached_model')
    @patch('app.inference.run_inference')
    @patch('app.inference.match_ppe')
    @patch('app.inference.calculate_violations_summary')
    def test_run_inference_with_ppe_analysis(
        self, mock_violations_summary, mock_match_ppe, 
        mock_run_inference, mock_get_model
    ):
        """Test integrated PPE analysis pipeline."""
        # Mock model
        mock_model = Mock()
        mock_get_model.return_value = mock_model
        
        # Mock raw detections
        mock_raw_detections = [
            {'category': 'person', 'confidence': 0.8, 'bbox': [100, 100, 200, 300]},
            {'category': 'helmet', 'confidence': 0.7, 'bbox': [120, 100, 160, 140]},
            {'category': 'vest', 'confidence': 0.6, 'bbox': [110, 150, 190, 220]}
        ]
        mock_run_inference.return_value = mock_raw_detections
        
        # Mock PPE matching
        mock_person = PersonStatus(
            person_id="person_1",
            bbox=[100, 100, 200, 300],
            has_helmet=True,
            has_vest=True,
            violations=[]
        )
        mock_match_ppe.return_value = [mock_person]
        
        # Mock violations summary
        mock_summary = Mock()
        mock_summary.total_persons = 1
        mock_summary.violation_persons = 0
        mock_violations_summary.return_value = mock_summary
        
        # Test the integration
        test_image = np.zeros((480, 640, 3), dtype=np.uint8)
        
        result = run_inference_with_ppe_analysis(
            test_image, draw=False, conf_threshold=0.35, iou_threshold=0.5
        )
        
        # Verify the pipeline was called correctly
        mock_run_inference.assert_called_once_with(
            mock_model, test_image, 0.35, 0.5
        )
        mock_match_ppe.assert_called_once()
        mock_violations_summary.assert_called_once_with([mock_person])
        
        # Check result structure
        assert 'detections' in result
        assert 'persons' in result
        assert 'violations' in result
        assert 'violations_summary' in result
        assert 'total_persons' in result
        assert 'total_violations' in result
        
        assert result['detections'] == mock_raw_detections
        assert result['persons'] == [mock_person]
        assert result['total_persons'] == 1
    
    def test_run_inference_with_ppe_analysis_error_handling(self):
        """Test error handling in PPE analysis pipeline."""
        # Test with invalid image
        with pytest.raises(ValueError, match="Image must be a numpy array"):
            run_inference_with_ppe_analysis(None)
        
        # Test with empty image
        empty_image = np.array([])
        with pytest.raises(ValueError, match="Image cannot be empty"):
            run_inference_with_ppe_analysis(empty_image)
    
    @patch('app.inference.get_cached_model')
    def test_run_inference_with_ppe_analysis_no_model(self, mock_get_model):
        """Test behavior when no model is available."""
        mock_get_model.return_value = None
        
        # Mock load_model to also fail
        with patch('app.inference.load_model', side_effect=Exception("Model load failed")):
            test_image = np.zeros((480, 640, 3), dtype=np.uint8)
            
            result = run_inference_with_ppe_analysis(test_image)
            
            # Should return error result
            assert 'error' in result
            assert result['total_persons'] == 0
            assert result['total_violations'] == 0


if __name__ == "__main__":
    pytest.main([__file__])