"""
PPE Logic Module for Person and Equipment Matching

This module contains the core logic for matching detected persons with their
Personal Protective Equipment (PPE) items such as helmets and safety vests.
It uses geometric intersection algorithms to determine PPE compliance.

Author: PPE Detection System
Date: 2025-08-26
"""

import math
import uuid
from typing import Dict, List, Tuple, Optional, Union, Any

import numpy as np

from .schemas import (
    PersonStatus, 
    Detection, 
    ViolationType,
    ViolationsSummary
)


def calculate_iou(box1: List[float], box2: List[float]) -> float:
    """
    Calculate Intersection over Union (IoU) between two bounding boxes.
    
    IoU is a standard metric for measuring overlap between two bounding boxes.
    It's calculated as the area of intersection divided by the area of union.
    
    Args:
        box1 (List[float]): First bounding box [x1, y1, x2, y2]
        box2 (List[float]): Second bounding box [x1, y1, x2, y2]
    
    Returns:
        float: IoU value between 0.0 and 1.0
        
    Example:
        >>> box1 = [0, 0, 10, 10]  # 10x10 box at origin
        >>> box2 = [5, 5, 15, 15]  # 10x10 box offset by 5
        >>> iou = calculate_iou(box1, box2)
        >>> print(f"IoU: {iou:.2f}")  # Should be 0.14 (25/175)
    """
    # Extract coordinates
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    
    # Calculate intersection coordinates
    x1_inter = max(x1_1, x1_2)
    y1_inter = max(y1_1, y1_2)
    x2_inter = min(x2_1, x2_2)
    y2_inter = min(y2_1, y2_2)
    
    # Check if there's no intersection
    if x2_inter <= x1_inter or y2_inter <= y1_inter:
        return 0.0
    
    # Calculate intersection area
    intersection_area = (x2_inter - x1_inter) * (y2_inter - y1_inter)
    
    # Calculate areas of both boxes
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    
    # Calculate union area
    union_area = area1 + area2 - intersection_area
    
    # Avoid division by zero
    if union_area <= 0:
        return 0.0
    
    return intersection_area / union_area


def calculate_overlap_ratio(person_box: List[float], ppe_box: List[float]) -> float:
    """
    Calculate overlap ratio of PPE item relative to person box.
    
    This calculates what fraction of the PPE item overlaps with the person,
    which is useful for determining if PPE belongs to a specific person.
    
    Args:
        person_box (List[float]): Person bounding box [x1, y1, x2, y2]
        ppe_box (List[float]): PPE item bounding box [x1, y1, x2, y2]
    
    Returns:
        float: Overlap ratio (intersection_area / ppe_area)
    """
    # Extract coordinates
    px1, py1, px2, py2 = person_box
    ppx1, ppy1, ppx2, ppy2 = ppe_box
    
    # Calculate intersection coordinates
    x1_inter = max(px1, ppx1)
    y1_inter = max(py1, ppy1)
    x2_inter = min(px2, ppx2)
    y2_inter = min(py2, ppy2)
    
    # Check if there's no intersection
    if x2_inter <= x1_inter or y2_inter <= y1_inter:
        return 0.0
    
    # Calculate intersection area
    intersection_area = (x2_inter - x1_inter) * (y2_inter - y1_inter)
    
    # Calculate PPE area
    ppe_area = (ppx2 - ppx1) * (ppy2 - ppy1)
    
    # Avoid division by zero
    if ppe_area <= 0:
        return 0.0
    
    return intersection_area / ppe_area


def calculate_area_ratio(person_box: List[float], ppe_box: List[float]) -> float:
    """
    Calculate the ratio of PPE area to person area.
    
    This helps validate that the PPE item is reasonably sized relative
    to the person (e.g., a helmet shouldn't be larger than the person).
    
    Args:
        person_box (List[float]): Person bounding box [x1, y1, x2, y2]
        ppe_box (List[float]): PPE item bounding box [x1, y1, x2, y2]
    
    Returns:
        float: Area ratio (ppe_area / person_area)
    """
    # Calculate person area
    px1, py1, px2, py2 = person_box
    person_area = (px2 - px1) * (py2 - py1)
    
    # Calculate PPE area
    ppx1, ppy1, ppx2, ppy2 = ppe_box
    ppe_area = (ppx2 - ppx1) * (ppy2 - ppy1)
    
    # Avoid division by zero
    if person_area <= 0:
        return 0.0
    
    return ppe_area / person_area


def is_ppe_position_valid(person_box: List[float], ppe_box: List[float], ppe_type: str) -> bool:
    """
    Validate PPE position relative to person based on expected anatomy.
    
    Different PPE items should be in specific positions relative to the person:
    - Helmets should be in the upper portion (head area)
    - Vests should be in the middle portion (torso area)
    
    Args:
        person_box (List[float]): Person bounding box [x1, y1, x2, y2]
        ppe_box (List[float]): PPE item bounding box [x1, y1, x2, y2]
        ppe_type (str): Type of PPE ('helmet', 'vest', etc.)
    
    Returns:
        bool: True if PPE is in anatomically correct position
    """
    px1, py1, px2, py2 = person_box
    ppx1, ppy1, ppx2, ppy2 = ppe_box
    
    person_height = py2 - py1
    person_width = px2 - px1
    
    # Calculate PPE center point
    ppe_center_x = (ppx1 + ppx2) / 2
    ppe_center_y = (ppy1 + ppy2) / 2
    
    # Check if PPE center is within person's horizontal bounds (with tolerance)
    horizontal_tolerance = person_width * 0.3  # 30% tolerance
    if not (px1 - horizontal_tolerance <= ppe_center_x <= px2 + horizontal_tolerance):
        return False
    
    # Position validation based on PPE type
    if ppe_type.lower() in ['helmet', 'hard_hat', 'hardhat']:
        # Helmet should be in upper 40% of person
        upper_bound = py1 + person_height * 0.4
        return ppy1 >= py1 and ppe_center_y <= upper_bound
        
    elif ppe_type.lower() in ['vest', 'safety_vest', 'safetyvest']:
        # Vest should be in middle 60% of person (from 20% to 80%)
        upper_bound = py1 + person_height * 0.2
        lower_bound = py1 + person_height * 0.8
        return upper_bound <= ppe_center_y <= lower_bound
    
    # For unknown PPE types, allow anywhere within person bounds
    return py1 <= ppe_center_y <= py2


def find_best_ppe_match(
    person_box: List[float], 
    ppe_boxes: List[List[float]], 
    ppe_type: str,
    thresholds: Dict[str, float]
) -> Tuple[Optional[int], float, Dict[str, float]]:
    """
    Find the best PPE match for a given person.
    
    Evaluates all available PPE items and returns the best match based on
    multiple criteria including IoU, overlap ratio, area ratio, and position.
    
    Args:
        person_box (List[float]): Person bounding box [x1, y1, x2, y2]
        ppe_boxes (List[List[float]]): List of PPE bounding boxes
        ppe_type (str): Type of PPE to match ('helmet', 'vest')
        thresholds (Dict[str, float]): Matching thresholds
    
    Returns:
        Tuple containing:
        - Optional[int]: Index of best matching PPE (None if no match)
        - float: Best matching score
        - Dict[str, float]: Detailed metrics for the best match
    """
    if not ppe_boxes:
        return None, 0.0, {}
    
    best_match_idx = None
    best_score = 0.0
    best_metrics = {}
    
    # Thresholds for matching
    min_iou = thresholds.get('min_iou_threshold', 0.1)
    min_overlap = thresholds.get('ppe_overlap_threshold', 0.3)
    min_area_ratio = thresholds.get('min_ppe_area_ratio', 0.01)
    max_area_ratio = thresholds.get('max_ppe_area_ratio', 0.5)
    
    for idx, ppe_box in enumerate(ppe_boxes):
        # Calculate various metrics
        iou = calculate_iou(person_box, ppe_box)
        overlap_ratio = calculate_overlap_ratio(person_box, ppe_box)
        area_ratio = calculate_area_ratio(person_box, ppe_box)
        position_valid = is_ppe_position_valid(person_box, ppe_box, ppe_type)
        
        metrics = {
            'iou': iou,
            'overlap_ratio': overlap_ratio,
            'area_ratio': area_ratio,
            'position_valid': position_valid
        }
        
        # Check if this PPE meets minimum requirements
        meets_requirements = (
            iou >= min_iou and
            overlap_ratio >= min_overlap and
            min_area_ratio <= area_ratio <= max_area_ratio and
            position_valid
        )
        
        if not meets_requirements:
            continue
        
        # Calculate composite score (weighted combination of metrics)
        score = (
            iou * 0.3 +  # IoU weight: 30%
            overlap_ratio * 0.4 +  # Overlap weight: 40%
            (1.0 - abs(area_ratio - 0.1)) * 0.2 +  # Area ratio weight: 20% (prefer ~10% of person area)
            (1.0 if position_valid else 0.0) * 0.1  # Position weight: 10%
        )
        
        # Update best match if this score is better
        if score > best_score:
            best_score = score
            best_match_idx = idx
            best_metrics = metrics
    
    return best_match_idx, best_score, best_metrics


def match_ppe(
    person_boxes: List[List[float]], 
    helmet_boxes: List[List[float]], 
    vest_boxes: List[List[float]], 
    thresholds: Dict[str, float]
) -> List[PersonStatus]:
    """
    Match detected persons with their PPE items (helmets and vests).
    
    This is the main function that implements the PPE matching algorithm.
    For each detected person, it attempts to find the best matching helmet
    and vest based on geometric overlap and position criteria.
    
    Algorithm:
    1. For each person, calculate IoU/overlap with all PPE items
    2. If intersection > ppe_overlap_threshold and area_ratio is valid → PPE assigned
    3. If no intersection found → violation recorded
    4. Return PersonStatus list with compliance information
    
    Args:
        person_boxes (List[List[float]]): List of person bounding boxes
        helmet_boxes (List[List[float]]): List of helmet bounding boxes
        vest_boxes (List[List[float]]): List of vest bounding boxes
        thresholds (Dict[str, float]): Matching thresholds including:
            - ppe_overlap_threshold: Minimum overlap ratio (default: 0.3)
            - min_ppe_area_ratio: Minimum PPE/person area ratio (default: 0.01)
            - max_ppe_area_ratio: Maximum PPE/person area ratio (default: 0.5)
            - min_iou_threshold: Minimum IoU for matching (default: 0.1)
    
    Returns:
        List[PersonStatus]: List of PersonStatus objects with PPE compliance info
    
    Example:
        >>> persons = [[100, 100, 200, 300]]  # One person
        >>> helmets = [[120, 100, 180, 150]]  # One helmet
        >>> vests = [[110, 150, 190, 220]]    # One vest
        >>> thresholds = {'ppe_overlap_threshold': 0.3}
        >>> results = match_ppe(persons, helmets, vests, thresholds)
        >>> print(f"Person compliant: {results[0].is_compliant}")
    """
    person_statuses = []
    
    # Keep track of used PPE items to avoid double assignment
    used_helmet_indices = set()
    used_vest_indices = set()
    
    # Process each detected person
    for person_idx, person_box in enumerate(person_boxes):
        # Generate unique person ID
        person_id = f"person_{person_idx}_{uuid.uuid4().hex[:8]}"
        
        # Initialize person status
        has_helmet = False
        has_vest = False
        violations = []
        
        # Find available helmet boxes (not already used)
        available_helmets = [
            (idx, box) for idx, box in enumerate(helmet_boxes) 
            if idx not in used_helmet_indices
        ]
        available_helmet_boxes = [box for _, box in available_helmets]
        
        # Find best helmet match
        if available_helmet_boxes:
            helmet_match_idx, helmet_score, helmet_metrics = find_best_ppe_match(
                person_box, available_helmet_boxes, 'helmet', thresholds
            )
            
            if helmet_match_idx is not None:
                # Mark helmet as used
                actual_helmet_idx = available_helmets[helmet_match_idx][0]
                used_helmet_indices.add(actual_helmet_idx)
                has_helmet = True
            else:
                violations.append(ViolationType.NO_HELMET.value)
        else:
            violations.append(ViolationType.NO_HELMET.value)
        
        # Find available vest boxes (not already used)
        available_vests = [
            (idx, box) for idx, box in enumerate(vest_boxes) 
            if idx not in used_vest_indices
        ]
        available_vest_boxes = [box for _, box in available_vests]
        
        # Find best vest match
        if available_vest_boxes:
            vest_match_idx, vest_score, vest_metrics = find_best_ppe_match(
                person_box, available_vest_boxes, 'vest', thresholds
            )
            
            if vest_match_idx is not None:
                # Mark vest as used
                actual_vest_idx = available_vests[vest_match_idx][0]
                used_vest_indices.add(actual_vest_idx)
                has_vest = True
            else:
                violations.append(ViolationType.NO_VEST.value)
        else:
            violations.append(ViolationType.NO_VEST.value)
        
        # Add additional violation if neither PPE item is found
        if not has_helmet and not has_vest:
            violations.append(ViolationType.INCOMPLETE_PPE.value)
        elif not has_helmet or not has_vest:
            # Partial compliance - already have specific violations
            pass
        
        # Create PersonStatus object
        person_status = PersonStatus(
            person_id=person_id,
            bbox=person_box,
            has_helmet=has_helmet,
            has_vest=has_vest,
            violations=violations
        )
        
        person_statuses.append(person_status)
    
    return person_statuses


def calculate_violations_summary(person_statuses: List[PersonStatus]) -> ViolationsSummary:
    """
    Calculate summary statistics for PPE violations.
    
    Analyzes the list of PersonStatus objects to provide aggregate
    statistics about PPE compliance across all detected persons.
    
    Args:
        person_statuses (List[PersonStatus]): List of person compliance status
    
    Returns:
        ViolationsSummary: Aggregate violation statistics
    """
    total_persons = len(person_statuses)
    
    if total_persons == 0:
        return ViolationsSummary(
            total_persons=0,
            compliant_persons=0,
            violations_count={},
            compliance_rate=1.0
        )
    
    # Count compliant persons
    compliant_persons = sum(1 for person in person_statuses if person.is_compliant)
    
    # Count violations by type
    violations_count = {}
    for person in person_statuses:
        for violation in person.violations:
            violations_count[violation] = violations_count.get(violation, 0) + 1
    
    # Calculate compliance rate
    compliance_rate = compliant_persons / total_persons if total_persons > 0 else 1.0
    
    return ViolationsSummary(
        total_persons=total_persons,
        compliant_persons=compliant_persons,
        violations_count=violations_count,
        compliance_rate=compliance_rate
    )


def get_default_thresholds() -> Dict[str, float]:
    """
    Get default threshold values for PPE matching.
    
    Returns:
        Dict[str, float]: Default threshold configuration
    """
    return {
        'ppe_overlap_threshold': 0.3,      # Minimum overlap ratio for PPE assignment
        'min_ppe_area_ratio': 0.01,        # Minimum PPE area relative to person
        'max_ppe_area_ratio': 0.5,         # Maximum PPE area relative to person
        'min_iou_threshold': 0.1,          # Minimum IoU for matching
        'helmet_position_tolerance': 0.4,   # Tolerance for helmet position
        'vest_position_tolerance': 0.6,     # Tolerance for vest position
    }


def validate_thresholds(thresholds: Dict[str, float]) -> Dict[str, float]:
    """
    Validate and sanitize threshold values.
    
    Ensures that threshold values are within reasonable ranges and
    provides defaults for missing values.
    
    Args:
        thresholds (Dict[str, float]): Input thresholds
    
    Returns:
        Dict[str, float]: Validated thresholds
    """
    defaults = get_default_thresholds()
    validated = defaults.copy()
    
    for key, value in thresholds.items():
        if key in defaults:
            # Clamp values to reasonable ranges
            if 'threshold' in key or 'ratio' in key:
                validated[key] = max(0.0, min(1.0, value))
            else:
                validated[key] = max(0.0, value)
    
    return validated


# Utility functions for debugging and analysis

def analyze_matching_quality(
    person_statuses: List[PersonStatus],
    person_boxes: List[List[float]],
    helmet_boxes: List[List[float]],
    vest_boxes: List[List[float]]
) -> Dict[str, Any]:
    """
    Analyze the quality of PPE matching for debugging purposes.
    
    Provides detailed metrics about the matching process to help
    tune thresholds and improve algorithm performance.
    
    Args:
        person_statuses (List[PersonStatus]): Matching results
        person_boxes (List[List[float]]): Original person boxes
        helmet_boxes (List[List[float]]): Original helmet boxes
        vest_boxes (List[List[float]]): Original vest boxes
    
    Returns:
        Dict[str, Any]: Analysis metrics and statistics
    """
    analysis = {
        'total_persons': len(person_boxes),
        'total_helmets': len(helmet_boxes),
        'total_vests': len(vest_boxes),
        'matched_persons': len(person_statuses),
        'helmet_match_rate': 0.0,
        'vest_match_rate': 0.0,
        'overall_compliance': 0.0,
        'unmatched_helmets': len(helmet_boxes),
        'unmatched_vests': len(vest_boxes)
    }
    
    if person_statuses:
        helmet_matches = sum(1 for p in person_statuses if p.has_helmet)
        vest_matches = sum(1 for p in person_statuses if p.has_vest)
        compliant_persons = sum(1 for p in person_statuses if p.is_compliant)
        
        analysis.update({
            'helmet_match_rate': helmet_matches / len(person_statuses),
            'vest_match_rate': vest_matches / len(person_statuses),
            'overall_compliance': compliant_persons / len(person_statuses),
            'unmatched_helmets': len(helmet_boxes) - helmet_matches,
            'unmatched_vests': len(vest_boxes) - vest_matches
        })
    
    return analysis