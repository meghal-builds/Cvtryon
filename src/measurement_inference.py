"""Measurement Inference Module"""

import numpy as np
from typing import Optional

from src.models import Measurements, PoseResult, SegmentationResult
from src.pose_detection import get_keypoint_coordinate, calculate_torso_length


def infer_measurements(
    pose_result: PoseResult,
    segmentation_result: SegmentationResult,
    pixels_per_cm: float = 10.0
) -> Measurements:
    """
    Infer body measurements from pose and segmentation
    
    Args:
        pose_result: Result from pose detection
        segmentation_result: Result from segmentation
        pixels_per_cm: Conversion factor from pixels to cm
        
    Returns:
        Measurements object
    """
    # Calculate shoulder width
    shoulder_width_cm = (pose_result.shoulder_width_px / pixels_per_cm)
    
    # Calculate chest circumference (estimated from shoulder and torso)
    torso_mask = segmentation_result.body_parts.get('torso')
    chest_circumference_cm = 0.0
    
    if torso_mask is not None:
        # Estimate from torso width
        torso_width = np.sum(np.any(torso_mask > 0, axis=0))
        chest_circumference_cm = (torso_width / pixels_per_cm) * 2.5  # Empirical factor
    
    # Calculate torso length
    torso_length_cm = 0.0
    torso_length_px = calculate_torso_length(pose_result)
    
    if torso_length_px is not None:
        torso_length_cm = torso_length_px / pixels_per_cm
    
    # Calculate confidence
    confidence = calculate_measurement_confidence(
        pose_result,
        segmentation_result
    )
    
    return Measurements(
        shoulder_width_cm=shoulder_width_cm,
        chest_circumference_cm=chest_circumference_cm,
        torso_length_cm=torso_length_cm,
        source='inferred',
        confidence=confidence
    )


def calculate_measurement_confidence(
    pose_result: PoseResult,
    segmentation_result: SegmentationResult
) -> float:
    """
    Calculate confidence in measurements
    
    Args:
        pose_result: Result from pose detection
        segmentation_result: Result from segmentation
        
    Returns:
        Confidence score (0.0 to 1.0)
    """
    confidence = 1.0
    
    # Reduce confidence if pose is not frontal
    if not pose_result.is_frontal:
        confidence *= 0.7
    
    # Reduce confidence if segmentation confidence is low
    confidence *= segmentation_result.confidence
    
    # Reduce confidence if too few keypoints
    if len(pose_result.keypoints) < 10:
        confidence *= 0.9
    
    return max(0.0, min(1.0, confidence))


def validate_measurements(measurements: Measurements) -> tuple:
    """
    Validate measurements are in reasonable ranges
    
    Args:
        measurements: Measurements object
        
    Returns:
        Tuple of (is_valid, errors)
    """
    errors = []
    
    # Validate shoulder width (30-60 cm)
    if measurements.shoulder_width_cm < 30 or measurements.shoulder_width_cm > 60:
        errors.append(
            f"Invalid shoulder width: {measurements.shoulder_width_cm:.1f}cm "
            f"(expected: 30-60cm)"
        )
    
    # Validate chest circumference (70-150 cm)
    if measurements.chest_circumference_cm < 70 or measurements.chest_circumference_cm > 150:
        errors.append(
            f"Invalid chest circumference: {measurements.chest_circumference_cm:.1f}cm "
            f"(expected: 70-150cm)"
        )
    
    # Validate torso length (40-80 cm)
    if measurements.torso_length_cm < 40 or measurements.torso_length_cm > 80:
        errors.append(
            f"Invalid torso length: {measurements.torso_length_cm:.1f}cm "
            f"(expected: 40-80cm)"
        )
    
    is_valid = len(errors) == 0
    
    return is_valid, errors