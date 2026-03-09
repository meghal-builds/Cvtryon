"""Pose Detection Processing Module"""

import numpy as np
from typing import Dict, Optional, List

from src.models import PoseResult, Keypoint
from src.model_layer import PoseModel


CRITICAL_KEYPOINTS = {'left_shoulder', 'right_shoulder', 'neck'}
MIN_KEYPOINT_COUNT = 7
MAX_SHOULDER_ANGLE = 25.0  # degrees


def detect_pose(
    image: np.ndarray,
    model: PoseModel,
    require_frontal: bool = True
) -> PoseResult:
    """
    Detect pose in image
    
    Args:
        image: Input image (H x W x 3)
        model: Pose model instance
        require_frontal: Require frontal pose
        
    Returns:
        PoseResult with keypoints and validation
    """
    if not isinstance(image, np.ndarray):
        raise TypeError("Image must be numpy array")
    
    if len(image.shape) != 3 or image.shape[2] != 3:
        raise ValueError("Image must be (H x W x 3)")
    
    # Run pose detection
    result = model.predict(image)
    
    # Check for critical keypoints
    critical_missing = check_critical_keypoints(result)
    if critical_missing:
        raise RuntimeError(
            f"Critical keypoints not detected: {', '.join(critical_missing)}. "
            "Please ensure your upper body is clearly visible in the photo "
            "with both shoulders and neck visible from the front."
        )
    
    # Check sufficient keypoints
    if len(result.keypoints) < MIN_KEYPOINT_COUNT:
        result.warnings.append(
            f"Insufficient keypoints: {len(result.keypoints)} "
            f"(min: {MIN_KEYPOINT_COUNT})"
        )
    
    # Check pose quality
    is_valid, warnings = validate_pose_quality(result)
    result.is_frontal = is_valid
    result.warnings.extend(warnings)
    
    return result


def check_critical_keypoints(result: PoseResult) -> List[str]:
    """
    Check if critical keypoints are detected
    
    Args:
        result: Pose result
        
    Returns:
        List of missing critical keypoints
    """
    detected = {k.name for k in result.keypoints}
    missing = CRITICAL_KEYPOINTS - detected
    return list(missing)


def validate_pose_quality(result: PoseResult) -> tuple:
    """
    Validate pose quality
    
    Args:
        result: Pose result
        
    Returns:
        Tuple of (is_frontal, warnings)
    """
    warnings = []
    is_frontal = True
    
    # Check shoulder angle
    if result.shoulder_angle_degrees > MAX_SHOULDER_ANGLE:
        warnings.append(
            f"Non-frontal pose detected: shoulder angle "
            f"{result.shoulder_angle_degrees:.1f}° "
            f"(max: {MAX_SHOULDER_ANGLE}°)"
        )
        is_frontal = False
    
    # Check keypoint count
    if len(result.keypoints) < MIN_KEYPOINT_COUNT:
        warnings.append(
            f"Insufficient keypoints: {len(result.keypoints)} "
            f"(min: {MIN_KEYPOINT_COUNT})"
        )
        is_frontal = False
    
    return is_frontal, warnings


def get_keypoint_coordinate(result: PoseResult, keypoint_name: str) -> Optional[tuple]:
    """
    Get coordinate of specific keypoint
    
    Args:
        result: Pose result
        keypoint_name: Name of keypoint
        
    Returns:
        Tuple of (x, y) or None
    """
    for keypoint in result.keypoints:
        if keypoint.name == keypoint_name:
            return (keypoint.x, keypoint.y)
    
    return None


def calculate_torso_length(result: PoseResult) -> Optional[float]:
    """
    Calculate torso length from keypoints
    
    Args:
        result: Pose result
        
    Returns:
        Torso length in pixels or None
    """
    neck_coord = get_keypoint_coordinate(result, 'neck')
    
    if neck_coord is None:
        return None
    
    # Try to find hip keypoints
    left_hip = get_keypoint_coordinate(result, 'left_hip')
    right_hip = get_keypoint_coordinate(result, 'right_hip')
    
    if left_hip is None and right_hip is None:
        return None
    
    # Use average of available hips
    if left_hip is not None and right_hip is not None:
        hip_y = (left_hip[1] + right_hip[1]) / 2
    elif left_hip is not None:
        hip_y = left_hip[1]
    else:
        hip_y = right_hip[1]
    
    torso_length = abs(hip_y - neck_coord[1])
    
    return torso_length