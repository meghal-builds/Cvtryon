"""Tests for Processing Pipeline"""

import pytest
import numpy as np

from src.model_layer import UNetSegmentationModel, MediaPipePoseModel
from src.segmentation import segment_body, validate_segmentation_quality
from src.pose_detection import detect_pose, validate_pose_quality
from src.measurement_inference import (
    infer_measurements,
    calculate_measurement_confidence,
    validate_measurements,
)


class TestSegmentation:
    """Test segmentation processing"""
    
    def test_segment_body_valid_image(self):
        """Test segmentation on valid image"""
        model = UNetSegmentationModel()
        image = np.ones((512, 512, 3), dtype=np.uint8) * 128
        
        result = segment_body(image, model)
        
        assert result is not None
        assert result.mask is not None
        assert result.torso_percentage > 0
    
    def test_segment_body_invalid_input(self):
        """Test segmentation with invalid input"""
        model = UNetSegmentationModel()
        
        with pytest.raises(TypeError):
            segment_body([1, 2, 3], model)
    
    def test_validate_segmentation_quality(self):
        """Test segmentation validation"""
        model = UNetSegmentationModel()
        image = np.ones((512, 512, 3), dtype=np.uint8) * 128
        
        result = model.predict(image)
        is_valid, warnings = validate_segmentation_quality(result)
        
        assert isinstance(is_valid, bool)
        assert isinstance(warnings, list)


class TestPoseDetection:
    """Test pose detection"""
    
    def test_detect_pose_valid_image(self):
        """Test pose detection on valid image"""
        model = MediaPipePoseModel()
        image = np.ones((512, 512, 3), dtype=np.uint8) * 128
        
        result = detect_pose(image, model)
        
        assert result is not None
        assert len(result.keypoints) > 0
        assert result.shoulder_width_px > 0
    
    def test_detect_pose_invalid_input(self):
        """Test pose detection with invalid input"""
        model = MediaPipePoseModel()
        
        with pytest.raises(TypeError):
            detect_pose([1, 2, 3], model)
    
    def test_validate_pose_quality(self):
        """Test pose validation"""
        model = MediaPipePoseModel()
        image = np.ones((512, 512, 3), dtype=np.uint8) * 128
        
        result = model.predict(image)
        is_frontal, warnings = validate_pose_quality(result)
        
        assert isinstance(is_frontal, bool)
        assert isinstance(warnings, list)


class TestMeasurementInference:
    """Test measurement inference"""
    
    def test_infer_measurements(self):
        """Test measurement inference"""
        seg_model = UNetSegmentationModel()
        pose_model = MediaPipePoseModel()
        
        image = np.ones((512, 512, 3), dtype=np.uint8) * 128
        
        seg_result = seg_model.predict(image)
        pose_result = pose_model.predict(image)
        
        measurements = infer_measurements(pose_result, seg_result)
        
        assert measurements is not None
        assert measurements.shoulder_width_cm > 0
        assert measurements.confidence >= 0
    
    def test_calculate_confidence(self):
        """Test confidence calculation"""
        seg_model = UNetSegmentationModel()
        pose_model = MediaPipePoseModel()
        
        image = np.ones((512, 512, 3), dtype=np.uint8) * 128
        
        seg_result = seg_model.predict(image)
        pose_result = pose_model.predict(image)
        
        confidence = calculate_measurement_confidence(pose_result, seg_result)
        
        assert 0 <= confidence <= 1
    
    def test_validate_measurements_valid(self):
        """Test measurement validation with valid measurements"""
        seg_model = UNetSegmentationModel()
        pose_model = MediaPipePoseModel()
        
        image = np.ones((512, 512, 3), dtype=np.uint8) * 128
        
        seg_result = seg_model.predict(image)
        pose_result = pose_model.predict(image)
        
        measurements = infer_measurements(pose_result, seg_result)
        is_valid, errors = validate_measurements(measurements)
        
        assert isinstance(is_valid, bool)
        assert isinstance(errors, list)