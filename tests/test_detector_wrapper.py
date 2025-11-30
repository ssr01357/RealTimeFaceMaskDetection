"""
Unit tests for detector wrapper module
"""

import unittest
import numpy as np
import cv2
import tempfile
import os
from unittest.mock import Mock, patch, MagicMock

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from evaluation.detectors.detector_wrapper import FaceDetectorWrapper, YuNetDetector, HaarDetector


class TestYuNetDetector(unittest.TestCase):
    """Test YuNet detector implementation"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
    @patch('os.path.exists')
    @patch('cv2.FaceDetectorYN_create')
    def test_yunet_detector_initialization(self, mock_create, mock_exists):
        """Test YuNet detector initialization"""
        mock_exists.return_value = True
        mock_detector = Mock()
        mock_create.return_value = mock_detector
        
        detector = YuNetDetector("dummy_model.onnx")
        
        self.assertIsNotNone(detector.detector)
        mock_create.assert_called_once()
    
    @patch('os.path.exists')
    @patch('cv2.FaceDetectorYN_create')
    def test_yunet_detect_faces(self, mock_create, mock_exists):
        """Test YuNet face detection"""
        mock_exists.return_value = True
        # Mock detector and its detect method
        mock_detector = Mock()
        mock_create.return_value = mock_detector
        mock_detector.setInputSize = Mock()
        
        # Mock detection results: [x, y, w, h, score, ...]
        mock_faces = np.array([[100, 100, 50, 60, 0.9, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
        mock_detector.detect.return_value = (1, mock_faces)
        
        detector = YuNetDetector("dummy_model.onnx")
        results = detector.detect(self.test_image)
        
        self.assertEqual(len(results), 1)
        self.assertIn('box', results[0])
        self.assertIn('score', results[0])
        self.assertEqual(results[0]['box'], [100, 100, 150, 160])  # [x1, y1, x2, y2]
        self.assertEqual(results[0]['score'], 0.9)
    
    @patch('os.path.exists')
    @patch('cv2.FaceDetectorYN_create')
    def test_yunet_no_faces_detected(self, mock_create, mock_exists):
        """Test YuNet when no faces are detected"""
        mock_exists.return_value = True
        mock_detector = Mock()
        mock_create.return_value = mock_detector
        mock_detector.setInputSize = Mock()
        mock_detector.detect.return_value = (0, None)
        
        detector = YuNetDetector("dummy_model.onnx")
        results = detector.detect(self.test_image)
        
        self.assertEqual(len(results), 0)
    
    @patch('os.path.exists')
    @patch('cv2.FaceDetectorYN_create')
    def test_yunet_get_info(self, mock_create, mock_exists):
        """Test YuNet detector info"""
        mock_exists.return_value = True
        mock_detector = Mock()
        mock_create.return_value = mock_detector
        
        detector = YuNetDetector("dummy_model.onnx")
        
        # Test that detector was created successfully
        self.assertIsNotNone(detector.detector)
        self.assertEqual(detector.score_threshold, 0.6)  # Default threshold


class TestHaarDetector(unittest.TestCase):
    """Test Haar cascade detector implementation"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
    @patch('os.path.exists')
    @patch('cv2.CascadeClassifier')
    def test_haar_detector_initialization(self, mock_cascade, mock_exists):
        """Test Haar detector initialization"""
        mock_exists.return_value = True
        mock_classifier = Mock()
        mock_cascade.return_value = mock_classifier
        
        detector = HaarDetector("dummy_cascade.xml")
        
        self.assertIsNotNone(detector.detector)
        mock_cascade.assert_called_once_with("dummy_cascade.xml")
    
    @patch('os.path.exists')
    @patch('cv2.CascadeClassifier')
    def test_haar_detect_faces(self, mock_cascade, mock_exists):
        """Test Haar face detection"""
        mock_exists.return_value = True
        mock_classifier = Mock()
        mock_cascade.return_value = mock_classifier
        
        # Mock detection results: [(x, y, w, h), ...]
        mock_faces = np.array([[100, 100, 50, 60], [200, 150, 45, 55]])
        mock_classifier.detectMultiScale.return_value = mock_faces
        
        detector = HaarDetector("dummy_cascade.xml")
        results = detector.detect(self.test_image)
        
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]['box'], [100, 100, 150, 160])
        self.assertEqual(results[1]['box'], [200, 150, 245, 205])
        # Haar doesn't provide confidence scores, so should be 1.0
        self.assertEqual(results[0]['score'], 1.0)
        self.assertEqual(results[1]['score'], 1.0)
    
    @patch('os.path.exists')
    @patch('cv2.CascadeClassifier')
    def test_haar_no_faces_detected(self, mock_cascade, mock_exists):
        """Test Haar when no faces are detected"""
        mock_exists.return_value = True
        mock_classifier = Mock()
        mock_cascade.return_value = mock_classifier
        mock_classifier.detectMultiScale.return_value = np.array([])
        
        detector = HaarDetector("dummy_cascade.xml")
        results = detector.detect(self.test_image)
        
        self.assertEqual(len(results), 0)
    
    @patch('os.path.exists')
    @patch('cv2.CascadeClassifier')
    def test_haar_get_info(self, mock_cascade, mock_exists):
        """Test Haar detector info"""
        mock_exists.return_value = True
        mock_classifier = Mock()
        mock_cascade.return_value = mock_classifier
        
        detector = HaarDetector("dummy_cascade.xml")
        
        # Test that detector was created successfully
        self.assertIsNotNone(detector.detector)
        self.assertEqual(detector.scale_factor, 1.1)  # Default scale factor
        self.assertEqual(detector.min_neighbors, 5)   # Default min neighbors


class TestFaceDetectorWrapper(unittest.TestCase):
    """Test face detector wrapper"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    @patch('os.path.exists')
    @patch('cv2.FaceDetectorYN_create')
    def test_wrapper_yunet_creation(self, mock_create, mock_exists):
        """Test wrapper creates YuNet detector correctly"""
        mock_exists.return_value = True
        mock_detector = Mock()
        mock_create.return_value = mock_detector
        
        wrapper = FaceDetectorWrapper('yunet', model_path='dummy.onnx')
        
        self.assertIsInstance(wrapper.detector, YuNetDetector)
    
    @patch('os.path.exists')
    @patch('cv2.CascadeClassifier')
    def test_wrapper_haar_creation(self, mock_cascade, mock_exists):
        """Test wrapper creates Haar detector correctly"""
        mock_exists.return_value = True
        mock_classifier = Mock()
        mock_cascade.return_value = mock_classifier
        
        wrapper = FaceDetectorWrapper('haar', cascade_path='dummy.xml')
        
        self.assertIsInstance(wrapper.detector, HaarDetector)
    
    def test_wrapper_invalid_detector_type(self):
        """Test wrapper raises error for invalid detector type"""
        with self.assertRaises(ValueError):
            FaceDetectorWrapper('invalid_detector')
    
    @patch('os.path.exists')
    @patch('cv2.FaceDetectorYN_create')
    def test_wrapper_detect_method(self, mock_create, mock_exists):
        """Test wrapper detect method"""
        mock_exists.return_value = True
        mock_detector = Mock()
        mock_create.return_value = mock_detector
        mock_detector.setInputSize = Mock()
        mock_faces = np.array([[100, 100, 50, 60, 0.8, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
        mock_detector.detect.return_value = (1, mock_faces)
        
        wrapper = FaceDetectorWrapper('yunet', model_path='dummy.onnx')
        results = wrapper.detect(self.test_image)
        
        self.assertEqual(len(results), 1)
        self.assertIn('box', results[0])
        self.assertIn('score', results[0])
    
    @patch('os.path.exists')
    @patch('cv2.FaceDetectorYN_create')
    def test_wrapper_get_detector_info(self, mock_create, mock_exists):
        """Test wrapper get_detector_info method"""
        mock_exists.return_value = True
        mock_detector = Mock()
        mock_create.return_value = mock_detector
        
        wrapper = FaceDetectorWrapper('yunet', model_path='dummy.onnx')
        info = wrapper.get_detector_info()
        
        self.assertIn('type', info)


if __name__ == '__main__':
    unittest.main()
