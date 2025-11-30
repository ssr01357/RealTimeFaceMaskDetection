"""
Unified wrapper for different face detectors (YuNet, Haar, etc.)
Provides a consistent interface for evaluation.
"""

import os
import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from abc import ABC, abstractmethod


class BaseDetector(ABC):
    """Abstract base class for face detectors"""
    
    @abstractmethod
    def detect(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect faces in image
        
        Args:
            image: BGR image as numpy array
            
        Returns:
            List of detections, each containing:
            - 'box': [x1, y1, x2, y2] 
            - 'score': confidence score (0-1)
        """
        pass


class YuNetDetector(BaseDetector):
    """YuNet face detector wrapper"""
    
    def __init__(self, model_path: str, score_threshold: float = 0.6, 
                 nms_threshold: float = 0.3, top_k: int = 5000):
        """
        Initialize YuNet detector
        
        Args:
            model_path: Path to YuNet ONNX model
            score_threshold: Minimum confidence threshold
            nms_threshold: NMS threshold
            top_k: Maximum number of detections
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"YuNet model not found: {model_path}")
            
        self.detector = cv2.FaceDetectorYN_create(
            model_path, "", (320, 320),
            score_threshold=score_threshold,
            nms_threshold=nms_threshold,
            top_k=top_k
        )
        self.score_threshold = score_threshold
        
    def detect(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """Detect faces using YuNet"""
        h, w = image.shape[:2]
        self.detector.setInputSize((w, h))
        
        result = self.detector.detect(image)
        detections = []
        
        if result[1] is not None:
            for det in result[1]:
                x, y, w_box, h_box, score = det[:5]
                if score >= self.score_threshold:
                    detections.append({
                        'box': [int(x), int(y), int(x + w_box), int(y + h_box)],
                        'score': float(score)
                    })
        
        return detections


class HaarDetector(BaseDetector):
    """Haar cascade face detector wrapper"""
    
    def __init__(self, cascade_path: str, scale_factor: float = 1.1, 
                 min_neighbors: int = 5, min_size: Tuple[int, int] = (60, 60)):
        """
        Initialize Haar cascade detector
        
        Args:
            cascade_path: Path to Haar cascade XML file
            scale_factor: Scale factor for multi-scale detection
            min_neighbors: Minimum neighbors for detection
            min_size: Minimum face size (width, height)
        """
        if not os.path.exists(cascade_path):
            raise FileNotFoundError(f"Haar cascade not found: {cascade_path}")
            
        self.detector = cv2.CascadeClassifier(cascade_path)
        self.scale_factor = scale_factor
        self.min_neighbors = min_neighbors
        self.min_size = min_size
        
    def detect(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """Detect faces using Haar cascade"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        faces = self.detector.detectMultiScale(
            gray,
            scaleFactor=self.scale_factor,
            minNeighbors=self.min_neighbors,
            minSize=self.min_size
        )
        
        detections = []
        for (x, y, w, h) in faces:
            detections.append({
                'box': [int(x), int(y), int(x + w), int(y + h)],
                'score': 1.0  # Haar doesn't provide confidence scores
            })
            
        return detections


class FaceDetectorWrapper:
    """
    Unified wrapper that can use different face detectors
    """
    
    def __init__(self, detector_type: str, **kwargs):
        """
        Initialize face detector wrapper
        
        Args:
            detector_type: Type of detector ('yunet', 'haar')
            **kwargs: Detector-specific arguments
        """
        self.detector_type = detector_type.lower()
        
        if self.detector_type == 'yunet':
            self.detector = YuNetDetector(**kwargs)
        elif self.detector_type == 'haar':
            self.detector = HaarDetector(**kwargs)
        else:
            raise ValueError(f"Unsupported detector type: {detector_type}")
    
    def __call__(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect faces in image
        
        Args:
            image: BGR image as numpy array
            
        Returns:
            List of detections with 'box' and 'score' keys
        """
        return self.detector.detect(image)
    
    def detect(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """Alias for __call__ method"""
        return self(image)
    
    @staticmethod
    def create_yunet_detector(model_path: str, score_threshold: float = 0.6) -> 'FaceDetectorWrapper':
        """
        Factory method to create YuNet detector
        
        Args:
            model_path: Path to YuNet ONNX model
            score_threshold: Minimum confidence threshold
            
        Returns:
            FaceDetectorWrapper instance
        """
        return FaceDetectorWrapper('yunet', model_path=model_path, score_threshold=score_threshold)
    
    @staticmethod
    def create_haar_detector(cascade_path: str) -> 'FaceDetectorWrapper':
        """
        Factory method to create Haar detector
        
        Args:
            cascade_path: Path to Haar cascade XML file
            
        Returns:
            FaceDetectorWrapper instance
        """
        return FaceDetectorWrapper('haar', cascade_path=cascade_path)
    
    def get_detector_info(self) -> Dict[str, Any]:
        """Get information about the current detector"""
        info = {'type': self.detector_type}
        
        if self.detector_type == 'yunet':
            info.update({
                'score_threshold': self.detector.score_threshold,
                'model_type': 'YuNet ONNX'
            })
        elif self.detector_type == 'haar':
            info.update({
                'scale_factor': self.detector.scale_factor,
                'min_neighbors': self.detector.min_neighbors,
                'min_size': self.detector.min_size,
                'model_type': 'Haar Cascade'
            })
            
        return info
