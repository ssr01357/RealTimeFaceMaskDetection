"""
Detection-only filter for face detection without classification
"""

import cv2
import numpy as np
import sys
import os
from typing import List, Dict, Any, Optional, Tuple

# Add yewon_pipeline to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'yewon_pipeline'))

try:
    from detectors_2 import build_detector, BaseFaceDetector, FaceBox
except ImportError:
    print("Warning: Could not import yewon_pipeline detectors")
    BaseFaceDetector = None
    build_detector = None


class DetectionOnlyFilter:
    """
    Filter that performs face detection only, without classification
    """
    
    def __init__(self, 
                 detector_name: str = 'haar',
                 device: str = 'cuda',
                 yunet_onnx: str = None,
                 haar_xml: str = None,
                 retina_thresh: float = 0.8,
                 confidence_threshold: float = 0.6):
        """
        Initialize detection-only filter
        
        Args:
            detector_name: Name of detector ('yunet', 'haar', 'mtcnn', 'retinaface')
            device: Device for computation ('cuda' or 'cpu')
            yunet_onnx: Path to YuNet ONNX model
            haar_xml: Path to Haar cascade XML
            retina_thresh: RetinaFace threshold
            confidence_threshold: Minimum confidence for detections
        """
        self.detector_name = detector_name
        self.device = device
        self.confidence_threshold = confidence_threshold
        
        # Set default model paths
        if yunet_onnx is None:
            yunet_onnx = os.path.join(os.path.dirname(__file__), '..', '..', 
                                    'yewon_pipeline', 'face_detection_yunet_2023mar.onnx')
        if haar_xml is None:
            haar_xml = os.path.join(os.path.dirname(__file__), '..', '..', 
                                  'yewon_pipeline', 'haarcascade_frontalface_default.xml')
        
        self.yunet_onnx = yunet_onnx
        self.haar_xml = haar_xml
        self.retina_thresh = retina_thresh
        
        # Initialize detector
        self.detector = self._setup_detector()
        
        # Color scheme for different confidence levels
        self.colors = {
            'high': (0, 255, 0),      # Green for high confidence (>0.8)
            'medium': (0, 255, 255),  # Yellow for medium confidence (0.6-0.8)
            'low': (0, 165, 255),     # Orange for low confidence (<0.6)
            'default': (255, 255, 0)  # Cyan for no confidence score
        }
        
        # Performance tracking
        self.detection_count = 0
        self.total_detections = 0
    
    def _setup_detector(self) -> Optional[BaseFaceDetector]:
        """Setup face detector based on configuration"""
        if build_detector is None:
            print("Warning: yewon_pipeline detectors not available, using fallback")
            return self._setup_fallback_detector()
        
        try:
            detector = build_detector(
                name=self.detector_name,
                device=self.device,
                yunet_onnx=self.yunet_onnx,
                haar_xml=self.haar_xml,
                retina_thresh=self.retina_thresh
            )
            print(f"Initialized {self.detector_name} detector successfully")
            return detector
        except Exception as e:
            print(f"Error initializing {self.detector_name} detector: {e}")
            print("Falling back to basic Haar cascade")
            return self._setup_fallback_detector()
    
    def _setup_fallback_detector(self):
        """Setup basic OpenCV Haar cascade as fallback"""
        try:
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            return cv2.CascadeClassifier(cascade_path)
        except Exception as e:
            print(f"Error setting up fallback detector: {e}")
            return None
    
    def detect_faces(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect faces in frame
        
        Args:
            frame: BGR image frame
            
        Returns:
            List of detection dictionaries with keys: 'box', 'score', 'confidence_level'
        """
        if self.detector is None:
            return []
        
        detections = []
        
        try:
            if hasattr(self.detector, 'detect'):
                # Using yewon_pipeline detector
                face_boxes = self.detector.detect(frame)
                
                for face_box in face_boxes:
                    x, y, w, h, score = face_box
                    
                    # Filter by confidence threshold
                    if score < self.confidence_threshold:
                        continue
                    
                    # Convert to x1, y1, x2, y2 format
                    box = [int(x), int(y), int(x + w), int(y + h)]
                    
                    # Determine confidence level for coloring
                    if score >= 0.8:
                        confidence_level = 'high'
                    elif score >= 0.6:
                        confidence_level = 'medium'
                    else:
                        confidence_level = 'low'
                    
                    detections.append({
                        'box': box,
                        'score': float(score),
                        'confidence_level': confidence_level
                    })
            
            else:
                # Using fallback Haar cascade
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = self.detector.detectMultiScale(
                    gray,
                    scaleFactor=1.1,
                    minNeighbors=5,
                    minSize=(60, 60)
                )
                
                for (x, y, w, h) in faces:
                    box = [int(x), int(y), int(x + w), int(y + h)]
                    detections.append({
                        'box': box,
                        'score': 1.0,  # Haar doesn't provide confidence
                        'confidence_level': 'default'
                    })
        
        except Exception as e:
            print(f"Error in face detection: {e}")
            return []
        
        # Update statistics
        self.detection_count = len(detections)
        self.total_detections += self.detection_count
        
        return detections
    
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Process frame with detection-only filter
        
        Args:
            frame: Input BGR frame
            
        Returns:
            Annotated frame with detection boxes
        """
        # Create a copy to avoid modifying original
        output_frame = frame.copy()
        
        # Detect faces
        detections = self.detect_faces(frame)
        
        # Draw detections
        for detection in detections:
            box = detection['box']
            score = detection['score']
            confidence_level = detection['confidence_level']
            
            x1, y1, x2, y2 = box
            color = self.colors[confidence_level]
            
            # Draw bounding box
            cv2.rectangle(output_frame, (x1, y1), (x2, y2), color, 2)
            
            # Prepare label text
            if confidence_level == 'default':
                label_text = "Face Detected"
            else:
                label_text = f"Face: {score:.2f}"
            
            # Calculate text size and background
            (text_width, text_height), baseline = cv2.getTextSize(
                label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            )
            
            # Draw background rectangle for text
            cv2.rectangle(
                output_frame,
                (x1, y1 - text_height - 10),
                (x1 + text_width, y1),
                color,
                -1
            )
            
            # Draw text
            cv2.putText(
                output_frame,
                label_text,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )
        
        # Add filter info
        self._add_filter_info(output_frame)
        
        return output_frame
    
    def _add_filter_info(self, frame: np.ndarray):
        """Add filter information to frame"""
        # Filter title
        cv2.putText(
            frame,
            f"Detection Only ({self.detector_name.upper()})",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )
        
        # Detection count
        cv2.putText(
            frame,
            f"Faces: {self.detection_count}",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )
        
        # Legend
        legend_y = frame.shape[0] - 80
        cv2.putText(
            frame,
            "Legend:",
            (10, legend_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1
        )
        
        # Color legend
        legend_items = [
            ("High (>0.8)", self.colors['high']),
            ("Med (0.6-0.8)", self.colors['medium']),
            ("Low (<0.6)", self.colors['low'])
        ]
        
        for i, (text, color) in enumerate(legend_items):
            y_pos = legend_y + 20 + (i * 15)
            cv2.rectangle(frame, (10, y_pos - 10), (25, y_pos), color, -1)
            cv2.putText(
                frame,
                text,
                (30, y_pos - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (255, 255, 255),
                1
            )
    
    def get_stats(self) -> Dict[str, Any]:
        """Get filter statistics"""
        return {
            'detector_name': self.detector_name,
            'current_detections': self.detection_count,
            'total_detections': self.total_detections,
            'confidence_threshold': self.confidence_threshold
        }
    
    def switch_detector(self, new_detector_name: str) -> bool:
        """
        Switch to a different detector
        
        Args:
            new_detector_name: Name of new detector
            
        Returns:
            True if switch successful, False otherwise
        """
        try:
            old_detector = self.detector_name
            self.detector_name = new_detector_name
            self.detector = self._setup_detector()
            
            if self.detector is not None:
                print(f"Switched from {old_detector} to {new_detector_name}")
                return True
            else:
                # Revert on failure
                self.detector_name = old_detector
                self.detector = self._setup_detector()
                print(f"Failed to switch to {new_detector_name}, reverted to {old_detector}")
                return False
                
        except Exception as e:
            print(f"Error switching detector: {e}")
            return False
