#!/usr/bin/env python3
"""
Unified Face Mask Detection System
Single entry point for all face detectors and mask classifiers

Supports:
- Face Detectors: Haar, YuNet, MTCNN, RetinaFace
- Classifiers: Custom PyTorch, Yewon pipeline, sklearn models, or detection-only
- Display Modes: Single view, dual comparison, overlay, difference
- Runtime switching between any detector/classifier combination

Usage:
    python face_mask_detector.py --detector haar --classifier custom_pytorch
    python face_mask_detector.py --detector yunet --classifier yewon
    python face_mask_detector.py --detector mtcnn  # detection only
"""

import cv2
import numpy as np
import time
import argparse
import os
import sys
from typing import Optional, Dict, Any, Tuple, List
from enum import Enum

# Add paths for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'evaluation'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'dual_filter_system'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'model_artifacts'))

try:
    from evaluation.detectors.detector_wrapper import FaceDetectorWrapper
    from evaluation.classifiers.classifier_wrapper import FaceClassifierWrapper
except ImportError:
    print("Warning: Could not import evaluation modules")
    FaceDetectorWrapper = None
    FaceClassifierWrapper = None

# Import custom PyTorch model loader directly
try:
    from model_artifacts.pytorch_model_loader import load_custom_pytorch_model
    CUSTOM_PYTORCH_AVAILABLE = True
except ImportError:
    print("Warning: Could not import custom PyTorch model loader")
    CUSTOM_PYTORCH_AVAILABLE = False

try:
    from dual_filter_system.ui.controls import ControlsManager, ControlAction
    from dual_filter_system.filters.filter_display import FilterDisplay
except ImportError:
    print("Warning: Could not import dual filter system modules")
    ControlsManager = None
    FilterDisplay = None


class DisplayMode(Enum):
    """Display mode options"""
    SINGLE = "single"
    DUAL_SIDE_BY_SIDE = "dual_side_by_side"
    OVERLAY = "overlay"
    DIFFERENCE = "difference"
    DETECTION_ONLY = "detection_only"
    CLASSIFICATION_ONLY = "classification_only"


class DetectionResult:
    """Container for detection results"""
    def __init__(self, box: List[int], score: float, label: Optional[str] = None, confidence: Optional[float] = None):
        self.box = box  # [x1, y1, x2, y2]
        self.score = score
        self.label = label
        self.confidence = confidence


class UnifiedFaceMaskDetector:
    """
    Unified face mask detection system supporting all detectors and classifiers
    """
    
    def __init__(self,
                 detector_name: str = 'haar',
                 classifier_type: Optional[str] = None,
                 classifier_model_path: Optional[str] = None,
                 camera_index: int = 0,
                 confidence_threshold: float = 0.6,
                 device: str = 'cpu',
                 frame_width: int = 640,
                 frame_height: int = 480):
        """
        Initialize unified face mask detector
        
        Args:
            detector_name: Face detector type ('haar', 'yunet', 'mtcnn', 'retinaface')
            classifier_type: Classifier type ('custom_pytorch', 'yewon', 'sklearn', None)
            classifier_model_path: Path to classifier model file
            camera_index: Camera device index
            confidence_threshold: Detection confidence threshold
            device: Computation device ('cuda' or 'cpu')
            frame_width: Width of video frames
            frame_height: Height of video frames
        """
        self.detector_name = detector_name
        self.classifier_type = classifier_type
        self.classifier_model_path = classifier_model_path
        self.camera_index = camera_index
        self.confidence_threshold = confidence_threshold
        self.device = device
        self.frame_width = frame_width
        self.frame_height = frame_height
        
        # Initialize detector
        self.face_detector = self._create_face_detector()
        
        # Initialize classifier (optional)
        self.mask_classifier = None
        self.classifier_enabled = False
        if classifier_type and classifier_model_path:
            self.mask_classifier = self._create_mask_classifier()
            self.classifier_enabled = True
        
        # Display and UI - Use proper FilterDisplay
        self.display_mode = DisplayMode.DUAL_SIDE_BY_SIDE  # Default to dual filter mode
        if FilterDisplay:
            self.display = FilterDisplay(
                window_name="Unified Face Mask Detection System",
                frame_width=frame_width, 
                frame_height=frame_height,
                show_fps=True,
                show_stats=True
            )
        else:
            self.display = None
        
        if ControlsManager:
            self.controls = ControlsManager()
            self._setup_controls()
        else:
            self.controls = None
        
        # Camera and state
        self.cap = None
        self.is_running = False
        
        # Performance tracking
        self.frame_count = 0
        self.start_time = None
        self.fps = 0.0
        self.total_detections = 0
        self.current_detections = 0
        self.current_classifications = 0
        self.class_distribution = {'with_mask': 0, 'without_mask': 0, 'unknown': 0}
        
        # Available detectors and classifiers for cycling
        self.available_detectors = ['haar', 'yunet', 'mtcnn', 'retinaface']
        self.available_classifiers = ['none', 'custom_pytorch', 'yewon', 'sklearn']
        
        # Current indices for cycling
        self.current_detector_index = 0
        self.current_classifier_index = 0
        
        # Set initial indices based on current settings
        if detector_name in self.available_detectors:
            self.current_detector_index = self.available_detectors.index(detector_name)
        
        if classifier_type in self.available_classifiers:
            self.current_classifier_index = self.available_classifiers.index(classifier_type)
        elif classifier_type is None:
            self.current_classifier_index = 0  # 'none'
        
        # Colors for visualization
        self.colors = {
            'with_mask': (0, 255, 0),      # Green
            'without_mask': (0, 0, 255),   # Red
            'incorrect_mask': (0, 165, 255), # Orange
            'unknown': (255, 255, 0),      # Yellow
            'face_detected': (255, 0, 255) # Magenta
        }
        
        print(f"Unified Face Mask Detector initialized:")
        print(f"  Detector: {detector_name}")
        print(f"  Classifier: {classifier_type if classifier_type else 'None (detection only)'}")
        print(f"  Device: {device}")
        print(f"  Camera: {camera_index}")
    
    def _create_face_detector(self):
        """Create face detector based on type"""
        if not FaceDetectorWrapper:
            return self._create_fallback_detector()
        
        try:
            if self.detector_name.lower() == 'haar':
                return FaceDetectorWrapper.create_haar_detector()
            elif self.detector_name.lower() == 'yunet':
                # Look for YuNet model in common locations
                yunet_paths = [
                    'yewon_pipeline/face_detection_yunet_2023mar.onnx',
                    'models/face_detection_yunet_2023mar.onnx',
                    'face_detection_yunet_2023mar.onnx'
                ]
                for path in yunet_paths:
                    if os.path.exists(path):
                        return FaceDetectorWrapper.create_yunet_detector(
                            model_path=path,
                            score_threshold=self.confidence_threshold
                        )
                print(f"YuNet model not found, falling back to Haar")
                return FaceDetectorWrapper.create_haar_detector()
            elif self.detector_name.lower() == 'mtcnn':
                return FaceDetectorWrapper.create_mtcnn_detector(device=self.device)
            elif self.detector_name.lower() == 'retinaface':
                return FaceDetectorWrapper.create_retinaface_detector(device=self.device)
            else:
                print(f"Unknown detector type: {self.detector_name}, using Haar")
                return FaceDetectorWrapper.create_haar_detector()
        except Exception as e:
            print(f"Error creating detector: {e}, falling back to basic Haar")
            return self._create_fallback_detector()
    
    def _create_fallback_detector(self):
        """Create basic OpenCV Haar cascade as fallback"""
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        return cv2.CascadeClassifier(cascade_path)
    
    def _create_mask_classifier(self):
        """Create mask classifier based on type"""
        try:
            if self.classifier_type == 'custom_pytorch':
                # Use direct custom PyTorch model loader if available
                if CUSTOM_PYTORCH_AVAILABLE:
                    print(f"Loading custom PyTorch model directly from: {self.classifier_model_path}")
                    return load_custom_pytorch_model(self.classifier_model_path, self.device)
                elif FaceClassifierWrapper:
                    return FaceClassifierWrapper.create_custom_pytorch_classifier(
                        model_path=self.classifier_model_path,
                        device=self.device,
                        class_names=['without_mask', 'with_mask']
                    )
                else:
                    print("Neither custom PyTorch loader nor FaceClassifierWrapper available")
                    return None
            
            elif self.classifier_type == 'yewon':
                if FaceClassifierWrapper:
                    return FaceClassifierWrapper(
                        classifier_type='pytorch',
                        model_path=self.classifier_model_path,
                        class_names=['without_mask', 'with_mask', 'incorrect_mask']
                    )
                else:
                    print("FaceClassifierWrapper not available for yewon classifier")
                    return None
            
            elif self.classifier_type == 'sklearn':
                if FaceClassifierWrapper:
                    return FaceClassifierWrapper(
                        classifier_type='sklearn',
                        model_path=self.classifier_model_path,
                        class_names=['without_mask', 'with_mask']
                    )
                else:
                    print("FaceClassifierWrapper not available for sklearn classifier")
                    return None
            
            else:
                print(f"Unknown classifier type: {self.classifier_type}")
                return None
                
        except Exception as e:
            print(f"Error creating classifier: {e}")
            return None
    
    def _setup_controls(self):
        """Setup keyboard controls"""
        if not self.controls:
            return
        
        # Register control callbacks
        self.controls.register_callback(ControlAction.QUIT, self._quit)
        self.controls.register_callback(ControlAction.SCREENSHOT, self._take_screenshot)
        self.controls.register_callback(ControlAction.SWITCH_DETECTOR_HAAR, 
                                      lambda: self._switch_detector('haar'))
        self.controls.register_callback(ControlAction.SWITCH_DETECTOR_YUNET, 
                                      lambda: self._switch_detector('yunet'))
        self.controls.register_callback(ControlAction.SWITCH_DETECTOR_MTCNN, 
                                      lambda: self._switch_detector('mtcnn'))
        self.controls.register_callback(ControlAction.SWITCH_DETECTOR_RETINAFACE, 
                                      lambda: self._switch_detector('retinaface'))
        self.controls.register_callback(ControlAction.TOGGLE_CLASSIFIER, self._toggle_classifier)
        self.controls.register_callback(ControlAction.SWITCH_CLASSIFIER_CUSTOM, 
                                      lambda: self._switch_classifier('custom_pytorch'))
        self.controls.register_callback(ControlAction.SWITCH_CLASSIFIER_YEWON, 
                                      lambda: self._switch_classifier('yewon'))
        self.controls.register_callback(ControlAction.SWITCH_CLASSIFIER_NONE, 
                                      lambda: self._switch_classifier('none'))
        self.controls.register_callback(ControlAction.TOGGLE_OVERLAY, self._toggle_display_mode)
        self.controls.register_callback(ControlAction.INCREASE_CONFIDENCE, self._increase_confidence)
        self.controls.register_callback(ControlAction.DECREASE_CONFIDENCE, self._decrease_confidence)
        self.controls.register_callback(ControlAction.RESET_STATS, self._reset_stats)
        
        # Note: D/C cycling is handled directly in the main loop
    
    def _quit(self):
        """Quit application"""
        self.is_running = False
    
    def _take_screenshot(self):
        """Take screenshot of current display"""
        if hasattr(self, '_current_frame') and self._current_frame is not None:
            if self.display:
                filename = self.display.save_screenshot(self._current_frame)
                print(f"Screenshot saved: {filename}")
            else:
                timestamp = int(time.time())
                filename = f"face_mask_detection_screenshot_{timestamp}.jpg"
                cv2.imwrite(filename, self._current_frame)
                print(f"Screenshot saved: {filename}")
    
    def _switch_detector(self, detector_name: str):
        """Switch to different detector"""
        print(f"Switching to {detector_name} detector...")
        self.detector_name = detector_name
        try:
            self.face_detector = self._create_face_detector()
            print(f"Successfully switched to {detector_name}")
        except Exception as e:
            print(f"Failed to switch to {detector_name}: {e}")
    
    def _switch_classifier(self, classifier_type: str):
        """Switch to different classifier"""
        if classifier_type == 'none':
            self.classifier_enabled = False
            self.mask_classifier = None
            self.classifier_type = None
            print("Classifier disabled")
            return
        
        print(f"Switching to {classifier_type} classifier...")
        
        # Set default model paths with better fallback options
        model_paths = {
            'custom_pytorch': [
                'model_artifacts/best_pytorch_model_custom.pth',
                'models/best_pytorch_model_custom.pth',
                'best_pytorch_model_custom.pth'
            ],
            'yewon': [
                'yewon_pipeline/best_model.pth',
                'models/yewon_best_model.pth',
                'best_model.pth'
            ],
            'sklearn': [
                'models/sklearn_classifier.pkl',
                'sklearn_classifier.pkl'
            ]
        }
        
        model_path = None
        possible_paths = model_paths.get(classifier_type, [])
        
        for path in possible_paths:
            if os.path.exists(path):
                model_path = path
                break
        
        if not model_path:
            print(f"Model file not found for {classifier_type}. Tried: {possible_paths}")
            print("Disabling classifier...")
            self.classifier_enabled = False
            self.mask_classifier = None
            self.classifier_type = None
            return
        
        self.classifier_type = classifier_type
        self.classifier_model_path = model_path
        
        try:
            self.mask_classifier = self._create_mask_classifier()
            self.classifier_enabled = True
            print(f"Successfully switched to {classifier_type} using {model_path}")
        except Exception as e:
            print(f"Failed to switch to {classifier_type}: {e}")
            self.classifier_enabled = False
            self.mask_classifier = None
            self.classifier_type = None
    
    def _toggle_classifier(self):
        """Toggle classifier on/off"""
        self.classifier_enabled = not self.classifier_enabled
        status = "enabled" if self.classifier_enabled else "disabled"
        print(f"Classifier {status}")
    
    def _toggle_display_mode(self):
        """Toggle display mode"""
        modes = list(DisplayMode)
        current_idx = modes.index(self.display_mode)
        self.display_mode = modes[(current_idx + 1) % len(modes)]
        print(f"Display mode: {self.display_mode.value}")
    
    def _increase_confidence(self):
        """Increase confidence threshold"""
        self.confidence_threshold = min(1.0, self.confidence_threshold + 0.05)
        print(f"Confidence threshold: {self.confidence_threshold:.2f}")
    
    def _decrease_confidence(self):
        """Decrease confidence threshold"""
        self.confidence_threshold = max(0.1, self.confidence_threshold - 0.05)
        print(f"Confidence threshold: {self.confidence_threshold:.2f}")
    
    def _reset_stats(self):
        """Reset statistics"""
        self.total_detections = 0
        self.class_distribution = {'with_mask': 0, 'without_mask': 0, 'unknown': 0}
        self.frame_count = 0
        self.start_time = time.time()
        print("Statistics reset")
    
    def _cycle_detector(self):
        """Cycle to next detector"""
        self.current_detector_index = (self.current_detector_index + 1) % len(self.available_detectors)
        new_detector = self.available_detectors[self.current_detector_index]
        self._switch_detector(new_detector)
        print(f"Cycled to detector: {new_detector} ({self.current_detector_index + 1}/{len(self.available_detectors)})")
    
    def _cycle_classifier(self):
        """Cycle to next classifier"""
        self.current_classifier_index = (self.current_classifier_index + 1) % len(self.available_classifiers)
        new_classifier = self.available_classifiers[self.current_classifier_index]
        self._switch_classifier(new_classifier)
        print(f"Cycled to classifier: {new_classifier} ({self.current_classifier_index + 1}/{len(self.available_classifiers)})")
    
    def detect_faces(self, frame: np.ndarray) -> List[DetectionResult]:
        """Detect faces in frame"""
        detections = []
        
        try:
            if hasattr(self.face_detector, 'detect'):
                # Using FaceDetectorWrapper
                results = self.face_detector.detect(frame)
                for result in results:
                    if result['score'] >= self.confidence_threshold:
                        detections.append(DetectionResult(
                            box=result['box'],
                            score=result['score']
                        ))
            else:
                # Using basic Haar cascade
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = self.face_detector.detectMultiScale(
                    gray,
                    scaleFactor=1.1,
                    minNeighbors=5,
                    minSize=(60, 60)
                )
                
                for (x, y, w, h) in faces:
                    detections.append(DetectionResult(
                        box=[x, y, x + w, y + h],
                        score=1.0
                    ))
        except Exception as e:
            print(f"Error in face detection: {e}")
        
        return detections
    
    def classify_mask(self, face_roi: np.ndarray) -> Tuple[str, float]:
        """Classify mask wearing status for face ROI"""
        if not self.classifier_enabled or self.mask_classifier is None:
            return 'face_detected', 0.8
        
        try:
            # Handle direct custom PyTorch classifier (from pytorch_model_loader)
            if hasattr(self.mask_classifier, 'predict') and hasattr(self.mask_classifier, 'class_names'):
                prediction = self.mask_classifier.predict(face_roi)
                
                if isinstance(prediction, dict):
                    label = prediction.get('class', 'unknown')
                    confidence = prediction.get('confidence', 0.0)
                    return label, confidence
            
            # Handle FaceClassifierWrapper
            elif hasattr(self.mask_classifier, 'predict'):
                prediction = self.mask_classifier.predict(face_roi)
                
                if isinstance(prediction, dict):
                    label = prediction.get('class', 'unknown')
                    confidence = prediction.get('confidence', 0.0)
                elif isinstance(prediction, tuple) and len(prediction) == 2:
                    class_idx, probabilities = prediction
                    if hasattr(self.mask_classifier, 'get_class_names'):
                        class_names = self.mask_classifier.get_class_names()
                        label = class_names[class_idx] if class_idx < len(class_names) else 'unknown'
                    else:
                        # Default class names for custom PyTorch
                        class_names = ['with_mask', 'without_mask']
                        label = class_names[class_idx] if class_idx < len(class_names) else 'unknown'
                    confidence = float(probabilities[class_idx]) if hasattr(probabilities, '__getitem__') else 0.8
                else:
                    label = str(prediction)
                    confidence = 0.8
                
                return label, confidence
            
            else:
                print("Unknown classifier interface")
                return 'unknown', 0.0
            
        except Exception as e:
            print(f"Error in mask classification: {e}")
            return 'unknown', 0.0
    
    def process_detection_only_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Process frame for detection only (left side of dual filter)"""
        detection_frame = frame.copy()
        
        # Detect faces
        detections = self.detect_faces(detection_frame)
        self.current_detections = len(detections)
        
        # Draw detection boxes only (no classification) - Clean minimal style
        for detection in detections:
            x1, y1, x2, y2 = detection.box
            
            # Draw bounding box in blue for detection only
            cv2.rectangle(detection_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            
            # Draw detection score - smaller, cleaner text
            label_text = f"Face: {detection.score:.2f}"
            
            # Calculate text size and background
            (text_width, text_height), baseline = cv2.getTextSize(
                label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1
            )
            
            # Draw background rectangle for text
            cv2.rectangle(
                detection_frame,
                (x1, y1 - text_height - 6),
                (x1 + text_width, y1),
                (255, 0, 0),
                -1
            )
            
            # Draw text
            cv2.putText(
                detection_frame,
                label_text,
                (x1, y1 - 3),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (255, 255, 255),
                1
            )
        
        # Create stats for FilterDisplay
        stats = {
            'current_detections': len(detections),
            'detector_name': self.detector_name,
            'total_detections': self.total_detections
        }
        
        return detection_frame, stats
    
    def process_classification_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Process frame for detection + classification (right side of dual filter)"""
        classification_frame = frame.copy()
        
        # Detect faces
        detections = self.detect_faces(classification_frame)
        self.total_detections += len(detections)
        
        classified_count = 0
        
        # Process each detection with classification
        for detection in detections:
            x1, y1, x2, y2 = detection.box
            
            # Extract face ROI
            face_roi = classification_frame[y1:y2, x1:x2]
            
            if face_roi.size > 0:
                # Classify mask if classifier is available
                label, confidence = self.classify_mask(face_roi)
                detection.label = label
                detection.confidence = confidence
                
                if self.classifier_enabled:
                    classified_count += 1
                    # Update statistics
                    if label in self.class_distribution:
                        self.class_distribution[label] += 1
                    else:
                        self.class_distribution['unknown'] += 1
                
                # Draw bounding box with enhanced color coding
                if self.classifier_enabled and label in ['with_mask', 'without_mask']:
                    # Use green for mask, red for no mask when classifier is active
                    if label == 'with_mask':
                        color = (0, 255, 0)  # Green for mask
                    else:  # without_mask
                        color = (0, 0, 255)  # Red for no mask
                else:
                    # Use default face detection color when no classifier or unknown result
                    color = self.colors.get(label, self.colors['face_detected'])
                
                cv2.rectangle(classification_frame, (x1, y1), (x2, y2), color, 2)
                
                # Draw label - cleaner text
                if self.classifier_enabled:
                    label_text = f"{label.replace('_', ' ')}: {confidence:.2f}"
                else:
                    label_text = f"Face: {detection.score:.2f}"
                
                # Calculate text size and background
                (text_width, text_height), baseline = cv2.getTextSize(
                    label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1
                )
                
                # Draw background rectangle for text
                cv2.rectangle(
                    classification_frame,
                    (x1, y1 - text_height - 6),
                    (x1 + text_width, y1),
                    color,
                    -1
                )
                
                # Draw text
                cv2.putText(
                    classification_frame,
                    label_text,
                    (x1, y1 - 3),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (255, 255, 255),
                    1
                )
        
        self.current_classifications = classified_count
        
        # Create stats for FilterDisplay
        stats = {
            'current_detections': len(detections),
            'current_classifications': classified_count,
            'detector_name': self.detector_name,
            'classifier_enabled': self.classifier_enabled,
            'classifier_type': self.classifier_type if self.classifier_type else 'None',
            'class_distribution': self.class_distribution.copy(),
            'total_detections': self.total_detections
        }
        
        return classification_frame, stats
    
    def create_dual_display(self, detection_frame: np.ndarray, classification_frame: np.ndarray, 
                           detection_stats: Dict[str, Any], classification_stats: Dict[str, Any]) -> np.ndarray:
        """Create side-by-side dual filter display using FilterDisplay"""
        if self.display and hasattr(self.display, 'create_combined_display'):
            # Create detector info for FilterDisplay
            detector_info = {
                'available_detectors': self.available_detectors,
                'available_classifiers': self.available_classifiers,
                'current_detector_index': self.current_detector_index,
                'current_classifier_index': self.current_classifier_index,
                'confidence_threshold': self.confidence_threshold
            }
            
            # Use the existing FilterDisplay for proper UI
            return self.display.create_combined_display(
                detection_frame,
                classification_frame,
                detection_stats,
                classification_stats,
                detector_info
            )
        else:
            # Fallback: simple side-by-side concatenation with basic UI
            h, w = detection_frame.shape[:2]
            
            # Resize frames to half width
            left_frame = cv2.resize(detection_frame, (w//2, h))
            right_frame = cv2.resize(classification_frame, (w//2, h))
            
            # Concatenate horizontally
            combined = np.hstack([left_frame, right_frame])
            
            # Add basic titles
            cv2.putText(combined, f"Detection ({detection_stats['detector_name'].upper()})", 
                       (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            classifier_name = classification_stats.get('classifier_type', 'None')
            cv2.putText(combined, f"Classification ({classifier_name.upper()})", 
                       (w//2 + 10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            return combined
    
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Process frame based on current display mode"""
        if self.display_mode == DisplayMode.DUAL_SIDE_BY_SIDE:
            # Create dual filter display
            detection_frame, detection_stats = self.process_detection_only_frame(frame)
            classification_frame, classification_stats = self.process_classification_frame(frame)
            return self.create_dual_display(detection_frame, classification_frame, detection_stats, classification_stats)
        
        elif self.display_mode == DisplayMode.DETECTION_ONLY:
            detection_frame, _ = self.process_detection_only_frame(frame)
            return detection_frame
        
        elif self.display_mode == DisplayMode.CLASSIFICATION_ONLY:
            classification_frame, _ = self.process_classification_frame(frame)
            return classification_frame
        
        elif self.display_mode == DisplayMode.OVERLAY and self.display:
            # Use FilterDisplay overlay mode
            detection_frame, _ = self.process_detection_only_frame(frame)
            classification_frame, _ = self.process_classification_frame(frame)
            return self.display.create_comparison_overlay(detection_frame, classification_frame)
        
        elif self.display_mode == DisplayMode.DIFFERENCE and self.display:
            # Use FilterDisplay difference mode
            detection_frame, _ = self.process_detection_only_frame(frame)
            classification_frame, _ = self.process_classification_frame(frame)
            return self.display.create_difference_view(detection_frame, classification_frame)
        
        else:
            # Single mode or fallback - use classification frame
            classification_frame, _ = self.process_classification_frame(frame)
            return classification_frame
    
    def start_camera(self) -> bool:
        """Initialize camera capture"""
        self.cap = cv2.VideoCapture(self.camera_index)
        
        if not self.cap.isOpened():
            print(f"Error: Could not open camera {self.camera_index}")
            return False
        
        # Set camera properties
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        print(f"Camera {self.camera_index} opened successfully")
        return True
    
    def stop_camera(self):
        """Release camera resources"""
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        self.is_running = False
    
    def run(self):
        """Run the face mask detection system"""
        if not self.start_camera():
            return
        
        self.is_running = True
        self.start_time = time.time()
        self.frame_count = 0
        
        print("\n" + "="*60)
        print("UNIFIED FACE MASK DETECTION SYSTEM")
        print("="*60)
        if self.controls:
            self.controls.print_help()
        else:
            print("Controls: Press 'q' to quit, 's' for screenshot")
        print("="*60)
        print("Starting detection loop...")
        
        try:
            while self.is_running:
                ret, frame = self.cap.read()
                if not ret:
                    print("Error: Could not read frame from camera")
                    break
                
                # Process frame using clean FilterDisplay UI
                processed_frame = self.process_frame(frame)
                
                # Store current frame for screenshot
                self._current_frame = processed_frame
                
                # Display frame using FilterDisplay if available, otherwise fallback
                if self.display and self.display_mode == DisplayMode.DUAL_SIDE_BY_SIDE:
                    # FilterDisplay handles all UI elements automatically
                    key = self.display.show_frame(processed_frame)
                else:
                    # Fallback display for non-dual modes
                    window_name = self.display.window_name if self.display else 'Unified Face Mask Detection'
                    cv2.imshow(window_name, processed_frame)
                    key = cv2.waitKey(1) & 0xFF
                
                # Handle key presses
                if key == ord('q') or key == 27:  # 'q' or ESC
                    break
                elif key == ord('s'):
                    self._take_screenshot()
                elif key == ord('d') or key == ord('D'):  # Cycle detector
                    self._cycle_detector()
                elif key == ord('c') or key == ord('C'):  # Cycle classifier
                    self._cycle_classifier()
                elif self.controls:
                    self.controls.handle_key(key)
                
                # Check if quit was requested
                if not self.is_running:
                    break
        
        except KeyboardInterrupt:
            print("\nInterrupted by user")
        
        except Exception as e:
            print(f"Error during detection loop: {e}")
        
        finally:
            self.stop_camera()
            if self.display:
                self.display.close()
            self._print_session_summary()
    
    def _print_session_summary(self):
        """Print session summary statistics"""
        if self.start_time:
            duration = time.time() - self.start_time
            
            print("\n" + "="*60)
            print("SESSION SUMMARY")
            print("="*60)
            print(f"Duration: {duration:.1f} seconds")
            print(f"Total frames: {self.frame_count}")
            print(f"Average FPS: {self.fps:.1f}")
            print(f"Detector used: {self.detector_name}")
            print(f"Classifier: {self.classifier_type if self.classifier_enabled else 'None'}")
            print(f"Total detections: {self.total_detections}")
            
            if self.classifier_enabled:
                print("Class distribution:")
                for class_name, count in self.class_distribution.items():
                    if count > 0:
                        print(f"  {class_name}: {count}")
            
            print("="*60)


def main():
    """Main function with command line interface"""
    parser = argparse.ArgumentParser(description='Unified Face Mask Detection System')
    
    parser.add_argument('--detector', type=str, default='haar',
                       choices=['haar', 'yunet', 'mtcnn', 'retinaface'],
                       help='Face detector type (default: haar)')
    
    parser.add_argument('--classifier', type=str, default=None,
                       choices=['custom_pytorch', 'yewon', 'sklearn'],
                       help='Classifier type (default: None for detection only)')
    
    parser.add_argument('--classifier-model', type=str, default=None,
                       help='Path to classifier model file')
    
    parser.add_argument('--camera', type=int, default=0,
                       help='Camera index (default: 0)')
    
    parser.add_argument('--confidence', type=float, default=0.6,
                       help='Detection confidence threshold (default: 0.6)')
    
    parser.add_argument('--device', type=str, default='cpu',
                       choices=['cuda', 'cpu'],
                       help='Computation device (default: cpu)')
    
    parser.add_argument('--width', type=int, default=640,
                       help='Frame width (default: 640)')
    
    parser.add_argument('--height', type=int, default=480,
                       help='Frame height (default: 480)')
    
    args = parser.parse_args()
    
    # Auto-detect model path if classifier specified but no model path given
    if args.classifier and not args.classifier_model:
        model_paths = {
            'custom_pytorch': 'model_artifacts/best_pytorch_model_custom.pth',
            'yewon': 'yewon_pipeline/best_model.pth',
            'sklearn': 'models/sklearn_classifier.pkl'
        }
        args.classifier_model = model_paths.get(args.classifier)
        
        if args.classifier_model and not os.path.exists(args.classifier_model):
            print(f"Warning: Default model path not found: {args.classifier_model}")
            print("Please specify --classifier-model or ensure the model file exists")
            args.classifier = None
            args.classifier_model = None
    
    print("Unified Face Mask Detection System")
    print("=" * 40)
    print(f"Detector: {args.detector}")
    print(f"Classifier: {args.classifier if args.classifier else 'None (detection only)'}")
    print(f"Camera: {args.camera}")
    print(f"Confidence threshold: {args.confidence}")
    print(f"Device: {args.device}")
    
    if args.classifier_model:
        print(f"Model: {args.classifier_model}")
    
    # Initialize detector
    detector = UnifiedFaceMaskDetector(
        detector_name=args.detector,
        classifier_type=args.classifier,
        classifier_model_path=args.classifier_model,
        camera_index=args.camera,
        confidence_threshold=args.confidence,
        device=args.device,
        frame_width=args.width,
        frame_height=args.height
    )
    
    # Run detection
    detector.run()


if __name__ == '__main__':
    main()
