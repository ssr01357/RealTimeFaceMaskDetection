"""
Classification filter for face detection with mask classification
"""

import cv2
import numpy as np
import sys
import os
import torch
from typing import List, Dict, Any, Optional, Tuple

# Add paths for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'yewon_pipeline'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'evaluation'))

try:
    from detectors_2 import build_detector, BaseFaceDetector, FaceBox
except ImportError:
    print("Warning: Could not import yewon_pipeline detectors")
    BaseFaceDetector = None
    build_detector = None

try:
    from evaluation.classifiers.classifier_wrapper import FaceClassifierWrapper
except ImportError:
    print("Warning: Could not import classifier wrapper")
    FaceClassifierWrapper = None

# Try to import yewon_pipeline classifier components
try:
    from pipeline_1 import build_model, TrainConfig, build_transforms
except ImportError:
    print("Warning: Could not import yewon_pipeline classifier components")
    build_model = None
    TrainConfig = None
    build_transforms = None

# Import custom PyTorch model loader
try:
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'model_artifacts'))
    from pytorch_model_loader import load_custom_pytorch_model, CustomPyTorchClassifier
except ImportError:
    print("Warning: Could not import custom PyTorch model loader")
    load_custom_pytorch_model = None
    CustomPyTorchClassifier = None


class ClassificationFilter:
    """
    Filter that performs face detection and mask classification
    """
    
    def __init__(self, 
                 detector_name: str = 'haar',
                 classifier_model_path: str = None,
                 device: str = 'cuda',
                 yunet_onnx: str = None,
                 haar_xml: str = None,
                 retina_thresh: float = 0.8,
                 confidence_threshold: float = 0.6,
                 num_classes: int = 2):
        """
        Initialize classification filter
        
        Args:
            detector_name: Name of detector ('yunet', 'haar', 'mtcnn', 'retinaface')
            classifier_model_path: Path to trained classifier model
            device: Device for computation ('cuda' or 'cpu')
            yunet_onnx: Path to YuNet ONNX model
            haar_xml: Path to Haar cascade XML
            retina_thresh: RetinaFace threshold
            confidence_threshold: Minimum confidence for detections
            num_classes: Number of classification classes (2 or 3)
        """
        self.detector_name = detector_name
        self.classifier_model_path = classifier_model_path
        self.device = device
        self.confidence_threshold = confidence_threshold
        self.num_classes = num_classes
        
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
        
        # Initialize classifier
        self.classifier = self._setup_classifier()
        self.transform = self._setup_transforms()
        
        # Color scheme for mask classification
        if num_classes == 2:
            self.colors = {
                'with_mask': (0, 255, 0),      # Green
                'without_mask': (0, 0, 255),   # Red
                'unknown': (255, 255, 0)       # Yellow
            }
            self.class_names = ['with_mask', 'without_mask']
        else:  # num_classes == 3
            self.colors = {
                'with_mask': (0, 255, 0),         # Green
                'without_mask': (0, 0, 255),      # Red
                'incorrect_mask': (0, 165, 255),  # Orange
                'unknown': (255, 255, 0)          # Yellow
            }
            self.class_names = ['with_mask', 'without_mask', 'incorrect_mask']
        
        # Performance tracking
        self.detection_count = 0
        self.classification_count = 0
        self.total_detections = 0
        self.class_distribution = {name: 0 for name in self.class_names}
    
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
    
    def _setup_classifier(self):
        """Setup mask classifier"""
        if self.classifier_model_path is None or not os.path.exists(self.classifier_model_path):
            print("Warning: No classifier model provided or file doesn't exist")
            return None
        
        try:
            # Check if this is the custom PyTorch model
            if self.classifier_model_path.endswith('best_pytorch_model_custom.pth'):
                return self._load_custom_pytorch_classifier()
            
            # Try to load using yewon_pipeline components first
            elif build_model is not None and TrainConfig is not None:
                return self._load_yewon_classifier()
            
            # Fallback to evaluation classifier wrapper
            elif FaceClassifierWrapper is not None:
                return self._load_wrapper_classifier()
            
            else:
                print("Warning: No classifier loading method available")
                return None
                
        except Exception as e:
            print(f"Error loading classifier: {e}")
            return None
    
    def _load_yewon_classifier(self):
        """Load classifier using yewon_pipeline components"""
        try:
            # Load checkpoint
            checkpoint = torch.load(self.classifier_model_path, map_location=self.device)
            
            # Get config from checkpoint or create default
            if 'config' in checkpoint:
                config_dict = checkpoint['config']
                config = TrainConfig(**config_dict)
            else:
                # Create default config
                config = TrainConfig(
                    data_root="",  # Not needed for inference
                    num_classes=self.num_classes,
                    model_name="resnet18"  # Default
                )
            
            # Build model
            model = build_model(config)
            model.load_state_dict(checkpoint['model_state'])
            model.to(self.device)
            model.eval()
            
            print(f"Loaded yewon_pipeline classifier: {config.model_name}")
            return model
            
        except Exception as e:
            print(f"Error loading yewon_pipeline classifier: {e}")
            return None
    
    def _load_custom_pytorch_classifier(self):
        """Load custom PyTorch classifier"""
        try:
            if load_custom_pytorch_model is None:
                print("Custom PyTorch model loader not available")
                return None
            
            classifier = load_custom_pytorch_model(
                model_path=self.classifier_model_path,
                device=self.device
            )
            
            print(f"Loaded custom PyTorch classifier from: {self.classifier_model_path}")
            return classifier
            
        except Exception as e:
            print(f"Error loading custom PyTorch classifier: {e}")
            return None
    
    def _load_wrapper_classifier(self):
        """Load classifier using evaluation wrapper"""
        try:
            # Determine classifier type from file extension
            if self.classifier_model_path.endswith('.pth') or self.classifier_model_path.endswith('.pt'):
                classifier_type = 'pytorch'
            elif self.classifier_model_path.endswith('.pkl'):
                classifier_type = 'sklearn'
            else:
                classifier_type = 'pytorch'  # Default
            
            classifier = FaceClassifierWrapper(
                classifier_type=classifier_type,
                model_path=self.classifier_model_path,
                class_names=self.class_names
            )
            
            print(f"Loaded wrapper classifier: {classifier_type}")
            return classifier
            
        except Exception as e:
            print(f"Error loading wrapper classifier: {e}")
            return None
    
    def _setup_transforms(self):
        """Setup image transforms for classifier"""
        if build_transforms is not None:
            return build_transforms(img_size=224, is_train=False)
        else:
            # Fallback transforms
            from torchvision import transforms
            return transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
    
    def detect_faces(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect faces in frame
        
        Args:
            frame: BGR image frame
            
        Returns:
            List of detection dictionaries
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
                    
                    detections.append({
                        'box': box,
                        'score': float(score)
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
                        'score': 1.0
                    })
        
        except Exception as e:
            print(f"Error in face detection: {e}")
            return []
        
        return detections
    
    def classify_mask(self, face_roi: np.ndarray) -> Tuple[str, float]:
        """
        Classify mask wearing status for face ROI
        
        Args:
            face_roi: Face region of interest (BGR image)
            
        Returns:
            Tuple of (label, confidence)
        """
        if self.classifier is None:
            return 'unknown', 0.0
        
        try:
            # Convert BGR to RGB
            face_rgb = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
            
            if hasattr(self.classifier, 'predict'):
                # Using wrapper classifier
                prediction = self.classifier.predict(face_rgb)
                
                if isinstance(prediction, dict):
                    label = prediction.get('class', 'unknown')
                    confidence = prediction.get('confidence', 0.0)
                else:
                    label = str(prediction)
                    confidence = 0.8
                    
            else:
                # Using yewon_pipeline model directly
                # Preprocess
                if self.transform is not None:
                    input_tensor = self.transform(face_rgb).unsqueeze(0).to(self.device)
                else:
                    # Basic preprocessing
                    face_resized = cv2.resize(face_rgb, (224, 224))
                    input_tensor = torch.from_numpy(face_resized).permute(2, 0, 1).float()
                    input_tensor = input_tensor.unsqueeze(0).to(self.device) / 255.0
                
                # Inference
                with torch.no_grad():
                    outputs = self.classifier(input_tensor)
                    probabilities = torch.softmax(outputs, dim=1)
                    confidence, predicted = torch.max(probabilities, 1)
                    
                    label = self.class_names[predicted.item()]
                    confidence = confidence.item()
            
            return label, confidence
            
        except Exception as e:
            print(f"Error in mask classification: {e}")
            return 'unknown', 0.0
    
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Process frame with detection and classification
        
        Args:
            frame: Input BGR frame
            
        Returns:
            Annotated frame with detection boxes and classification labels
        """
        # Create a copy to avoid modifying original
        output_frame = frame.copy()
        
        # Detect faces
        detections = self.detect_faces(frame)
        self.detection_count = len(detections)
        self.total_detections += self.detection_count
        
        classified_count = 0
        
        # Process each detection
        for detection in detections:
            box = detection['box']
            score = detection['score']
            
            x1, y1, x2, y2 = box
            
            # Extract face ROI
            face_roi = frame[y1:y2, x1:x2]
            
            if face_roi.size > 0:
                # Classify mask
                label, confidence = self.classify_mask(face_roi)
                
                if label != 'unknown':
                    classified_count += 1
                    # Update class distribution
                    if label in self.class_distribution:
                        self.class_distribution[label] += 1
                
                # Get color for this classification
                color = self.colors.get(label, self.colors['unknown'])
                
                # Draw bounding box
                cv2.rectangle(output_frame, (x1, y1), (x2, y2), color, 2)
                
                # Prepare label text
                if label == 'unknown':
                    label_text = f"Face: {score:.2f}"
                else:
                    label_text = f"{label}: {confidence:.2f}"
                
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
        
        self.classification_count = classified_count
        
        # Add filter info
        self._add_filter_info(output_frame)
        
        return output_frame
    
    def _add_filter_info(self, frame: np.ndarray):
        """Add filter information to frame"""
        # Filter title
        cv2.putText(
            frame,
            f"Detection + Classification ({self.detector_name.upper()})",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )
        
        # Detection and classification counts
        cv2.putText(
            frame,
            f"Faces: {self.detection_count} | Classified: {self.classification_count}",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )
        
        # Legend
        legend_y = frame.shape[0] - 100
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
        legend_items = []
        for class_name in self.class_names:
            color = self.colors[class_name]
            count = self.class_distribution[class_name]
            legend_items.append((f"{class_name} ({count})", color))
        
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
            'classifier_available': self.classifier is not None,
            'current_detections': self.detection_count,
            'current_classifications': self.classification_count,
            'total_detections': self.total_detections,
            'class_distribution': self.class_distribution.copy(),
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
