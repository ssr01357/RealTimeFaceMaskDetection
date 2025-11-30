"""
Unified wrapper for different face mask classifiers
Supports both NumPy CNN and PyTorch models
"""

import os
import pickle
import numpy as np
import cv2
import torch
import torch.nn as nn
from PIL import Image
from typing import List, Dict, Tuple, Optional, Any, Union
from abc import ABC, abstractmethod

# Import transforms for PyTorch models
try:
    from torchvision import transforms
    TORCHVISION_AVAILABLE = True
except ImportError:
    TORCHVISION_AVAILABLE = False


class BaseClassifier(ABC):
    """Abstract base class for face mask classifiers"""
    
    @abstractmethod
    def predict(self, face_crops: List[np.ndarray]) -> List[Tuple[int, List[float]]]:
        """
        Classify face crops
        
        Args:
            face_crops: List of face crop images (BGR format)
            
        Returns:
            List of (predicted_class, probabilities) tuples
        """
        pass
    
    @abstractmethod
    def get_class_names(self) -> List[str]:
        """Get list of class names"""
        pass


class NumPyCNNClassifier(BaseClassifier):
    """Wrapper for NumPy-based CNN classifier (your custom implementation)"""
    
    def __init__(self, model_path: str, class_names: List[str] = None):
        """
        Initialize NumPy CNN classifier
        
        Args:
            model_path: Path to pickled model file
            class_names: List of class names (default: ['No Mask', 'Mask'])
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)
        
        self.class_names = class_names or ['No Mask', 'Mask']
        self.input_size = 32  # Based on your implementation
    
    def predict(self, face_crops: List[np.ndarray]) -> List[Tuple[int, List[float]]]:
        """Predict using NumPy CNN model"""
        if not face_crops:
            return []
        
        # Preprocess crops
        processed_crops = []
        for crop in face_crops:
            if crop.size == 0:
                # Handle empty crops
                processed_crops.append(np.zeros((self.input_size, self.input_size, 1), dtype=np.float32))
                continue
            
            # Resize and convert to grayscale
            resized = cv2.resize(crop, (self.input_size, self.input_size)).astype(np.float32) / 255.0
            gray = np.mean(resized, axis=2, keepdims=True)
            processed_crops.append(gray)
        
        # Stack into batch
        batch = np.stack(processed_crops, axis=0)
        
        # Get predictions
        logits = self.model.forward(batch)
        
        # Convert to probabilities
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        
        # Get predicted classes
        pred_classes = np.argmax(probs, axis=1)
        
        results = []
        for i in range(len(face_crops)):
            results.append((int(pred_classes[i]), probs[i].tolist()))
        
        return results
    
    def get_class_names(self) -> List[str]:
        """Get class names"""
        return self.class_names


class PyTorchClassifier(BaseClassifier):
    """Wrapper for PyTorch-based classifiers (ResNet18, etc.)"""
    
    def __init__(self, model: nn.Module, device: str = 'cpu', 
                 transform: Optional[transforms.Compose] = None,
                 class_names: List[str] = None):
        """
        Initialize PyTorch classifier
        
        Args:
            model: PyTorch model
            device: Device to run inference on ('cpu' or 'cuda')
            transform: Preprocessing transforms
            class_names: List of class names
        """
        if not TORCHVISION_AVAILABLE:
            raise ImportError("torchvision is required for PyTorch classifiers")
        
        self.model = model.to(device)
        self.model.eval()
        self.device = device
        self.class_names = class_names or ['No Mask', 'Mask']
        
        # Default transform if none provided
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
        else:
            self.transform = transform
    
    def predict(self, face_crops: List[np.ndarray]) -> List[Tuple[int, List[float]]]:
        """Predict using PyTorch model"""
        if not face_crops:
            return []
        
        # Preprocess crops
        batch_tensors = []
        for crop in face_crops:
            if crop.size == 0:
                # Handle empty crops with black image
                crop = np.zeros((64, 64, 3), dtype=np.uint8)
            
            # Convert BGR to RGB
            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            crop_pil = Image.fromarray(crop_rgb)
            
            # Apply transforms
            tensor = self.transform(crop_pil)
            batch_tensors.append(tensor)
        
        # Stack into batch
        batch = torch.stack(batch_tensors).to(self.device)
        
        # Get predictions
        with torch.no_grad():
            logits = self.model(batch)
            probs = torch.softmax(logits, dim=1)
            pred_classes = torch.argmax(probs, dim=1)
        
        # Convert to CPU and numpy
        probs_np = probs.cpu().numpy()
        pred_classes_np = pred_classes.cpu().numpy()
        
        results = []
        for i in range(len(face_crops)):
            results.append((int(pred_classes_np[i]), probs_np[i].tolist()))
        
        return results
    
    def get_class_names(self) -> List[str]:
        """Get class names"""
        return self.class_names


class CustomPyTorchClassifierWrapper(BaseClassifier):
    """Wrapper for custom PyTorch classifier from model_artifacts"""
    
    def __init__(self, model_path: str, device: str = 'cpu', class_names: List[str] = None):
        """
        Initialize custom PyTorch classifier wrapper
        
        Args:
            model_path: Path to custom PyTorch model file
            device: Device to run inference on
            class_names: List of class names
        """
        # Import the custom PyTorch model loader
        import sys
        import os
        sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'model_artifacts'))
        
        try:
            from pytorch_model_loader import load_custom_pytorch_model
            self.classifier = load_custom_pytorch_model(model_path, device)
            self.class_names = class_names or ['with_mask', 'without_mask']
        except ImportError as e:
            raise ImportError(f"Could not import custom PyTorch model loader: {e}")
    
    def predict(self, face_crops: List[np.ndarray]) -> List[Tuple[int, List[float]]]:
        """Predict using custom PyTorch model"""
        if not face_crops:
            return []
        
        results = []
        for crop in face_crops:
            if crop.size == 0:
                # Handle empty crops
                results.append((0, [1.0, 0.0]))
                continue
            
            # Get prediction from custom classifier
            prediction = self.classifier.predict(crop)
            
            if isinstance(prediction, dict):
                predicted_class = prediction.get('predicted_index', 0)
                probabilities = list(prediction.get('probabilities', {}).values())
                if len(probabilities) != len(self.class_names):
                    probabilities = [1.0, 0.0] if predicted_class == 0 else [0.0, 1.0]
            else:
                predicted_class = 0
                probabilities = [1.0, 0.0]
            
            results.append((predicted_class, probabilities))
        
        return results
    
    def get_class_names(self) -> List[str]:
        """Get class names"""
        return self.class_names


class ClassifierWrapper:
    """
    Unified wrapper that can use different face mask classifiers
    """
    
    def __init__(self, classifier_type: str, **kwargs):
        """
        Initialize classifier wrapper
        
        Args:
            classifier_type: Type of classifier ('numpy_cnn', 'pytorch', 'custom_pytorch')
            **kwargs: Classifier-specific arguments
        """
        self.classifier_type = classifier_type.lower()
        
        if self.classifier_type == 'numpy_cnn':
            self.classifier = NumPyCNNClassifier(**kwargs)
        elif self.classifier_type == 'pytorch':
            self.classifier = PyTorchClassifier(**kwargs)
        elif self.classifier_type == 'custom_pytorch':
            self.classifier = CustomPyTorchClassifierWrapper(**kwargs)
        else:
            raise ValueError(f"Unsupported classifier type: {classifier_type}")
    
    def __call__(self, face_crops: List[np.ndarray]) -> List[Tuple[int, List[float]]]:
        """
        Classify face crops
        
        Args:
            face_crops: List of face crop images (BGR format)
            
        Returns:
            List of (predicted_class, probabilities) tuples
        """
        return self.classifier.predict(face_crops)
    
    def predict(self, face_crops: List[np.ndarray]) -> List[Tuple[int, List[float]]]:
        """Alias for __call__ method"""
        return self(face_crops)
    
    def predict_single(self, face_crop: np.ndarray) -> Tuple[int, List[float]]:
        """
        Classify a single face crop
        
        Args:
            face_crop: Single face crop image (BGR format)
            
        Returns:
            Tuple of (predicted_class, probabilities)
        """
        results = self.predict([face_crop])
        return results[0] if results else (0, [1.0, 0.0])
    
    def get_class_names(self) -> List[str]:
        """Get class names"""
        return self.classifier.get_class_names()
    
    @staticmethod
    def create_numpy_cnn_classifier(model_path: str, class_names: List[str] = None) -> 'ClassifierWrapper':
        """
        Factory method to create NumPy CNN classifier
        
        Args:
            model_path: Path to pickled model file
            class_names: List of class names
            
        Returns:
            ClassifierWrapper instance
        """
        return ClassifierWrapper('numpy_cnn', model_path=model_path, class_names=class_names)
    
    @staticmethod
    def create_pytorch_classifier(model: nn.Module, device: str = 'cpu',
                                 transform: Optional[transforms.Compose] = None,
                                 class_names: List[str] = None) -> 'ClassifierWrapper':
        """
        Factory method to create PyTorch classifier
        
        Args:
            model: PyTorch model
            device: Device to run inference on
            transform: Preprocessing transforms
            class_names: List of class names
            
        Returns:
            ClassifierWrapper instance
        """
        return ClassifierWrapper('pytorch', model=model, device=device, 
                               transform=transform, class_names=class_names)
    
    @staticmethod
    def create_custom_pytorch_classifier(model_path: str, device: str = 'cpu',
                                        class_names: List[str] = None) -> 'ClassifierWrapper':
        """
        Factory method to create custom PyTorch classifier
        
        Args:
            model_path: Path to custom PyTorch model file
            device: Device to run inference on
            class_names: List of class names
            
        Returns:
            ClassifierWrapper instance
        """
        return ClassifierWrapper('custom_pytorch', model_path=model_path, 
                               device=device, class_names=class_names)
    
    def get_classifier_info(self) -> Dict[str, Any]:
        """Get information about the current classifier"""
        info = {
            'type': self.classifier_type,
            'class_names': self.get_class_names(),
            'num_classes': len(self.get_class_names())
        }
        
        if self.classifier_type == 'numpy_cnn':
            info.update({
                'input_size': self.classifier.input_size,
                'framework': 'NumPy'
            })
        elif self.classifier_type == 'pytorch':
            info.update({
                'device': self.classifier.device,
                'framework': 'PyTorch'
            })
        
        return info
