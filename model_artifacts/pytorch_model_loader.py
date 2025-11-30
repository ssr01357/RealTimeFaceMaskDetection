"""
PyTorch Model Loader for Custom Face Mask Classification Model
Loads the custom PyTorchCNN model trained in CS583 Project
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
from typing import Tuple, Dict, Any, Optional
import os


class DepthwiseSeparableConv(nn.Module):
    """Depthwise Separable Convolution block"""
    
    def __init__(self, in_channels, out_channels, kernel_size, padding):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, 
                                 padding=padding, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class PyTorchCNN(nn.Module):
    """Custom PyTorch CNN for Face Mask Classification"""
    
    def __init__(self, in_channels, num_classes):
        super().__init__()
        # Convolutional Stem
        self.conv_stem = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Depthwise-separable convolution stages
        self.depthwise_block1 = DepthwiseSeparableConv(32, 64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.depthwise_block2 = DepthwiseSeparableConv(64, 128, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Global Average Pooling
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Dropout
        self.dropout = nn.Dropout(0.5)

        # Final Fully Connected Layer
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        # Convolutional Stem
        x = self.conv_stem(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool1(x)

        # Depthwise-separable convolution stages
        x = self.depthwise_block1(x)
        x = self.pool2(x)
        x = self.depthwise_block2(x)
        x = self.pool3(x)

        # Global Average Pooling
        x = self.global_avg_pool(x)

        # Flatten the output for the fully connected layer
        x = torch.flatten(x, 1)

        # Apply Dropout
        x = self.dropout(x)

        # Final Fully Connected Layer
        x = self.fc(x)
        return x


class CustomPyTorchClassifier:
    """Wrapper for the custom PyTorch face mask classifier"""
    
    def __init__(self, model_path: str, device: str = 'cuda'):
        """
        Initialize the custom PyTorch classifier
        
        Args:
            model_path: Path to the saved model file
            device: Device to run inference on ('cuda' or 'cpu')
        """
        self.model_path = model_path
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # Model parameters (from notebook)
        self.input_channels = 3  # RGB
        self.num_classes = 2     # with_mask, without_mask
        self.input_size = (40, 40)  # Model was trained on 40x40 images
        
        # Class names (index 0 = with_mask, index 1 = without_mask)
        self.class_names = ['with_mask', 'without_mask']
        
        # Load model
        self.model = self._load_model()
        
        print(f"Custom PyTorch classifier loaded on {self.device}")
        print(f"Input size: {self.input_size}")
        print(f"Classes: {self.class_names}")
    
    def _load_model(self) -> PyTorchCNN:
        """Load the trained model"""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        
        # Create model instance
        model = PyTorchCNN(self.input_channels, self.num_classes)
        
        # Load state dict
        try:
            # Try loading as state dict first
            state_dict = torch.load(self.model_path, map_location=self.device)
            model.load_state_dict(state_dict)
        except Exception as e:
            print(f"Error loading as state dict: {e}")
            # If that fails, try loading as full model
            try:
                checkpoint = torch.load(self.model_path, map_location=self.device)
                if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'])
                elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['state_dict'])
                else:
                    # Assume it's just the state dict
                    model.load_state_dict(checkpoint)
            except Exception as e2:
                raise RuntimeError(f"Could not load model: {e2}")
        
        # Move to device and set to eval mode
        model.to(self.device)
        model.eval()
        
        return model
    
    def preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        """
        Preprocess image for model input
        
        Args:
            image: Input image (BGR or RGB format)
            
        Returns:
            Preprocessed tensor ready for model
        """
        # Convert BGR to RGB if needed (OpenCV uses BGR by default)
        if len(image.shape) == 3 and image.shape[2] == 3:
            # Assume it's BGR from OpenCV, convert to RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = image
        
        # Resize to model input size
        image_resized = cv2.resize(image_rgb, self.input_size)
        
        # Convert to float and normalize to [0, 1]
        image_float = image_resized.astype(np.float32) / 255.0
        
        # Convert to tensor and change from HWC to CHW format
        image_tensor = torch.from_numpy(image_float).permute(2, 0, 1)
        
        # Add batch dimension
        image_tensor = image_tensor.unsqueeze(0)
        
        # Move to device
        image_tensor = image_tensor.to(self.device)
        
        return image_tensor
    
    def predict(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Predict mask wearing status for input image
        
        Args:
            image: Input face image (BGR format from OpenCV)
            
        Returns:
            Dictionary with prediction results
        """
        try:
            # Preprocess image
            input_tensor = self.preprocess_image(image)
            
            # Run inference
            with torch.no_grad():
                outputs = self.model(input_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                confidence, predicted_idx = torch.max(probabilities, 1)
                
                predicted_class = self.class_names[predicted_idx.item()]
                confidence_score = confidence.item()
                
                # Get probabilities for all classes
                all_probs = probabilities.squeeze().cpu().numpy()
                
                return {
                    'class': predicted_class,
                    'confidence': confidence_score,
                    'predicted_index': predicted_idx.item(),
                    'probabilities': {
                        self.class_names[i]: float(all_probs[i]) 
                        for i in range(len(self.class_names))
                    }
                }
        
        except Exception as e:
            print(f"Error in prediction: {e}")
            return {
                'class': 'unknown',
                'confidence': 0.0,
                'predicted_index': -1,
                'probabilities': {}
            }
    
    def predict_batch(self, images: list) -> list:
        """
        Predict for a batch of images
        
        Args:
            images: List of input images
            
        Returns:
            List of prediction dictionaries
        """
        results = []
        for image in images:
            results.append(self.predict(image))
        return results


def load_custom_pytorch_model(model_path: str, device: str = 'cuda') -> CustomPyTorchClassifier:
    """
    Load the custom PyTorch face mask classification model
    
    Args:
        model_path: Path to the saved model file
        device: Device to run inference on
        
    Returns:
        Loaded classifier instance
    """
    return CustomPyTorchClassifier(model_path, device)


def test_model_loading(model_path: str = None):
    """Test function to verify model loading works correctly"""
    if model_path is None:
        model_path = os.path.join(os.path.dirname(__file__), 'best_pytorch_model_custom.pth')
    
    print(f"Testing model loading from: {model_path}")
    
    try:
        classifier = load_custom_pytorch_model(model_path)
        print("✓ Model loaded successfully")
        
        # Test with dummy image
        dummy_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        result = classifier.predict(dummy_image)
        
        print(f"✓ Test prediction: {result}")
        print("Model is ready for use!")
        
    except Exception as e:
        print(f"✗ Error testing model: {e}")


if __name__ == '__main__':
    # Test the model loading
    test_model_loading()
