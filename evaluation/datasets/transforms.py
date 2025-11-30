"""
Transform utilities for face mask detection datasets
Provides standard transforms for detection and classification tasks
"""

import torch
import torchvision.transforms as T
import torchvision.transforms.functional as F
import numpy as np
from PIL import Image
import cv2
from typing import Dict, Any, Tuple, List, Optional, Union
import random


class DetectionTransforms:
    """Transforms for object detection tasks that handle both images and bounding boxes"""
    
    def __init__(self, image_size: int = 416, augment: bool = True):
        """
        Initialize detection transforms
        
        Args:
            image_size: Target image size (square)
            augment: Whether to apply data augmentation
        """
        self.image_size = image_size
        self.augment = augment
    
    def __call__(self, image: Union[Image.Image, np.ndarray], 
                 target: Dict[str, Any]) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Apply transforms to image and target
        
        Args:
            image: PIL Image or numpy array
            target: Dictionary with 'boxes' and 'labels' keys
            
        Returns:
            Transformed image tensor and target dictionary
        """
        # Convert to PIL if numpy array
        if isinstance(image, np.ndarray):
            if image.dtype == np.uint8:
                image = Image.fromarray(image)
            else:
                image = Image.fromarray((image * 255).astype(np.uint8))
        
        # Get original image size
        orig_w, orig_h = image.size
        
        # Convert boxes to tensor if not already
        boxes = target['boxes']
        if not isinstance(boxes, torch.Tensor):
            boxes = torch.tensor(boxes, dtype=torch.float32)
        
        labels = target['labels']
        if not isinstance(labels, torch.Tensor):
            labels = torch.tensor(labels, dtype=torch.int64)
        
        # Apply augmentations if enabled
        if self.augment:
            # Random horizontal flip
            if random.random() < 0.5:
                image = F.hflip(image)
                # Flip boxes horizontally
                boxes[:, [0, 2]] = orig_w - boxes[:, [2, 0]]
            
            # Color jitter
            if random.random() < 0.5:
                color_jitter = T.ColorJitter(brightness=0.2, contrast=0.2, 
                                           saturation=0.2, hue=0.1)
                image = color_jitter(image)
        
        # Resize image and adjust boxes
        image = F.resize(image, (self.image_size, self.image_size))
        
        # Scale boxes to new image size
        scale_x = self.image_size / orig_w
        scale_y = self.image_size / orig_h
        
        boxes[:, [0, 2]] *= scale_x  # x coordinates
        boxes[:, [1, 3]] *= scale_y  # y coordinates
        
        # Convert to tensor and normalize
        image = F.to_tensor(image)
        image = F.normalize(image, mean=[0.485, 0.456, 0.406], 
                          std=[0.229, 0.224, 0.225])
        
        # Update target
        target = {
            'boxes': boxes,
            'labels': labels,
            'image_id': target.get('image_id', 0),
            'area': (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0]),
            'iscrowd': torch.zeros(len(boxes), dtype=torch.int64)
        }
        
        return image, target


class ClassificationTransforms:
    """Transforms for classification tasks"""
    
    def __init__(self, image_size: int = 224, augment: bool = True):
        """
        Initialize classification transforms
        
        Args:
            image_size: Target image size
            augment: Whether to apply data augmentation
        """
        self.image_size = image_size
        self.augment = augment
        
        # Base transforms
        self.base_transforms = T.Compose([
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Augmentation transforms
        if augment:
            self.augment_transforms = T.Compose([
                T.Resize((int(image_size * 1.1), int(image_size * 1.1))),
                T.RandomCrop((image_size, image_size)),
                T.RandomHorizontalFlip(p=0.5),
                T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                T.RandomRotation(degrees=10),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            self.augment_transforms = self.base_transforms
    
    def __call__(self, image: Union[Image.Image, np.ndarray]) -> torch.Tensor:
        """
        Apply transforms to image
        
        Args:
            image: PIL Image or numpy array
            
        Returns:
            Transformed image tensor
        """
        # Convert to PIL if numpy array
        if isinstance(image, np.ndarray):
            if len(image.shape) == 3 and image.shape[2] == 3:
                # BGR to RGB conversion if needed
                if image.max() <= 1.0:
                    image = (image * 255).astype(np.uint8)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(image)
            else:
                image = Image.fromarray(image)
        
        # Apply transforms
        if self.augment:
            return self.augment_transforms(image)
        else:
            return self.base_transforms(image)


class InferenceTransforms:
    """Transforms for inference (no augmentation)"""
    
    def __init__(self, image_size: int = 224, task: str = 'classification'):
        """
        Initialize inference transforms
        
        Args:
            image_size: Target image size
            task: 'classification' or 'detection'
        """
        self.image_size = image_size
        self.task = task
        
        if task == 'classification':
            self.transforms = T.Compose([
                T.Resize((image_size, image_size)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:  # detection
            self.detection_transforms = DetectionTransforms(image_size, augment=False)
    
    def __call__(self, image: Union[Image.Image, np.ndarray], 
                 target: Optional[Dict[str, Any]] = None) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, Any]]]:
        """Apply inference transforms"""
        if self.task == 'classification':
            if isinstance(image, np.ndarray):
                if len(image.shape) == 3 and image.shape[2] == 3:
                    if image.max() <= 1.0:
                        image = (image * 255).astype(np.uint8)
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    image = Image.fromarray(image)
                else:
                    image = Image.fromarray(image)
            return self.transforms(image)
        else:  # detection
            if target is None:
                target = {'boxes': [], 'labels': []}
            return self.detection_transforms(image, target)


class FaceMaskTransforms:
    """Factory class for creating face mask detection transforms"""
    
    @staticmethod
    def get_detection_transforms(image_size: int = 416, augment: bool = True) -> DetectionTransforms:
        """
        Get transforms for detection tasks
        
        Args:
            image_size: Target image size (square)
            augment: Whether to apply data augmentation
            
        Returns:
            DetectionTransforms instance
        """
        return DetectionTransforms(image_size=image_size, augment=augment)
    
    @staticmethod
    def get_classification_transforms(image_size: int = 224, augment: bool = True) -> ClassificationTransforms:
        """
        Get transforms for classification tasks
        
        Args:
            image_size: Target image size
            augment: Whether to apply data augmentation
            
        Returns:
            ClassificationTransforms instance
        """
        return ClassificationTransforms(image_size=image_size, augment=augment)
    
    @staticmethod
    def get_inference_transforms(image_size: int = 224, task: str = 'classification') -> InferenceTransforms:
        """
        Get transforms for inference (no augmentation)
        
        Args:
            image_size: Target image size
            task: 'classification' or 'detection'
            
        Returns:
            InferenceTransforms instance
        """
        return InferenceTransforms(image_size=image_size, task=task)
    
    @staticmethod
    def get_train_transforms(task: str = 'classification', image_size: Optional[int] = None) -> Union[DetectionTransforms, ClassificationTransforms]:
        """
        Get training transforms with augmentation
        
        Args:
            task: 'classification' or 'detection'
            image_size: Target image size (defaults: 224 for classification, 416 for detection)
            
        Returns:
            Appropriate transforms instance
        """
        if task == 'classification':
            size = image_size or 224
            return FaceMaskTransforms.get_classification_transforms(size, augment=True)
        else:  # detection
            size = image_size or 416
            return FaceMaskTransforms.get_detection_transforms(size, augment=True)
    
    @staticmethod
    def get_val_transforms(task: str = 'classification', image_size: Optional[int] = None) -> Union[DetectionTransforms, ClassificationTransforms]:
        """
        Get validation transforms without augmentation
        
        Args:
            task: 'classification' or 'detection'
            image_size: Target image size (defaults: 224 for classification, 416 for detection)
            
        Returns:
            Appropriate transforms instance
        """
        if task == 'classification':
            size = image_size or 224
            return FaceMaskTransforms.get_classification_transforms(size, augment=False)
        else:  # detection
            size = image_size or 416
            return FaceMaskTransforms.get_detection_transforms(size, augment=False)


# Utility functions for custom transforms
def denormalize_tensor(tensor: torch.Tensor, mean: List[float] = [0.485, 0.456, 0.406], 
                      std: List[float] = [0.229, 0.224, 0.225]) -> torch.Tensor:
    """
    Denormalize a tensor for visualization
    
    Args:
        tensor: Normalized tensor
        mean: Normalization mean
        std: Normalization std
        
    Returns:
        Denormalized tensor
    """
    mean = torch.tensor(mean).view(3, 1, 1)
    std = torch.tensor(std).view(3, 1, 1)
    return tensor * std + mean


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    """
    Convert tensor to PIL Image
    
    Args:
        tensor: Image tensor (C, H, W)
        
    Returns:
        PIL Image
    """
    # Denormalize if needed
    if tensor.min() < 0:
        tensor = denormalize_tensor(tensor)
    
    # Clamp values
    tensor = torch.clamp(tensor, 0, 1)
    
    # Convert to PIL
    tensor = (tensor * 255).byte()
    return T.ToPILImage()(tensor)


def visualize_detection_sample(image: torch.Tensor, target: Dict[str, Any], 
                             class_names: List[str]) -> Image.Image:
    """
    Visualize a detection sample with bounding boxes
    
    Args:
        image: Image tensor
        target: Target dictionary with boxes and labels
        class_names: List of class names
        
    Returns:
        PIL Image with bounding boxes drawn
    """
    # Convert to PIL
    pil_image = tensor_to_pil(image)
    
    # Draw bounding boxes
    from PIL import ImageDraw, ImageFont
    draw = ImageDraw.Draw(pil_image)
    
    boxes = target['boxes']
    labels = target['labels']
    
    colors = ['red', 'green', 'blue', 'yellow', 'purple', 'orange']
    
    for i, (box, label) in enumerate(zip(boxes, labels)):
        x1, y1, x2, y2 = box.tolist()
        color = colors[label % len(colors)]
        
        # Draw box
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
        
        # Draw label
        if label < len(class_names):
            label_text = class_names[label]
            draw.text((x1, y1 - 15), label_text, fill=color)
    
    return pil_image
