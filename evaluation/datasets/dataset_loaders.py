"""
PyTorch Dataset classes for Face Mask Detection
Supports multiple Kaggle datasets with automatic downloading and caching
"""

import os
import xml.etree.ElementTree as ET
import cv2
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from typing import List, Dict, Tuple, Optional, Any, Union
import random
import json
from pathlib import Path
import time

from .kaggle_manager import KaggleDatasetManager
from .transforms import FaceMaskTransforms
from .cache_manager import DatasetCacheManager


class AndrewMVDPyTorchDataset(Dataset):
    """
    PyTorch Dataset for andrewmvd/face-mask-detection from Kaggle
    Supports both detection (full images + boxes) and classification (cropped faces) modes
    """
    
    def __init__(self, 
                 kaggle_dataset_id: str = 'andrewmvd/face-mask-detection',
                 mode: str = 'detection',
                 split: str = 'train',
                 split_ratios: Tuple[float, float, float] = (0.7, 0.15, 0.15),
                 transforms=None,
                 cache_processed: bool = True,
                 cache_dir: str = "/tmp/face_mask_datasets",
                 download_to_tmp: bool = True,
                 random_seed: int = 42):
        """
        Initialize AndrewMVD PyTorch dataset
        
        Args:
            kaggle_dataset_id: Kaggle dataset identifier
            mode: 'detection' for full images + boxes, 'classification' for cropped faces
            split: 'train', 'val', 'test', or 'all'
            split_ratios: (train, val, test) ratios
            transforms: Transform pipeline
            cache_processed: Whether to cache processed samples
            cache_dir: Directory for caching
            download_to_tmp: Whether to download to tmp directory
            random_seed: Random seed for reproducible splits
        """
        self.kaggle_dataset_id = kaggle_dataset_id
        self.mode = mode.lower()
        self.split = split.lower()
        self.split_ratios = split_ratios
        self.transforms = transforms
        self.cache_processed = cache_processed
        self.random_seed = random_seed
        
        # Set random seed for reproducible splits
        random.seed(random_seed)
        np.random.seed(random_seed)
        
        # Initialize managers
        self.kaggle_manager = KaggleDatasetManager(cache_dir)
        if cache_processed:
            self.cache_manager = DatasetCacheManager()
        
        # Download dataset
        print(f"Initializing AndrewMVD dataset (mode: {mode}, split: {split})")
        self.dataset_path = self.kaggle_manager.download_dataset(kaggle_dataset_id)
        
        # Set up paths
        self.images_path = os.path.join(self.dataset_path, 'images')
        self.annotations_path = os.path.join(self.dataset_path, 'annotations')
        
        # Load and process samples
        self._load_samples()
        
        # Apply split
        if split != 'all':
            self._apply_split()
        
        # Set up transforms
        if transforms is None:
            if mode == 'detection':
                self.transforms = FaceMaskTransforms.get_detection_transforms(augment=(split == 'train'))
            else:
                self.transforms = FaceMaskTransforms.get_classification_transforms(augment=(split == 'train'))
        
        print(f"Dataset initialized with {len(self)} samples")
    
    def _load_samples(self):
        """Load samples from dataset"""
        # Check cache first
        if self.cache_processed and hasattr(self, 'cache_manager'):
            cached_samples = self.cache_manager.load_from_cache(
                self.kaggle_dataset_id, 'all', self.mode, None
            )
            if cached_samples is not None:
                self.samples = cached_samples
                return
        
        print("Processing dataset samples...")
        self.samples = []
        self.classes = set()
        
        # Get all annotation files
        annotation_files = [f for f in os.listdir(self.annotations_path) if f.endswith('.xml')]
        
        for anno_file in annotation_files:
            image_id = os.path.splitext(anno_file)[0]
            
            # Find corresponding image
            img_path = None
            for ext in ['.png', '.jpg', '.jpeg']:
                potential_path = os.path.join(self.images_path, f"{image_id}{ext}")
                if os.path.exists(potential_path):
                    img_path = potential_path
                    break
            
            if img_path is None:
                continue
            
            # Parse annotation
            anno_path = os.path.join(self.annotations_path, anno_file)
            try:
                tree = ET.parse(anno_path)
                root = tree.getroot()
                
                boxes = []
                labels = []
                
                for obj in root.findall('object'):
                    class_name = obj.find('name').text
                    self.classes.add(class_name)
                    
                    bndbox = obj.find('bndbox')
                    xmin = int(float(bndbox.find('xmin').text))
                    ymin = int(float(bndbox.find('ymin').text))
                    xmax = int(float(bndbox.find('xmax').text))
                    ymax = int(float(bndbox.find('ymax').text))
                    
                    boxes.append([xmin, ymin, xmax, ymax])
                    labels.append(class_name)
                
                if boxes:  # Only add samples with faces
                    self.samples.append({
                        'image_path': img_path,
                        'boxes': boxes,
                        'labels': labels,
                        'image_id': len(self.samples)
                    })
                    
            except ET.ParseError:
                print(f"Warning: Could not parse annotation {anno_file}")
                continue
        
        # Create class mapping
        sorted_classes = sorted(list(self.classes))
        self.class_mapping = {name: idx for idx, name in enumerate(sorted_classes)}
        self.class_names = sorted_classes
        
        # For classification mode, create cropped samples
        if self.mode == 'classification':
            self._create_cropped_samples()
        
        # Cache processed samples
        if self.cache_processed and hasattr(self, 'cache_manager'):
            self.cache_manager.save_to_cache(
                self.kaggle_dataset_id, 'all', self.mode, None, self.samples
            )
    
    def _create_cropped_samples(self):
        """Create cropped face samples for classification mode"""
        print("Creating cropped face samples...")
        cropped_samples = []
        
        for sample in self.samples:
            try:
                image = cv2.imread(sample['image_path'])
                if image is None:
                    continue
                
                for box, label in zip(sample['boxes'], sample['labels']):
                    xmin, ymin, xmax, ymax = box
                    
                    # Ensure coordinates are within image bounds
                    h, w = image.shape[:2]
                    xmin = max(0, min(xmin, w-1))
                    xmax = max(0, min(xmax, w))
                    ymin = max(0, min(ymin, h-1))
                    ymax = max(0, min(ymax, h))
                    
                    if xmax > xmin and ymax > ymin:
                        crop = image[ymin:ymax, xmin:xmax]
                        cropped_samples.append({
                            'crop': crop,
                            'label': self.class_mapping[label],
                            'original_image_id': sample['image_id']
                        })
            except Exception as e:
                print(f"Warning: Could not process image {sample['image_path']}: {e}")
                continue
        
        self.samples = cropped_samples
        print(f"Created {len(cropped_samples)} cropped face samples")
    
    def _apply_split(self):
        """Apply train/val/test split"""
        total_samples = len(self.samples)
        
        # Shuffle samples for random split
        random.shuffle(self.samples)
        
        # Calculate split indices
        train_size = int(self.split_ratios[0] * total_samples)
        val_size = int(self.split_ratios[1] * total_samples)
        
        if self.split == 'train':
            self.samples = self.samples[:train_size]
        elif self.split == 'val':
            self.samples = self.samples[train_size:train_size + val_size]
        elif self.split == 'test':
            self.samples = self.samples[train_size + val_size:]
    
    def __len__(self) -> int:
        """Return number of samples"""
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Any]:
        """Get sample at index"""
        sample = self.samples[idx]
        
        if self.mode == 'detection':
            # Load image
            image = cv2.imread(sample['image_path'])
            if image is None:
                # Return dummy data if image loading fails
                image = np.zeros((416, 416, 3), dtype=np.uint8)
            
            # Convert BGR to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(image)
            
            # Prepare target
            boxes = torch.tensor(sample['boxes'], dtype=torch.float32)
            labels = torch.tensor([self.class_mapping[label] for label in sample['labels']], dtype=torch.int64)
            
            target = {
                'boxes': boxes,
                'labels': labels,
                'image_id': sample['image_id']
            }
            
            # Apply transforms
            if self.transforms:
                image, target = self.transforms(image, target)
            
            return image, target
        
        else:  # classification mode
            # Get cropped image
            crop = sample['crop']
            
            # Convert BGR to RGB
            crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            crop = Image.fromarray(crop)
            
            label = sample['label']
            
            # Apply transforms
            if self.transforms:
                crop = self.transforms(crop)
            
            return crop, label
    
    def get_class_names(self) -> List[str]:
        """Get list of class names"""
        return self.class_names.copy()
    
    def get_dataset_info(self) -> Dict[str, Any]:
        """Get dataset information"""
        return {
            'dataset_id': self.kaggle_dataset_id,
            'mode': self.mode,
            'split': self.split,
            'num_samples': len(self),
            'num_classes': len(self.class_names),
            'class_names': self.class_names,
            'split_ratios': self.split_ratios
        }


class Face12kPyTorchDataset(Dataset):
    """
    PyTorch Dataset for ashishjangra27/face-mask-12k-images-dataset from Kaggle
    Classification dataset with folder structure: with_mask/, without_mask/, mask_weared_incorrect/
    """
    
    def __init__(self,
                 kaggle_dataset_id: str = 'ashishjangra27/face-mask-12k-images-dataset',
                 split: str = 'train',
                 split_ratios: Tuple[float, float, float] = (0.7, 0.15, 0.15),
                 transforms=None,
                 cache_processed: bool = True,
                 cache_dir: str = "/tmp/face_mask_datasets",
                 download_to_tmp: bool = True,
                 random_seed: int = 42):
        """
        Initialize Face12k PyTorch dataset
        
        Args:
            kaggle_dataset_id: Kaggle dataset identifier
            split: 'train', 'val', 'test', or 'all'
            split_ratios: (train, val, test) ratios
            transforms: Transform pipeline
            cache_processed: Whether to cache processed samples
            cache_dir: Directory for caching
            download_to_tmp: Whether to download to tmp directory
            random_seed: Random seed for reproducible splits
        """
        self.kaggle_dataset_id = kaggle_dataset_id
        self.split = split.lower()
        self.split_ratios = split_ratios
        self.transforms = transforms
        self.cache_processed = cache_processed
        self.random_seed = random_seed
        
        # Set random seed
        random.seed(random_seed)
        np.random.seed(random_seed)
        
        # Initialize managers
        self.kaggle_manager = KaggleDatasetManager(cache_dir)
        if cache_processed:
            self.cache_manager = DatasetCacheManager()
        
        # Download dataset
        print(f"Initializing Face12k dataset (split: {split})")
        self.dataset_path = self.kaggle_manager.download_dataset(kaggle_dataset_id)
        
        # Class mapping
        self.class_folders = {
            'without_mask': 0,
            'with_mask': 1,
            'mask_weared_incorrect': 2
        }
        self.class_names = ['without_mask', 'with_mask', 'mask_weared_incorrect']
        
        # Load samples
        self._load_samples()
        
        # Apply split
        if split != 'all':
            self._apply_split()
        
        # Set up transforms
        if transforms is None:
            self.transforms = FaceMaskTransforms.get_classification_transforms(augment=(split == 'train'))
        
        print(f"Dataset initialized with {len(self)} samples")
    
    def _load_samples(self):
        """Load samples from dataset"""
        # Check cache first
        if self.cache_processed and hasattr(self, 'cache_manager'):
            cached_samples = self.cache_manager.load_from_cache(
                self.kaggle_dataset_id, 'all', 'classification', None
            )
            if cached_samples is not None:
                self.samples = cached_samples
                return
        
        print("Processing Face12k samples...")
        self.samples = []
        
        # Load samples from each class folder
        for class_name, class_idx in self.class_folders.items():
            class_path = os.path.join(self.dataset_path, class_name)
            
            if not os.path.exists(class_path):
                print(f"Warning: Class folder {class_path} not found")
                continue
            
            for img_file in os.listdir(class_path):
                if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(class_path, img_file)
                    self.samples.append({
                        'image_path': img_path,
                        'label': class_idx,
                        'class_name': class_name
                    })
        
        # Shuffle for better distribution
        random.shuffle(self.samples)
        
        # Cache processed samples
        if self.cache_processed and hasattr(self, 'cache_manager'):
            self.cache_manager.save_to_cache(
                self.kaggle_dataset_id, 'all', 'classification', None, self.samples
            )
    
    def _apply_split(self):
        """Apply train/val/test split"""
        total_samples = len(self.samples)
        
        # Calculate split indices
        train_size = int(self.split_ratios[0] * total_samples)
        val_size = int(self.split_ratios[1] * total_samples)
        
        if self.split == 'train':
            self.samples = self.samples[:train_size]
        elif self.split == 'val':
            self.samples = self.samples[train_size:train_size + val_size]
        elif self.split == 'test':
            self.samples = self.samples[train_size + val_size:]
    
    def __len__(self) -> int:
        """Return number of samples"""
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """Get sample at index"""
        sample = self.samples[idx]
        
        # Load image
        try:
            image = cv2.imread(sample['image_path'])
            if image is None:
                # Return dummy image if loading fails
                image = np.zeros((224, 224, 3), dtype=np.uint8)
            else:
                # Convert BGR to RGB
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            image = Image.fromarray(image)
        except Exception as e:
            print(f"Warning: Could not load image {sample['image_path']}: {e}")
            image = Image.new('RGB', (224, 224), color=(0, 0, 0))
        
        label = sample['label']
        
        # Apply transforms
        if self.transforms:
            image = self.transforms(image)
        
        return image, label
    
    def get_class_names(self) -> List[str]:
        """Get list of class names"""
        return self.class_names.copy()
    
    def get_dataset_info(self) -> Dict[str, Any]:
        """Get dataset information"""
        return {
            'dataset_id': self.kaggle_dataset_id,
            'mode': 'classification',
            'split': self.split,
            'num_samples': len(self),
            'num_classes': len(self.class_names),
            'class_names': self.class_names,
            'split_ratios': self.split_ratios
        }


class MedicalMaskPyTorchDataset(Dataset):
    """
    PyTorch Dataset for wobotintelligence/face-mask-detection-dataset from Kaggle
    Medical mask detection dataset (similar structure to AndrewMVD)
    """
    
    def __init__(self,
                 kaggle_dataset_id: str = 'wobotintelligence/face-mask-detection-dataset',
                 mode: str = 'detection',
                 split: str = 'train',
                 split_ratios: Tuple[float, float, float] = (0.7, 0.15, 0.15),
                 transforms=None,
                 cache_processed: bool = True,
                 cache_dir: str = "/tmp/face_mask_datasets",
                 download_to_tmp: bool = True,
                 random_seed: int = 42):
        """
        Initialize Medical Mask PyTorch dataset
        
        Args:
            kaggle_dataset_id: Kaggle dataset identifier
            mode: 'detection' or 'classification'
            split: 'train', 'val', 'test', or 'all'
            split_ratios: (train, val, test) ratios
            transforms: Transform pipeline
            cache_processed: Whether to cache processed samples
            cache_dir: Directory for caching
            download_to_tmp: Whether to download to tmp directory
            random_seed: Random seed for reproducible splits
        """
        self.kaggle_dataset_id = kaggle_dataset_id
        self.mode = mode.lower()
        self.split = split.lower()
        self.split_ratios = split_ratios
        self.transforms = transforms
        self.cache_processed = cache_processed
        self.random_seed = random_seed
        
        # Set random seed
        random.seed(random_seed)
        np.random.seed(random_seed)
        
        # Initialize managers
        self.kaggle_manager = KaggleDatasetManager(cache_dir)
        if cache_processed:
            self.cache_manager = DatasetCacheManager()
        
        # Download dataset
        print(f"Initializing Medical Mask dataset (mode: {mode}, split: {split})")
        self.dataset_path = self.kaggle_manager.download_dataset(kaggle_dataset_id)
        
        # Default class mapping (will be updated based on actual data)
        self.class_names = ['without_mask', 'with_mask']
        self.class_mapping = {'without_mask': 0, 'with_mask': 1}
        
        # Load samples (implementation depends on actual dataset structure)
        self._load_samples()
        
        # Apply split
        if split != 'all':
            self._apply_split()
        
        # Set up transforms
        if transforms is None:
            if mode == 'detection':
                self.transforms = FaceMaskTransforms.get_detection_transforms(augment=(split == 'train'))
            else:
                self.transforms = FaceMaskTransforms.get_classification_transforms(augment=(split == 'train'))
        
        print(f"Dataset initialized with {len(self)} samples")
    
    def _load_samples(self):
        """Load samples from dataset (placeholder implementation)"""
        # This is a placeholder - actual implementation would depend on the dataset structure
        print("Warning: MedicalMaskPyTorchDataset is a placeholder implementation")
        print("Please check the actual dataset structure and update this class accordingly")
        
        self.samples = []
        # Add dummy samples to prevent errors
        for i in range(10):
            self.samples.append({
                'image_path': None,
                'boxes': [[10, 10, 50, 50]],
                'labels': ['with_mask'],
                'image_id': i
            })
    
    def _apply_split(self):
        """Apply train/val/test split"""
        total_samples = len(self.samples)
        
        # Calculate split indices
        train_size = int(self.split_ratios[0] * total_samples)
        val_size = int(self.split_ratios[1] * total_samples)
        
        if self.split == 'train':
            self.samples = self.samples[:train_size]
        elif self.split == 'val':
            self.samples = self.samples[train_size:train_size + val_size]
        elif self.split == 'test':
            self.samples = self.samples[train_size + val_size:]
    
    def __len__(self) -> int:
        """Return number of samples"""
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Any]:
        """Get sample at index (placeholder implementation)"""
        # Placeholder implementation - return dummy data
        if self.mode == 'detection':
            image = torch.zeros(3, 416, 416)
            target = {
                'boxes': torch.tensor([[10, 10, 50, 50]], dtype=torch.float32),
                'labels': torch.tensor([1], dtype=torch.int64),
                'image_id': idx
            }
            return image, target
        else:
            image = torch.zeros(3, 224, 224)
            label = 1
            return image, label
    
    def get_class_names(self) -> List[str]:
        """Get list of class names"""
        return self.class_names.copy()
    
    def get_dataset_info(self) -> Dict[str, Any]:
        """Get dataset information"""
        return {
            'dataset_id': self.kaggle_dataset_id,
            'mode': self.mode,
            'split': self.split,
            'num_samples': len(self),
            'num_classes': len(self.class_names),
            'class_names': self.class_names,
            'split_ratios': self.split_ratios
        }


class DatasetFactory:
    """Factory class for creating face mask detection datasets"""
    
    SUPPORTED_DATASETS = {
        'andrewmvd': AndrewMVDPyTorchDataset,
        'face12k': Face12kPyTorchDataset,
        'medical_mask': MedicalMaskPyTorchDataset
    }
    
    @staticmethod
    def create_dataset(dataset_type: str, **kwargs) -> Dataset:
        """
        Create a dataset instance
        
        Args:
            dataset_type: Type of dataset ('andrewmvd', 'face12k', 'medical_mask')
            **kwargs: Dataset-specific arguments
            
        Returns:
            Dataset instance
            
        Raises:
            ValueError: If dataset type is not supported
        """
        dataset_type = dataset_type.lower()
        
        if dataset_type not in DatasetFactory.SUPPORTED_DATASETS:
            raise ValueError(f"Unsupported dataset type: {dataset_type}. "
                           f"Supported: {list(DatasetFactory.SUPPORTED_DATASETS.keys())}")
        
        dataset_class = DatasetFactory.SUPPORTED_DATASETS[dataset_type]
        return dataset_class(**kwargs)
    
    @staticmethod
    def list_supported_datasets() -> List[str]:
        """List all supported dataset types"""
        return list(DatasetFactory.SUPPORTED_DATASETS.keys())


# Convenience functions
def create_andrewmvd_dataset(mode: str = 'detection', split: str = 'train', **kwargs) -> AndrewMVDPyTorchDataset:
    """Create AndrewMVD dataset"""
    return AndrewMVDPyTorchDataset(mode=mode, split=split, **kwargs)


def create_face12k_dataset(split: str = 'train', **kwargs) -> Face12kPyTorchDataset:
    """Create Face12k dataset"""
    return Face12kPyTorchDataset(split=split, **kwargs)


def create_medical_mask_dataset(mode: str = 'detection', split: str = 'train', **kwargs) -> MedicalMaskPyTorchDataset:
    """Create Medical Mask dataset"""
    return MedicalMaskPyTorchDataset(mode=mode, split=split, **kwargs)
