"""
Example script demonstrating how to use the PyTorch dataset classes
for face mask detection with Kaggle integration.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from evaluation.datasets import (
    AndrewMVDPyTorchDataset,
    Face12kPyTorchDataset,
    DatasetFactory,
    FaceMaskTransforms
)


def main():
    print("Face Mask Detection Dataset Example")
    print("=" * 40)
    
    # Example 1: Create AndrewMVD detection dataset
    print("\n1. Creating AndrewMVD Detection Dataset")
    print("-" * 40)
    
    try:
        detection_dataset = AndrewMVDPyTorchDataset(
            mode='detection',
            split='train',
            split_ratios=(0.7, 0.15, 0.15),
            cache_dir='/tmp/face_mask_datasets'
        )
        
        print(f"✓ Detection dataset created with {len(detection_dataset)} samples")
        print(f"  Classes: {detection_dataset.get_class_names()}")
        
        # Test loading a sample
        if len(detection_dataset) > 0:
            image, target = detection_dataset[0]
            print(f"  Sample image shape: {image.shape}")
            print(f"  Number of faces: {len(target['boxes'])}")
            
    except Exception as e:
        print(f"✗ Error creating detection dataset: {e}")
    
    # Example 2: Create AndrewMVD classification dataset
    print("\n2. Creating AndrewMVD Classification Dataset")
    print("-" * 45)
    
    try:
        classification_dataset = AndrewMVDPyTorchDataset(
            mode='classification',
            split='train',
            split_ratios=(0.7, 0.15, 0.15),
            cache_dir='/tmp/face_mask_datasets'
        )
        
        print(f"✓ Classification dataset created with {len(classification_dataset)} samples")
        print(f"  Classes: {classification_dataset.get_class_names()}")
        
        # Test loading a sample
        if len(classification_dataset) > 0:
            image, label = classification_dataset[0]
            print(f"  Sample image shape: {image.shape}")
            print(f"  Sample label: {label} ({classification_dataset.get_class_names()[label]})")
            
    except Exception as e:
        print(f"✗ Error creating classification dataset: {e}")
    
    # Example 3: Using DatasetFactory
    print("\n3. Using DatasetFactory")
    print("-" * 25)
    
    try:
        # Create dataset using factory
        factory_dataset = DatasetFactory.create_dataset(
            'andrewmvd',
            mode='classification',
            split='val',
            cache_dir='/tmp/face_mask_datasets'
        )
        
        print(f"✓ Factory dataset created with {len(factory_dataset)} samples")
        
    except Exception as e:
        print(f"✗ Error using DatasetFactory: {e}")
    
    # Example 4: DataLoader integration
    print("\n4. DataLoader Integration")
    print("-" * 25)
    
    try:
        # Create a small dataset for DataLoader testing
        test_dataset = AndrewMVDPyTorchDataset(
            mode='classification',
            split='val',
            cache_dir='/tmp/face_mask_datasets'
        )
        
        if len(test_dataset) > 0:
            # Create DataLoader
            dataloader = DataLoader(
                test_dataset,
                batch_size=4,
                shuffle=True,
                num_workers=0
            )
            
            print(f"✓ DataLoader created")
            print(f"  Dataset size: {len(test_dataset)}")
            print(f"  Batch size: {dataloader.batch_size}")
            print(f"  Number of batches: {len(dataloader)}")
            
            # Test loading a batch
            batch_images, batch_labels = next(iter(dataloader))
            print(f"  Batch shape: {batch_images.shape}")
            print(f"  Batch labels: {batch_labels.tolist()}")
            
        else:
            print("✗ Dataset is empty, skipping DataLoader test")
            
    except Exception as e:
        print(f"✗ Error with DataLoader: {e}")
    
    # Example 5: Custom transforms
    print("\n5. Custom Transforms")
    print("-" * 20)
    
    try:
        # Create custom transforms
        custom_transforms = FaceMaskTransforms.get_classification_transforms(
            image_size=256,
            augment=True
        )
        
        # Create dataset with custom transforms
        custom_dataset = AndrewMVDPyTorchDataset(
            mode='classification',
            split='val',
            transforms=custom_transforms,
            cache_dir='/tmp/face_mask_datasets'
        )
        
        print(f"✓ Custom transforms dataset created with {len(custom_dataset)} samples")
        
        if len(custom_dataset) > 0:
            image, label = custom_dataset[0]
            print(f"  Custom image shape: {image.shape}")
            
    except Exception as e:
        print(f"✗ Error with custom transforms: {e}")
    
    # Example 6: Try Face12k dataset
    print("\n6. Face12k Dataset")
    print("-" * 20)
    
    try:
        face12k_dataset = Face12kPyTorchDataset(
            split='train',
            cache_dir='/tmp/face_mask_datasets'
        )
        
        print(f"✓ Face12k dataset created with {len(face12k_dataset)} samples")
        print(f"  Classes: {face12k_dataset.get_class_names()}")
        
    except Exception as e:
        print(f"✗ Face12k dataset not available: {e}")
    
    print("\n" + "=" * 40)
    print("Example completed!")
    print("\nNext steps:")
    print("- Use these datasets to train face mask detection models")
    print("- Experiment with different augmentation strategies")
    print("- Compare performance across different datasets")


if __name__ == "__main__":
    main()
