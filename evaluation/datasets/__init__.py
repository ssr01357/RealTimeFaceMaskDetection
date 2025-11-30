"""
Face Mask Detection Datasets
PyTorch Dataset classes with Kaggle integration and caching
"""

from .dataset_loaders import (
    AndrewMVDPyTorchDataset,
    Face12kPyTorchDataset,
    MedicalMaskPyTorchDataset,
    DatasetFactory,
    create_andrewmvd_dataset,
    create_face12k_dataset,
    create_medical_mask_dataset
)

from .kaggle_manager import (
    KaggleDatasetManager,
    get_dataset_path
)

from .transforms import (
    FaceMaskTransforms,
    DetectionTransforms,
    ClassificationTransforms,
    InferenceTransforms,
    denormalize_tensor,
    tensor_to_pil,
    visualize_detection_sample
)

from .cache_manager import (
    DatasetCacheManager,
    create_cache_manager,
    clear_all_cache,
    get_cache_stats
)

__all__ = [
    # Dataset classes
    'AndrewMVDPyTorchDataset',
    'Face12kPyTorchDataset', 
    'MedicalMaskPyTorchDataset',
    'DatasetFactory',
    
    # Convenience functions
    'create_andrewmvd_dataset',
    'create_face12k_dataset',
    'create_medical_mask_dataset',
    
    # Kaggle management
    'KaggleDatasetManager',
    'get_dataset_path',
    
    # Transforms
    'FaceMaskTransforms',
    'DetectionTransforms',
    'ClassificationTransforms',
    'InferenceTransforms',
    'denormalize_tensor',
    'tensor_to_pil',
    'visualize_detection_sample',
    
    # Cache management
    'DatasetCacheManager',
    'create_cache_manager',
    'clear_all_cache',
    'get_cache_stats'
]

# Quick access functions for common use cases
def get_andrewmvd_detection_dataset(split='train', **kwargs):
    """Quick access to AndrewMVD detection dataset"""
    return create_andrewmvd_dataset(mode='detection', split=split, **kwargs)

def get_andrewmvd_classification_dataset(split='train', **kwargs):
    """Quick access to AndrewMVD classification dataset"""
    return create_andrewmvd_dataset(mode='classification', split=split, **kwargs)

def get_face12k_dataset(split='train', **kwargs):
    """Quick access to Face12k classification dataset"""
    return create_face12k_dataset(split=split, **kwargs)

# Add quick access functions to __all__
__all__.extend([
    'get_andrewmvd_detection_dataset',
    'get_andrewmvd_classification_dataset',
    'get_face12k_dataset'
])
