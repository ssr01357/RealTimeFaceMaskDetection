"""
Kaggle Dataset Manager for Face Mask Detection Datasets
Handles downloading, caching, and validation of Kaggle datasets
"""

import os
import shutil
import tempfile
import hashlib
from typing import Optional, Dict, Any
from pathlib import Path
import json
import time


class KaggleDatasetManager:
    """Manages Kaggle dataset downloads and caching"""
    
    # Supported datasets with their metadata
    SUPPORTED_DATASETS = {
        'andrewmvd/face-mask-detection': {
            'name': 'andrewmvd_face_mask',
            'type': 'detection',
            'expected_folders': ['images', 'annotations'],
            'description': 'Face mask detection with bounding boxes'
        },
        'ashishjangra27/face-mask-12k-images-dataset': {
            'name': 'face_mask_12k',
            'type': 'classification',
            'expected_folders': ['without_mask', 'with_mask', 'mask_weared_incorrect'],
            'description': '12k face mask classification images'
        },
        'wobotintelligence/face-mask-detection-dataset': {
            'name': 'medical_mask',
            'type': 'detection',
            'expected_folders': ['images', 'annotations'],
            'description': 'Medical mask detection dataset'
        }
    }
    
    def __init__(self, cache_dir: str = "/tmp/face_mask_datasets"):
        """
        Initialize Kaggle dataset manager
        
        Args:
            cache_dir: Directory to cache downloaded datasets
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Create metadata file for tracking downloads
        self.metadata_file = self.cache_dir / "dataset_metadata.json"
        self.metadata = self._load_metadata()
    
    def _load_metadata(self) -> Dict[str, Any]:
        """Load dataset metadata from cache"""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                pass
        return {}
    
    def _save_metadata(self):
        """Save dataset metadata to cache"""
        try:
            with open(self.metadata_file, 'w') as f:
                json.dump(self.metadata, f, indent=2)
        except IOError:
            print(f"Warning: Could not save metadata to {self.metadata_file}")
    
    def _get_dataset_cache_path(self, dataset_id: str) -> Path:
        """Get cache path for a dataset"""
        if dataset_id not in self.SUPPORTED_DATASETS:
            raise ValueError(f"Unsupported dataset: {dataset_id}")
        
        dataset_name = self.SUPPORTED_DATASETS[dataset_id]['name']
        return self.cache_dir / dataset_name
    
    def _verify_dataset_structure(self, dataset_path: Path, dataset_id: str) -> bool:
        """Verify that downloaded dataset has expected structure"""
        if not dataset_path.exists():
            return False
        
        expected_folders = self.SUPPORTED_DATASETS[dataset_id]['expected_folders']
        
        for folder in expected_folders:
            folder_path = dataset_path / folder
            if not folder_path.exists() or not folder_path.is_dir():
                return False
            
            # Check if folder has any files
            if not any(folder_path.iterdir()):
                return False
        
        return True
    
    def is_dataset_cached(self, dataset_id: str) -> bool:
        """Check if dataset is already cached and valid"""
        if dataset_id not in self.SUPPORTED_DATASETS:
            return False
        
        cache_path = self._get_dataset_cache_path(dataset_id)
        
        # Check if path exists and has valid structure
        if not self._verify_dataset_structure(cache_path, dataset_id):
            return False
        
        # Check metadata
        if dataset_id in self.metadata:
            metadata = self.metadata[dataset_id]
            if metadata.get('status') == 'complete' and metadata.get('path') == str(cache_path):
                return True
        
        return False
    
    def get_cached_dataset_path(self, dataset_id: str) -> Optional[str]:
        """Get path to cached dataset if available"""
        if self.is_dataset_cached(dataset_id):
            return str(self._get_dataset_cache_path(dataset_id))
        return None
    
    def download_dataset(self, dataset_id: str, force_redownload: bool = False) -> str:
        """
        Download dataset from Kaggle
        
        Args:
            dataset_id: Kaggle dataset identifier (e.g., 'andrewmvd/face-mask-detection')
            force_redownload: Force redownload even if cached
            
        Returns:
            Path to downloaded dataset
            
        Raises:
            ValueError: If dataset is not supported
            ImportError: If kagglehub is not available
            RuntimeError: If download fails
        """
        if dataset_id not in self.SUPPORTED_DATASETS:
            raise ValueError(f"Unsupported dataset: {dataset_id}. "
                           f"Supported: {list(self.SUPPORTED_DATASETS.keys())}")
        
        # Check if already cached and valid
        if not force_redownload and self.is_dataset_cached(dataset_id):
            cached_path = self.get_cached_dataset_path(dataset_id)
            print(f"Using cached dataset: {cached_path}")
            return cached_path
        
        try:
            import kagglehub
        except ImportError:
            raise ImportError("kagglehub is required for dataset downloading. "
                            "Install with: pip install kagglehub")
        
        print(f"Downloading dataset: {dataset_id}")
        
        try:
            # Download to temporary location first
            temp_path = kagglehub.dataset_download(dataset_id)
            temp_path = Path(temp_path)
            
            # Verify download
            if not self._verify_dataset_structure(temp_path, dataset_id):
                raise RuntimeError(f"Downloaded dataset has invalid structure: {temp_path}")
            
            # Move to cache location
            cache_path = self._get_dataset_cache_path(dataset_id)
            
            # Remove existing cache if present
            if cache_path.exists():
                shutil.rmtree(cache_path)
            
            # Move downloaded data to cache
            shutil.move(str(temp_path), str(cache_path))
            
            # Update metadata
            self.metadata[dataset_id] = {
                'path': str(cache_path),
                'download_time': time.time(),
                'status': 'complete',
                'dataset_info': self.SUPPORTED_DATASETS[dataset_id]
            }
            self._save_metadata()
            
            print(f"Dataset cached at: {cache_path}")
            return str(cache_path)
            
        except Exception as e:
            raise RuntimeError(f"Failed to download dataset {dataset_id}: {str(e)}")
    
    def clear_cache(self, dataset_id: Optional[str] = None):
        """
        Clear cached datasets
        
        Args:
            dataset_id: Specific dataset to clear, or None to clear all
        """
        if dataset_id is None:
            # Clear all cached datasets
            if self.cache_dir.exists():
                shutil.rmtree(self.cache_dir)
                self.cache_dir.mkdir(parents=True, exist_ok=True)
            self.metadata = {}
            self._save_metadata()
            print("Cleared all cached datasets")
        else:
            # Clear specific dataset
            if dataset_id in self.SUPPORTED_DATASETS:
                cache_path = self._get_dataset_cache_path(dataset_id)
                if cache_path.exists():
                    shutil.rmtree(cache_path)
                
                if dataset_id in self.metadata:
                    del self.metadata[dataset_id]
                    self._save_metadata()
                
                print(f"Cleared cached dataset: {dataset_id}")
            else:
                print(f"Unknown dataset: {dataset_id}")
    
    def get_cache_info(self) -> Dict[str, Any]:
        """Get information about cached datasets"""
        info = {
            'cache_dir': str(self.cache_dir),
            'total_datasets': len(self.metadata),
            'datasets': {}
        }
        
        total_size = 0
        for dataset_id, metadata in self.metadata.items():
            cache_path = Path(metadata['path'])
            size = 0
            if cache_path.exists():
                size = sum(f.stat().st_size for f in cache_path.rglob('*') if f.is_file())
            
            info['datasets'][dataset_id] = {
                'path': metadata['path'],
                'size_mb': size / (1024 * 1024),
                'download_time': metadata.get('download_time'),
                'status': metadata.get('status'),
                'valid': self.is_dataset_cached(dataset_id)
            }
            total_size += size
        
        info['total_size_mb'] = total_size / (1024 * 1024)
        return info
    
    def list_supported_datasets(self) -> Dict[str, Dict[str, Any]]:
        """List all supported datasets with their information"""
        return self.SUPPORTED_DATASETS.copy()


# Convenience function for quick dataset access
def get_dataset_path(dataset_id: str, cache_dir: str = "/tmp/face_mask_datasets", 
                    force_redownload: bool = False) -> str:
    """
    Convenience function to get dataset path, downloading if necessary
    
    Args:
        dataset_id: Kaggle dataset identifier
        cache_dir: Cache directory
        force_redownload: Force redownload even if cached
        
    Returns:
        Path to dataset
    """
    manager = KaggleDatasetManager(cache_dir)
    return manager.download_dataset(dataset_id, force_redownload)
