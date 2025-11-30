"""
Cache Manager for Face Mask Detection Datasets
Handles caching of processed dataset samples for faster loading
"""

import os
import pickle
import hashlib
import json
import time
import shutil
from typing import Any, Dict, Optional, List, Tuple
from pathlib import Path
import torch
import numpy as np


class DatasetCacheManager:
    """Manages caching of processed dataset samples"""
    
    def __init__(self, cache_dir: str = "/tmp/face_mask_cache", 
                 max_cache_size_gb: float = 5.0):
        """
        Initialize cache manager
        
        Args:
            cache_dir: Directory to store cached data
            max_cache_size_gb: Maximum cache size in GB
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_cache_size = max_cache_size_gb * 1024 * 1024 * 1024  # Convert to bytes
        
        # Cache metadata
        self.metadata_file = self.cache_dir / "cache_metadata.json"
        self.metadata = self._load_metadata()
    
    def _load_metadata(self) -> Dict[str, Any]:
        """Load cache metadata"""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                pass
        return {
            'datasets': {},
            'total_size': 0,
            'last_cleanup': time.time()
        }
    
    def _save_metadata(self):
        """Save cache metadata"""
        try:
            with open(self.metadata_file, 'w') as f:
                json.dump(self.metadata, f, indent=2)
        except IOError:
            print(f"Warning: Could not save cache metadata to {self.metadata_file}")
    
    def _get_cache_key(self, dataset_id: str, split: str, mode: str, 
                      transform_hash: str) -> str:
        """Generate cache key for dataset configuration"""
        key_string = f"{dataset_id}_{split}_{mode}_{transform_hash}"
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def _get_transform_hash(self, transforms) -> str:
        """Generate hash for transform configuration"""
        if transforms is None:
            return "no_transforms"
        
        # Create a string representation of transforms
        transform_str = str(transforms.__class__.__name__)
        if hasattr(transforms, 'image_size'):
            transform_str += f"_size{transforms.image_size}"
        if hasattr(transforms, 'augment'):
            transform_str += f"_aug{transforms.augment}"
        
        return hashlib.md5(transform_str.encode()).hexdigest()[:8]
    
    def _get_cache_path(self, cache_key: str) -> Path:
        """Get cache file path for a cache key"""
        return self.cache_dir / f"{cache_key}.pkl"
    
    def _calculate_directory_size(self, directory: Path) -> int:
        """Calculate total size of directory in bytes"""
        total_size = 0
        try:
            for file_path in directory.rglob('*'):
                if file_path.is_file():
                    total_size += file_path.stat().st_size
        except (OSError, IOError):
            pass
        return total_size
    
    def _cleanup_old_cache(self):
        """Remove old cache files if cache size exceeds limit"""
        current_size = self._calculate_directory_size(self.cache_dir)
        
        if current_size <= self.max_cache_size:
            return
        
        print(f"Cache size ({current_size / (1024**3):.2f} GB) exceeds limit "
              f"({self.max_cache_size / (1024**3):.2f} GB). Cleaning up...")
        
        # Get all cache files with their access times
        cache_files = []
        for cache_file in self.cache_dir.glob("*.pkl"):
            try:
                stat = cache_file.stat()
                cache_files.append((cache_file, stat.st_atime))
            except OSError:
                continue
        
        # Sort by access time (oldest first)
        cache_files.sort(key=lambda x: x[1])
        
        # Remove oldest files until under limit
        for cache_file, _ in cache_files:
            try:
                cache_file.unlink()
                current_size -= cache_file.stat().st_size
                
                # Remove from metadata
                cache_key = cache_file.stem
                for dataset_id in list(self.metadata['datasets'].keys()):
                    if cache_key in self.metadata['datasets'][dataset_id].get('cache_keys', []):
                        self.metadata['datasets'][dataset_id]['cache_keys'].remove(cache_key)
                        break
                
                if current_size <= self.max_cache_size * 0.8:  # Leave some buffer
                    break
            except OSError:
                continue
        
        self.metadata['last_cleanup'] = time.time()
        self._save_metadata()
        print(f"Cache cleanup completed. New size: {current_size / (1024**3):.2f} GB")
    
    def is_cached(self, dataset_id: str, split: str, mode: str, transforms) -> bool:
        """Check if dataset configuration is cached"""
        transform_hash = self._get_transform_hash(transforms)
        cache_key = self._get_cache_key(dataset_id, split, mode, transform_hash)
        cache_path = self._get_cache_path(cache_key)
        
        return cache_path.exists()
    
    def save_to_cache(self, dataset_id: str, split: str, mode: str, transforms,
                     samples: List[Tuple[Any, Any]]):
        """
        Save processed samples to cache
        
        Args:
            dataset_id: Dataset identifier
            split: Dataset split (train/val/test)
            mode: Dataset mode (detection/classification)
            transforms: Transform configuration
            samples: List of (image, target) tuples
        """
        transform_hash = self._get_transform_hash(transforms)
        cache_key = self._get_cache_key(dataset_id, split, mode, transform_hash)
        cache_path = self._get_cache_path(cache_key)
        
        try:
            # Save samples to cache
            with open(cache_path, 'wb') as f:
                pickle.dump(samples, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            # Update metadata
            if dataset_id not in self.metadata['datasets']:
                self.metadata['datasets'][dataset_id] = {
                    'cache_keys': [],
                    'splits': {}
                }
            
            if cache_key not in self.metadata['datasets'][dataset_id]['cache_keys']:
                self.metadata['datasets'][dataset_id]['cache_keys'].append(cache_key)
            
            self.metadata['datasets'][dataset_id]['splits'][f"{split}_{mode}"] = {
                'cache_key': cache_key,
                'transform_hash': transform_hash,
                'num_samples': len(samples),
                'cached_time': time.time(),
                'file_size': cache_path.stat().st_size
            }
            
            self._save_metadata()
            
            # Check if cleanup is needed
            self._cleanup_old_cache()
            
            print(f"Cached {len(samples)} samples for {dataset_id} ({split}, {mode})")
            
        except Exception as e:
            print(f"Failed to cache samples: {e}")
            if cache_path.exists():
                cache_path.unlink()
    
    def load_from_cache(self, dataset_id: str, split: str, mode: str, 
                       transforms) -> Optional[List[Tuple[Any, Any]]]:
        """
        Load processed samples from cache
        
        Args:
            dataset_id: Dataset identifier
            split: Dataset split
            mode: Dataset mode
            transforms: Transform configuration
            
        Returns:
            List of cached samples or None if not cached
        """
        transform_hash = self._get_transform_hash(transforms)
        cache_key = self._get_cache_key(dataset_id, split, mode, transform_hash)
        cache_path = self._get_cache_path(cache_key)
        
        if not cache_path.exists():
            return None
        
        try:
            with open(cache_path, 'rb') as f:
                samples = pickle.load(f)
            
            # Update access time
            cache_path.touch()
            
            print(f"Loaded {len(samples)} cached samples for {dataset_id} ({split}, {mode})")
            return samples
            
        except Exception as e:
            print(f"Failed to load cached samples: {e}")
            # Remove corrupted cache file
            try:
                cache_path.unlink()
            except OSError:
                pass
            return None
    
    def clear_cache(self, dataset_id: Optional[str] = None):
        """
        Clear cache for specific dataset or all datasets
        
        Args:
            dataset_id: Dataset to clear, or None for all
        """
        if dataset_id is None:
            # Clear all cache
            shutil.rmtree(self.cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self.metadata = {
                'datasets': {},
                'total_size': 0,
                'last_cleanup': time.time()
            }
            self._save_metadata()
            print("Cleared all cache")
        else:
            # Clear specific dataset cache
            if dataset_id in self.metadata['datasets']:
                cache_keys = self.metadata['datasets'][dataset_id]['cache_keys']
                for cache_key in cache_keys:
                    cache_path = self._get_cache_path(cache_key)
                    if cache_path.exists():
                        cache_path.unlink()
                
                del self.metadata['datasets'][dataset_id]
                self._save_metadata()
                print(f"Cleared cache for dataset: {dataset_id}")
    
    def get_cache_info(self) -> Dict[str, Any]:
        """Get cache statistics"""
        total_size = self._calculate_directory_size(self.cache_dir)
        
        info = {
            'cache_dir': str(self.cache_dir),
            'total_size_mb': total_size / (1024 * 1024),
            'max_size_gb': self.max_cache_size / (1024 * 1024 * 1024),
            'usage_percent': (total_size / self.max_cache_size) * 100,
            'num_datasets': len(self.metadata['datasets']),
            'datasets': {}
        }
        
        for dataset_id, dataset_info in self.metadata['datasets'].items():
            dataset_size = 0
            total_samples = 0
            
            for split_info in dataset_info['splits'].values():
                dataset_size += split_info.get('file_size', 0)
                total_samples += split_info.get('num_samples', 0)
            
            info['datasets'][dataset_id] = {
                'size_mb': dataset_size / (1024 * 1024),
                'total_samples': total_samples,
                'num_splits': len(dataset_info['splits']),
                'splits': list(dataset_info['splits'].keys())
            }
        
        return info


# Utility functions
def create_cache_manager(cache_dir: str = "/tmp/face_mask_cache", 
                        max_size_gb: float = 5.0) -> DatasetCacheManager:
    """
    Create a cache manager instance
    
    Args:
        cache_dir: Cache directory
        max_size_gb: Maximum cache size in GB
        
    Returns:
        DatasetCacheManager instance
    """
    return DatasetCacheManager(cache_dir, max_size_gb)


def clear_all_cache(cache_dir: str = "/tmp/face_mask_cache"):
    """
    Clear all cached data
    
    Args:
        cache_dir: Cache directory to clear
    """
    cache_manager = DatasetCacheManager(cache_dir)
    cache_manager.clear_cache()


def get_cache_stats(cache_dir: str = "/tmp/face_mask_cache") -> Dict[str, Any]:
    """
    Get cache statistics
    
    Args:
        cache_dir: Cache directory
        
    Returns:
        Cache statistics dictionary
    """
    cache_manager = DatasetCacheManager(cache_dir)
    return cache_manager.get_cache_info()
