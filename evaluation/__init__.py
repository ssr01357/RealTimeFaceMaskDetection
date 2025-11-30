"""
Face Mask Detection Evaluation Pipeline

A comprehensive evaluation suite for face mask detection systems including:
- Face detection evaluation (YuNet, Haar, etc.)
- Classification evaluation (custom CNNs)
- Full pipeline evaluation (detection + classification)
- Real-time performance benchmarking
- Multiple dataset support
"""

__version__ = "1.0.0"
__author__ = "CS583 Face Mask Detection Team"

from .eval_pipeline import EvaluationPipeline
from .detectors.detector_wrapper import FaceDetectorWrapper
from .classifiers.classifier_wrapper import ClassifierWrapper
from .datasets.dataset_loaders import DatasetLoader

__all__ = [
    'EvaluationPipeline',
    'FaceDetectorWrapper', 
    'ClassifierWrapper',
    'DatasetLoader'
]
