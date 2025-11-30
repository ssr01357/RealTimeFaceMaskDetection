"""
Dual Filter Face Detection System

A real-time face detection and mask classification system that displays
two different processing pipelines side by side for comparison.

Main Components:
- DetectionOnlyFilter: Pure face detection without classification
- ClassificationFilter: Face detection with mask classification
- FilterDisplay: Side-by-side video display manager
- ControlsManager: Keyboard controls and user interactions
- DualFilterDetector: Main coordinator class

Usage:
    from dual_filter_system import DualFilterDetector
    
    detector = DualFilterDetector(
        detector_name='haar',
        classifier_model_path='path/to/model.pth',
        camera_index=0
    )
    detector.run()
"""

from .dual_filter_detection import DualFilterDetector, DisplayMode
from .filters.detection_filter import DetectionOnlyFilter
from .filters.classification_filter import ClassificationFilter
from .filters.filter_display import FilterDisplay
from .ui.controls import ControlsManager, ControlAction

__version__ = "1.0.0"
__author__ = "Face Mask Detection Team"

__all__ = [
    'DualFilterDetector',
    'DisplayMode',
    'DetectionOnlyFilter',
    'ClassificationFilter',
    'FilterDisplay',
    'ControlsManager',
    'ControlAction'
]
