# Dual Filter Face Detection System

A real-time face detection and mask classification system that displays two different processing pipelines side by side for comparison and evaluation.

## Overview

The Dual Filter System allows you to compare face detection and classification algorithms in real-time by showing two video streams side by side:

- **Filter 1 (Left)**: Face detection only - shows detected faces with bounding boxes
- **Filter 2 (Right)**: Face detection + mask classification - shows detected faces with mask status classification

## Features

### 🎯 Core Functionality
- **Real-time Processing**: Live camera feed processing at 30+ FPS
- **Side-by-Side Comparison**: Visual comparison of detection vs classification
- **Multiple Display Modes**: Side-by-side, overlay, and difference views
- **Interactive Controls**: Runtime switching of detectors and parameters

### 🔧 Supported Detectors
- **Haar Cascade**: Fast, CPU-friendly, good for basic detection
- **YuNet**: Accurate ONNX-based detector with good performance
- **MTCNN**: Precise multi-task CNN for high-quality detection
- **RetinaFace**: State-of-the-art detector for best accuracy

### 🎭 Mask Classification
- **2-Class Mode**: With mask / Without mask
- **3-Class Mode**: With mask / Without mask / Incorrect mask
- **Confidence Scoring**: Real-time confidence display
- **Statistics Tracking**: Class distribution and performance metrics

### 🎮 Interactive Controls
- **Real-time Detector Switching**: Press 1-4 to switch detectors
- **Confidence Adjustment**: +/- keys to adjust detection threshold
- **Display Modes**: Toggle between side-by-side, overlay, and difference views
- **Recording**: Save video output and screenshots
- **Statistics**: Real-time performance monitoring

## Installation

### Prerequisites
```bash
# Core dependencies
pip install opencv-python numpy torch torchvision

# Optional GPU acceleration
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For advanced detectors
pip install facenet-pytorch retinaface-pytorch
```

### Setup
1. Clone or copy the `dual_filter_system` directory to your project
2. Ensure the `yewon_pipeline` directory is available (for detector models)
3. Install required dependencies listed above

## Quick Start

### Basic Usage (Detection Only)
```python
from dual_filter_system import DualFilterDetector

# Initialize with basic Haar cascade detector
detector = DualFilterDetector(
    detector_name='haar',
    camera_index=0,
    confidence_threshold=0.6
)

# Run the dual filter system
detector.run()
```

### Advanced Usage (With Classification)
```python
from dual_filter_system import DualFilterDetector

# Initialize with YuNet detector and mask classifier
detector = DualFilterDetector(
    detector_name='yunet',
    classifier_model_path='path/to/your/trained_model.pth',
    camera_index=0,
    confidence_threshold=0.6,
    device='cuda',  # Use GPU if available
    num_classes=2   # 2 for with/without mask, 3 for with/without/incorrect
)

detector.run()
```

### Command Line Interface
```bash
# Basic usage
python dual_filter_detection.py

# With specific detector
python dual_filter_detection.py --detector yunet --confidence 0.7

# With classifier
python dual_filter_detection.py --detector yunet --classifier path/to/model.pth --device cuda

# Full options
python dual_filter_detection.py \
    --detector yunet \
    --classifier path/to/model.pth \
    --camera 0 \
    --confidence 0.6 \
    --device cuda \
    --width 640 \
    --height 480 \
    --classes 2
```

## Examples

### Run Example Scripts
```bash
# Interactive examples menu
python examples/dual_filter_example.py

# Specific examples
python examples/dual_filter_example.py --example basic
python examples/dual_filter_example.py --example advanced
python examples/dual_filter_example.py --example comparison
python examples/dual_filter_example.py --example test
```

## Controls Reference

### Keyboard Controls
| Key | Action |
|-----|--------|
| `Q` / `ESC` | Quit application |
| `S` | Save screenshot |
| `V` | Toggle video recording |
| `1` | Switch to Haar Cascade detector |
| `2` | Switch to YuNet detector |
| `3` | Switch to MTCNN detector |
| `4` | Switch to RetinaFace detector |
| `C` | Toggle classifier on/off |
| `O` | Toggle overlay comparison mode |
| `D` | Toggle difference view |
| `+` / `=` | Increase confidence threshold |
| `-` / `_` | Decrease confidence threshold |
| `T` | Toggle statistics display |
| `F` | Toggle FPS counter |
| `R` | Reset statistics |

### Display Modes
- **Side-by-Side**: Default mode showing both filters
- **Overlay**: Blended view of both filters
- **Difference**: Highlights differences between filters

## Architecture

### Core Components

```
dual_filter_system/
├── __init__.py                 # Package initialization
├── dual_filter_detection.py   # Main coordinator class
├── filters/                    # Filter implementations
│   ├── detection_filter.py    # Detection-only filter
│   ├── classification_filter.py # Detection + classification filter
│   └── filter_display.py      # Display management
├── ui/                         # User interface
│   └── controls.py            # Keyboard controls
└── examples/                   # Example scripts
    └── dual_filter_example.py # Usage examples
```

### Class Hierarchy

```python
DualFilterDetector
├── DetectionOnlyFilter
├── ClassificationFilter
├── FilterDisplay
└── ControlsManager
```

## Integration with Yewon Pipeline

The system integrates seamlessly with the existing `yewon_pipeline` scaffolding:

### Detector Integration
- Uses `detectors_2.py` for face detection implementations
- Supports all detectors: Haar, YuNet, MTCNN, RetinaFace
- Automatic fallback to OpenCV Haar cascade if yewon detectors unavailable

### Classifier Integration
- Loads trained models from `pipeline_1.py` training output
- Supports PyTorch models (.pth, .pt files)
- Compatible with ResNet, MobileNet, EfficientNet architectures
- Automatic preprocessing and inference

### Model Files
- YuNet: `yewon_pipeline/face_detection_yunet_2023mar.onnx`
- Haar: `yewon_pipeline/haarcascade_frontalface_default.xml`
- Trained classifiers: Output from `pipeline_1.py` training

## Performance Optimization

### GPU Acceleration
```python
# Enable GPU processing
detector = DualFilterDetector(
    detector_name='yunet',
    device='cuda',
    classifier_model_path='model.pth'
)
```

### CPU Optimization
```python
# Optimize for CPU
detector = DualFilterDetector(
    detector_name='haar',  # Fastest detector
    device='cpu',
    frame_width=320,       # Smaller resolution
    frame_height=240
)
```

### Memory Management
- Automatic frame resizing for consistent performance
- Efficient memory usage with frame copying
- GPU memory management for CUDA operations

## Troubleshooting

### Common Issues

**Camera not opening:**
```bash
# Test camera access
python examples/dual_filter_example.py --example test
```

**Import errors:**
- Ensure `yewon_pipeline` is in the correct relative path
- Install missing dependencies: `pip install opencv-python torch`

**GPU issues:**
- Check CUDA installation: `torch.cuda.is_available()`
- Fall back to CPU: `--device cpu`

**Model loading errors:**
- Verify model file paths exist
- Check model compatibility with PyTorch version
- Ensure model was trained with compatible architecture

### Performance Issues

**Low FPS:**
- Reduce frame resolution: `--width 320 --height 240`
- Use faster detector: `--detector haar`
- Disable classifier temporarily: Don't specify `--classifier`

**High memory usage:**
- Use CPU instead of GPU: `--device cpu`
- Reduce batch processing in classifier
- Close other applications using camera

## API Reference

### DualFilterDetector
Main class coordinating the dual filter system.

```python
class DualFilterDetector:
    def __init__(self,
                 detector_name: str = 'haar',
                 classifier_model_path: Optional[str] = None,
                 camera_index: int = 0,
                 confidence_threshold: float = 0.6,
                 device: str = 'cuda',
                 frame_width: int = 640,
                 frame_height: int = 480,
                 num_classes: int = 2)
    
    def run(self) -> None
    def start_camera(self) -> bool
    def stop_camera(self) -> None
```

### DetectionOnlyFilter
Filter for face detection without classification.

```python
class DetectionOnlyFilter:
    def __init__(self, detector_name: str, device: str, confidence_threshold: float)
    def process_frame(self, frame: np.ndarray) -> np.ndarray
    def switch_detector(self, new_detector_name: str) -> bool
    def get_stats(self) -> Dict[str, Any]
```

### ClassificationFilter
Filter for face detection with mask classification.

```python
class ClassificationFilter:
    def __init__(self, detector_name: str, classifier_model_path: str, ...)
    def process_frame(self, frame: np.ndarray) -> np.ndarray
    def classify_mask(self, face_roi: np.ndarray) -> Tuple[str, float]
    def get_stats(self) -> Dict[str, Any]
```

## Contributing

### Adding New Detectors
1. Implement detector in `yewon_pipeline/detectors_2.py`
2. Add to `build_detector()` factory function
3. Update `ControlsManager.available_detectors`

### Adding New Classifiers
1. Train model using `yewon_pipeline/pipeline_1.py`
2. Save model with proper state dict format
3. Test loading in `ClassificationFilter._load_yewon_classifier()`

### Adding New Display Modes
1. Add mode to `DisplayMode` enum
2. Implement in `FilterDisplay` class
3. Add keyboard control in `ControlsManager`

## License

This project is part of the Real-Time Face Mask Detection system. Please refer to the main project license.

## Acknowledgments

- Built on top of the `yewon_pipeline` scaffolding
- Integrates with existing evaluation framework
- Uses OpenCV for computer vision operations
- PyTorch for deep learning inference
