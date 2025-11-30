# Real-Time Face Mask Detection System

A comprehensive face mask detection system with unified interface supporting multiple face detectors and mask classifiers for real-time applications.

## 🚀 Quick Start

### Simple Usage
```bash
# Face detection only
python face_mask_detector.py

# With mask classification using your custom PyTorch model
python face_mask_detector.py --classifier custom_pytorch

# Advanced detection with YuNet
python face_mask_detector.py --detector yunet --classifier custom_pytorch
```

### Interactive Demo
```bash
# Run interactive demo menu
python demo_unified_system.py

# Or run specific demos directly
python demo_unified_system.py --demo pytorch
python demo_unified_system.py --demo guide
```

## 🎯 Features

### **Unified Interface**
- **Single Entry Point**: One command for all detector/classifier combinations
- **Runtime Switching**: Change detectors and classifiers without restarting
- **Consistent API**: Same interface for all models

### **Face Detectors**
- **Haar Cascade**: Fast, CPU-friendly (OpenCV built-in)
- **YuNet**: Accurate ONNX-based detector
- **MTCNN**: Precise multi-task CNN (requires GPU)
- **RetinaFace**: State-of-the-art quality (requires GPU)

### **Mask Classifiers**
- **Custom PyTorch**: Your trained PyTorch model
- **Yewon Pipeline**: Research pipeline models
- **sklearn Models**: Traditional ML classifiers
- **Detection Only**: Face detection without classification

### **Real-Time Performance**
- **Live Camera Feed**: Real-time webcam processing
- **Performance Tracking**: FPS counter and statistics
- **Interactive Controls**: Runtime parameter adjustment

## 📁 Project Structure

```
RealTimeFaceMaskDetection/
├── face_mask_detector.py          # 🎯 Main unified system
├── demo_unified_system.py         # 🎮 Interactive demo
├── evaluation/                    # 📊 Evaluation framework
│   ├── detectors/                # Face detection wrappers
│   ├── classifiers/              # Classification wrappers
│   ├── datasets/                 # Dataset utilities
│   └── benchmarks/               # Performance testing
├── dual_filter_system/           # 🎨 UI and display components
│   ├── ui/controls.py            # Keyboard controls
│   └── filters/filter_display.py # Display utilities
├── model_artifacts/              # 🧠 Custom model support
│   └── pytorch_model_loader.py   # PyTorch model loader
├── yewon_pipeline/               # 🔬 Research pipeline
├── reference_code/               # 📚 Reference implementations
├── notebooks/                    # 📓 Jupyter notebooks
└── tests/                        # 🧪 Test suite
```

## 🛠️ Installation

```bash
# Clone repository
git clone https://github.com/clydebaron2000/RealTimeFaceMaskDetection.git
cd RealTimeFaceMaskDetection

# Install dependencies
pip install -r requirements_realtime.txt

# Additional dependencies for advanced features
pip install torch torchvision  # For PyTorch models
pip install onnxruntime        # For YuNet detector
```

## 📖 Usage Guide

### Command Line Options

```bash
python face_mask_detector.py [OPTIONS]

Face Detectors:
  --detector haar        # Haar cascade (default, fast)
  --detector yunet       # YuNet ONNX (accurate)
  --detector mtcnn       # MTCNN (precise, GPU)
  --detector retinaface  # RetinaFace (best quality, GPU)

Classifiers:
  --classifier custom_pytorch  # Your PyTorch model
  --classifier yewon          # Yewon pipeline model
  --classifier sklearn        # sklearn model
  (no classifier)             # Detection only

Other Options:
  --camera 0             # Camera index
  --confidence 0.6       # Detection threshold
  --device cpu           # cpu or cuda
  --width 640            # Frame width
  --height 480           # Frame height
```

### Runtime Controls

During execution, use these keyboard shortcuts:

| Key | Action |
|-----|--------|
| **1-4** | Switch detectors (Haar, YuNet, MTCNN, RetinaFace) |
| **5-8** | Switch classifiers (Custom, Yewon, Wrapper, None) |
| **C** | Toggle classifier on/off |
| **S** | Take screenshot |
| **+/-** | Adjust confidence threshold |
| **Q/ESC** | Quit |

### Examples

```bash
# Basic face detection
python face_mask_detector.py --detector haar

# Custom PyTorch model with Haar detector
python face_mask_detector.py --detector haar --classifier custom_pytorch

# Advanced setup with YuNet and custom model
python face_mask_detector.py --detector yunet --classifier custom_pytorch --device cuda

# Detection only with high-quality RetinaFace
python face_mask_detector.py --detector retinaface --confidence 0.8
```

## 🧠 Model Requirements

### Custom PyTorch Model
- **Location**: `model_artifacts/best_pytorch_model_custom.pth`
- **Input**: 40x40 RGB images
- **Output**: 2 classes (without_mask=0, with_mask=1)

### YuNet Detector
- **Location**: `yewon_pipeline/face_detection_yunet_2023mar.onnx`
- **Format**: ONNX model file
- **Download**: Available from OpenCV model zoo

### Yewon Pipeline Models
- **Location**: `yewon_pipeline/` directory
- **Format**: PyTorch models (.pth files)
- **Classes**: 3-class support (with_mask, without_mask, incorrect_mask)

## 🎮 Interactive Demo

The demo system provides guided examples:

```bash
python demo_unified_system.py
```

**Demo Options:**
1. **Face Detection Only** - Basic Haar cascade detection
2. **Custom PyTorch Model** - Your trained classifier
3. **Advanced YuNet** - High-accuracy detection
4. **Detector Comparison** - Runtime switching demo
5. **Usage Guide** - Comprehensive help

## 🔬 Research Components

### Evaluation Framework
Comprehensive evaluation system for testing different detector/classifier combinations:

```python
from evaluation.eval_pipeline import EvaluationPipeline

pipeline = EvaluationPipeline(output_dir="results")
pipeline.setup_detector('yunet', model_path='models/yunet.onnx')
pipeline.setup_classifier('pytorch', model_path='models/classifier.pth')
results = pipeline.run_comprehensive_evaluation()
```

### Yewon Pipeline
Research pipeline for training and evaluating models:

```bash
cd yewon_pipeline
python run_experiments.py --data_root /path/to/data
python eval_detector_classifier.py --detector yunet --classifier best_model.pth
```

### Jupyter Notebooks
Interactive analysis and experimentation:
- `notebooks/CS_583_Final_Project_Face_detection.ipynb` - Main project notebook
- `notebooks/complete_dataset_notebook.ipynb` - Dataset analysis
- `notebooks/dataset_testing_notebook.ipynb` - Testing utilities

## 🧪 Testing

```bash
# Run all tests
python tests/run_tests.py

# Test specific components
python test_custom_pytorch_integration.py

# Run evaluation examples
python examples/basic_evaluation_example.py
```

## 🎨 Visualization

The system provides real-time visualization with:
- **Color-coded bounding boxes**: Green (with mask), Red (without mask), Orange (incorrect mask)
- **Confidence scores**: Real-time probability display
- **Performance metrics**: FPS counter and detection statistics
- **System information**: Current detector and classifier display

## 🚀 Performance

### Real-Time Capabilities
- **Target**: 30 FPS for real-time applications
- **Haar Cascade**: ~45-60 FPS (CPU)
- **YuNet**: ~25-35 FPS (CPU), ~60+ FPS (GPU)
- **MTCNN**: ~15-25 FPS (GPU required)
- **RetinaFace**: ~10-20 FPS (GPU required)

### Memory Usage
- **Haar + Custom PyTorch**: ~200-300 MB
- **YuNet + Custom PyTorch**: ~300-400 MB
- **MTCNN/RetinaFace**: ~500-800 MB (GPU memory)

## 🔧 Troubleshooting

### Common Issues

**Camera not opening:**
```bash
# Check camera permissions and availability
python -c "import cv2; cap = cv2.VideoCapture(0); print('Camera OK' if cap.isOpened() else 'Camera Error')"
```

**Model not found:**
```bash
# Check model file exists
ls -la model_artifacts/best_pytorch_model_custom.pth
ls -la yewon_pipeline/face_detection_yunet_2023mar.onnx
```

**Import errors:**
```bash
# Install missing dependencies
pip install opencv-python torch torchvision scikit-learn
```

**Performance issues:**
- Use `--device cpu` for CPU-only systems
- Try `--detector haar` for fastest performance
- Reduce `--width` and `--height` for better FPS

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Run tests (`python tests/run_tests.py`)
4. Commit changes (`git commit -m 'Add amazing feature'`)
5. Push to branch (`git push origin feature/amazing-feature`)
6. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **OpenCV**: Computer vision library and YuNet model
- **PyTorch**: Deep learning framework
- **Kaggle Datasets**: Face mask detection datasets
- **Research Community**: Face detection and classification advances

## 📚 Citation

If you use this system in your research, please cite:

```bibtex
@misc{realtime_facemask_2024,
  title={Real-Time Face Mask Detection System},
  author={Your Name},
  year={2024},
  url={https://github.com/clydebaron2000/RealTimeFaceMaskDetection}
}
```

---

## 🎯 Quick Reference

### Most Common Commands
```bash
# Start with defaults (Haar detector, no classifier)
python face_mask_detector.py

# Use your custom PyTorch model
python face_mask_detector.py --classifier custom_pytorch

# Best quality setup (if you have GPU)
python face_mask_detector.py --detector retinaface --classifier custom_pytorch --device cuda

# Run interactive demo
python demo_unified_system.py
```

### Key Files
- `face_mask_detector.py` - Main unified system
- `demo_unified_system.py` - Interactive demo
- `model_artifacts/best_pytorch_model_custom.pth` - Your PyTorch model
- `yewon_pipeline/face_detection_yunet_2023mar.onnx` - YuNet detector

**Ready to detect masks in real-time! 🎭**
