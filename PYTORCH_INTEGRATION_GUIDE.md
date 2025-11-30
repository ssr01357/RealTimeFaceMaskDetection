# PyTorch Model Integration Guide

## Overview

Your custom PyTorch face mask classification model has been successfully integrated into the real-time detection system. The integration supports both the basic real-time detection script and the advanced dual filter system.

## What Was Implemented

### 1. Custom PyTorch Model Loader (`model_artifacts/pytorch_model_loader.py`)
- **PyTorchCNN**: Your exact model architecture from the notebook
- **DepthwiseSeparableConv**: Custom convolution blocks
- **CustomPyTorchClassifier**: Wrapper class for inference
- **Preprocessing**: Handles 40x40 RGB input, BGR→RGB conversion, normalization

### 2. Updated Classification Filter (`dual_filter_system/filters/classification_filter.py`)
- Automatically detects `best_pytorch_model_custom.pth` files
- Loads your custom model with proper preprocessing
- Integrates seamlessly with existing detection pipeline

### 3. Updated Real-time Detection (`real_time_detection.py`)
- Added support for custom PyTorch models
- Automatic model type detection
- Proper error handling and fallbacks

## Model Specifications

**Architecture**: Custom CNN with Depthwise Separable Convolutions
- **Input Size**: 40x40x3 (RGB)
- **Classes**: 2 (without_mask=0, with_mask=1)
- **Device**: CUDA/CPU auto-detection
- **Preprocessing**: BGR→RGB, resize to 40x40, normalize to [0,1]

**Color Coding**:
- 🟢 **Green**: with_mask (Class 1)
- 🔴 **Red**: without_mask (Class 0)

## Usage Examples

### 1. Basic Real-time Detection

```bash
# Run with your custom PyTorch model
python3 real_time_detection.py --classifier-model model_artifacts/best_pytorch_model_custom.pth

# With specific detector
python3 real_time_detection.py \
    --detector haar \
    --classifier-model model_artifacts/best_pytorch_model_custom.pth \
    --confidence 0.6
```

### 2. Advanced Dual Filter System

```bash
# Run dual filter system with your model
python3 dual_filter_system/dual_filter_detection.py \
    --classifier model_artifacts/best_pytorch_model_custom.pth

# With specific settings
python3 dual_filter_system/dual_filter_detection.py \
    --detector haar \
    --classifier model_artifacts/best_pytorch_model_custom.pth \
    --confidence 0.6 \
    --device cuda
```

### 3. Interactive Controls (Dual Filter System)

**Keyboard Controls**:
- `q`: Quit
- `s`: Screenshot
- `1-4`: Switch detectors (Haar, YuNet, MTCNN, RetinaFace)
- `c`: Toggle classifier on/off
- `+/-`: Adjust confidence threshold
- `o`: Toggle overlay mode
- `d`: Toggle difference view
- `f`: Toggle FPS counter
- `t`: Toggle statistics
- `r`: Start/stop video recording
- `x`: Reset statistics

## File Structure

```
RealTimeFaceMaskDetection/
├── model_artifacts/
│   ├── best_pytorch_model_custom.pth          # Your trained model
│   └── pytorch_model_loader.py                # Model loader (NEW)
├── dual_filter_system/
│   ├── filters/classification_filter.py       # Updated for your model
│   └── dual_filter_detection.py              # Main dual filter app
├── real_time_detection.py                     # Updated basic detection
└── PYTORCH_INTEGRATION_GUIDE.md              # This guide
```

## Technical Details

### Model Loading Process

1. **Detection**: System automatically detects `best_pytorch_model_custom.pth`
2. **Architecture**: Recreates your exact PyTorchCNN model
3. **Loading**: Loads state dict with error handling
4. **Device**: Auto-selects CUDA/CPU
5. **Preprocessing**: Configures 40x40 RGB input pipeline

### Preprocessing Pipeline

```python
# Your model expects:
Input: BGR image (any size) → 
Resize: 40x40 → 
Convert: BGR→RGB → 
Normalize: [0,1] → 
Tensor: (1,3,40,40) → 
Model: PyTorchCNN → 
Output: [without_mask_prob, with_mask_prob]
```

### Integration Points

1. **Classification Filter**: `_load_custom_pytorch_classifier()`
2. **Real-time Detection**: `_setup_custom_pytorch_classifier()`
3. **Model Loader**: `load_custom_pytorch_model()`

## Troubleshooting

### Common Issues

1. **"Custom PyTorch model loader not available"**
   - Ensure PyTorch is installed: `pip install torch torchvision`
   - Check model_artifacts path is accessible

2. **"Model file not found"**
   - Verify `model_artifacts/best_pytorch_model_custom.pth` exists
   - Check file permissions

3. **CUDA/CPU Issues**
   - Model auto-detects available device
   - Force CPU: modify device parameter in code

4. **Import Errors**
   - Ensure all dependencies: `pip install opencv-python numpy torch`
   - Check Python path configuration

### Performance Tips

1. **GPU Acceleration**: Ensure CUDA is available for faster inference
2. **Confidence Threshold**: Adjust for your use case (default: 0.6)
3. **Input Size**: Model optimized for 40x40, don't change preprocessing
4. **Batch Processing**: Current implementation processes one face at a time

## Testing Your Integration

### Quick Test

```bash
# Test model loading
cd model_artifacts
python3 -c "
from pytorch_model_loader import load_custom_pytorch_model
import numpy as np
model = load_custom_pytorch_model('best_pytorch_model_custom.pth', 'cpu')
dummy = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
result = model.predict(dummy)
print('Test successful:', result)
"
```

### Full System Test

```bash
# Test real-time detection (without camera)
python3 real_time_detection.py \
    --classifier-model model_artifacts/best_pytorch_model_custom.pth \
    --help
```

## Next Steps

1. **Run the System**: Use the commands above to start real-time detection
2. **Adjust Settings**: Modify confidence threshold, detector type as needed
3. **Collect Data**: Use screenshot/video features to save results
4. **Monitor Performance**: Check FPS and accuracy in real-time

## Model Performance

Based on your notebook training:
- **Architecture**: Efficient depthwise separable convolutions
- **Input Size**: 40x40 (lightweight for real-time)
- **Classes**: Binary classification (mask/no-mask)
- **Training**: Trained on balanced dataset with data augmentation

The integration maintains your model's performance while providing real-time inference capabilities with proper preprocessing and error handling.
