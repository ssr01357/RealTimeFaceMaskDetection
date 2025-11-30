#!/usr/bin/env python3
"""
Test script for custom PyTorch model integration
Tests the complete pipeline from model loading to real-time detection
"""

import os
import sys
import cv2
import numpy as np
import torch
from typing import Dict, Any

# Add paths for imports
sys.path.append('model_artifacts')
sys.path.append('evaluation/classifiers')
sys.path.append('dual_filter_system')

def test_model_loading():
    """Test loading the custom PyTorch model"""
    print("="*60)
    print("TESTING CUSTOM PYTORCH MODEL LOADING")
    print("="*60)
    
    model_path = os.path.join('model_artifacts', 'best_pytorch_model_custom.pth')
    
    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        return False
    
    try:
        from pytorch_model_loader import load_custom_pytorch_model, test_model_loading
        
        # Test the model loader directly
        print("Testing model loader...")
        test_model_loading(model_path)
        
        # Load the model
        print("\nLoading custom PyTorch classifier...")
        classifier = load_custom_pytorch_model(model_path, device='cpu')
        
        print(f"✅ Model loaded successfully!")
        print(f"   Device: {classifier.device}")
        print(f"   Input size: {classifier.input_size}")
        print(f"   Classes: {classifier.class_names}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return False


def test_classifier_wrapper():
    """Test the classifier wrapper with custom PyTorch model"""
    print("\n" + "="*60)
    print("TESTING CLASSIFIER WRAPPER")
    print("="*60)
    
    model_path = os.path.join('model_artifacts', 'best_pytorch_model_custom.pth')
    
    try:
        from classifier_wrapper import ClassifierWrapper
        
        # Test creating custom PyTorch classifier wrapper
        print("Creating custom PyTorch classifier wrapper...")
        wrapper = ClassifierWrapper.create_custom_pytorch_classifier(
            model_path=model_path,
            device='cpu',
            class_names=['without_mask', 'with_mask']
        )
        
        print(f"✅ Wrapper created successfully!")
        print(f"   Type: {wrapper.classifier_type}")
        print(f"   Classes: {wrapper.get_class_names()}")
        
        # Test with dummy image
        print("\nTesting prediction with dummy image...")
        dummy_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        
        result = wrapper.predict_single(dummy_image)
        predicted_class, probabilities = result
        
        print(f"✅ Prediction successful!")
        print(f"   Predicted class: {predicted_class} ({wrapper.get_class_names()[predicted_class]})")
        print(f"   Probabilities: {probabilities}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing classifier wrapper: {e}")
        return False


def test_classification_filter():
    """Test the classification filter with custom PyTorch model"""
    print("\n" + "="*60)
    print("TESTING CLASSIFICATION FILTER")
    print("="*60)
    
    model_path = os.path.join('model_artifacts', 'best_pytorch_model_custom.pth')
    
    try:
        from filters.classification_filter import ClassificationFilter
        
        print("Creating classification filter...")
        filter_obj = ClassificationFilter(
            detector_name='haar',
            classifier_model_path=model_path,
            device='cpu',
            confidence_threshold=0.6,
            num_classes=2
        )
        
        print(f"✅ Classification filter created successfully!")
        print(f"   Detector: {filter_obj.detector_name}")
        print(f"   Classifier available: {filter_obj.classifier is not None}")
        print(f"   Classes: {filter_obj.class_names}")
        
        # Test with dummy frame
        print("\nTesting with dummy frame...")
        dummy_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        # Add a simple rectangle to simulate a face
        cv2.rectangle(dummy_frame, (200, 150), (400, 350), (255, 255, 255), -1)
        
        processed_frame = filter_obj.process_frame(dummy_frame)
        
        print(f"✅ Frame processing successful!")
        print(f"   Output shape: {processed_frame.shape}")
        print(f"   Detections: {filter_obj.detection_count}")
        print(f"   Classifications: {filter_obj.classification_count}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing classification filter: {e}")
        return False


def test_dual_filter_system():
    """Test the dual filter system with custom PyTorch model"""
    print("\n" + "="*60)
    print("TESTING DUAL FILTER SYSTEM")
    print("="*60)
    
    model_path = os.path.join('model_artifacts', 'best_pytorch_model_custom.pth')
    
    try:
        from dual_filter_detection import DualFilterDetector
        
        print("Creating dual filter detector...")
        detector = DualFilterDetector(
            detector_name='haar',
            classifier_model_path=model_path,
            camera_index=0,
            confidence_threshold=0.6,
            device='cpu',
            frame_width=640,
            frame_height=480,
            num_classes=2
        )
        
        print(f"✅ Dual filter detector created successfully!")
        print(f"   Detector: {detector.detector_name}")
        print(f"   Classifier enabled: {detector.classifier_enabled}")
        print(f"   Current classifier type: {detector.current_classifier_type}")
        
        # Test frame processing
        print("\nTesting frame processing...")
        dummy_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        # Add a simple rectangle to simulate a face
        cv2.rectangle(dummy_frame, (200, 150), (400, 350), (255, 255, 255), -1)
        
        detection_frame, classification_frame = detector.process_frame(dummy_frame)
        
        print(f"✅ Frame processing successful!")
        print(f"   Detection frame shape: {detection_frame.shape}")
        print(f"   Classification frame shape: {classification_frame.shape}")
        
        # Test classifier switching
        print("\nTesting classifier switching...")
        detector._switch_classifier('none')
        print(f"   Switched to none: {detector.current_classifier_type}")
        
        detector._switch_classifier('custom')
        print(f"   Switched back to custom: {detector.current_classifier_type}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing dual filter system: {e}")
        return False


def test_real_time_compatibility():
    """Test compatibility with real-time detection system"""
    print("\n" + "="*60)
    print("TESTING REAL-TIME COMPATIBILITY")
    print("="*60)
    
    try:
        # Test if we can import the real-time detection module
        import real_time_detection
        
        print("✅ Real-time detection module imported successfully!")
        
        # Check if the model path is accessible
        model_path = os.path.join('model_artifacts', 'best_pytorch_model_custom.pth')
        if os.path.exists(model_path):
            print(f"✅ Model file accessible: {model_path}")
        else:
            print(f"❌ Model file not found: {model_path}")
            return False
        
        print("✅ Real-time compatibility test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Error testing real-time compatibility: {e}")
        return False


def print_usage_instructions():
    """Print usage instructions for the integrated system"""
    print("\n" + "="*60)
    print("USAGE INSTRUCTIONS")
    print("="*60)
    print()
    print("Your custom PyTorch model has been successfully integrated!")
    print()
    print("To use the real-time detection system with your model:")
    print()
    print("1. Run the dual filter system:")
    print("   python dual_filter_system/dual_filter_detection.py --classifier model_artifacts/best_pytorch_model_custom.pth")
    print()
    print("2. Use keyboard controls during runtime:")
    print("   - Press '5' to switch to your custom PyTorch model")
    print("   - Press '6' to switch to Yewon pipeline model (if available)")
    print("   - Press '8' to disable classifier")
    print("   - Press 'C' to toggle classifier on/off")
    print("   - Press 'Q' or ESC to quit")
    print()
    print("3. Alternative: Run the basic real-time detection:")
    print("   python real_time_detection.py")
    print()
    print("Your model expects:")
    print("   - Input: 40x40 RGB images")
    print("   - Output: 2 classes (without_mask=0, with_mask=1)")
    print("   - Classes: ['without_mask', 'with_mask']")
    print()
    print("="*60)


def main():
    """Run all tests"""
    print("CUSTOM PYTORCH MODEL INTEGRATION TEST")
    print("="*60)
    
    tests = [
        ("Model Loading", test_model_loading),
        ("Classifier Wrapper", test_classifier_wrapper),
        ("Classification Filter", test_classification_filter),
        ("Dual Filter System", test_dual_filter_system),
        ("Real-time Compatibility", test_real_time_compatibility)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results[test_name] = False
    
    # Print summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    passed = 0
    total = len(tests)
    
    for test_name, result in results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name:.<30} {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED! Your custom PyTorch model is fully integrated!")
        print_usage_instructions()
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Please check the errors above.")
    
    return passed == total


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
