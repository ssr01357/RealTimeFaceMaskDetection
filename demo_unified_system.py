#!/usr/bin/env python3
"""
Demo script for the unified face mask detection system
Shows how to use the new unified interface for all detectors and classifiers
"""

import os
import sys
import argparse

def demo_detection_only():
    """Demo with face detection only (no classification)"""
    print("="*60)
    print("DEMO: Face Detection Only")
    print("="*60)
    print("This demo shows face detection without mask classification")
    print("Uses Haar cascade detector (most compatible)")
    print()
    
    os.system("python face_mask_detector.py --detector haar")

def demo_custom_pytorch():
    """Demo with custom PyTorch model"""
    print("="*60)
    print("DEMO: Custom PyTorch Model")
    print("="*60)
    print("This demo uses your custom PyTorch model for mask classification")
    print("Detector: Haar cascade")
    print("Classifier: Custom PyTorch model")
    print()
    
    model_path = "model_artifacts/best_pytorch_model_custom.pth"
    if not os.path.exists(model_path):
        print(f"❌ Custom PyTorch model not found: {model_path}")
        print("Please ensure your trained model is available.")
        return
    
    os.system(f"python face_mask_detector.py --detector haar --classifier custom_pytorch")

def demo_yunet_advanced():
    """Demo with YuNet detector"""
    print("="*60)
    print("DEMO: Advanced YuNet Detection")
    print("="*60)
    print("This demo uses YuNet for more accurate face detection")
    print("Detector: YuNet (if available)")
    print("Classifier: Custom PyTorch (if available)")
    print()
    
    yunet_path = "yewon_pipeline/face_detection_yunet_2023mar.onnx"
    model_path = "model_artifacts/best_pytorch_model_custom.pth"
    
    if not os.path.exists(yunet_path):
        print(f"⚠️  YuNet model not found: {yunet_path}")
        print("Will fall back to Haar cascade detector")
    
    if not os.path.exists(model_path):
        print(f"⚠️  Custom PyTorch model not found: {model_path}")
        print("Will run detection only")
        os.system("python face_mask_detector.py --detector yunet")
    else:
        os.system("python face_mask_detector.py --detector yunet --classifier custom_pytorch")

def demo_comparison():
    """Demo showing how to compare different detectors"""
    print("="*60)
    print("DEMO: Detector Comparison")
    print("="*60)
    print("This demo shows how to switch between detectors at runtime")
    print("Start with Haar, then use keyboard controls to switch:")
    print("  1 - Haar cascade")
    print("  2 - YuNet")
    print("  3 - MTCNN")
    print("  4 - RetinaFace")
    print("  C - Toggle classifier on/off")
    print("  Q - Quit")
    print()
    
    os.system("python face_mask_detector.py --detector haar --classifier custom_pytorch")

def print_usage_guide():
    """Print comprehensive usage guide"""
    print("="*60)
    print("UNIFIED FACE MASK DETECTION SYSTEM - USAGE GUIDE")
    print("="*60)
    print()
    print("BASIC USAGE:")
    print("  python face_mask_detector.py                    # Detection only with Haar")
    print("  python face_mask_detector.py --classifier custom_pytorch  # With classification")
    print()
    print("DETECTOR OPTIONS:")
    print("  --detector haar        # OpenCV Haar cascade (fast, CPU-friendly)")
    print("  --detector yunet       # YuNet ONNX model (accurate)")
    print("  --detector mtcnn       # MTCNN (precise, requires GPU)")
    print("  --detector retinaface  # RetinaFace (best quality, requires GPU)")
    print()
    print("CLASSIFIER OPTIONS:")
    print("  --classifier custom_pytorch  # Your custom PyTorch model")
    print("  --classifier yewon          # Yewon pipeline model")
    print("  --classifier sklearn        # sklearn model")
    print("  (no classifier)             # Detection only")
    print()
    print("RUNTIME CONTROLS:")
    print("  1-4: Switch detectors")
    print("  5-8: Switch classifiers")
    print("  C:   Toggle classifier on/off")
    print("  S:   Take screenshot")
    print("  Q:   Quit")
    print("  +/-: Adjust confidence threshold")
    print()
    print("EXAMPLES:")
    print("  python face_mask_detector.py --detector haar --classifier custom_pytorch")
    print("  python face_mask_detector.py --detector yunet")
    print("  python face_mask_detector.py --detector mtcnn --classifier yewon")
    print()
    print("REQUIREMENTS:")
    print("  - Custom PyTorch model: model_artifacts/best_pytorch_model_custom.pth")
    print("  - YuNet model: yewon_pipeline/face_detection_yunet_2023mar.onnx")
    print("  - Working webcam")
    print()

def main():
    """Main demo menu"""
    parser = argparse.ArgumentParser(description='Demo for Unified Face Mask Detection System')
    parser.add_argument('--demo', type=str, 
                       choices=['detection', 'pytorch', 'yunet', 'comparison', 'guide'],
                       help='Run specific demo directly')
    
    args = parser.parse_args()
    
    if args.demo:
        if args.demo == 'detection':
            demo_detection_only()
        elif args.demo == 'pytorch':
            demo_custom_pytorch()
        elif args.demo == 'yunet':
            demo_yunet_advanced()
        elif args.demo == 'comparison':
            demo_comparison()
        elif args.demo == 'guide':
            print_usage_guide()
        return
    
    print("="*60)
    print("UNIFIED FACE MASK DETECTION SYSTEM - DEMO")
    print("="*60)
    print("Choose a demo to run:")
    print()
    print("1. Face Detection Only (Haar cascade)")
    print("2. Custom PyTorch Model Demo")
    print("3. Advanced YuNet Detection")
    print("4. Detector Comparison (runtime switching)")
    print("5. Usage Guide")
    print("0. Exit")
    print()
    
    while True:
        try:
            choice = input("Enter your choice (0-5): ").strip()
            
            if choice == '0':
                print("Goodbye!")
                break
            elif choice == '1':
                demo_detection_only()
            elif choice == '2':
                demo_custom_pytorch()
            elif choice == '3':
                demo_yunet_advanced()
            elif choice == '4':
                demo_comparison()
            elif choice == '5':
                print_usage_guide()
            else:
                print("Invalid choice. Please enter 0-5.")
                
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")

if __name__ == '__main__':
    main()
