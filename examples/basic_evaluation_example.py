"""
Basic example of using the evaluation pipeline
"""

import os
import sys

# Add parent directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from evaluation.eval_pipeline import EvaluationPipeline

def main():
    """
    Example of running a basic evaluation
    """
    print("Face Mask Detection System - Basic Evaluation Example")
    print("=" * 60)
    
    # Initialize evaluation pipeline
    pipeline = EvaluationPipeline(output_dir="example_results")
    
    try:
        # Setup detector (YuNet)
        print("\n1. Setting up detector...")
        detector = pipeline.setup_detector(
            detector_type='yunet',
            model_path='models/face_detection_yunet_2023mar.onnx'  # Update path as needed
        )
        
        # Setup classifier (PyTorch)
        print("\n2. Setting up classifier...")
        classifier = pipeline.setup_classifier(
            classifier_type='pytorch',
            model_path='models/mask_classifier.pth',  # Update path as needed
            class_names=['no_mask', 'with_mask', 'incorrect_mask']
        )
        
        # Setup dataset
        print("\n3. Setting up dataset...")
        dataset = pipeline.setup_dataset(
            dataset_type='andrewmvd',
            dataset_path='datasets/andrewmvd_face_mask_detection'  # Update path as needed
        )
        
        # Run individual evaluations
        print("\n4. Running detection evaluation...")
        detection_results = pipeline.evaluate_detection()
        print(f"Detection F1 Score: {detection_results.get('f1_score', 0):.3f}")
        
        print("\n5. Running classification evaluation...")
        classification_results = pipeline.evaluate_classification()
        print(f"Classification Accuracy: {classification_results.get('accuracy', 0):.3f}")
        
        print("\n6. Running full pipeline evaluation...")
        pipeline_results = pipeline.evaluate_full_pipeline()
        pipeline_fps = pipeline_results.get('pipeline_performance', {}).get('pipeline_fps', 0)
        print(f"Pipeline FPS: {pipeline_fps:.1f}")
        
        print("\n7. Running speed benchmarks...")
        speed_results = pipeline.benchmark_speed()
        
        print("\n8. Running memory benchmarks...")
        memory_results = pipeline.benchmark_memory()
        
        print("\nEvaluation complete! Check 'example_results' directory for detailed results.")
        
    except Exception as e:
        print(f"Error during evaluation: {e}")
        print("Make sure you have:")
        print("1. YuNet ONNX model file")
        print("2. Trained classifier model")
        print("3. Dataset in the correct format")
        return 1
    
    return 0

def comprehensive_evaluation_example():
    """
    Example of running comprehensive evaluation
    """
    print("Face Mask Detection System - Comprehensive Evaluation Example")
    print("=" * 70)
    
    # Initialize evaluation pipeline
    pipeline = EvaluationPipeline(output_dir="comprehensive_results")
    
    try:
        # Setup components
        pipeline.setup_detector('yunet', model_path='models/yunet.onnx')
        pipeline.setup_classifier('pytorch', 
                                 model_path='models/classifier.pth',
                                 class_names=['no_mask', 'with_mask', 'incorrect_mask'])
        pipeline.setup_dataset('andrewmvd', 'datasets/andrewmvd')
        
        # Run comprehensive evaluation (all tests)
        results = pipeline.run_comprehensive_evaluation()
        
        print("\nComprehensive evaluation complete!")
        print("Check 'comprehensive_results' directory for:")
        print("- comprehensive_evaluation.json (all results)")
        print("- evaluation_summary_report.txt (human-readable summary)")
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0

if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == '--comprehensive':
        exit(comprehensive_evaluation_example())
    else:
        exit(main())
