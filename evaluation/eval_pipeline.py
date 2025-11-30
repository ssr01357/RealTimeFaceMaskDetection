"""
Main evaluation pipeline orchestrator for face mask detection system
"""

import os
import json
import time
from typing import Dict, List, Optional, Any, Tuple
import argparse
import numpy as np
import cv2

from .detectors.detector_wrapper import FaceDetectorWrapper
from .classifiers.classifier_wrapper import ClassifierWrapper
from .datasets.dataset_loaders import DatasetLoader
from .detectors.metrics import DetectionMetrics, evaluate_detector_on_dataset
from .classifiers.metrics import ClassificationMetrics, evaluate_classifier_on_dataset
from .benchmarks.speed_benchmark import SpeedBenchmark
from .benchmarks.memory_benchmark import MemoryBenchmark


class EvaluationPipeline:
    """
    Main evaluation pipeline for face mask detection system
    """
    
    def __init__(self, output_dir: str = "evaluation_results"):
        """
        Initialize evaluation pipeline
        
        Args:
            output_dir: Directory to save evaluation results
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        self.detector = None
        self.classifier = None
        self.dataset_loader = None
        
        # Initialize benchmarking tools
        self.speed_benchmark = SpeedBenchmark()
        self.memory_benchmark = MemoryBenchmark()
    
    def setup_detector(self, detector_type: str, **kwargs) -> FaceDetectorWrapper:
        """
        Setup face detector
        
        Args:
            detector_type: Type of detector ('yunet', 'haar')
            **kwargs: Detector-specific arguments
            
        Returns:
            Configured detector wrapper
        """
        self.detector = FaceDetectorWrapper(detector_type, **kwargs)
        print(f"Detector setup complete: {self.detector.get_detector_info()}")
        return self.detector
    
    def setup_classifier(self, classifier_type: str, **kwargs) -> ClassifierWrapper:
        """
        Setup face mask classifier
        
        Args:
            classifier_type: Type of classifier ('numpy_cnn', 'pytorch')
            **kwargs: Classifier-specific arguments
            
        Returns:
            Configured classifier wrapper
        """
        self.classifier = ClassifierWrapper(classifier_type, **kwargs)
        print(f"Classifier setup complete: {self.classifier.get_classifier_info()}")
        return self.classifier
    
    def setup_dataset(self, dataset_type: str, dataset_path: str, **kwargs) -> DatasetLoader:
        """
        Setup dataset loader
        
        Args:
            dataset_type: Type of dataset ('andrewmvd', 'face12k', 'medical_mask')
            dataset_path: Path to dataset
            **kwargs: Dataset-specific arguments
            
        Returns:
            Configured dataset loader
        """
        self.dataset_loader = DatasetLoader(dataset_type, dataset_path, **kwargs)
        print(f"Dataset setup complete: {self.dataset_loader.get_dataset_info()}")
        return self.dataset_loader
    
    def evaluate_detection(self, save_results: bool = True) -> Dict[str, Any]:
        """
        Evaluate face detection performance
        
        Args:
            save_results: Whether to save results to file
            
        Returns:
            Detection evaluation results
        """
        if self.detector is None:
            raise ValueError("Detector not setup. Call setup_detector() first.")
        
        if self.dataset_loader is None:
            raise ValueError("Dataset not setup. Call setup_dataset() first.")
        
        print("Starting detection evaluation...")
        
        # Initialize detection metrics
        detection_metrics = DetectionMetrics()
        
        # Evaluate on dataset
        num_samples = min(100, len(self.dataset_loader))  # Limit for faster evaluation
        
        for i in range(num_samples):
            image, target = self.dataset_loader[i]
            
            # Get ground truth boxes
            if isinstance(target, dict) and 'boxes' in target:
                gt_boxes = [{'box': box} for box in target['boxes']]
            else:
                continue  # Skip if no bounding box info
            
            # Get predictions
            predictions = self.detector.detect(image)
            
            # Update metrics
            detection_metrics.update(predictions, gt_boxes)
            
            if (i + 1) % 20 == 0:
                print(f"Processed {i + 1}/{num_samples} images...")
        
        # Get results
        results = detection_metrics.get_metrics_summary()
        results['evaluation_info'] = {
            'detector_info': self.detector.get_detector_info(),
            'dataset_info': self.dataset_loader.get_dataset_info(),
            'samples_evaluated': num_samples,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        if save_results:
            output_path = os.path.join(self.output_dir, 'detection_evaluation.json')
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"Detection evaluation results saved to {output_path}")
        
        return results
    
    def evaluate_classification(self, save_results: bool = True) -> Dict[str, Any]:
        """
        Evaluate face mask classification performance
        
        Args:
            save_results: Whether to save results to file
            
        Returns:
            Classification evaluation results
        """
        if self.classifier is None:
            raise ValueError("Classifier not setup. Call setup_classifier() first.")
        
        if self.dataset_loader is None:
            raise ValueError("Dataset not setup. Call setup_dataset() first.")
        
        print("Starting classification evaluation...")
        
        # Initialize classification metrics
        class_names = self.classifier.get_class_names()
        classification_metrics = ClassificationMetrics(class_names)
        
        # Evaluate on dataset
        num_samples = min(500, len(self.dataset_loader))  # Limit for faster evaluation
        
        for i in range(num_samples):
            try:
                if hasattr(self.dataset_loader.loader, 'mode') and self.dataset_loader.loader.mode == 'classification':
                    # Dataset provides cropped faces
                    face_crop, label = self.dataset_loader[i]
                    pred_class, probs = self.classifier.predict_single(face_crop)
                    
                    classification_metrics.update([pred_class], [label], [probs])
                
                else:
                    # Dataset provides full images - need to crop faces
                    image, target = self.dataset_loader[i]
                    
                    if isinstance(target, dict) and 'boxes' in target and 'labels' in target:
                        boxes = target['boxes']
                        labels = target['labels']
                        
                        for box, label in zip(boxes, labels):
                            x1, y1, x2, y2 = map(int, box)
                            crop = image[max(0, y1):max(0, y2), max(0, x1):max(0, x2)]
                            
                            if crop.size > 0:
                                pred_class, probs = self.classifier.predict_single(crop)
                                classification_metrics.update([pred_class], [label], [probs])
                
                if (i + 1) % 50 == 0:
                    print(f"Processed {i + 1}/{num_samples} samples...")
                    
            except Exception as e:
                print(f"Error processing sample {i}: {e}")
                continue
        
        # Get results
        results = classification_metrics.get_metrics_summary()
        results['evaluation_info'] = {
            'classifier_info': self.classifier.get_classifier_info(),
            'dataset_info': self.dataset_loader.get_dataset_info(),
            'samples_evaluated': num_samples,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        if save_results:
            output_path = os.path.join(self.output_dir, 'classification_evaluation.json')
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"Classification evaluation results saved to {output_path}")
        
        return results
    
    def evaluate_full_pipeline(self, save_results: bool = True) -> Dict[str, Any]:
        """
        Evaluate full detection + classification pipeline
        
        Args:
            save_results: Whether to save results to file
            
        Returns:
            Pipeline evaluation results
        """
        if self.detector is None or self.classifier is None:
            raise ValueError("Both detector and classifier must be setup.")
        
        if self.dataset_loader is None:
            raise ValueError("Dataset not setup. Call setup_dataset() first.")
        
        print("Starting full pipeline evaluation...")
        
        # Initialize metrics
        detection_metrics = DetectionMetrics()
        classification_metrics = ClassificationMetrics(self.classifier.get_class_names())
        
        pipeline_stats = {
            'total_images': 0,
            'total_faces_detected': 0,
            'total_faces_classified': 0,
            'images_with_faces': 0,
            'detection_times': [],
            'classification_times': [],
            'total_times': []
        }
        
        # Evaluate on dataset
        num_samples = min(200, len(self.dataset_loader))
        
        for i in range(num_samples):
            try:
                image, target = self.dataset_loader[i]
                
                # Get ground truth
                if isinstance(target, dict) and 'boxes' in target:
                    gt_boxes = [{'box': box} for box in target['boxes']]
                    gt_labels = target.get('labels', [])
                else:
                    continue
                
                pipeline_stats['total_images'] += 1
                
                # Time detection
                start_time = time.perf_counter()
                detections = self.detector.detect(image)
                detection_time = time.perf_counter() - start_time
                pipeline_stats['detection_times'].append(detection_time)
                
                # Update detection metrics
                detection_metrics.update(detections, gt_boxes)
                pipeline_stats['total_faces_detected'] += len(detections)
                
                if detections:
                    pipeline_stats['images_with_faces'] += 1
                    
                    # Time classification
                    start_time = time.perf_counter()
                    
                    # Crop faces and classify
                    crops = []
                    pred_classes = []
                    pred_probs = []
                    
                    for det in detections:
                        x1, y1, x2, y2 = map(int, det['box'])
                        crop = image[max(0, y1):max(0, y2), max(0, x1):max(0, x2)]
                        
                        if crop.size > 0:
                            crops.append(crop)
                    
                    if crops:
                        predictions = self.classifier.predict(crops)
                        for pred_class, probs in predictions:
                            pred_classes.append(pred_class)
                            pred_probs.append(probs)
                    
                    classification_time = time.perf_counter() - start_time
                    pipeline_stats['classification_times'].append(classification_time)
                    pipeline_stats['total_faces_classified'] += len(pred_classes)
                    
                    # Update classification metrics (match with ground truth)
                    if len(pred_classes) > 0 and len(gt_labels) > 0:
                        # Simple matching - assumes same order
                        min_len = min(len(pred_classes), len(gt_labels))
                        classification_metrics.update(
                            pred_classes[:min_len], 
                            gt_labels[:min_len], 
                            pred_probs[:min_len]
                        )
                else:
                    classification_time = 0
                    pipeline_stats['classification_times'].append(0)
                
                total_time = detection_time + classification_time
                pipeline_stats['total_times'].append(total_time)
                
                if (i + 1) % 25 == 0:
                    print(f"Processed {i + 1}/{num_samples} images...")
                    
            except Exception as e:
                print(f"Error processing sample {i}: {e}")
                continue
        
        # Calculate pipeline statistics
        avg_detection_time = np.mean(pipeline_stats['detection_times']) if pipeline_stats['detection_times'] else 0
        avg_classification_time = np.mean(pipeline_stats['classification_times']) if pipeline_stats['classification_times'] else 0
        avg_total_time = np.mean(pipeline_stats['total_times']) if pipeline_stats['total_times'] else 0
        
        pipeline_fps = 1.0 / avg_total_time if avg_total_time > 0 else 0
        
        # Compile results
        results = {
            'detection_metrics': detection_metrics.get_metrics_summary(),
            'classification_metrics': classification_metrics.get_metrics_summary(),
            'pipeline_performance': {
                'avg_detection_time_ms': avg_detection_time * 1000,
                'avg_classification_time_ms': avg_classification_time * 1000,
                'avg_total_time_ms': avg_total_time * 1000,
                'pipeline_fps': pipeline_fps,
                'detection_percentage': (avg_detection_time / avg_total_time * 100) if avg_total_time > 0 else 0,
                'classification_percentage': (avg_classification_time / avg_total_time * 100) if avg_total_time > 0 else 0
            },
            'pipeline_stats': {
                'total_images_processed': pipeline_stats['total_images'],
                'total_faces_detected': pipeline_stats['total_faces_detected'],
                'total_faces_classified': pipeline_stats['total_faces_classified'],
                'images_with_faces': pipeline_stats['images_with_faces'],
                'avg_faces_per_image': pipeline_stats['total_faces_detected'] / pipeline_stats['total_images'] if pipeline_stats['total_images'] > 0 else 0,
                'face_detection_rate': pipeline_stats['images_with_faces'] / pipeline_stats['total_images'] if pipeline_stats['total_images'] > 0 else 0
            },
            'evaluation_info': {
                'detector_info': self.detector.get_detector_info(),
                'classifier_info': self.classifier.get_classifier_info(),
                'dataset_info': self.dataset_loader.get_dataset_info(),
                'samples_evaluated': num_samples,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            }
        }
        
        if save_results:
            output_path = os.path.join(self.output_dir, 'pipeline_evaluation.json')
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"Pipeline evaluation results saved to {output_path}")
        
        return results
    
    def benchmark_speed(self, save_results: bool = True) -> Dict[str, Any]:
        """
        Benchmark speed performance of the system
        
        Args:
            save_results: Whether to save results to file
            
        Returns:
            Speed benchmark results
        """
        if self.detector is None or self.classifier is None:
            raise ValueError("Both detector and classifier must be setup.")
        
        if self.dataset_loader is None:
            raise ValueError("Dataset not setup. Call setup_dataset() first.")
        
        print("Starting speed benchmarking...")
        
        # Prepare test data
        test_images = []
        test_crops = []
        
        num_samples = min(50, len(self.dataset_loader))
        for i in range(num_samples):
            try:
                image, target = self.dataset_loader[i]
                test_images.append(image)
                
                # Extract crops if available
                if isinstance(target, dict) and 'boxes' in target:
                    for box in target['boxes']:
                        x1, y1, x2, y2 = map(int, box)
                        crop = image[max(0, y1):max(0, y2), max(0, x1):max(0, x2)]
                        if crop.size > 0:
                            test_crops.append(crop)
                            
            except Exception as e:
                print(f"Error preparing test data from sample {i}: {e}")
                continue
        
        if not test_images:
            raise ValueError("No test images available for benchmarking")
        
        # Ensure we have some crops for classifier benchmarking
        if not test_crops:
            # Generate crops using detector
            for img in test_images[:10]:
                detections = self.detector.detect(img)
                for det in detections:
                    x1, y1, x2, y2 = map(int, det['box'])
                    crop = img[max(0, y1):max(0, y2), max(0, x1):max(0, x2)]
                    if crop.size > 0:
                        test_crops.append(crop)
        
        results = {}
        
        # Benchmark detector
        print("Benchmarking detector...")
        detector_results = self.speed_benchmark.benchmark_detector(self.detector, test_images)
        results['detector_benchmark'] = detector_results
        
        # Benchmark classifier
        if test_crops:
            print("Benchmarking classifier...")
            classifier_results = self.speed_benchmark.benchmark_classifier(self.classifier, test_crops)
            results['classifier_benchmark'] = classifier_results
        
        # Benchmark full pipeline
        print("Benchmarking full pipeline...")
        pipeline_results = self.speed_benchmark.benchmark_full_pipeline(
            self.detector, self.classifier, test_images
        )
        results['pipeline_benchmark'] = pipeline_results
        
        # Benchmark scalability
        print("Benchmarking scalability...")
        scalability_results = self.speed_benchmark.benchmark_scalability(
            self.detector, self.classifier, test_images
        )
        results['scalability_benchmark'] = scalability_results
        
        # Add evaluation info
        results['evaluation_info'] = {
            'detector_info': self.detector.get_detector_info(),
            'classifier_info': self.classifier.get_classifier_info(),
            'dataset_info': self.dataset_loader.get_dataset_info(),
            'test_images_count': len(test_images),
            'test_crops_count': len(test_crops),
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        if save_results:
            output_path = os.path.join(self.output_dir, 'speed_benchmark.json')
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"Speed benchmark results saved to {output_path}")
            
            # Generate and save performance report
            if 'classifier_benchmark' in results:
                report = self.speed_benchmark.generate_performance_report(
                    detector_results, classifier_results, pipeline_results, scalability_results
                )
            else:
                report = "Classifier benchmark not available - insufficient test crops"
            
            report_path = os.path.join(self.output_dir, 'speed_benchmark_report.txt')
            with open(report_path, 'w') as f:
                f.write(report)
            print(f"Speed benchmark report saved to {report_path}")
        
        return results
    
    def benchmark_memory(self, save_results: bool = True) -> Dict[str, Any]:
        """
        Benchmark memory usage of the system
        
        Args:
            save_results: Whether to save results to file
            
        Returns:
            Memory benchmark results
        """
        if self.detector is None or self.classifier is None:
            raise ValueError("Both detector and classifier must be setup.")
        
        if self.dataset_loader is None:
            raise ValueError("Dataset not setup. Call setup_dataset() first.")
        
        print("Starting memory benchmarking...")
        
        # Prepare test data
        test_images = []
        test_crops = []
        
        num_samples = min(30, len(self.dataset_loader))  # Fewer samples for memory testing
        for i in range(num_samples):
            try:
                image, target = self.dataset_loader[i]
                test_images.append(image)
                
                # Extract crops if available
                if isinstance(target, dict) and 'boxes' in target:
                    for box in target['boxes']:
                        x1, y1, x2, y2 = map(int, box)
                        crop = image[max(0, y1):max(0, y2), max(0, x1):max(0, x2)]
                        if crop.size > 0:
                            test_crops.append(crop)
                            
            except Exception as e:
                print(f"Error preparing test data from sample {i}: {e}")
                continue
        
        if not test_images:
            raise ValueError("No test images available for benchmarking")
        
        # Ensure we have some crops
        if not test_crops:
            for img in test_images[:5]:
                detections = self.detector.detect(img)
                for det in detections:
                    x1, y1, x2, y2 = map(int, det['box'])
                    crop = img[max(0, y1):max(0, y2), max(0, x1):max(0, x2)]
                    if crop.size > 0:
                        test_crops.append(crop)
        
        results = {}
        
        # Benchmark detector memory
        print("Benchmarking detector memory...")
        detector_results = self.memory_benchmark.benchmark_detector_memory(self.detector, test_images)
        results['detector_memory'] = detector_results
        
        # Benchmark classifier memory
        if test_crops:
            print("Benchmarking classifier memory...")
            classifier_results = self.memory_benchmark.benchmark_classifier_memory(self.classifier, test_crops)
            results['classifier_memory'] = classifier_results
        
        # Benchmark pipeline memory
        print("Benchmarking pipeline memory...")
        pipeline_results = self.memory_benchmark.benchmark_pipeline_memory(
            self.detector, self.classifier, test_images
        )
        results['pipeline_memory'] = pipeline_results
        
        # Test for memory leaks
        print("Testing for memory leaks...")
        leak_results = self.memory_benchmark.benchmark_memory_leaks(
            self.detector, self.classifier, test_images, iterations=50
        )
        results['memory_leak_test'] = leak_results
        
        # Add evaluation info
        results['evaluation_info'] = {
            'detector_info': self.detector.get_detector_info(),
            'classifier_info': self.classifier.get_classifier_info(),
            'dataset_info': self.dataset_loader.get_dataset_info(),
            'test_images_count': len(test_images),
            'test_crops_count': len(test_crops),
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        if save_results:
            output_path = os.path.join(self.output_dir, 'memory_benchmark.json')
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"Memory benchmark results saved to {output_path}")
            
            # Generate and save memory report
            if 'classifier_memory' in results:
                report = self.memory_benchmark.generate_memory_report(
                    detector_results, classifier_results, pipeline_results, leak_results
                )
            else:
                report = "Classifier memory benchmark not available - insufficient test crops"
            
            report_path = os.path.join(self.output_dir, 'memory_benchmark_report.txt')
            with open(report_path, 'w') as f:
                f.write(report)
            print(f"Memory benchmark report saved to {report_path}")
        
        return results
    
    def run_comprehensive_evaluation(self, save_results: bool = True) -> Dict[str, Any]:
        """
        Run comprehensive evaluation including all metrics and benchmarks
        
        Args:
            save_results: Whether to save results to file
            
        Returns:
            Complete evaluation results
        """
        print("Starting comprehensive evaluation...")
        print("=" * 60)
        
        results = {
            'evaluation_timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'system_info': {
                'detector_info': self.detector.get_detector_info() if self.detector else None,
                'classifier_info': self.classifier.get_classifier_info() if self.classifier else None,
                'dataset_info': self.dataset_loader.get_dataset_info() if self.dataset_loader else None
            }
        }
        
        try:
            # 1. Detection evaluation
            print("\n1. Detection Evaluation")
            print("-" * 30)
            detection_results = self.evaluate_detection(save_results=False)
            results['detection_evaluation'] = detection_results
            
            # 2. Classification evaluation
            print("\n2. Classification Evaluation")
            print("-" * 30)
            classification_results = self.evaluate_classification(save_results=False)
            results['classification_evaluation'] = classification_results
            
            # 3. Full pipeline evaluation
            print("\n3. Full Pipeline Evaluation")
            print("-" * 30)
            pipeline_results = self.evaluate_full_pipeline(save_results=False)
            results['pipeline_evaluation'] = pipeline_results
            
            # 4. Speed benchmarking
            print("\n4. Speed Benchmarking")
            print("-" * 30)
            speed_results = self.benchmark_speed(save_results=False)
            results['speed_benchmark'] = speed_results
            
            # 5. Memory benchmarking
            print("\n5. Memory Benchmarking")
            print("-" * 30)
            memory_results = self.benchmark_memory(save_results=False)
            results['memory_benchmark'] = memory_results
            
        except Exception as e:
            print(f"Error during comprehensive evaluation: {e}")
            results['error'] = str(e)
        
        if save_results:
            output_path = os.path.join(self.output_dir, 'comprehensive_evaluation.json')
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\nComprehensive evaluation results saved to {output_path}")
            
            # Generate summary report
            self._generate_summary_report(results)
        
        print("\nComprehensive evaluation complete!")
        print("=" * 60)
        
        return results
    
    def _generate_summary_report(self, results: Dict[str, Any]):
        """Generate a summary report of all evaluation results"""
        report = []
        report.append("=" * 80)
        report.append("FACE MASK DETECTION SYSTEM - COMPREHENSIVE EVALUATION REPORT")
        report.append("=" * 80)
        report.append(f"Evaluation Date: {results.get('evaluation_timestamp', 'Unknown')}")
        report.append("")
        
        # System Information
        system_info = results.get('system_info', {})
        report.append("SYSTEM CONFIGURATION")
        report.append("-" * 40)
        
        if system_info.get('detector_info'):
            detector_info = system_info['detector_info']
            report.append(f"Detector: {detector_info.get('type', 'Unknown')} ({detector_info.get('model_type', 'Unknown')})")
        
        if system_info.get('classifier_info'):
            classifier_info = system_info['classifier_info']
            report.append(f"Classifier: {classifier_info.get('type', 'Unknown')} ({classifier_info.get('framework', 'Unknown')})")
            report.append(f"Classes: {', '.join(classifier_info.get('class_names', []))}")
        
        if system_info.get('dataset_info'):
            dataset_info = system_info['dataset_info']
            report.append(f"Dataset: {dataset_info.get('type', 'Unknown')} ({dataset_info.get('num_samples', 0)} samples)")
        
        report.append("")
        
        # Detection Performance
        detection_eval = results.get('detection_evaluation', {})
        if detection_eval:
            report.append("DETECTION PERFORMANCE")
            report.append("-" * 40)
            report.append(f"Precision: {detection_eval.get('precision', 0):.3f}")
            report.append(f"Recall: {detection_eval.get('recall', 0):.3f}")
            report.append(f"F1 Score: {detection_eval.get('f1_score', 0):.3f}")
            report.append(f"Average Precision: {detection_eval.get('average_precision', 0):.3f}")
            report.append("")
        
        # Classification Performance
        classification_eval = results.get('classification_evaluation', {})
        if classification_eval:
            report.append("CLASSIFICATION PERFORMANCE")
            report.append("-" * 40)
            report.append(f"Overall Accuracy: {classification_eval.get('accuracy', 0):.3f}")
            
            prf = classification_eval.get('precision_recall_f1', {})
            if prf.get('macro_avg'):
                macro = prf['macro_avg']
                report.append(f"Macro F1: {macro.get('f1_score', 0):.3f}")
                report.append(f"Macro Precision: {macro.get('precision', 0):.3f}")
                report.append(f"Macro Recall: {macro.get('recall', 0):.3f}")
            report.append("")
        
        # Speed Performance
        speed_benchmark = results.get('speed_benchmark', {})
        if speed_benchmark:
            pipeline_bench = speed_benchmark.get('pipeline_benchmark', {})
            pipeline_stats = pipeline_bench.get('pipeline_stats', {})
            
            report.append("SPEED PERFORMANCE")
            report.append("-" * 40)
            report.append(f"Pipeline FPS: {pipeline_stats.get('pipeline_fps', 0):.1f}")
            report.append(f"Average Processing Time: {pipeline_stats.get('avg_total_time_ms', 0):.2f} ms")
            
            target_fps = pipeline_bench.get('target_fps', 30)
            meets_target = pipeline_stats.get('meets_target_fps', False)
            report.append(f"Real-time Target ({target_fps} FPS): {'✓ ACHIEVED' if meets_target else '✗ NOT ACHIEVED'}")
            report.append("")
        
        # Memory Usage
        memory_benchmark = results.get('memory_benchmark', {})
        if memory_benchmark:
            pipeline_memory = memory_benchmark.get('pipeline_memory', {})
            pipeline_stats = pipeline_memory.get('pipeline_stats', {})
            
            report.append("MEMORY USAGE")
            report.append("-" * 40)
            report.append(f"Peak Memory: {pipeline_stats.get('peak_memory_mb', 0):.1f} MB")
            report.append(f"Average Memory per Image: {pipeline_stats.get('avg_total_increase_per_image_mb', 0):.2f} MB")
            report.append(f"Average Memory per Face: {pipeline_stats.get('avg_memory_per_face_mb', 0):.2f} MB")
            
            leak_test = memory_benchmark.get('memory_leak_test', {})
            leak_analysis = leak_test.get('leak_analysis', {})
            has_leak = leak_analysis.get('has_memory_leak', False)
            report.append(f"Memory Leak Detection: {'⚠ POTENTIAL LEAK' if has_leak else '✓ NO LEAKS DETECTED'}")
            report.append("")
        
        # Overall Assessment
        report.append("OVERALL ASSESSMENT")
        report.append("-" * 40)
        
        # Determine overall system readiness
        readiness_score = 0
        total_checks = 0
        
        # Detection performance check
        if detection_eval.get('f1_score', 0) >= 0.7:
            readiness_score += 1
        total_checks += 1
        
        # Classification performance check
        if classification_eval.get('accuracy', 0) >= 0.8:
            readiness_score += 1
        total_checks += 1
        
        # Speed performance check
        if speed_benchmark:
            pipeline_bench = speed_benchmark.get('pipeline_benchmark', {})
            if pipeline_bench.get('pipeline_stats', {}).get('meets_target_fps', False):
                readiness_score += 1
            total_checks += 1
        
        # Memory performance check
        if memory_benchmark:
            leak_test = memory_benchmark.get('memory_leak_test', {})
            if not leak_test.get('leak_analysis', {}).get('has_memory_leak', True):
                readiness_score += 1
            total_checks += 1
        
        readiness_percentage = (readiness_score / total_checks * 100) if total_checks > 0 else 0
        
        if readiness_percentage >= 75:
            status = "✓ READY FOR DEPLOYMENT"
        elif readiness_percentage >= 50:
            status = "⚠ NEEDS OPTIMIZATION"
        else:
            status = "✗ REQUIRES SIGNIFICANT IMPROVEMENT"
        
        report.append(f"System Readiness: {readiness_percentage:.0f}% ({readiness_score}/{total_checks} checks passed)")
        report.append(f"Status: {status}")
        report.append("")
        
        # Recommendations
        report.append("RECOMMENDATIONS")
        report.append("-" * 40)
        
        recommendations = []
        
        # Detection recommendations
        if detection_eval.get('f1_score', 0) < 0.7:
            recommendations.append("• Improve face detection accuracy - consider different detector or fine-tuning")
        
        # Classification recommendations
        if classification_eval.get('accuracy', 0) < 0.8:
            recommendations.append("• Enhance mask classification - add more training data or improve model architecture")
        
        # Speed recommendations
        if speed_benchmark:
            pipeline_bench = speed_benchmark.get('pipeline_benchmark', {})
            if not pipeline_bench.get('pipeline_stats', {}).get('meets_target_fps', True):
                recommendations.append("• Optimize for real-time performance - consider model quantization or hardware acceleration")
        
        # Memory recommendations
        if memory_benchmark:
            leak_test = memory_benchmark.get('memory_leak_test', {})
            if leak_test.get('leak_analysis', {}).get('has_memory_leak', False):
                recommendations.append("• Address potential memory leaks - review resource cleanup")
            
            pipeline_memory = memory_benchmark.get('pipeline_memory', {})
            peak_memory = pipeline_memory.get('pipeline_stats', {}).get('peak_memory_mb', 0)
            if peak_memory > 1000:  # > 1GB
                recommendations.append("• Optimize memory usage - consider batch processing or model compression")
        
        if not recommendations:
            recommendations.append("• System performance is satisfactory for current requirements")
        
        for rec in recommendations:
            report.append(rec)
        
        report.append("")
        report.append("=" * 80)
        report.append("End of Report")
        report.append("=" * 80)
        
        # Save report
        report_text = "\n".join(report)
        report_path = os.path.join(self.output_dir, 'evaluation_summary_report.txt')
        with open(report_path, 'w') as f:
            f.write(report_text)
        
        print(f"Summary report saved to {report_path}")
        print("\nSUMMARY:")
        print(f"System Readiness: {readiness_percentage:.0f}% - {status}")


def main():
    """
    Command-line interface for running evaluations
    """
    parser = argparse.ArgumentParser(description='Face Mask Detection System Evaluation Pipeline')
    
    # Setup arguments
    parser.add_argument('--detector', choices=['yunet', 'haar'], default='yunet',
                       help='Face detector type')
    parser.add_argument('--detector-model', type=str,
                       help='Path to detector model file')
    parser.add_argument('--classifier', choices=['numpy_cnn', 'pytorch'], default='pytorch',
                       help='Classifier type')
    parser.add_argument('--classifier-model', type=str,
                       help='Path to classifier model file')
    parser.add_argument('--dataset', choices=['andrewmvd', 'face12k', 'medical_mask'], default='andrewmvd',
                       help='Dataset type')
    parser.add_argument('--dataset-path', type=str, required=True,
                       help='Path to dataset')
    parser.add_argument('--output-dir', type=str, default='evaluation_results',
                       help='Output directory for results')
    
    # Evaluation options
    parser.add_argument('--eval-detection', action='store_true',
                       help='Run detection evaluation')
    parser.add_argument('--eval-classification', action='store_true',
                       help='Run classification evaluation')
    parser.add_argument('--eval-pipeline', action='store_true',
                       help='Run full pipeline evaluation')
    parser.add_argument('--benchmark-speed', action='store_true',
                       help='Run speed benchmarking')
    parser.add_argument('--benchmark-memory', action='store_true',
                       help='Run memory benchmarking')
    parser.add_argument('--comprehensive', action='store_true',
                       help='Run comprehensive evaluation (all tests)')
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = EvaluationPipeline(args.output_dir)
    
    try:
        # Setup detector
        detector_kwargs = {}
        if args.detector_model:
            detector_kwargs['model_path'] = args.detector_model
        pipeline.setup_detector(args.detector, **detector_kwargs)
        
        # Setup classifier
        classifier_kwargs = {}
        if args.classifier_model:
            classifier_kwargs['model_path'] = args.classifier_model
        pipeline.setup_classifier(args.classifier, **classifier_kwargs)
        
        # Setup dataset
        pipeline.setup_dataset(args.dataset, args.dataset_path)
        
        # Run evaluations
        if args.comprehensive:
            pipeline.run_comprehensive_evaluation()
        else:
            if args.eval_detection:
                pipeline.evaluate_detection()
            if args.eval_classification:
                pipeline.evaluate_classification()
            if args.eval_pipeline:
                pipeline.evaluate_full_pipeline()
            if args.benchmark_speed:
                pipeline.benchmark_speed()
            if args.benchmark_memory:
                pipeline.benchmark_memory()
            
            # If no specific evaluation selected, run comprehensive
            if not any([args.eval_detection, args.eval_classification, args.eval_pipeline,
                       args.benchmark_speed, args.benchmark_memory]):
                print("No specific evaluation selected. Running comprehensive evaluation...")
                pipeline.run_comprehensive_evaluation()
        
    except Exception as e:
        print(f"Error during evaluation: {e}")
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())
