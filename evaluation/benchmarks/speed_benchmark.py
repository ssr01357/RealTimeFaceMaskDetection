"""
Speed and performance benchmarking for face detection and classification
"""

import time
import numpy as np
import cv2
from typing import List, Dict, Tuple, Optional, Any
import statistics
from contextlib import contextmanager


@contextmanager
def timer():
    """Context manager for timing code blocks"""
    start = time.perf_counter()
    yield lambda: time.perf_counter() - start
    end = time.perf_counter()


class SpeedBenchmark:
    """
    Comprehensive speed benchmarking for face mask detection pipeline
    """
    
    def __init__(self, warmup_iterations: int = 10, timing_iterations: int = 100):
        """
        Initialize speed benchmark
        
        Args:
            warmup_iterations: Number of warmup runs
            timing_iterations: Number of timing runs for averaging
        """
        self.warmup_iterations = warmup_iterations
        self.timing_iterations = timing_iterations
    
    def benchmark_detector(self, detector, test_images: List[np.ndarray], 
                          image_sizes: List[Tuple[int, int]] = None) -> Dict[str, Any]:
        """
        Benchmark face detector performance
        
        Args:
            detector: Face detector with detect() method
            test_images: List of test images
            image_sizes: List of (width, height) tuples to test different resolutions
            
        Returns:
            Dictionary with benchmark results
        """
        if image_sizes is None:
            image_sizes = [(640, 480), (1280, 720), (1920, 1080)]
        
        results = {
            'detector_info': detector.get_detector_info() if hasattr(detector, 'get_detector_info') else {},
            'image_sizes': {},
            'overall_stats': {}
        }
        
        all_times = []
        all_detections = []
        
        for width, height in image_sizes:
            print(f"Benchmarking detector at {width}x{height}...")
            
            # Resize test images to target resolution
            resized_images = []
            for img in test_images:
                resized = cv2.resize(img, (width, height))
                resized_images.append(resized)
            
            # Warmup
            for i in range(self.warmup_iterations):
                img = resized_images[i % len(resized_images)]
                _ = detector.detect(img)
            
            # Timing runs
            times = []
            detection_counts = []
            
            for i in range(self.timing_iterations):
                img = resized_images[i % len(resized_images)]
                
                start_time = time.perf_counter()
                detections = detector.detect(img)
                end_time = time.perf_counter()
                
                elapsed = end_time - start_time
                times.append(elapsed)
                detection_counts.append(len(detections))
                all_times.append(elapsed)
                all_detections.append(len(detections))
            
            # Calculate statistics for this resolution
            avg_time = statistics.mean(times)
            std_time = statistics.stdev(times) if len(times) > 1 else 0
            min_time = min(times)
            max_time = max(times)
            fps = 1.0 / avg_time if avg_time > 0 else float('inf')
            avg_detections = statistics.mean(detection_counts)
            
            results['image_sizes'][f'{width}x{height}'] = {
                'avg_time_ms': avg_time * 1000,
                'std_time_ms': std_time * 1000,
                'min_time_ms': min_time * 1000,
                'max_time_ms': max_time * 1000,
                'fps': fps,
                'avg_detections_per_image': avg_detections,
                'total_iterations': self.timing_iterations
            }
        
        # Overall statistics
        overall_avg_time = statistics.mean(all_times)
        overall_std_time = statistics.stdev(all_times) if len(all_times) > 1 else 0
        overall_fps = 1.0 / overall_avg_time if overall_avg_time > 0 else float('inf')
        overall_avg_detections = statistics.mean(all_detections)
        
        results['overall_stats'] = {
            'avg_time_ms': overall_avg_time * 1000,
            'std_time_ms': overall_std_time * 1000,
            'fps': overall_fps,
            'avg_detections_per_image': overall_avg_detections,
            'total_images_processed': len(all_times)
        }
        
        return results
    
    def benchmark_classifier(self, classifier, test_crops: List[np.ndarray],
                           batch_sizes: List[int] = None) -> Dict[str, Any]:
        """
        Benchmark face mask classifier performance
        
        Args:
            classifier: Face mask classifier with predict() method
            test_crops: List of face crop images
            batch_sizes: List of batch sizes to test
            
        Returns:
            Dictionary with benchmark results
        """
        if batch_sizes is None:
            batch_sizes = [1, 4, 8, 16, 32]
        
        results = {
            'classifier_info': classifier.get_classifier_info() if hasattr(classifier, 'get_classifier_info') else {},
            'batch_sizes': {},
            'overall_stats': {}
        }
        
        all_times = []
        all_throughputs = []
        
        for batch_size in batch_sizes:
            print(f"Benchmarking classifier with batch size {batch_size}...")
            
            # Warmup
            for i in range(self.warmup_iterations):
                start_idx = (i * batch_size) % len(test_crops)
                end_idx = min(start_idx + batch_size, len(test_crops))
                batch = test_crops[start_idx:end_idx]
                _ = classifier.predict(batch)
            
            # Timing runs
            times = []
            sample_counts = []
            
            for i in range(self.timing_iterations):
                start_idx = (i * batch_size) % len(test_crops)
                end_idx = min(start_idx + batch_size, len(test_crops))
                batch = test_crops[start_idx:end_idx]
                
                start_time = time.perf_counter()
                predictions = classifier.predict(batch)
                end_time = time.perf_counter()
                
                elapsed = end_time - start_time
                times.append(elapsed)
                sample_counts.append(len(predictions))
                all_times.append(elapsed)
            
            # Calculate statistics for this batch size
            avg_time = statistics.mean(times)
            std_time = statistics.stdev(times) if len(times) > 1 else 0
            total_samples = sum(sample_counts)
            total_time = sum(times)
            throughput = total_samples / total_time if total_time > 0 else 0
            avg_time_per_sample = avg_time / batch_size if batch_size > 0 else 0
            
            all_throughputs.append(throughput)
            
            results['batch_sizes'][f'batch_{batch_size}'] = {
                'avg_time_per_batch_ms': avg_time * 1000,
                'std_time_per_batch_ms': std_time * 1000,
                'avg_time_per_sample_ms': avg_time_per_sample * 1000,
                'throughput_samples_per_sec': throughput,
                'total_samples_processed': total_samples,
                'total_iterations': self.timing_iterations
            }
        
        # Overall statistics
        overall_avg_time = statistics.mean(all_times)
        overall_std_time = statistics.stdev(all_times) if len(all_times) > 1 else 0
        overall_avg_throughput = statistics.mean(all_throughputs)
        
        results['overall_stats'] = {
            'avg_time_ms': overall_avg_time * 1000,
            'std_time_ms': overall_std_time * 1000,
            'avg_throughput_samples_per_sec': overall_avg_throughput,
            'total_batches_processed': len(all_times)
        }
        
        return results
    
    def benchmark_full_pipeline(self, detector, classifier, test_images: List[np.ndarray],
                               target_fps: float = 30.0) -> Dict[str, Any]:
        """
        Benchmark full detection + classification pipeline
        
        Args:
            detector: Face detector
            classifier: Face mask classifier
            test_images: List of test images
            target_fps: Target FPS for real-time performance assessment
            
        Returns:
            Dictionary with benchmark results
        """
        print("Benchmarking full pipeline...")
        
        results = {
            'detector_info': detector.get_detector_info() if hasattr(detector, 'get_detector_info') else {},
            'classifier_info': classifier.get_classifier_info() if hasattr(classifier, 'get_classifier_info') else {},
            'target_fps': target_fps,
            'pipeline_stats': {},
            'stage_breakdown': {}
        }
        
        # Warmup
        for i in range(self.warmup_iterations):
            img = test_images[i % len(test_images)]
            detections = detector.detect(img)
            
            if detections:
                crops = []
                for det in detections:
                    x1, y1, x2, y2 = det['box']
                    crop = img[max(0, y1):max(0, y2), max(0, x1):max(0, x2)]
                    if crop.size > 0:
                        crops.append(crop)
                
                if crops:
                    _ = classifier.predict(crops)
        
        # Timing runs
        total_times = []
        detection_times = []
        classification_times = []
        total_faces_detected = []
        total_faces_classified = []
        
        for i in range(self.timing_iterations):
            img = test_images[i % len(test_images)]
            
            # Time detection
            start_time = time.perf_counter()
            detections = detector.detect(img)
            detection_end = time.perf_counter()
            
            detection_time = detection_end - start_time
            detection_times.append(detection_time)
            total_faces_detected.append(len(detections))
            
            # Time classification
            classification_time = 0
            faces_classified = 0
            
            if detections:
                crops = []
                for det in detections:
                    x1, y1, x2, y2 = det['box']
                    crop = img[max(0, y1):max(0, y2), max(0, x1):max(0, x2)]
                    if crop.size > 0:
                        crops.append(crop)
                
                if crops:
                    classification_start = time.perf_counter()
                    predictions = classifier.predict(crops)
                    classification_end = time.perf_counter()
                    
                    classification_time = classification_end - classification_start
                    faces_classified = len(predictions)
            
            classification_times.append(classification_time)
            total_faces_classified.append(faces_classified)
            
            total_time = detection_time + classification_time
            total_times.append(total_time)
        
        # Calculate statistics
        avg_total_time = statistics.mean(total_times)
        std_total_time = statistics.stdev(total_times) if len(total_times) > 1 else 0
        avg_detection_time = statistics.mean(detection_times)
        avg_classification_time = statistics.mean(classification_times)
        
        pipeline_fps = 1.0 / avg_total_time if avg_total_time > 0 else float('inf')
        detection_fps = 1.0 / avg_detection_time if avg_detection_time > 0 else float('inf')
        
        avg_faces_detected = statistics.mean(total_faces_detected)
        avg_faces_classified = statistics.mean(total_faces_classified)
        
        # Check if meets real-time requirements
        meets_target_fps = pipeline_fps >= target_fps
        
        results['pipeline_stats'] = {
            'avg_total_time_ms': avg_total_time * 1000,
            'std_total_time_ms': std_total_time * 1000,
            'pipeline_fps': pipeline_fps,
            'meets_target_fps': meets_target_fps,
            'avg_faces_detected_per_image': avg_faces_detected,
            'avg_faces_classified_per_image': avg_faces_classified,
            'total_iterations': self.timing_iterations
        }
        
        results['stage_breakdown'] = {
            'detection': {
                'avg_time_ms': avg_detection_time * 1000,
                'fps': detection_fps,
                'percentage_of_total': (avg_detection_time / avg_total_time * 100) if avg_total_time > 0 else 0
            },
            'classification': {
                'avg_time_ms': avg_classification_time * 1000,
                'percentage_of_total': (avg_classification_time / avg_total_time * 100) if avg_total_time > 0 else 0
            }
        }
        
        return results
    
    def benchmark_scalability(self, detector, classifier, test_images: List[np.ndarray],
                            face_counts: List[int] = None) -> Dict[str, Any]:
        """
        Benchmark how performance scales with number of faces in image
        
        Args:
            detector: Face detector
            classifier: Face mask classifier
            test_images: List of test images
            face_counts: List of target face counts to test
            
        Returns:
            Dictionary with scalability results
        """
        if face_counts is None:
            face_counts = [1, 2, 5, 10, 20]
        
        print("Benchmarking scalability...")
        
        results = {
            'face_counts': {},
            'scalability_analysis': {}
        }
        
        # Group images by approximate face count
        images_by_face_count = {count: [] for count in face_counts}
        
        for img in test_images:
            detections = detector.detect(img)
            actual_face_count = len(detections)
            
            # Find closest target face count
            closest_count = min(face_counts, key=lambda x: abs(x - actual_face_count))
            if len(images_by_face_count[closest_count]) < 10:  # Limit per category
                images_by_face_count[closest_count].append(img)
        
        # Benchmark each face count category
        for target_count in face_counts:
            if not images_by_face_count[target_count]:
                continue
            
            category_images = images_by_face_count[target_count]
            times = []
            actual_face_counts = []
            
            # Warmup
            for i in range(min(5, len(category_images))):
                img = category_images[i]
                detections = detector.detect(img)
                if detections:
                    crops = [img[max(0, det['box'][1]):max(0, det['box'][3]), 
                               max(0, det['box'][0]):max(0, det['box'][2])] 
                            for det in detections if det['box'][2] > det['box'][0] and det['box'][3] > det['box'][1]]
                    if crops:
                        _ = classifier.predict(crops)
            
            # Timing runs
            iterations = min(self.timing_iterations, len(category_images) * 5)
            for i in range(iterations):
                img = category_images[i % len(category_images)]
                
                start_time = time.perf_counter()
                detections = detector.detect(img)
                
                if detections:
                    crops = []
                    for det in detections:
                        x1, y1, x2, y2 = det['box']
                        crop = img[max(0, y1):max(0, y2), max(0, x1):max(0, x2)]
                        if crop.size > 0:
                            crops.append(crop)
                    
                    if crops:
                        _ = classifier.predict(crops)
                
                end_time = time.perf_counter()
                
                times.append(end_time - start_time)
                actual_face_counts.append(len(detections))
            
            if times:
                avg_time = statistics.mean(times)
                avg_face_count = statistics.mean(actual_face_counts)
                fps = 1.0 / avg_time if avg_time > 0 else float('inf')
                
                results['face_counts'][f'{target_count}_faces'] = {
                    'target_face_count': target_count,
                    'actual_avg_face_count': avg_face_count,
                    'avg_time_ms': avg_time * 1000,
                    'fps': fps,
                    'images_tested': len(category_images),
                    'iterations': len(times)
                }
        
        # Analyze scalability trends
        if len(results['face_counts']) >= 2:
            face_counts_data = [(data['actual_avg_face_count'], data['avg_time_ms']) 
                              for data in results['face_counts'].values()]
            face_counts_data.sort()
            
            # Simple linear regression to estimate time complexity
            if len(face_counts_data) >= 2:
                x_vals = [x[0] for x in face_counts_data]
                y_vals = [x[1] for x in face_counts_data]
                
                # Calculate correlation coefficient
                n = len(x_vals)
                sum_x = sum(x_vals)
                sum_y = sum(y_vals)
                sum_xy = sum(x * y for x, y in zip(x_vals, y_vals))
                sum_x2 = sum(x * x for x in x_vals)
                
                correlation = ((n * sum_xy - sum_x * sum_y) / 
                             ((n * sum_x2 - sum_x * sum_x) * (n * sum(y * y for y in y_vals) - sum_y * sum_y)) ** 0.5
                             if (n * sum_x2 - sum_x * sum_x) > 0 and (n * sum(y * y for y in y_vals) - sum_y * sum_y) > 0 else 0)
                
                results['scalability_analysis'] = {
                    'correlation_faces_vs_time': correlation,
                    'min_faces_tested': min(x_vals),
                    'max_faces_tested': max(x_vals),
                    'time_increase_factor': max(y_vals) / min(y_vals) if min(y_vals) > 0 else 0
                }
        
        return results
    
    def generate_performance_report(self, detector_results: Dict, classifier_results: Dict,
                                  pipeline_results: Dict, scalability_results: Dict = None) -> str:
        """
        Generate a comprehensive performance report
        
        Args:
            detector_results: Results from benchmark_detector
            classifier_results: Results from benchmark_classifier
            pipeline_results: Results from benchmark_full_pipeline
            scalability_results: Results from benchmark_scalability (optional)
            
        Returns:
            Formatted performance report as string
        """
        report = []
        report.append("=" * 80)
        report.append("FACE MASK DETECTION SYSTEM - PERFORMANCE REPORT")
        report.append("=" * 80)
        report.append("")
        
        # Detector performance
        report.append("FACE DETECTOR PERFORMANCE")
        report.append("-" * 40)
        detector_stats = detector_results.get('overall_stats', {})
        report.append(f"Average Detection Time: {detector_stats.get('avg_time_ms', 0):.2f} ms")
        report.append(f"Detection FPS: {detector_stats.get('fps', 0):.1f}")
        report.append(f"Average Faces per Image: {detector_stats.get('avg_detections_per_image', 0):.1f}")
        report.append("")
        
        # Resolution breakdown
        report.append("Performance by Resolution:")
        for resolution, stats in detector_results.get('image_sizes', {}).items():
            report.append(f"  {resolution}: {stats['avg_time_ms']:.2f} ms ({stats['fps']:.1f} FPS)")
        report.append("")
        
        # Classifier performance
        report.append("FACE MASK CLASSIFIER PERFORMANCE")
        report.append("-" * 40)
        classifier_stats = classifier_results.get('overall_stats', {})
        report.append(f"Average Classification Time: {classifier_stats.get('avg_time_ms', 0):.2f} ms")
        report.append(f"Throughput: {classifier_stats.get('avg_throughput_samples_per_sec', 0):.1f} samples/sec")
        report.append("")
        
        # Batch size breakdown
        report.append("Performance by Batch Size:")
        for batch_size, stats in classifier_results.get('batch_sizes', {}).items():
            report.append(f"  {batch_size}: {stats['avg_time_per_sample_ms']:.2f} ms/sample "
                         f"({stats['throughput_samples_per_sec']:.1f} samples/sec)")
        report.append("")
        
        # Pipeline performance
        report.append("FULL PIPELINE PERFORMANCE")
        report.append("-" * 40)
        pipeline_stats = pipeline_results.get('pipeline_stats', {})
        stage_breakdown = pipeline_results.get('stage_breakdown', {})
        
        report.append(f"End-to-End Processing Time: {pipeline_stats.get('avg_total_time_ms', 0):.2f} ms")
        report.append(f"Pipeline FPS: {pipeline_stats.get('pipeline_fps', 0):.1f}")
        report.append(f"Target FPS ({pipeline_results.get('target_fps', 30)} FPS): "
                     f"{'✓ ACHIEVED' if pipeline_stats.get('meets_target_fps', False) else '✗ NOT ACHIEVED'}")
        report.append("")
        
        report.append("Stage Breakdown:")
        detection_stats = stage_breakdown.get('detection', {})
        classification_stats = stage_breakdown.get('classification', {})
        report.append(f"  Detection: {detection_stats.get('avg_time_ms', 0):.2f} ms "
                     f"({detection_stats.get('percentage_of_total', 0):.1f}%)")
        report.append(f"  Classification: {classification_stats.get('avg_time_ms', 0):.2f} ms "
                     f"({classification_stats.get('percentage_of_total', 0):.1f}%)")
        report.append("")
        
        # Scalability analysis
        if scalability_results:
            report.append("SCALABILITY ANALYSIS")
            report.append("-" * 40)
            scalability_analysis = scalability_results.get('scalability_analysis', {})
            
            if scalability_analysis:
                correlation = scalability_analysis.get('correlation_faces_vs_time', 0)
                report.append(f"Correlation (Faces vs Time): {correlation:.3f}")
                report.append(f"Time Increase Factor: {scalability_analysis.get('time_increase_factor', 0):.2f}x")
                
                if correlation > 0.8:
                    report.append("  → Strong positive correlation: Performance degrades significantly with more faces")
                elif correlation > 0.5:
                    report.append("  → Moderate correlation: Performance degrades moderately with more faces")
                else:
                    report.append("  → Weak correlation: Performance scales well with number of faces")
            
            report.append("")
            report.append("Performance by Face Count:")
            for face_count, stats in scalability_results.get('face_counts', {}).items():
                report.append(f"  {stats['actual_avg_face_count']:.1f} faces: "
                             f"{stats['avg_time_ms']:.2f} ms ({stats['fps']:.1f} FPS)")
            report.append("")
        
        # Performance recommendations
        report.append("PERFORMANCE RECOMMENDATIONS")
        report.append("-" * 40)
        
        pipeline_fps = pipeline_stats.get('pipeline_fps', 0)
        if pipeline_fps < 15:
            report.append("⚠️  CRITICAL: Very low FPS - not suitable for real-time applications")
            report.append("   Recommendations:")
            report.append("   - Consider using a faster detector (e.g., Haar instead of YuNet)")
            report.append("   - Reduce input image resolution")
            report.append("   - Optimize classifier architecture")
        elif pipeline_fps < 30:
            report.append("⚠️  WARNING: Below target FPS for smooth real-time performance")
            report.append("   Recommendations:")
            report.append("   - Optimize detection confidence threshold")
            report.append("   - Use batch processing for classification when possible")
        else:
            report.append("✅ GOOD: Achieves real-time performance requirements")
        
        report.append("")
        report.append("=" * 80)
        
        return "\n".join(report)
