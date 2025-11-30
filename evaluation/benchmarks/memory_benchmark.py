"""
Memory usage benchmarking for face detection and classification
"""

import psutil
import os
import gc
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
import time
from contextlib import contextmanager


@contextmanager
def memory_monitor():
    """Context manager for monitoring memory usage"""
    process = psutil.Process(os.getpid())
    
    # Force garbage collection before measurement
    gc.collect()
    
    # Get initial memory usage
    initial_memory = process.memory_info()
    peak_memory = initial_memory
    
    def get_current_memory():
        nonlocal peak_memory
        current = process.memory_info()
        if current.rss > peak_memory.rss:
            peak_memory = current
        return current
    
    yield get_current_memory
    
    # Final memory measurement
    final_memory = process.memory_info()
    
    return {
        'initial_rss_mb': initial_memory.rss / 1024 / 1024,
        'final_rss_mb': final_memory.rss / 1024 / 1024,
        'peak_rss_mb': peak_memory.rss / 1024 / 1024,
        'memory_increase_mb': (final_memory.rss - initial_memory.rss) / 1024 / 1024
    }


class MemoryBenchmark:
    """
    Memory usage benchmarking for face mask detection pipeline
    """
    
    def __init__(self):
        """Initialize memory benchmark"""
        self.process = psutil.Process(os.getpid())
    
    def get_current_memory_usage(self) -> Dict[str, float]:
        """
        Get current memory usage statistics
        
        Returns:
            Dictionary with memory usage in MB
        """
        memory_info = self.process.memory_info()
        memory_percent = self.process.memory_percent()
        
        return {
            'rss_mb': memory_info.rss / 1024 / 1024,  # Resident Set Size
            'vms_mb': memory_info.vms / 1024 / 1024,  # Virtual Memory Size
            'percent': memory_percent,
            'available_mb': psutil.virtual_memory().available / 1024 / 1024,
            'total_mb': psutil.virtual_memory().total / 1024 / 1024
        }
    
    def benchmark_detector_memory(self, detector, test_images: List[np.ndarray],
                                image_sizes: List[Tuple[int, int]] = None) -> Dict[str, Any]:
        """
        Benchmark memory usage of face detector
        
        Args:
            detector: Face detector with detect() method
            test_images: List of test images
            image_sizes: List of (width, height) tuples to test
            
        Returns:
            Dictionary with memory usage results
        """
        if image_sizes is None:
            image_sizes = [(640, 480), (1280, 720), (1920, 1080)]
        
        results = {
            'detector_info': detector.get_detector_info() if hasattr(detector, 'get_detector_info') else {},
            'image_sizes': {},
            'overall_stats': {}
        }
        
        all_peak_memory = []
        all_memory_increases = []
        
        for width, height in image_sizes:
            print(f"Benchmarking detector memory usage at {width}x{height}...")
            
            # Resize test images
            resized_images = []
            for img in test_images[:10]:  # Limit to 10 images for memory testing
                resized = cv2.resize(img, (width, height))
                resized_images.append(resized)
            
            # Force garbage collection
            gc.collect()
            initial_memory = self.get_current_memory_usage()
            
            peak_memory = initial_memory['rss_mb']
            memory_samples = []
            
            # Run detection and monitor memory
            for img in resized_images:
                pre_detection_memory = self.get_current_memory_usage()
                
                # Run detection
                detections = detector.detect(img)
                
                post_detection_memory = self.get_current_memory_usage()
                
                # Track peak memory
                if post_detection_memory['rss_mb'] > peak_memory:
                    peak_memory = post_detection_memory['rss_mb']
                
                memory_samples.append({
                    'pre_detection_mb': pre_detection_memory['rss_mb'],
                    'post_detection_mb': post_detection_memory['rss_mb'],
                    'detection_increase_mb': post_detection_memory['rss_mb'] - pre_detection_memory['rss_mb'],
                    'num_detections': len(detections)
                })
            
            # Force garbage collection and get final memory
            gc.collect()
            final_memory = self.get_current_memory_usage()
            
            # Calculate statistics
            avg_detection_increase = np.mean([s['detection_increase_mb'] for s in memory_samples])
            max_detection_increase = max([s['detection_increase_mb'] for s in memory_samples])
            total_memory_increase = final_memory['rss_mb'] - initial_memory['rss_mb']
            
            all_peak_memory.append(peak_memory)
            all_memory_increases.append(total_memory_increase)
            
            results['image_sizes'][f'{width}x{height}'] = {
                'initial_memory_mb': initial_memory['rss_mb'],
                'peak_memory_mb': peak_memory,
                'final_memory_mb': final_memory['rss_mb'],
                'total_increase_mb': total_memory_increase,
                'avg_detection_increase_mb': avg_detection_increase,
                'max_detection_increase_mb': max_detection_increase,
                'images_processed': len(resized_images)
            }
        
        # Overall statistics
        results['overall_stats'] = {
            'avg_peak_memory_mb': np.mean(all_peak_memory),
            'max_peak_memory_mb': max(all_peak_memory),
            'avg_memory_increase_mb': np.mean(all_memory_increases),
            'max_memory_increase_mb': max(all_memory_increases)
        }
        
        return results
    
    def benchmark_classifier_memory(self, classifier, test_crops: List[np.ndarray],
                                  batch_sizes: List[int] = None) -> Dict[str, Any]:
        """
        Benchmark memory usage of face mask classifier
        
        Args:
            classifier: Face mask classifier with predict() method
            test_crops: List of face crop images
            batch_sizes: List of batch sizes to test
            
        Returns:
            Dictionary with memory usage results
        """
        if batch_sizes is None:
            batch_sizes = [1, 4, 8, 16, 32]
        
        results = {
            'classifier_info': classifier.get_classifier_info() if hasattr(classifier, 'get_classifier_info') else {},
            'batch_sizes': {},
            'overall_stats': {}
        }
        
        all_peak_memory = []
        all_memory_increases = []
        
        for batch_size in batch_sizes:
            print(f"Benchmarking classifier memory usage with batch size {batch_size}...")
            
            # Force garbage collection
            gc.collect()
            initial_memory = self.get_current_memory_usage()
            
            peak_memory = initial_memory['rss_mb']
            memory_samples = []
            
            # Test multiple batches
            num_batches = min(20, len(test_crops) // batch_size)
            for i in range(num_batches):
                start_idx = i * batch_size
                end_idx = min(start_idx + batch_size, len(test_crops))
                batch = test_crops[start_idx:end_idx]
                
                pre_classification_memory = self.get_current_memory_usage()
                
                # Run classification
                predictions = classifier.predict(batch)
                
                post_classification_memory = self.get_current_memory_usage()
                
                # Track peak memory
                if post_classification_memory['rss_mb'] > peak_memory:
                    peak_memory = post_classification_memory['rss_mb']
                
                memory_samples.append({
                    'pre_classification_mb': pre_classification_memory['rss_mb'],
                    'post_classification_mb': post_classification_memory['rss_mb'],
                    'classification_increase_mb': post_classification_memory['rss_mb'] - pre_classification_memory['rss_mb'],
                    'batch_size': len(batch),
                    'num_predictions': len(predictions)
                })
            
            # Force garbage collection and get final memory
            gc.collect()
            final_memory = self.get_current_memory_usage()
            
            # Calculate statistics
            avg_classification_increase = np.mean([s['classification_increase_mb'] for s in memory_samples])
            max_classification_increase = max([s['classification_increase_mb'] for s in memory_samples])
            total_memory_increase = final_memory['rss_mb'] - initial_memory['rss_mb']
            
            all_peak_memory.append(peak_memory)
            all_memory_increases.append(total_memory_increase)
            
            results['batch_sizes'][f'batch_{batch_size}'] = {
                'initial_memory_mb': initial_memory['rss_mb'],
                'peak_memory_mb': peak_memory,
                'final_memory_mb': final_memory['rss_mb'],
                'total_increase_mb': total_memory_increase,
                'avg_classification_increase_mb': avg_classification_increase,
                'max_classification_increase_mb': max_classification_increase,
                'batches_processed': len(memory_samples),
                'avg_memory_per_sample_mb': avg_classification_increase / batch_size if batch_size > 0 else 0
            }
        
        # Overall statistics
        results['overall_stats'] = {
            'avg_peak_memory_mb': np.mean(all_peak_memory),
            'max_peak_memory_mb': max(all_peak_memory),
            'avg_memory_increase_mb': np.mean(all_memory_increases),
            'max_memory_increase_mb': max(all_memory_increases)
        }
        
        return results
    
    def benchmark_pipeline_memory(self, detector, classifier, test_images: List[np.ndarray],
                                num_images: int = 50) -> Dict[str, Any]:
        """
        Benchmark memory usage of full pipeline
        
        Args:
            detector: Face detector
            classifier: Face mask classifier
            test_images: List of test images
            num_images: Number of images to process
            
        Returns:
            Dictionary with memory usage results
        """
        print("Benchmarking pipeline memory usage...")
        
        results = {
            'detector_info': detector.get_detector_info() if hasattr(detector, 'get_detector_info') else {},
            'classifier_info': classifier.get_classifier_info() if hasattr(classifier, 'get_classifier_info') else {},
            'pipeline_stats': {},
            'memory_timeline': []
        }
        
        # Force garbage collection
        gc.collect()
        initial_memory = self.get_current_memory_usage()
        
        peak_memory = initial_memory['rss_mb']
        memory_timeline = []
        total_faces_processed = 0
        
        # Process images and track memory
        num_images = min(num_images, len(test_images))
        for i in range(num_images):
            img = test_images[i]
            
            # Memory before detection
            pre_detection_memory = self.get_current_memory_usage()
            
            # Run detection
            detections = detector.detect(img)
            
            # Memory after detection
            post_detection_memory = self.get_current_memory_usage()
            
            # Run classification if faces detected
            post_classification_memory = post_detection_memory
            faces_classified = 0
            
            if detections:
                crops = []
                for det in detections:
                    x1, y1, x2, y2 = det['box']
                    crop = img[max(0, y1):max(0, y2), max(0, x1):max(0, x2)]
                    if crop.size > 0:
                        crops.append(crop)
                
                if crops:
                    predictions = classifier.predict(crops)
                    faces_classified = len(predictions)
                    post_classification_memory = self.get_current_memory_usage()
            
            # Track peak memory
            current_peak = max(post_detection_memory['rss_mb'], post_classification_memory['rss_mb'])
            if current_peak > peak_memory:
                peak_memory = current_peak
            
            total_faces_processed += faces_classified
            
            # Record memory timeline
            memory_timeline.append({
                'image_index': i,
                'pre_detection_mb': pre_detection_memory['rss_mb'],
                'post_detection_mb': post_detection_memory['rss_mb'],
                'post_classification_mb': post_classification_memory['rss_mb'],
                'faces_detected': len(detections),
                'faces_classified': faces_classified,
                'detection_increase_mb': post_detection_memory['rss_mb'] - pre_detection_memory['rss_mb'],
                'classification_increase_mb': post_classification_memory['rss_mb'] - post_detection_memory['rss_mb'],
                'total_increase_mb': post_classification_memory['rss_mb'] - pre_detection_memory['rss_mb']
            })
        
        # Force garbage collection and get final memory
        gc.collect()
        final_memory = self.get_current_memory_usage()
        
        # Calculate statistics
        total_memory_increase = final_memory['rss_mb'] - initial_memory['rss_mb']
        avg_detection_increase = np.mean([m['detection_increase_mb'] for m in memory_timeline])
        avg_classification_increase = np.mean([m['classification_increase_mb'] for m in memory_timeline])
        avg_total_increase = np.mean([m['total_increase_mb'] for m in memory_timeline])
        
        max_detection_increase = max([m['detection_increase_mb'] for m in memory_timeline])
        max_classification_increase = max([m['classification_increase_mb'] for m in memory_timeline])
        max_total_increase = max([m['total_increase_mb'] for m in memory_timeline])
        
        results['pipeline_stats'] = {
            'initial_memory_mb': initial_memory['rss_mb'],
            'peak_memory_mb': peak_memory,
            'final_memory_mb': final_memory['rss_mb'],
            'total_memory_increase_mb': total_memory_increase,
            'avg_detection_increase_mb': avg_detection_increase,
            'avg_classification_increase_mb': avg_classification_increase,
            'avg_total_increase_per_image_mb': avg_total_increase,
            'max_detection_increase_mb': max_detection_increase,
            'max_classification_increase_mb': max_classification_increase,
            'max_total_increase_per_image_mb': max_total_increase,
            'images_processed': num_images,
            'total_faces_processed': total_faces_processed,
            'avg_memory_per_face_mb': (total_memory_increase / total_faces_processed) if total_faces_processed > 0 else 0
        }
        
        results['memory_timeline'] = memory_timeline
        
        return results
    
    def benchmark_memory_leaks(self, detector, classifier, test_images: List[np.ndarray],
                             iterations: int = 100) -> Dict[str, Any]:
        """
        Test for memory leaks by running pipeline multiple times
        
        Args:
            detector: Face detector
            classifier: Face mask classifier
            test_images: List of test images
            iterations: Number of iterations to run
            
        Returns:
            Dictionary with memory leak analysis
        """
        print(f"Testing for memory leaks over {iterations} iterations...")
        
        results = {
            'detector_info': detector.get_detector_info() if hasattr(detector, 'get_detector_info') else {},
            'classifier_info': classifier.get_classifier_info() if hasattr(classifier, 'get_classifier_info') else {},
            'leak_analysis': {},
            'memory_progression': []
        }
        
        # Force garbage collection
        gc.collect()
        initial_memory = self.get_current_memory_usage()
        
        memory_progression = []
        
        # Run multiple iterations
        for iteration in range(iterations):
            # Process a few images
            for i in range(min(5, len(test_images))):
                img = test_images[i % len(test_images)]
                
                # Run full pipeline
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
            
            # Force garbage collection every 10 iterations
            if iteration % 10 == 0:
                gc.collect()
            
            # Record memory usage
            current_memory = self.get_current_memory_usage()
            memory_progression.append({
                'iteration': iteration,
                'memory_mb': current_memory['rss_mb'],
                'memory_increase_mb': current_memory['rss_mb'] - initial_memory['rss_mb']
            })
        
        # Final garbage collection
        gc.collect()
        final_memory = self.get_current_memory_usage()
        
        # Analyze memory progression for leaks
        memory_values = [m['memory_mb'] for m in memory_progression]
        memory_increases = [m['memory_increase_mb'] for m in memory_progression]
        
        # Simple linear regression to detect trend
        n = len(memory_values)
        if n >= 2:
            x_vals = list(range(n))
            y_vals = memory_values
            
            sum_x = sum(x_vals)
            sum_y = sum(y_vals)
            sum_xy = sum(x * y for x, y in zip(x_vals, y_vals))
            sum_x2 = sum(x * x for x in x_vals)
            
            # Calculate slope (memory increase per iteration)
            slope = ((n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)) if (n * sum_x2 - sum_x * sum_x) != 0 else 0
            
            # Calculate correlation coefficient
            mean_x = sum_x / n
            mean_y = sum_y / n
            
            numerator = sum((x - mean_x) * (y - mean_y) for x, y in zip(x_vals, y_vals))
            denominator_x = sum((x - mean_x) ** 2 for x in x_vals)
            denominator_y = sum((y - mean_y) ** 2 for y in y_vals)
            
            correlation = (numerator / (denominator_x * denominator_y) ** 0.5) if denominator_x > 0 and denominator_y > 0 else 0
        else:
            slope = 0
            correlation = 0
        
        # Determine if there's a memory leak
        leak_threshold_mb_per_iteration = 0.1  # 0.1 MB per iteration
        has_memory_leak = slope > leak_threshold_mb_per_iteration and correlation > 0.7
        
        results['leak_analysis'] = {
            'initial_memory_mb': initial_memory['rss_mb'],
            'final_memory_mb': final_memory['rss_mb'],
            'total_memory_increase_mb': final_memory['rss_mb'] - initial_memory['rss_mb'],
            'memory_increase_per_iteration_mb': slope,
            'correlation_iteration_vs_memory': correlation,
            'has_potential_memory_leak': has_memory_leak,
            'iterations_tested': iterations,
            'max_memory_increase_mb': max(memory_increases),
            'avg_memory_increase_mb': np.mean(memory_increases),
            'std_memory_increase_mb': np.std(memory_increases)
        }
        
        results['memory_progression'] = memory_progression
        
        return results
    
    def generate_memory_report(self, detector_results: Dict, classifier_results: Dict,
                             pipeline_results: Dict, leak_results: Dict = None) -> str:
        """
        Generate a comprehensive memory usage report
        
        Args:
            detector_results: Results from benchmark_detector_memory
            classifier_results: Results from benchmark_classifier_memory
            pipeline_results: Results from benchmark_pipeline_memory
            leak_results: Results from benchmark_memory_leaks (optional)
            
        Returns:
            Formatted memory report as string
        """
        report = []
        report.append("=" * 80)
        report.append("FACE MASK DETECTION SYSTEM - MEMORY USAGE REPORT")
        report.append("=" * 80)
        report.append("")
        
        # System memory info
        system_memory = psutil.virtual_memory()
        report.append("SYSTEM MEMORY INFORMATION")
        report.append("-" * 40)
        report.append(f"Total System Memory: {system_memory.total / 1024 / 1024 / 1024:.1f} GB")
        report.append(f"Available Memory: {system_memory.available / 1024 / 1024 / 1024:.1f} GB")
        report.append(f"Memory Usage: {system_memory.percent:.1f}%")
        report.append("")
        
        # Detector memory usage
        report.append("FACE DETECTOR MEMORY USAGE")
        report.append("-" * 40)
        detector_stats = detector_results.get('overall_stats', {})
        report.append(f"Average Peak Memory: {detector_stats.get('avg_peak_memory_mb', 0):.1f} MB")
        report.append(f"Maximum Peak Memory: {detector_stats.get('max_peak_memory_mb', 0):.1f} MB")
        report.append(f"Average Memory Increase: {detector_stats.get('avg_memory_increase_mb', 0):.1f} MB")
        report.append("")
        
        # Memory usage by resolution
        report.append("Memory Usage by Resolution:")
        for resolution, stats in detector_results.get('image_sizes', {}).items():
            report.append(f"  {resolution}: Peak {stats['peak_memory_mb']:.1f} MB "
                         f"(+{stats['total_increase_mb']:.1f} MB)")
        report.append("")
        
        # Classifier memory usage
        report.append("FACE MASK CLASSIFIER MEMORY USAGE")
        report.append("-" * 40)
        classifier_stats = classifier_results.get('overall_stats', {})
        report.append(f"Average Peak Memory: {classifier_stats.get('avg_peak_memory_mb', 0):.1f} MB")
        report.append(f"Maximum Peak Memory: {classifier_stats.get('max_peak_memory_mb', 0):.1f} MB")
        report.append(f"Average Memory Increase: {classifier_stats.get('avg_memory_increase_mb', 0):.1f} MB")
        report.append("")
        
        # Memory usage by batch size
        report.append("Memory Usage by Batch Size:")
        for batch_size, stats in classifier_results.get('batch_sizes', {}).items():
            report.append(f"  {batch_size}: Peak {stats['peak_memory_mb']:.1f} MB "
                         f"({stats['avg_memory_per_sample_mb']:.2f} MB/sample)")
        report.append("")
        
        # Pipeline memory usage
        report.append("FULL PIPELINE MEMORY USAGE")
        report.append("-" * 40)
        pipeline_stats = pipeline_results.get('pipeline_stats', {})
        report.append(f"Peak Memory Usage: {pipeline_stats.get('peak_memory_mb', 0):.1f} MB")
        report.append(f"Total Memory Increase: {pipeline_stats.get('total_memory_increase_mb', 0):.1f} MB")
        report.append(f"Average Memory per Image: {pipeline_stats.get('avg_total_increase_per_image_mb', 0):.2f} MB")
        report.append(f"Average Memory per Face: {pipeline_stats.get('avg_memory_per_face_mb', 0):.2f} MB")
        report.append("")
        
        # Memory leak analysis
        if leak_results:
            report.append("MEMORY LEAK ANALYSIS")
            report.append("-" * 40)
            leak_analysis = leak_results.get('leak_analysis', {})
            
            has_leak = leak_analysis.get('has_potential_memory_leak', False)
            memory_per_iteration = leak_analysis.get('memory_increase_per_iteration_mb', 0)
            correlation = leak_analysis.get('correlation_iteration_vs_memory', 0)
            
            if has_leak:
                report.append("⚠️  POTENTIAL MEMORY LEAK DETECTED")
                report.append(f"   Memory increase per iteration: {memory_per_iteration:.3f} MB")
                report.append(f"   Correlation: {correlation:.3f}")
                report.append("   Recommendation: Investigate memory management in models")
            else:
                report.append("✅ NO SIGNIFICANT MEMORY LEAKS DETECTED")
                report.append(f"   Memory increase per iteration: {memory_per_iteration:.3f} MB")
                report.append(f"   Correlation: {correlation:.3f}")
            
            report.append(f"Total memory increase over {leak_analysis.get('iterations_tested', 0)} iterations: "
                         f"{leak_analysis.get('total_memory_increase_mb', 0):.1f} MB")
            report.append("")
        
        # Memory recommendations
        report.append("MEMORY OPTIMIZATION RECOMMENDATIONS")
        report.append("-" * 40)
        
        peak_memory = pipeline_stats.get('peak_memory_mb', 0)
        if peak_memory > 2000:  # > 2GB
            report.append("⚠️  HIGH MEMORY USAGE: Consider optimization")
            report.append("   Recommendations:")
            report.append("   - Process images in smaller batches")
            report.append("   - Reduce input image resolution")
            report.append("   - Use memory-efficient model architectures")
        elif peak_memory > 1000:  # > 1GB
            report.append("⚠️  MODERATE MEMORY USAGE: Monitor for large deployments")
            report.append("   Recommendations:")
            report.append("   - Consider batch size optimization")
            report.append("   - Monitor memory usage in production")
        else:
            report.append("✅ ACCEPTABLE MEMORY USAGE for most systems")
        
        # Check if memory usage is reasonable for target systems
        memory_per_face = pipeline_stats.get('avg_memory_per_face_mb', 0)
        if memory_per_face > 50:
            report.append("⚠️  HIGH MEMORY PER FACE: May not scale well")
        elif memory_per_face > 20:
            report.append("⚠️  MODERATE MEMORY PER FACE: Consider optimization for high-throughput scenarios")
        else:
            report.append("✅ EFFICIENT MEMORY USAGE per face")
        
        report.append("")
        report.append("=" * 80)
        
        return "\n".join(report)
