"""
Detection evaluation metrics including IoU, mAP, precision, recall, etc.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from collections import defaultdict


def calculate_iou(box1: List[float], box2: List[float]) -> float:
    """
    Calculate Intersection over Union (IoU) between two bounding boxes
    
    Args:
        box1: [x1, y1, x2, y2] format
        box2: [x1, y1, x2, y2] format
        
    Returns:
        IoU value between 0 and 1
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    if x2 <= x1 or y2 <= y1:
        return 0.0
    
    intersection = (x2 - x1) * (y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.0


def match_detections_to_ground_truth(
    predictions: List[Dict], 
    ground_truth: List[Dict], 
    iou_threshold: float = 0.5
) -> Tuple[List[bool], List[bool]]:
    """
    Match predictions to ground truth boxes using IoU threshold
    
    Args:
        predictions: List of prediction dicts with 'box' and 'score' keys
        ground_truth: List of ground truth dicts with 'box' key
        iou_threshold: IoU threshold for positive match
        
    Returns:
        Tuple of (pred_matches, gt_matches) - boolean lists indicating matches
    """
    pred_matches = [False] * len(predictions)
    gt_matches = [False] * len(ground_truth)
    
    # Sort predictions by confidence score (descending)
    pred_indices = sorted(range(len(predictions)), 
                         key=lambda i: predictions[i]['score'], reverse=True)
    
    for pred_idx in pred_indices:
        pred_box = predictions[pred_idx]['box']
        best_iou = 0.0
        best_gt_idx = -1
        
        for gt_idx, gt_dict in enumerate(ground_truth):
            if gt_matches[gt_idx]:  # Already matched
                continue
                
            gt_box = gt_dict['box']
            iou = calculate_iou(pred_box, gt_box)
            
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx
        
        if best_iou >= iou_threshold and best_gt_idx >= 0:
            pred_matches[pred_idx] = True
            gt_matches[best_gt_idx] = True
    
    return pred_matches, gt_matches


class DetectionMetrics:
    """Calculate various detection evaluation metrics"""
    
    def __init__(self, iou_threshold: float = 0.5):
        """
        Initialize detection metrics calculator
        
        Args:
            iou_threshold: IoU threshold for positive detections
        """
        self.iou_threshold = iou_threshold
        self.reset()
    
    def reset(self):
        """Reset accumulated statistics"""
        self.total_predictions = 0
        self.total_ground_truth = 0
        self.total_true_positives = 0
        self.all_predictions = []
        self.all_ground_truth = []
        
    def update(self, predictions: List[Dict], ground_truth: List[Dict]):
        """
        Update metrics with new predictions and ground truth
        
        Args:
            predictions: List of prediction dicts with 'box' and 'score'
            ground_truth: List of ground truth dicts with 'box'
        """
        pred_matches, gt_matches = match_detections_to_ground_truth(
            predictions, ground_truth, self.iou_threshold
        )
        
        self.total_predictions += len(predictions)
        self.total_ground_truth += len(ground_truth)
        self.total_true_positives += sum(pred_matches)
        
        # Store for AP calculation
        for i, pred in enumerate(predictions):
            self.all_predictions.append({
                'score': pred['score'],
                'matched': pred_matches[i]
            })
    
    def calculate_precision_recall(self) -> Tuple[float, float]:
        """
        Calculate overall precision and recall
        
        Returns:
            Tuple of (precision, recall)
        """
        precision = (self.total_true_positives / self.total_predictions 
                    if self.total_predictions > 0 else 0.0)
        recall = (self.total_true_positives / self.total_ground_truth 
                 if self.total_ground_truth > 0 else 0.0)
        
        return precision, recall
    
    def calculate_f1_score(self) -> float:
        """Calculate F1 score"""
        precision, recall = self.calculate_precision_recall()
        
        if precision + recall == 0:
            return 0.0
        
        return 2 * (precision * recall) / (precision + recall)
    
    def calculate_ap(self) -> float:
        """
        Calculate Average Precision (AP)
        
        Returns:
            Average Precision value
        """
        if not self.all_predictions:
            return 0.0
        
        # Sort by confidence score (descending)
        sorted_preds = sorted(self.all_predictions, 
                             key=lambda x: x['score'], reverse=True)
        
        precisions = []
        recalls = []
        tp = 0
        
        for i, pred in enumerate(sorted_preds):
            if pred['matched']:
                tp += 1
            
            precision = tp / (i + 1)
            recall = tp / self.total_ground_truth if self.total_ground_truth > 0 else 0.0
            
            precisions.append(precision)
            recalls.append(recall)
        
        # Calculate AP using 11-point interpolation
        ap = 0.0
        for t in np.arange(0, 1.1, 0.1):
            # Find precisions for recalls >= t
            valid_precisions = [p for p, r in zip(precisions, recalls) if r >= t]
            max_precision = max(valid_precisions) if valid_precisions else 0.0
            ap += max_precision / 11.0
        
        return ap
    
    def get_metrics_summary(self) -> Dict[str, float]:
        """
        Get summary of all calculated metrics
        
        Returns:
            Dictionary with metric names and values
        """
        precision, recall = self.calculate_precision_recall()
        f1 = self.calculate_f1_score()
        ap = self.calculate_ap()
        
        return {
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'average_precision': ap,
            'total_predictions': self.total_predictions,
            'total_ground_truth': self.total_ground_truth,
            'true_positives': self.total_true_positives,
            'false_positives': self.total_predictions - self.total_true_positives,
            'false_negatives': self.total_ground_truth - self.total_true_positives
        }


def evaluate_detector_on_dataset(detector, dataset_loader, iou_threshold: float = 0.5) -> Dict[str, float]:
    """
    Evaluate a detector on a complete dataset
    
    Args:
        detector: Face detector with detect() method
        dataset_loader: Dataset loader yielding (image, ground_truth) pairs
        iou_threshold: IoU threshold for positive detections
        
    Returns:
        Dictionary with evaluation metrics
    """
    metrics = DetectionMetrics(iou_threshold)
    
    for image, gt_data in dataset_loader:
        # Convert ground truth to expected format
        ground_truth = []
        if 'boxes' in gt_data:
            for box in gt_data['boxes']:
                ground_truth.append({'box': box})
        
        # Get predictions
        predictions = detector.detect(image)
        
        # Update metrics
        metrics.update(predictions, ground_truth)
    
    return metrics.get_metrics_summary()


def calculate_detection_speed(detector, images: List[np.ndarray], 
                            num_warmup: int = 10, num_iterations: int = 100) -> Dict[str, float]:
    """
    Measure detection speed and throughput
    
    Args:
        detector: Face detector with detect() method
        images: List of test images
        num_warmup: Number of warmup iterations
        num_iterations: Number of timing iterations
        
    Returns:
        Dictionary with speed metrics
    """
    import time
    
    # Warmup
    for i in range(num_warmup):
        img = images[i % len(images)]
        _ = detector.detect(img)
    
    # Timing
    start_time = time.perf_counter()
    total_detections = 0
    
    for i in range(num_iterations):
        img = images[i % len(images)]
        detections = detector.detect(img)
        total_detections += len(detections)
    
    end_time = time.perf_counter()
    total_time = end_time - start_time
    
    avg_time_per_image = total_time / num_iterations
    fps = 1.0 / avg_time_per_image if avg_time_per_image > 0 else float('inf')
    avg_detections_per_image = total_detections / num_iterations
    
    return {
        'avg_time_per_image_ms': avg_time_per_image * 1000,
        'fps': fps,
        'total_time_sec': total_time,
        'avg_detections_per_image': avg_detections_per_image,
        'total_detections': total_detections
    }
