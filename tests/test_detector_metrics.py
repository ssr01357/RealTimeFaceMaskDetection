"""
Unit tests for detector metrics module
"""

import unittest
import numpy as np
from unittest.mock import Mock

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from evaluation.detectors.metrics import DetectionMetrics, calculate_iou, match_detections_to_ground_truth


class TestCalculateIoU(unittest.TestCase):
    """Test IoU calculation function"""
    
    def test_perfect_overlap(self):
        """Test IoU calculation for perfect overlap"""
        box1 = [10, 10, 50, 50]
        box2 = [10, 10, 50, 50]
        iou = calculate_iou(box1, box2)
        self.assertAlmostEqual(iou, 1.0, places=5)
    
    def test_no_overlap(self):
        """Test IoU calculation for no overlap"""
        box1 = [10, 10, 30, 30]
        box2 = [40, 40, 60, 60]
        iou = calculate_iou(box1, box2)
        self.assertEqual(iou, 0.0)
    
    def test_partial_overlap(self):
        """Test IoU calculation for partial overlap"""
        box1 = [10, 10, 30, 30]  # Area = 400
        box2 = [20, 20, 40, 40]  # Area = 400
        # Intersection = [20, 20, 30, 30] = 100
        # Union = 400 + 400 - 100 = 700
        # IoU = 100/700 ≈ 0.143
        iou = calculate_iou(box1, box2)
        self.assertAlmostEqual(iou, 100/700, places=3)
    
    def test_zero_area_boxes(self):
        """Test IoU calculation with zero area boxes"""
        box1 = [10, 10, 10, 10]  # Zero area
        box2 = [10, 10, 20, 20]
        iou = calculate_iou(box1, box2)
        self.assertEqual(iou, 0.0)
    
    def test_invalid_boxes(self):
        """Test IoU calculation with invalid boxes"""
        box1 = [30, 30, 10, 10]  # x2 < x1, y2 < y1
        box2 = [10, 10, 20, 20]
        iou = calculate_iou(box1, box2)
        self.assertEqual(iou, 0.0)


class TestMatchDetectionsToGroundTruth(unittest.TestCase):
    """Test detection matching function"""
    
    def test_perfect_matches(self):
        """Test matching with perfect IoU matches"""
        predictions = [
            {'box': [10, 10, 30, 30], 'score': 0.9},
            {'box': [50, 50, 70, 70], 'score': 0.8}
        ]
        ground_truth = [
            {'box': [10, 10, 30, 30]},
            {'box': [50, 50, 70, 70]}
        ]
        
        pred_matches, gt_matches = match_detections_to_ground_truth(predictions, ground_truth, iou_threshold=0.5)
        
        self.assertEqual(len(pred_matches), 2)
        self.assertEqual(len(gt_matches), 2)
        self.assertTrue(pred_matches[0])  # First prediction matched
        self.assertTrue(pred_matches[1])  # Second prediction matched
        self.assertTrue(gt_matches[0])    # First GT matched
        self.assertTrue(gt_matches[1])    # Second GT matched
    
    def test_no_matches(self):
        """Test matching with no overlapping boxes"""
        predictions = [
            {'box': [10, 10, 30, 30], 'score': 0.9}
        ]
        ground_truth = [
            {'box': [50, 50, 70, 70]}
        ]
        
        pred_matches, gt_matches = match_detections_to_ground_truth(predictions, ground_truth, iou_threshold=0.5)
        
        self.assertEqual(len(pred_matches), 1)
        self.assertEqual(len(gt_matches), 1)
        self.assertFalse(pred_matches[0])  # No prediction matched
        self.assertFalse(gt_matches[0])    # No GT matched
    
    def test_partial_matches(self):
        """Test matching with some overlapping boxes"""
        predictions = [
            {'box': [10, 10, 30, 30], 'score': 0.9},  # Should match GT 0
            {'box': [100, 100, 120, 120], 'score': 0.8}  # No match
        ]
        ground_truth = [
            {'box': [15, 15, 35, 35]}  # Overlaps with pred 0
        ]
        
        pred_matches, gt_matches = match_detections_to_ground_truth(predictions, ground_truth, iou_threshold=0.1)
        
        self.assertEqual(len(pred_matches), 2)
        self.assertEqual(len(gt_matches), 1)
        self.assertTrue(pred_matches[0])   # First prediction matched
        self.assertFalse(pred_matches[1])  # Second prediction didn't match
        self.assertTrue(gt_matches[0])     # GT matched
    
    def test_multiple_predictions_one_gt(self):
        """Test matching when multiple predictions match one GT (should pick best)"""
        predictions = [
            {'box': [10, 10, 30, 30], 'score': 0.7},  # Lower score
            {'box': [12, 12, 32, 32], 'score': 0.9}   # Higher score, should be chosen
        ]
        ground_truth = [
            {'box': [11, 11, 31, 31]}
        ]
        
        pred_matches, gt_matches = match_detections_to_ground_truth(predictions, ground_truth, iou_threshold=0.1)
        
        self.assertEqual(len(pred_matches), 2)
        self.assertEqual(len(gt_matches), 1)
        self.assertFalse(pred_matches[0])  # Lower score prediction not matched
        self.assertTrue(pred_matches[1])   # Higher score prediction matched
        self.assertTrue(gt_matches[0])     # GT matched


class TestDetectionMetrics(unittest.TestCase):
    """Test detection metrics accumulation and calculation"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.metrics = DetectionMetrics()
    
    def test_initialization(self):
        """Test metrics initialization"""
        self.assertEqual(self.metrics.total_predictions, 0)
        self.assertEqual(self.metrics.total_ground_truth, 0)
        self.assertEqual(self.metrics.total_true_positives, 0)
        self.assertEqual(len(self.metrics.all_predictions), 0)
    
    def test_update_perfect_matches(self):
        """Test updating metrics with perfect matches"""
        predictions = [
            {'box': [10, 10, 30, 30], 'score': 0.9},
            {'box': [50, 50, 70, 70], 'score': 0.8}
        ]
        ground_truth = [
            {'box': [10, 10, 30, 30]},
            {'box': [50, 50, 70, 70]}
        ]
        
        self.metrics.update(predictions, ground_truth)
        
        self.assertEqual(self.metrics.total_predictions, 2)
        self.assertEqual(self.metrics.total_ground_truth, 2)
        self.assertEqual(self.metrics.total_true_positives, 2)
        self.assertEqual(len(self.metrics.all_predictions), 2)
    
    def test_update_no_matches(self):
        """Test updating metrics with no matches"""
        predictions = [
            {'box': [10, 10, 30, 30], 'score': 0.9}
        ]
        ground_truth = [
            {'box': [50, 50, 70, 70]}
        ]
        
        self.metrics.update(predictions, ground_truth)
        
        self.assertEqual(self.metrics.total_predictions, 1)
        self.assertEqual(self.metrics.total_ground_truth, 1)
        self.assertEqual(self.metrics.total_true_positives, 0)
        self.assertEqual(len(self.metrics.all_predictions), 1)
    
    def test_get_metrics_summary_perfect_performance(self):
        """Test metrics summary with perfect performance"""
        predictions = [
            {'box': [10, 10, 30, 30], 'score': 0.9},
            {'box': [50, 50, 70, 70], 'score': 0.8}
        ]
        ground_truth = [
            {'box': [10, 10, 30, 30]},
            {'box': [50, 50, 70, 70]}
        ]
        
        self.metrics.update(predictions, ground_truth)
        summary = self.metrics.get_metrics_summary()
        
        self.assertEqual(summary['precision'], 1.0)
        self.assertEqual(summary['recall'], 1.0)
        self.assertEqual(summary['f1_score'], 1.0)
        self.assertGreater(summary['average_precision'], 0.9)
    
    def test_get_metrics_summary_no_predictions(self):
        """Test metrics summary with no predictions"""
        predictions = []
        ground_truth = [
            {'box': [10, 10, 30, 30]}
        ]
        
        self.metrics.update(predictions, ground_truth)
        summary = self.metrics.get_metrics_summary()
        
        self.assertEqual(summary['precision'], 0.0)
        self.assertEqual(summary['recall'], 0.0)
        self.assertEqual(summary['f1_score'], 0.0)
        self.assertEqual(summary['average_precision'], 0.0)
    
    def test_get_metrics_summary_no_ground_truth(self):
        """Test metrics summary with no ground truth"""
        predictions = [
            {'box': [10, 10, 30, 30], 'score': 0.9}
        ]
        ground_truth = []
        
        self.metrics.update(predictions, ground_truth)
        summary = self.metrics.get_metrics_summary()
        
        self.assertEqual(summary['precision'], 0.0)
        self.assertEqual(summary['recall'], 0.0)
        self.assertEqual(summary['f1_score'], 0.0)
        self.assertEqual(summary['average_precision'], 0.0)
    
    def test_multiple_updates(self):
        """Test multiple metric updates accumulate correctly"""
        # First batch
        predictions1 = [{'box': [10, 10, 30, 30], 'score': 0.9}]
        ground_truth1 = [{'box': [10, 10, 30, 30]}]
        
        # Second batch
        predictions2 = [{'box': [50, 50, 70, 70], 'score': 0.8}]
        ground_truth2 = [{'box': [50, 50, 70, 70]}]
        
        self.metrics.update(predictions1, ground_truth1)
        self.metrics.update(predictions2, ground_truth2)
        
        self.assertEqual(self.metrics.total_predictions, 2)
        self.assertEqual(self.metrics.total_ground_truth, 2)
        self.assertEqual(self.metrics.total_true_positives, 2)
        
        summary = self.metrics.get_metrics_summary()
        self.assertEqual(summary['precision'], 1.0)
        self.assertEqual(summary['recall'], 1.0)


if __name__ == '__main__':
    unittest.main()
