"""
Classification evaluation metrics including accuracy, precision, recall, F1, etc.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, 
    confusion_matrix, classification_report, roc_curve, auc
)
import matplotlib.pyplot as plt
import seaborn as sns


class ClassificationMetrics:
    """Calculate various classification evaluation metrics"""
    
    def __init__(self, class_names: List[str] = None):
        """
        Initialize classification metrics calculator
        
        Args:
            class_names: List of class names for reporting
        """
        self.class_names = class_names or ['No Mask', 'Mask']
        self.reset()
    
    def reset(self):
        """Reset accumulated statistics"""
        self.all_predictions = []
        self.all_ground_truth = []
        self.all_probabilities = []
    
    def update(self, predictions: List[int], ground_truth: List[int], 
               probabilities: List[List[float]] = None):
        """
        Update metrics with new predictions and ground truth
        
        Args:
            predictions: List of predicted class indices
            ground_truth: List of true class indices
            probabilities: List of probability distributions (optional)
        """
        self.all_predictions.extend(predictions)
        self.all_ground_truth.extend(ground_truth)
        
        if probabilities is not None:
            self.all_probabilities.extend(probabilities)
    
    def calculate_accuracy(self) -> float:
        """Calculate overall accuracy"""
        if not self.all_predictions:
            return 0.0
        
        return accuracy_score(self.all_ground_truth, self.all_predictions)
    
    def calculate_precision_recall_f1(self) -> Dict[str, Dict[str, float]]:
        """
        Calculate precision, recall, and F1 score per class and overall
        
        Returns:
            Dictionary with per-class and macro/micro averages
        """
        if not self.all_predictions:
            return {}
        
        # Calculate per-class metrics
        precision, recall, f1, support = precision_recall_fscore_support(
            self.all_ground_truth, self.all_predictions, 
            labels=list(range(len(self.class_names))), zero_division=0
        )
        
        # Calculate macro and micro averages
        macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
            self.all_ground_truth, self.all_predictions, average='macro', zero_division=0
        )
        
        micro_precision, micro_recall, micro_f1, _ = precision_recall_fscore_support(
            self.all_ground_truth, self.all_predictions, average='micro', zero_division=0
        )
        
        results = {
            'per_class': {},
            'macro_avg': {
                'precision': macro_precision,
                'recall': macro_recall,
                'f1_score': macro_f1
            },
            'micro_avg': {
                'precision': micro_precision,
                'recall': micro_recall,
                'f1_score': micro_f1
            }
        }
        
        # Add per-class results
        for i, class_name in enumerate(self.class_names):
            results['per_class'][class_name] = {
                'precision': precision[i],
                'recall': recall[i],
                'f1_score': f1[i],
                'support': int(support[i])
            }
        
        return results
    
    def calculate_confusion_matrix(self) -> np.ndarray:
        """Calculate confusion matrix"""
        if not self.all_predictions:
            return np.array([])
        
        return confusion_matrix(
            self.all_ground_truth, self.all_predictions,
            labels=list(range(len(self.class_names)))
        )
    
    def calculate_roc_auc(self) -> Dict[str, float]:
        """
        Calculate ROC AUC for each class (binary classification or one-vs-rest)
        
        Returns:
            Dictionary with AUC scores per class
        """
        if not self.all_probabilities or len(self.class_names) < 2:
            return {}
        
        y_true = np.array(self.all_ground_truth)
        y_probs = np.array(self.all_probabilities)
        
        auc_scores = {}
        
        if len(self.class_names) == 2:
            # Binary classification
            fpr, tpr, _ = roc_curve(y_true, y_probs[:, 1])
            auc_scores[self.class_names[1]] = auc(fpr, tpr)
        else:
            # Multi-class (one-vs-rest)
            for i, class_name in enumerate(self.class_names):
                y_true_binary = (y_true == i).astype(int)
                fpr, tpr, _ = roc_curve(y_true_binary, y_probs[:, i])
                auc_scores[class_name] = auc(fpr, tpr)
        
        return auc_scores
    
    def get_classification_report(self) -> str:
        """Get detailed classification report as string"""
        if not self.all_predictions:
            return "No predictions available"
        
        return classification_report(
            self.all_ground_truth, self.all_predictions,
            target_names=self.class_names, zero_division=0
        )
    
    def get_metrics_summary(self) -> Dict[str, any]:
        """
        Get comprehensive summary of all metrics
        
        Returns:
            Dictionary with all calculated metrics
        """
        if not self.all_predictions:
            return {}
        
        summary = {
            'accuracy': self.calculate_accuracy(),
            'precision_recall_f1': self.calculate_precision_recall_f1(),
            'confusion_matrix': self.calculate_confusion_matrix().tolist(),
            'classification_report': self.get_classification_report(),
            'total_samples': len(self.all_predictions),
            'class_names': self.class_names
        }
        
        # Add ROC AUC if probabilities are available
        roc_auc = self.calculate_roc_auc()
        if roc_auc:
            summary['roc_auc'] = roc_auc
        
        return summary
    
    def plot_confusion_matrix(self, save_path: Optional[str] = None, 
                            figsize: Tuple[int, int] = (8, 6)) -> plt.Figure:
        """
        Plot confusion matrix as heatmap
        
        Args:
            save_path: Path to save the plot (optional)
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        cm = self.calculate_confusion_matrix()
        
        fig, ax = plt.subplots(figsize=figsize)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.class_names,
                   yticklabels=self.class_names,
                   ax=ax)
        
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        ax.set_title('Confusion Matrix')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_roc_curves(self, save_path: Optional[str] = None,
                       figsize: Tuple[int, int] = (8, 6)) -> Optional[plt.Figure]:
        """
        Plot ROC curves for each class
        
        Args:
            save_path: Path to save the plot (optional)
            figsize: Figure size
            
        Returns:
            Matplotlib figure or None if no probabilities available
        """
        if not self.all_probabilities:
            return None
        
        y_true = np.array(self.all_ground_truth)
        y_probs = np.array(self.all_probabilities)
        
        fig, ax = plt.subplots(figsize=figsize)
        
        if len(self.class_names) == 2:
            # Binary classification
            fpr, tpr, _ = roc_curve(y_true, y_probs[:, 1])
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, label=f'{self.class_names[1]} (AUC = {roc_auc:.2f})')
        else:
            # Multi-class (one-vs-rest)
            for i, class_name in enumerate(self.class_names):
                y_true_binary = (y_true == i).astype(int)
                fpr, tpr, _ = roc_curve(y_true_binary, y_probs[:, i])
                roc_auc = auc(fpr, tpr)
                ax.plot(fpr, tpr, label=f'{class_name} (AUC = {roc_auc:.2f})')
        
        ax.plot([0, 1], [0, 1], 'k--', label='Random')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curves')
        ax.legend()
        ax.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig


def evaluate_classifier_on_dataset(classifier, dataset_loader, 
                                 class_names: List[str] = None) -> Dict[str, any]:
    """
    Evaluate a classifier on a complete dataset
    
    Args:
        classifier: Classifier with predict() method
        dataset_loader: Dataset loader yielding (face_crop, label) pairs
        class_names: List of class names
        
    Returns:
        Dictionary with evaluation metrics
    """
    metrics = ClassificationMetrics(class_names)
    
    for face_crop, label in dataset_loader:
        # Get prediction
        pred_class, probs = classifier.predict_single(face_crop)
        
        # Update metrics
        metrics.update([pred_class], [label], [probs])
    
    return metrics.get_metrics_summary()


def calculate_classification_speed(classifier, face_crops: List[np.ndarray],
                                 batch_sizes: List[int] = [1, 4, 8, 16],
                                 num_warmup: int = 10, num_iterations: int = 100) -> Dict[str, Dict[str, float]]:
    """
    Measure classification speed for different batch sizes
    
    Args:
        classifier: Classifier with predict() method
        face_crops: List of test face crops
        batch_sizes: List of batch sizes to test
        num_warmup: Number of warmup iterations
        num_iterations: Number of timing iterations
        
    Returns:
        Dictionary with speed metrics per batch size
    """
    import time
    
    results = {}
    
    for batch_size in batch_sizes:
        # Warmup
        for i in range(num_warmup):
            start_idx = (i * batch_size) % len(face_crops)
            end_idx = min(start_idx + batch_size, len(face_crops))
            batch = face_crops[start_idx:end_idx]
            _ = classifier.predict(batch)
        
        # Timing
        start_time = time.perf_counter()
        total_predictions = 0
        
        for i in range(num_iterations):
            start_idx = (i * batch_size) % len(face_crops)
            end_idx = min(start_idx + batch_size, len(face_crops))
            batch = face_crops[start_idx:end_idx]
            
            predictions = classifier.predict(batch)
            total_predictions += len(predictions)
        
        end_time = time.perf_counter()
        total_time = end_time - start_time
        
        avg_time_per_batch = total_time / num_iterations
        avg_time_per_sample = total_time / total_predictions if total_predictions > 0 else 0
        throughput = total_predictions / total_time if total_time > 0 else 0
        
        results[f'batch_size_{batch_size}'] = {
            'avg_time_per_batch_ms': avg_time_per_batch * 1000,
            'avg_time_per_sample_ms': avg_time_per_sample * 1000,
            'throughput_samples_per_sec': throughput,
            'total_predictions': total_predictions,
            'total_time_sec': total_time
        }
    
    return results
