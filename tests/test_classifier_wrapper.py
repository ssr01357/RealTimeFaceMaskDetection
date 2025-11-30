"""
Unit tests for classifier wrapper module
"""

import unittest
import numpy as np
import tempfile
import os
from unittest.mock import Mock, patch, MagicMock

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from evaluation.classifiers.classifier_wrapper import ClassifierWrapper, NumPyCNNClassifier, PyTorchClassifier


class TestNumPyCNNClassifier(unittest.TestCase):
    """Test NumPy CNN classifier implementation"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.test_image = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        self.class_names = ['no_mask', 'with_mask', 'incorrect_mask']
    
    @patch('os.path.exists')
    @patch('pickle.load')
    @patch('builtins.open', new_callable=MagicMock)
    def test_numpy_cnn_initialization(self, mock_open, mock_pickle_load, mock_exists):
        """Test NumPy CNN classifier initialization"""
        # Mock file operations
        mock_exists.return_value = True
        mock_model = Mock()
        mock_model.forward.return_value = np.array([[1.0, 2.0, 0.5]])
        mock_pickle_load.return_value = mock_model
        
        classifier = NumPyCNNClassifier("dummy_model.pkl", self.class_names)
        
        self.assertEqual(classifier.model, mock_model)
        self.assertEqual(classifier.class_names, self.class_names)
        self.assertEqual(classifier.input_size, 32)
    
    @patch('os.path.exists')
    @patch('pickle.load')
    @patch('builtins.open', new_callable=MagicMock)
    def test_numpy_cnn_predict_single(self, mock_open, mock_pickle_load, mock_exists):
        """Test NumPy CNN single prediction"""
        # Mock file operations
        mock_exists.return_value = True
        mock_model = Mock()
        mock_model.forward.return_value = np.array([[1.0, 2.0, 0.5]])
        mock_pickle_load.return_value = mock_model
        
        classifier = NumPyCNNClassifier("dummy_model.pkl", self.class_names)
        results = classifier.predict([self.test_image])
        
        self.assertEqual(len(results), 1)
        pred_class, probs = results[0]
        self.assertEqual(pred_class, 1)  # Index of max probability
        self.assertEqual(len(probs), 3)
    
    @patch('os.path.exists')
    @patch('pickle.load')
    @patch('builtins.open', new_callable=MagicMock)
    def test_numpy_cnn_predict_batch(self, mock_open, mock_pickle_load, mock_exists):
        """Test NumPy CNN batch prediction"""
        # Mock file operations
        mock_exists.return_value = True
        mock_model = Mock()
        mock_model.forward.return_value = np.array([
            [1.0, 2.0, 0.5],
            [2.0, 1.0, 0.5]
        ])
        mock_pickle_load.return_value = mock_model
        
        classifier = NumPyCNNClassifier("dummy_model.pkl", self.class_names)
        results = classifier.predict([self.test_image, self.test_image])
        
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0][0], 1)  # First prediction class
        self.assertEqual(results[1][0], 0)  # Second prediction class
    
    @patch('os.path.exists')
    @patch('pickle.load')
    @patch('builtins.open', new_callable=MagicMock)
    def test_numpy_cnn_preprocess_image(self, mock_open, mock_pickle_load, mock_exists):
        """Test NumPy CNN image preprocessing"""
        # Mock file operations
        mock_exists.return_value = True
        mock_model = Mock()
        mock_pickle_load.return_value = mock_model
        
        classifier = NumPyCNNClassifier("dummy_model.pkl", self.class_names)
        
        # Test preprocessing by calling predict with a single image
        # Since _preprocess_image is likely a private method, test through public interface
        mock_model.forward.return_value = np.array([[1.0, 2.0, 0.5]])
        results = classifier.predict([self.test_image])
        
        # Verify the model was called (indicating preprocessing worked)
        mock_model.forward.assert_called_once()
        self.assertEqual(len(results), 1)
    
    @patch('os.path.exists')
    @patch('pickle.load')
    @patch('builtins.open', new_callable=MagicMock)
    def test_numpy_cnn_get_info(self, mock_open, mock_pickle_load, mock_exists):
        """Test NumPy CNN classifier info"""
        # Mock file operations
        mock_exists.return_value = True
        mock_model = Mock()
        mock_pickle_load.return_value = mock_model
        
        classifier = NumPyCNNClassifier("dummy_model.pkl", self.class_names)
        
        class_names = classifier.get_class_names()
        self.assertEqual(class_names, self.class_names)


class TestPyTorchClassifier(unittest.TestCase):
    """Test PyTorch classifier implementation"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.test_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        self.class_names = ['no_mask', 'with_mask', 'incorrect_mask']
    
    def test_pytorch_classifier_initialization(self):
        """Test PyTorch classifier initialization"""
        # Mock model
        mock_model = Mock()
        mock_model.to.return_value = mock_model
        mock_model.eval.return_value = None
        
        with patch('torch.cuda.is_available', return_value=False):
            classifier = PyTorchClassifier(mock_model, device='cpu', class_names=self.class_names)
        
        self.assertEqual(classifier.class_names, self.class_names)
        self.assertEqual(classifier.device, 'cpu')
        mock_model.to.assert_called_with('cpu')
        mock_model.eval.assert_called_once()
    
    @patch('torch.no_grad')
    @patch('torch.stack')
    @patch('torch.softmax')
    def test_pytorch_predict_single(self, mock_softmax, mock_stack, mock_no_grad):
        """Test PyTorch single prediction"""
        # Mock model and its forward pass
        mock_model = Mock()
        mock_model.to.return_value = mock_model
        mock_model.eval.return_value = None
        
        # Mock tensor operations
        mock_tensor = Mock()
        mock_stack.return_value = mock_tensor
        mock_tensor.to.return_value = mock_tensor
        
        # Mock model output
        mock_logits = Mock()
        mock_model.return_value = mock_logits
        
        # Mock softmax output
        mock_probs_tensor = Mock()
        mock_softmax.return_value = mock_probs_tensor
        mock_probs_tensor.cpu.return_value.numpy.return_value = np.array([[0.1, 0.8, 0.1]])
        
        # Mock argmax
        mock_pred_tensor = Mock()
        with patch('torch.argmax', return_value=mock_pred_tensor):
            mock_pred_tensor.cpu.return_value.numpy.return_value = np.array([1])
            
            with patch('torch.cuda.is_available', return_value=False):
                classifier = PyTorchClassifier(mock_model, device='cpu', class_names=self.class_names)
                results = classifier.predict([self.test_image])
                
                self.assertEqual(len(results), 1)
                pred_class, probs = results[0]
                self.assertIsInstance(pred_class, (int, np.integer))
                self.assertIsInstance(probs, (list, np.ndarray))
    
    def test_pytorch_preprocess_image(self):
        """Test PyTorch image preprocessing"""
        mock_model = Mock()
        mock_model.to.return_value = mock_model
        mock_model.eval.return_value = None
        
        with patch('torch.cuda.is_available', return_value=False):
            classifier = PyTorchClassifier(mock_model, device='cpu', class_names=self.class_names)
            
            # Mock the transform
            mock_transform = Mock()
            mock_transform.return_value = Mock()
            classifier.transform = mock_transform
            
            # Test preprocessing through predict method since _preprocess_image might be private
            with patch('torch.stack') as mock_stack, \
                 patch('torch.no_grad'), \
                 patch('torch.softmax') as mock_softmax, \
                 patch('torch.argmax') as mock_argmax:
                
                mock_stack.return_value = Mock()
                mock_stack.return_value.to.return_value = Mock()
                mock_softmax.return_value = Mock()
                mock_softmax.return_value.cpu.return_value.numpy.return_value = np.array([[0.1, 0.8, 0.1]])
                mock_argmax.return_value = Mock()
                mock_argmax.return_value.cpu.return_value.numpy.return_value = np.array([1])
                
                classifier.predict([self.test_image])
                
                # Verify transform was called
                mock_transform.assert_called()
    
    def test_pytorch_get_info(self):
        """Test PyTorch classifier info"""
        mock_model = Mock()
        mock_model.to.return_value = mock_model
        mock_model.eval.return_value = None
        
        with patch('torch.cuda.is_available', return_value=False):
            classifier = PyTorchClassifier(mock_model, device='cpu', class_names=self.class_names)
            
            class_names = classifier.get_class_names()
            self.assertEqual(class_names, self.class_names)


class TestClassifierWrapper(unittest.TestCase):
    """Test classifier wrapper"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.test_image = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        self.class_names = ['no_mask', 'with_mask', 'incorrect_mask']
    
    @patch('os.path.exists')
    @patch('pickle.load')
    @patch('builtins.open', new_callable=MagicMock)
    def test_wrapper_numpy_cnn_creation(self, mock_open, mock_pickle_load, mock_exists):
        """Test wrapper creates NumPy CNN classifier correctly"""
        # Mock file operations
        mock_exists.return_value = True
        mock_model = Mock()
        mock_model.forward.return_value = np.array([[1.0, 2.0, 0.5]])
        mock_pickle_load.return_value = mock_model
        
        wrapper = ClassifierWrapper('numpy_cnn', model_path='dummy_model.pkl', class_names=self.class_names)
        
        self.assertIsInstance(wrapper.classifier, NumPyCNNClassifier)
    
    def test_wrapper_pytorch_creation(self):
        """Test wrapper creates PyTorch classifier correctly"""
        mock_model = Mock()
        mock_model.to.return_value = mock_model
        mock_model.eval.return_value = None
        
        with patch('torch.cuda.is_available', return_value=False):
            wrapper = ClassifierWrapper('pytorch', model=mock_model, device='cpu', class_names=self.class_names)
            
            self.assertIsInstance(wrapper.classifier, PyTorchClassifier)
    
    def test_wrapper_invalid_classifier_type(self):
        """Test wrapper raises error for invalid classifier type"""
        with self.assertRaises(ValueError):
            ClassifierWrapper('invalid_classifier')
    
    @patch('os.path.exists')
    @patch('pickle.load')
    @patch('builtins.open', new_callable=MagicMock)
    def test_wrapper_predict_method(self, mock_open, mock_pickle_load, mock_exists):
        """Test wrapper predict method"""
        # Mock file operations
        mock_exists.return_value = True
        mock_model = Mock()
        mock_model.forward.return_value = np.array([[1.0, 2.0, 0.5]])
        mock_pickle_load.return_value = mock_model
        
        wrapper = ClassifierWrapper('numpy_cnn', model_path='dummy_model.pkl', class_names=self.class_names)
        results = wrapper.predict([self.test_image])
        
        self.assertEqual(len(results), 1)
        self.assertIsInstance(results[0], tuple)
        self.assertEqual(len(results[0]), 2)  # (class, probs)
    
    @patch('os.path.exists')
    @patch('pickle.load')
    @patch('builtins.open', new_callable=MagicMock)
    def test_wrapper_predict_single_method(self, mock_open, mock_pickle_load, mock_exists):
        """Test wrapper predict_single method"""
        # Mock file operations
        mock_exists.return_value = True
        mock_model = Mock()
        mock_model.forward.return_value = np.array([[1.0, 2.0, 0.5]])
        mock_pickle_load.return_value = mock_model
        
        wrapper = ClassifierWrapper('numpy_cnn', model_path='dummy_model.pkl', class_names=self.class_names)
        pred_class, probs = wrapper.predict_single(self.test_image)
        
        self.assertIsInstance(pred_class, (int, np.integer))
        self.assertIsInstance(probs, (list, np.ndarray))
    
    @patch('os.path.exists')
    @patch('pickle.load')
    @patch('builtins.open', new_callable=MagicMock)
    def test_wrapper_get_class_names(self, mock_open, mock_pickle_load, mock_exists):
        """Test wrapper get_class_names method"""
        # Mock file operations
        mock_exists.return_value = True
        mock_model = Mock()
        mock_pickle_load.return_value = mock_model
        
        wrapper = ClassifierWrapper('numpy_cnn', model_path='dummy_model.pkl', class_names=self.class_names)
        
        names = wrapper.get_class_names()
        self.assertEqual(names, self.class_names)
    
    @patch('os.path.exists')
    @patch('pickle.load')
    @patch('builtins.open', new_callable=MagicMock)
    def test_wrapper_get_classifier_info(self, mock_open, mock_pickle_load, mock_exists):
        """Test wrapper get_classifier_info method"""
        # Mock file operations
        mock_exists.return_value = True
        mock_model = Mock()
        mock_pickle_load.return_value = mock_model
        
        wrapper = ClassifierWrapper('numpy_cnn', model_path='dummy_model.pkl', class_names=self.class_names)
        
        info = wrapper.get_classifier_info()
        self.assertIn('type', info)
        self.assertIn('class_names', info)


if __name__ == '__main__':
    unittest.main()
