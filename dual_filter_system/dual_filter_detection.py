"""
Main dual filter detection system
Coordinates face detection and classification filters with side-by-side display
"""

import cv2
import numpy as np
import time
import argparse
import os
import sys
from typing import Optional, Dict, Any, Tuple
from enum import Enum

# Add current directory to path for imports
sys.path.append(os.path.dirname(__file__))

from filters.detection_filter import DetectionOnlyFilter
from filters.classification_filter import ClassificationFilter
from filters.filter_display import FilterDisplay
from ui.controls import ControlsManager, ControlAction


class DisplayMode(Enum):
    """Display mode options"""
    SIDE_BY_SIDE = "side_by_side"
    OVERLAY = "overlay"
    DIFFERENCE = "difference"
    DETECTION_ONLY = "detection_only"
    CLASSIFICATION_ONLY = "classification_only"


class DualFilterDetector:
    """
    Main dual filter detection system that manages two filters side by side
    """
    
    def __init__(self,
                 detector_name: str = 'haar',
                 classifier_model_path: Optional[str] = None,
                 camera_index: int = 0,
                 confidence_threshold: float = 0.6,
                 device: str = 'cuda',
                 frame_width: int = 640,
                 frame_height: int = 480,
                 num_classes: int = 2,
                 additional_classifier_paths: Optional[Dict[str, str]] = None):
        """
        Initialize dual filter detector
        
        Args:
            detector_name: Name of face detector to use
            classifier_model_path: Path to trained classifier model
            camera_index: Camera device index
            confidence_threshold: Detection confidence threshold
            device: Computation device ('cuda' or 'cpu')
            frame_width: Width of video frames
            frame_height: Height of video frames
            num_classes: Number of classification classes
        """
        self.detector_name = detector_name
        self.classifier_model_path = classifier_model_path
        self.camera_index = camera_index
        self.confidence_threshold = confidence_threshold
        self.device = device
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.num_classes = num_classes
        
        # Classifier management
        self.current_classifier_type = 'custom' if classifier_model_path else 'none'
        self.classifier_paths = additional_classifier_paths or {}
        if classifier_model_path:
            self.classifier_paths['custom'] = classifier_model_path
        
        # Initialize filters
        self.detection_filter = self._create_detection_filter()
        self.classification_filter = self._create_classification_filter()
        
        # Initialize display manager
        self.display = FilterDisplay(
            frame_width=frame_width,
            frame_height=frame_height
        )
        
        # Initialize controls
        self.controls = ControlsManager()
        self._setup_controls()
        
        # Camera setup
        self.cap = None
        self.is_running = False
        
        # Display state
        self.display_mode = DisplayMode.SIDE_BY_SIDE
        self.classifier_enabled = classifier_model_path is not None
        self.show_stats = True
        self.show_fps = True
        
        # Video recording
        self.video_writer = None
        self.recording = False
        self.output_path = None
        
        # Performance tracking
        self.session_start_time = None
        self.total_frames = 0
        
        print(f"Dual Filter Detector initialized:")
        print(f"  Detector: {detector_name}")
        print(f"  Classifier: {'Enabled' if self.classifier_enabled else 'Disabled'}")
        print(f"  Device: {device}")
        print(f"  Camera: {camera_index}")
    
    def _create_detection_filter(self) -> DetectionOnlyFilter:
        """Create detection-only filter"""
        return DetectionOnlyFilter(
            detector_name=self.detector_name,
            device=self.device,
            confidence_threshold=self.confidence_threshold
        )
    
    def _create_classification_filter(self) -> ClassificationFilter:
        """Create classification filter"""
        return ClassificationFilter(
            detector_name=self.detector_name,
            classifier_model_path=self.classifier_model_path,
            device=self.device,
            confidence_threshold=self.confidence_threshold,
            num_classes=self.num_classes
        )
    
    def _setup_controls(self):
        """Setup keyboard controls and callbacks"""
        # Register control callbacks
        self.controls.register_callback(ControlAction.QUIT, self._quit)
        self.controls.register_callback(ControlAction.SCREENSHOT, self._take_screenshot)
        self.controls.register_callback(ControlAction.SWITCH_DETECTOR_HAAR, 
                                      lambda: self._switch_detector('haar'))
        self.controls.register_callback(ControlAction.SWITCH_DETECTOR_YUNET, 
                                      lambda: self._switch_detector('yunet'))
        self.controls.register_callback(ControlAction.SWITCH_DETECTOR_MTCNN, 
                                      lambda: self._switch_detector('mtcnn'))
        self.controls.register_callback(ControlAction.SWITCH_DETECTOR_RETINAFACE, 
                                      lambda: self._switch_detector('retinaface'))
        self.controls.register_callback(ControlAction.TOGGLE_CLASSIFIER, self._toggle_classifier)
        self.controls.register_callback(ControlAction.SWITCH_CLASSIFIER_CUSTOM, 
                                      lambda: self._switch_classifier('custom'))
        self.controls.register_callback(ControlAction.SWITCH_CLASSIFIER_YEWON, 
                                      lambda: self._switch_classifier('yewon'))
        self.controls.register_callback(ControlAction.SWITCH_CLASSIFIER_WRAPPER, 
                                      lambda: self._switch_classifier('wrapper'))
        self.controls.register_callback(ControlAction.SWITCH_CLASSIFIER_NONE, 
                                      lambda: self._switch_classifier('none'))
        self.controls.register_callback(ControlAction.TOGGLE_OVERLAY, self._toggle_overlay)
        self.controls.register_callback(ControlAction.TOGGLE_DIFFERENCE, self._toggle_difference)
        self.controls.register_callback(ControlAction.INCREASE_CONFIDENCE, self._increase_confidence)
        self.controls.register_callback(ControlAction.DECREASE_CONFIDENCE, self._decrease_confidence)
        self.controls.register_callback(ControlAction.TOGGLE_STATS, self._toggle_stats)
        self.controls.register_callback(ControlAction.TOGGLE_FPS, self._toggle_fps)
        self.controls.register_callback(ControlAction.SAVE_VIDEO, self._toggle_recording)
        self.controls.register_callback(ControlAction.RESET_STATS, self._reset_stats)
    
    def _quit(self):
        """Quit application"""
        self.is_running = False
    
    def _take_screenshot(self):
        """Take screenshot of current display"""
        if hasattr(self, '_current_display_frame'):
            filename = self.display.save_screenshot(self._current_display_frame)
            print(f"Screenshot saved: {filename}")
        else:
            print("No frame available for screenshot")
    
    def _switch_detector(self, detector_name: str):
        """Switch to different detector"""
        if not self.controls.is_detector_available(detector_name):
            print(f"Detector {detector_name} not available")
            return
        
        print(f"Switching to {detector_name} detector...")
        
        # Update both filters
        success1 = self.detection_filter.switch_detector(detector_name)
        success2 = self.classification_filter.switch_detector(detector_name)
        
        if success1 and success2:
            self.detector_name = detector_name
            print(f"Successfully switched to {detector_name}")
        else:
            print(f"Failed to switch to {detector_name}")
    
    def _switch_classifier(self, classifier_type: str):
        """Switch to different classifier"""
        if not self.controls.is_classifier_available(classifier_type):
            print(f"Classifier {classifier_type} not available")
            return
        
        print(f"Switching to {classifier_type} classifier...")
        
        # Update classifier path and type
        self.current_classifier_type = classifier_type
        
        if classifier_type == 'none':
            new_path = None
            self.classifier_enabled = False
        else:
            new_path = self.classifier_paths.get(classifier_type)
            if not new_path:
                # Set default paths for known classifiers
                if classifier_type == 'custom':
                    new_path = os.path.join('model_artifacts', 'best_pytorch_model_custom.pth')
                    self.classifier_paths['custom'] = new_path
                else:
                    print(f"No model path configured for {classifier_type} classifier")
                    return
            self.classifier_enabled = True
        
        # Recreate classification filter with new classifier
        try:
            self.classification_filter = ClassificationFilter(
                detector_name=self.detector_name,
                classifier_model_path=new_path,
                device=self.device,
                confidence_threshold=self.confidence_threshold,
                num_classes=self.num_classes
            )
            
            classifier_desc = self.controls.get_classifier_info(classifier_type)
            print(f"Successfully switched to {classifier_desc}")
            
        except Exception as e:
            print(f"Failed to switch to {classifier_type}: {e}")
            self.classifier_enabled = False
    
    def _toggle_classifier(self):
        """Toggle classifier on/off"""
        self.classifier_enabled = not self.classifier_enabled
        status = "enabled" if self.classifier_enabled else "disabled"
        print(f"Classifier {status}")
    
    def _toggle_overlay(self):
        """Toggle overlay display mode"""
        if self.display_mode == DisplayMode.SIDE_BY_SIDE:
            self.display_mode = DisplayMode.OVERLAY
            print("Switched to overlay mode")
        else:
            self.display_mode = DisplayMode.SIDE_BY_SIDE
            print("Switched to side-by-side mode")
    
    def _toggle_difference(self):
        """Toggle difference view"""
        if self.display_mode == DisplayMode.SIDE_BY_SIDE:
            self.display_mode = DisplayMode.DIFFERENCE
            print("Switched to difference view")
        else:
            self.display_mode = DisplayMode.SIDE_BY_SIDE
            print("Switched to side-by-side mode")
    
    def _increase_confidence(self):
        """Increase confidence threshold"""
        step = self.controls.get_confidence_step()
        new_threshold = self.controls.validate_confidence_threshold(
            self.confidence_threshold + step
        )
        
        if new_threshold != self.confidence_threshold:
            self.confidence_threshold = new_threshold
            self.detection_filter.confidence_threshold = new_threshold
            self.classification_filter.confidence_threshold = new_threshold
            print(f"Confidence threshold: {self.confidence_threshold:.2f}")
    
    def _decrease_confidence(self):
        """Decrease confidence threshold"""
        step = self.controls.get_confidence_step()
        new_threshold = self.controls.validate_confidence_threshold(
            self.confidence_threshold - step
        )
        
        if new_threshold != self.confidence_threshold:
            self.confidence_threshold = new_threshold
            self.detection_filter.confidence_threshold = new_threshold
            self.classification_filter.confidence_threshold = new_threshold
            print(f"Confidence threshold: {self.confidence_threshold:.2f}")
    
    def _toggle_stats(self):
        """Toggle statistics display"""
        self.show_stats = not self.show_stats
        self.display.show_stats = self.show_stats
        status = "enabled" if self.show_stats else "disabled"
        print(f"Statistics display {status}")
    
    def _toggle_fps(self):
        """Toggle FPS counter"""
        self.show_fps = not self.show_fps
        self.display.show_fps = self.show_fps
        status = "enabled" if self.show_fps else "disabled"
        print(f"FPS counter {status}")
    
    def _toggle_recording(self):
        """Toggle video recording"""
        if not self.recording:
            self._start_recording()
        else:
            self._stop_recording()
    
    def _start_recording(self):
        """Start video recording"""
        if self.recording:
            return
        
        timestamp = int(time.time())
        self.output_path = f"dual_filter_output_{timestamp}.avi"
        
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        self.video_writer = cv2.VideoWriter(
            self.output_path,
            fourcc,
            20.0,
            (self.display.display_width, self.display.display_height)
        )
        
        if self.video_writer.isOpened():
            self.recording = True
            print(f"Started recording: {self.output_path}")
        else:
            print("Failed to start recording")
            self.video_writer = None
    
    def _stop_recording(self):
        """Stop video recording"""
        if not self.recording or self.video_writer is None:
            return
        
        self.video_writer.release()
        self.video_writer = None
        self.recording = False
        print(f"Recording saved: {self.output_path}")
    
    def _reset_stats(self):
        """Reset statistics"""
        self.detection_filter.total_detections = 0
        self.classification_filter.total_detections = 0
        self.classification_filter.class_distribution = {
            name: 0 for name in self.classification_filter.class_names
        }
        self.total_frames = 0
        self.session_start_time = time.time()
        print("Statistics reset")
    
    def start_camera(self) -> bool:
        """Initialize camera capture"""
        self.cap = cv2.VideoCapture(self.camera_index)
        
        if not self.cap.isOpened():
            print(f"Error: Could not open camera {self.camera_index}")
            return False
        
        # Set camera properties
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        print(f"Camera {self.camera_index} opened successfully")
        return True
    
    def stop_camera(self):
        """Release camera resources"""
        if self.cap:
            self.cap.release()
        
        if self.recording:
            self._stop_recording()
        
        self.display.close()
        self.is_running = False
    
    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Process frame through both filters
        
        Args:
            frame: Input frame
            
        Returns:
            Tuple of (detection_frame, classification_frame)
        """
        # Process with detection filter
        detection_frame = self.detection_filter.process_frame(frame)
        
        # Process with classification filter (if enabled)
        if self.classifier_enabled:
            classification_frame = self.classification_filter.process_frame(frame)
        else:
            # Show detection only with different title
            classification_frame = detection_frame.copy()
            cv2.putText(
                classification_frame,
                "Classification Disabled",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2
            )
        
        return detection_frame, classification_frame
    
    def create_display_frame(self, 
                           detection_frame: np.ndarray, 
                           classification_frame: np.ndarray) -> np.ndarray:
        """
        Create display frame based on current mode
        
        Args:
            detection_frame: Detection filter output
            classification_frame: Classification filter output
            
        Returns:
            Display frame
        """
        # Get filter statistics
        detection_stats = self.detection_filter.get_stats()
        classification_stats = self.classification_filter.get_stats()
        
        if self.display_mode == DisplayMode.SIDE_BY_SIDE:
            return self.display.create_combined_display(
                detection_frame,
                classification_frame,
                detection_stats,
                classification_stats
            )
        
        elif self.display_mode == DisplayMode.OVERLAY:
            return self.display.create_comparison_overlay(
                detection_frame,
                classification_frame,
                alpha=0.5
            )
        
        elif self.display_mode == DisplayMode.DIFFERENCE:
            return self.display.create_difference_view(
                detection_frame,
                classification_frame
            )
        
        elif self.display_mode == DisplayMode.DETECTION_ONLY:
            return cv2.resize(detection_frame, 
                            (self.display.display_width, self.display.display_height))
        
        elif self.display_mode == DisplayMode.CLASSIFICATION_ONLY:
            return cv2.resize(classification_frame, 
                            (self.display.display_width, self.display.display_height))
        
        else:
            # Default to side by side
            return self.display.create_combined_display(
                detection_frame,
                classification_frame,
                detection_stats,
                classification_stats
            )
    
    def run(self):
        """Run the dual filter detection system"""
        if not self.start_camera():
            return
        
        self.is_running = True
        self.session_start_time = time.time()
        
        print("\n" + "="*60)
        print("DUAL FILTER FACE DETECTION SYSTEM")
        print("="*60)
        self.controls.print_help()
        print("="*60)
        print("Starting detection loop...")
        
        try:
            while self.is_running:
                ret, frame = self.cap.read()
                if not ret:
                    print("Error: Could not read frame from camera")
                    break
                
                # Process frame through both filters
                detection_frame, classification_frame = self.process_frame(frame)
                
                # Create display frame
                display_frame = self.create_display_frame(
                    detection_frame, classification_frame
                )
                
                # Store for screenshot functionality
                self._current_display_frame = display_frame
                
                # Record video if enabled
                if self.recording and self.video_writer:
                    self.video_writer.write(display_frame)
                
                # Show frame and handle input
                key = self.display.show_frame(display_frame)
                action = self.controls.handle_key(key)
                
                # Update frame counter
                self.total_frames += 1
                
                # Check if quit was requested
                if not self.is_running:
                    break
        
        except KeyboardInterrupt:
            print("\nInterrupted by user")
        
        except Exception as e:
            print(f"Error during detection loop: {e}")
        
        finally:
            self.stop_camera()
            self._print_session_summary()
    
    def _print_session_summary(self):
        """Print session summary statistics"""
        if self.session_start_time:
            duration = time.time() - self.session_start_time
            avg_fps = self.total_frames / duration if duration > 0 else 0
            
            print("\n" + "="*60)
            print("SESSION SUMMARY")
            print("="*60)
            print(f"Duration: {duration:.1f} seconds")
            print(f"Total frames: {self.total_frames}")
            print(f"Average FPS: {avg_fps:.1f}")
            print(f"Detector used: {self.detector_name}")
            print(f"Classifier: {'Enabled' if self.classifier_enabled else 'Disabled'}")
            
            # Detection stats
            detection_stats = self.detection_filter.get_stats()
            print(f"Total detections: {detection_stats['total_detections']}")
            
            # Classification stats
            if self.classifier_enabled:
                classification_stats = self.classification_filter.get_stats()
                print("Class distribution:")
                for class_name, count in classification_stats['class_distribution'].items():
                    print(f"  {class_name}: {count}")
            
            print("="*60)


def main():
    """Main function with command line interface"""
    parser = argparse.ArgumentParser(description='Dual Filter Face Detection System')
    
    parser.add_argument('--detector', type=str, default='haar',
                       choices=['haar', 'yunet', 'mtcnn', 'retinaface'],
                       help='Face detector type (default: haar)')
    
    parser.add_argument('--classifier', type=str, default=None,
                       help='Path to classifier model file')
    
    parser.add_argument('--camera', type=int, default=0,
                       help='Camera index (default: 0)')
    
    parser.add_argument('--confidence', type=float, default=0.6,
                       help='Detection confidence threshold (default: 0.6)')
    
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'],
                       help='Computation device (default: cuda)')
    
    parser.add_argument('--width', type=int, default=640,
                       help='Frame width (default: 640)')
    
    parser.add_argument('--height', type=int, default=480,
                       help='Frame height (default: 480)')
    
    parser.add_argument('--classes', type=int, default=2,
                       choices=[2, 3],
                       help='Number of classification classes (default: 2)')
    
    args = parser.parse_args()
    
    # Initialize dual filter detector
    detector = DualFilterDetector(
        detector_name=args.detector,
        classifier_model_path=args.classifier,
        camera_index=args.camera,
        confidence_threshold=args.confidence,
        device=args.device,
        frame_width=args.width,
        frame_height=args.height,
        num_classes=args.classes
    )
    
    # Run detection system
    detector.run()


if __name__ == '__main__':
    main()
