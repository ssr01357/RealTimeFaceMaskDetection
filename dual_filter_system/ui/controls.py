"""
UI controls for dual filter system
Handles keyboard input and user interactions
"""

from typing import Dict, Callable, Any, Optional
from enum import Enum


class ControlAction(Enum):
    """Enumeration of available control actions"""
    QUIT = "quit"
    SCREENSHOT = "screenshot"
    SWITCH_DETECTOR_HAAR = "switch_detector_haar"
    SWITCH_DETECTOR_YUNET = "switch_detector_yunet"
    SWITCH_DETECTOR_MTCNN = "switch_detector_mtcnn"
    SWITCH_DETECTOR_RETINAFACE = "switch_detector_retinaface"
    TOGGLE_CLASSIFIER = "toggle_classifier"
    SWITCH_CLASSIFIER_CUSTOM = "switch_classifier_custom"
    SWITCH_CLASSIFIER_YEWON = "switch_classifier_yewon"
    SWITCH_CLASSIFIER_WRAPPER = "switch_classifier_wrapper"
    SWITCH_CLASSIFIER_NONE = "switch_classifier_none"
    TOGGLE_OVERLAY = "toggle_overlay"
    TOGGLE_DIFFERENCE = "toggle_difference"
    INCREASE_CONFIDENCE = "increase_confidence"
    DECREASE_CONFIDENCE = "decrease_confidence"
    TOGGLE_STATS = "toggle_stats"
    TOGGLE_FPS = "toggle_fps"
    SAVE_VIDEO = "save_video"
    RESET_STATS = "reset_stats"


class ControlsManager:
    """
    Manages keyboard controls and user interactions for dual filter system
    """
    
    def __init__(self):
        """Initialize controls manager"""
        # Default key mappings
        self.key_mappings = {
            ord('q'): ControlAction.QUIT,
            ord('Q'): ControlAction.QUIT,
            27: ControlAction.QUIT,  # ESC key
            ord('s'): ControlAction.SCREENSHOT,
            ord('S'): ControlAction.SCREENSHOT,
            ord('1'): ControlAction.SWITCH_DETECTOR_HAAR,
            ord('2'): ControlAction.SWITCH_DETECTOR_YUNET,
            ord('3'): ControlAction.SWITCH_DETECTOR_MTCNN,
            ord('4'): ControlAction.SWITCH_DETECTOR_RETINAFACE,
            ord('c'): ControlAction.TOGGLE_CLASSIFIER,
            ord('C'): ControlAction.TOGGLE_CLASSIFIER,
            ord('5'): ControlAction.SWITCH_CLASSIFIER_CUSTOM,
            ord('6'): ControlAction.SWITCH_CLASSIFIER_YEWON,
            ord('7'): ControlAction.SWITCH_CLASSIFIER_WRAPPER,
            ord('8'): ControlAction.SWITCH_CLASSIFIER_NONE,
            ord('o'): ControlAction.TOGGLE_OVERLAY,
            ord('O'): ControlAction.TOGGLE_OVERLAY,
            ord('d'): ControlAction.TOGGLE_DIFFERENCE,
            ord('D'): ControlAction.TOGGLE_DIFFERENCE,
            ord('+'): ControlAction.INCREASE_CONFIDENCE,
            ord('='): ControlAction.INCREASE_CONFIDENCE,  # + without shift
            ord('-'): ControlAction.DECREASE_CONFIDENCE,
            ord('_'): ControlAction.DECREASE_CONFIDENCE,  # - with shift
            ord('t'): ControlAction.TOGGLE_STATS,
            ord('T'): ControlAction.TOGGLE_STATS,
            ord('f'): ControlAction.TOGGLE_FPS,
            ord('F'): ControlAction.TOGGLE_FPS,
            ord('v'): ControlAction.SAVE_VIDEO,
            ord('V'): ControlAction.SAVE_VIDEO,
            ord('r'): ControlAction.RESET_STATS,
            ord('R'): ControlAction.RESET_STATS,
        }
        # Available classifiers
        self.available_classifiers = ['custom', 'yewon', 'wrapper', 'none']
        self.classifier_descriptions = {
            'custom': 'Custom PyTorch (Your Model)',
            'yewon': 'Yewon Pipeline Model',
            'wrapper': 'Evaluation Wrapper',
            'none': 'No Classifier'
        }
        
        # Action callbacks
        self.action_callbacks: Dict[ControlAction, Callable] = {}
        
        # Control state
        self.enabled = True
        self.help_visible = True
        
        # Available detectors
        self.available_detectors = ['haar', 'yunet', 'mtcnn', 'retinaface']
        self.detector_descriptions = {
            'haar': 'Haar Cascade (Fast, CPU)',
            'yunet': 'YuNet (Accurate, GPU)',
            'mtcnn': 'MTCNN (Precise, GPU)',
            'retinaface': 'RetinaFace (Best, GPU)'
        }
    
    def register_callback(self, action: ControlAction, callback: Callable):
        """
        Register callback for control action
        
        Args:
            action: Control action to register
            callback: Function to call when action is triggered
        """
        self.action_callbacks[action] = callback
    
    def handle_key(self, key_code: int) -> Optional[ControlAction]:
        """
        Handle keyboard input
        
        Args:
            key_code: Key code from cv2.waitKey()
            
        Returns:
            ControlAction if key was handled, None otherwise
        """
        if not self.enabled or key_code == -1:
            return None
        
        # Check if key is mapped to an action
        if key_code in self.key_mappings:
            action = self.key_mappings[key_code]
            
            # Execute callback if registered
            if action in self.action_callbacks:
                try:
                    self.action_callbacks[action]()
                except Exception as e:
                    print(f"Error executing callback for {action}: {e}")
            
            return action
        
        return None
    
    def get_help_text(self) -> str:
        """
        Get formatted help text for controls
        
        Returns:
            Multi-line help text string
        """
        help_lines = [
            "=== DUAL FILTER CONTROLS ===",
            "",
            "Basic Controls:",
            "  Q / ESC    - Quit application",
            "  S          - Save screenshot",
            "  V          - Toggle video recording",
            "",
            "Detector Selection:",
            "  1          - Haar Cascade (Fast, CPU)",
            "  2          - YuNet (Accurate, GPU)",
            "  3          - MTCNN (Precise, GPU)",
            "  4          - RetinaFace (Best, GPU)",
            "",
            "Classifier Selection:",
            "  5          - Custom PyTorch (Your Model)",
            "  6          - Yewon Pipeline Model",
            "  7          - Evaluation Wrapper",
            "  8          - No Classifier",
            "  C          - Toggle classifier on/off",
            "",
            "Display Options:",
            "  O          - Toggle overlay comparison",
            "  D          - Toggle difference view",
            "  T          - Toggle statistics display",
            "  F          - Toggle FPS counter",
            "",
            "Adjustments:",
            "  + / =      - Increase confidence threshold",
            "  - / _      - Decrease confidence threshold",
            "  R          - Reset statistics",
            "",
            "================================"
        ]
        
        return "\n".join(help_lines)
    
    def print_help(self):
        """Print help text to console"""
        print(self.get_help_text())
    
    def get_detector_info(self, detector_name: str) -> str:
        """
        Get description for detector
        
        Args:
            detector_name: Name of detector
            
        Returns:
            Human-readable description
        """
        return self.detector_descriptions.get(detector_name, f"Unknown detector: {detector_name}")
    
    def is_detector_available(self, detector_name: str) -> bool:
        """
        Check if detector is available
        
        Args:
            detector_name: Name of detector to check
            
        Returns:
            True if detector is available
        """
        return detector_name in self.available_detectors
    
    def get_available_detectors(self) -> list:
        """Get list of available detectors"""
        return self.available_detectors.copy()
    
    def get_classifier_info(self, classifier_name: str) -> str:
        """
        Get description for classifier
        
        Args:
            classifier_name: Name of classifier
            
        Returns:
            Human-readable description
        """
        return self.classifier_descriptions.get(classifier_name, f"Unknown classifier: {classifier_name}")
    
    def is_classifier_available(self, classifier_name: str) -> bool:
        """
        Check if classifier is available
        
        Args:
            classifier_name: Name of classifier to check
            
        Returns:
            True if classifier is available
        """
        return classifier_name in self.available_classifiers
    
    def get_available_classifiers(self) -> list:
        """Get list of available classifiers"""
        return self.available_classifiers.copy()
    
    def set_enabled(self, enabled: bool):
        """Enable or disable controls"""
        self.enabled = enabled
    
    def add_custom_key(self, key_code: int, action: ControlAction):
        """
        Add custom key mapping
        
        Args:
            key_code: Key code to map
            action: Action to trigger
        """
        self.key_mappings[key_code] = action
    
    def remove_key(self, key_code: int):
        """
        Remove key mapping
        
        Args:
            key_code: Key code to remove
        """
        if key_code in self.key_mappings:
            del self.key_mappings[key_code]
    
    def get_key_for_action(self, action: ControlAction) -> Optional[int]:
        """
        Get key code for action
        
        Args:
            action: Action to find key for
            
        Returns:
            Key code if found, None otherwise
        """
        for key_code, mapped_action in self.key_mappings.items():
            if mapped_action == action:
                return key_code
        return None
    
    def create_status_message(self, 
                            current_detector: str,
                            classifier_enabled: bool,
                            confidence_threshold: float,
                            overlay_mode: str = "side_by_side") -> str:
        """
        Create status message for display
        
        Args:
            current_detector: Currently active detector
            classifier_enabled: Whether classifier is enabled
            confidence_threshold: Current confidence threshold
            overlay_mode: Current display mode
            
        Returns:
            Formatted status message
        """
        detector_desc = self.get_detector_info(current_detector)
        classifier_status = "ON" if classifier_enabled else "OFF"
        
        status_lines = [
            f"Detector: {detector_desc}",
            f"Classifier: {classifier_status}",
            f"Confidence: {confidence_threshold:.2f}",
            f"Display: {overlay_mode.replace('_', ' ').title()}"
        ]
        
        return " | ".join(status_lines)
    
    def validate_confidence_threshold(self, threshold: float) -> float:
        """
        Validate and clamp confidence threshold
        
        Args:
            threshold: Threshold value to validate
            
        Returns:
            Clamped threshold value (0.0 to 1.0)
        """
        return max(0.0, min(1.0, threshold))
    
    def get_confidence_step(self) -> float:
        """Get step size for confidence threshold adjustments"""
        return 0.05  # 5% steps
    
    def format_key_name(self, key_code: int) -> str:
        """
        Format key code as human-readable name
        
        Args:
            key_code: Key code to format
            
        Returns:
            Human-readable key name
        """
        if key_code == 27:
            return "ESC"
        elif 32 <= key_code <= 126:  # Printable ASCII
            return chr(key_code)
        else:
            return f"Key({key_code})"
    
    def get_controls_summary(self) -> Dict[str, str]:
        """
        Get summary of all controls
        
        Returns:
            Dictionary mapping action names to key names
        """
        summary = {}
        for key_code, action in self.key_mappings.items():
            key_name = self.format_key_name(key_code)
            action_name = action.value.replace('_', ' ').title()
            
            if action_name in summary:
                summary[action_name] += f", {key_name}"
            else:
                summary[action_name] = key_name
        
        return summary
