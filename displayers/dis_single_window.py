"""
Cursor Gaze Application

A real-time eye gaze redirection application that:
- Uses a single window for live feedback
- Allows real-time toggle of Cursor Gaze with 'g' key
- Supports injectable face predictor backends
- Decoupled architecture: FacePredictor -> CursorGazeCorrector
- Calibration mode for camera offset adjustment
"""

import cv2
from dataclasses import dataclass
from typing import Optional
import Quartz

from utils.logger import Logger
from displayers.face_predictor import (
    FacePredictor,
    EyeExtractionConfig,
    create_face_predictor,
)
from model_managers.cursor_gaze_v1 import CursorGazeCorrector


################################################################################
# Configuration
################################################################################


@dataclass
class DisplayConfig:
    """Configuration for the display application."""

    video_size: tuple[int, int] = (640, 480)
    face_detect_size: tuple[int, int] = (320, 240)
    window_name: str = "Cursor Gaze"

    @property
    def x_ratio(self) -> float:
        return self.video_size[0] / self.face_detect_size[0]

    @property
    def y_ratio(self) -> float:
        return self.video_size[1] / self.face_detect_size[1]


@dataclass
class CalibrationConfig:
    """Configuration for calibration mode."""

    step_xy: float = 0.5      # cm per key press for X/Y
    step_z: float = 0.5       # cm per key press for Z
    step_focal: float = 10.0  # pixels per key press for focal length


################################################################################
# Single Window Cursor Gaze App
################################################################################


class CursorGazeApp:
    """
    Main Cursor Gaze application with real-time toggle.

    This class orchestrates:
    - FacePredictor: Detects faces and extracts eye data
    - CursorGazeCorrector: Applies gaze redirection model

    Controls:
        - 'g': Toggle Cursor Gaze on/off
        - 'c': Toggle calibration mode
        - 'q': Quit

    Calibration Mode Controls:
        - Arrow keys: Adjust X/Y offset
        - '+'/'-': Adjust Z offset
        - '['/']': Adjust focal length
        - 'r': Reset to default
    """

    # Arrow key codes (platform-dependent)
    # KEY_UP = 82
    # KEY_DOWN = 84
    # KEY_LEFT = 81
    # KEY_RIGHT = 83
    KEY_UP = 0
    KEY_DOWN = 1
    KEY_LEFT = 2
    KEY_RIGHT = 3

    def __init__(
        self,
        face_predictor: Optional[FacePredictor] = None,
        cursor_gaze: Optional[CursorGazeCorrector] = None,
        display_config: Optional[DisplayConfig] = None,
        calibration_config: Optional[CalibrationConfig] = None,
        camera_id: int = 0,
        config_path: str = "./model_managers/cursor_gaze_v1_01.yaml",
        use_tflite: bool = True,
        use_coreml: bool = True,
        use_virtual_cam: bool = False,
    ):
        self.logger = Logger(self.__class__.__name__)
        self.display_cfg = display_config or DisplayConfig()
        self.calib_cfg = calibration_config or CalibrationConfig()

        # Initialize face predictor (injectable)
        self.face_predictor = face_predictor or create_face_predictor("dlib")
        self.logger.log(f"Using face predictor: {self.face_predictor.get_name()}")

        # Initialize Cursor Gaze engine (injectable)
        self.cursor_gaze = cursor_gaze or CursorGazeCorrector(
            config_path=config_path, 
            use_tflite=use_tflite,
            use_coreml=use_coreml
        )

        # Eye extraction config (matches model requirements)
        self.eye_config = EyeExtractionConfig()

        # State
        self.cursor_gaze_enabled = True
        self.calibration_mode = False
        self.camera_id = camera_id

        # Store default values for reset
        self.default_camera_offset = self.cursor_gaze.get_camera_offset()
        self.default_focal_length = self.cursor_gaze.get_focal_length()
        
        # Get screen resolution for cursor tracking
        self.screen_res = self._get_screen_resolution()
        self.logger.log(f"Detected screen resolution: {self.screen_res}")

        # Initialize Virtual Camera if requested
        self.vcam = None
        if use_virtual_cam:
            try:
                import pyvirtualcam
                width, height = self.display_cfg.video_size
                self.vcam = pyvirtualcam.Camera(width=width, height=height, fps=30)
                self.logger.log(f"[OK] Virtual Camera started: {self.vcam.device}")
            except ImportError:
                self.logger.log("[ERROR] pyvirtualcam not installed. Virtual camera disabled.")
            except Exception as e:
                self.logger.log(f"[ERROR] Failed to start Virtual Camera: {e}")
                self.logger.log("  (Make sure OBS Virtual Camera is installed and ready)")

    def _get_screen_resolution(self) -> tuple[int, int]:
        """Get main display resolution."""
        rect = Quartz.CGDisplayBounds(Quartz.CGMainDisplayID())
        return int(rect.size.width), int(rect.size.height)

    def _get_cursor_position(self) -> tuple[int, int]:
        """Get current global cursor position."""
        event = Quartz.CGEventCreate(None)
        point = Quartz.CGEventGetLocation(event)
        return int(point.x), int(point.y)

    def draw_status(self, frame) -> None:
        """Draw status overlay on frame."""
        color = (0, 255, 0) if self.cursor_gaze_enabled else (0, 0, 255)
        
        cv2.rectangle(frame, (10, 10), (180, 60), (0, 0, 0), -1)
        cv2.putText(
            frame, f"CURSOR GAZE: {'ON' if self.cursor_gaze_enabled else 'OFF'}", (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA
        )
        cv2.putText(
            frame, f"[{self.face_predictor.get_name()}]", (20, 52),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1, cv2.LINE_AA
        )

        if self.vcam:
            cv2.putText(
                frame, "VCAM ON", (20, 68),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 255, 100), 1, cv2.LINE_AA
            )

    def draw_calibration_overlay(self, frame) -> None:
        """Draw calibration mode overlay with camera offset visualization."""
        h, w = frame.shape[:2]

        # Semi-transparent overlay background
        overlay = frame.copy()
        cv2.rectangle(overlay, (w - 260, 10), (w - 10, 245), (40, 40, 40), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        # Title
        cv2.putText(
            frame, "CALIBRATION MODE", (w - 250, 35),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA
        )

        # Current camera offset
        offset = self.cursor_gaze.get_camera_offset()
        cv2.putText(
            frame, f"Camera Offset:", (w - 250, 60),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA
        )
        cv2.putText(
            frame, f"  X: {offset[0]:+.1f} cm", (w - 250, 80),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 200, 255), 1, cv2.LINE_AA
        )
        cv2.putText(
            frame, f"  Y: {offset[1]:+.1f} cm", (w - 250, 100),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 255, 100), 1, cv2.LINE_AA
        )
        cv2.putText(
            frame, f"  Z: {offset[2]:+.1f} cm", (w - 250, 120),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 150, 100), 1, cv2.LINE_AA
        )

        # Eye position (estimated)
        eye_pos = self.cursor_gaze.get_last_eye_position()
        cv2.putText(
            frame, f"Eye Position:", (w - 250, 145),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA
        )
        cv2.putText(
            frame, f"  X: {eye_pos[0]:+.1f} cm", (w - 250, 165),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 255), 1, cv2.LINE_AA
        )
        cv2.putText(
            frame, f"  Y: {eye_pos[1]:+.1f} cm", (w - 250, 180),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 255, 150), 1, cv2.LINE_AA
        )
        cv2.putText(
            frame, f"  Z: {eye_pos[2]:+.1f} cm", (w - 250, 195),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 200, 150), 1, cv2.LINE_AA
        )

        # Focal length
        focal = self.cursor_gaze.get_focal_length()
        cv2.putText(
            frame, f"Focal Length: {focal:.0f} px", (w - 250, 215),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 200, 255), 1, cv2.LINE_AA
        )

        # Controls hint
        cv2.putText(
            frame, "[Arrows:XY] [+/-:Z] [[/]:F] [R:Reset]", (w - 250, 235),
            cv2.FONT_HERSHEY_SIMPLEX, 0.32, (180, 180, 180), 1, cv2.LINE_AA
        )

        # Draw camera position diagram (bottom left)
        self._draw_camera_diagram(frame, offset, eye_pos)

    def _draw_camera_diagram(
        self, frame, camera_offset: tuple, eye_pos: list
    ) -> None:
        """Draw a simple diagram showing camera and eye positions."""
        h, w = frame.shape[:2]

        # Diagram area
        diagram_x, diagram_y = 10, h - 160
        diagram_w, diagram_h = 150, 150

        # Background
        overlay = frame.copy()
        cv2.rectangle(
            overlay,
            (diagram_x, diagram_y),
            (diagram_x + diagram_w, diagram_y + diagram_h),
            (30, 30, 30), -1
        )
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        # Title
        cv2.putText(
            frame, "Top View (X-Z)", (diagram_x + 10, diagram_y + 15),
            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1, cv2.LINE_AA
        )

        # Center point (screen center, Z=0)
        center_x = diagram_x + diagram_w // 2
        center_y = diagram_y + diagram_h - 30

        # Scale: 1cm = 2 pixels
        scale = 2

        # Draw screen line
        cv2.line(
            frame,
            (diagram_x + 10, center_y),
            (diagram_x + diagram_w - 10, center_y),
            (100, 100, 100), 2
        )
        cv2.putText(
            frame, "Screen", (diagram_x + 50, center_y + 12),
            cv2.FONT_HERSHEY_SIMPLEX, 0.3, (100, 100, 100), 1, cv2.LINE_AA
        )

        # Draw camera position (offset from screen center)
        cam_px = center_x + int(camera_offset[0] * scale)
        cam_py = center_y + int(camera_offset[2] * scale)  # Z goes up in diagram
        cv2.circle(frame, (cam_px, cam_py), 6, (0, 255, 255), -1)
        cv2.putText(
            frame, "Cam", (cam_px - 12, cam_py - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 255), 1, cv2.LINE_AA
        )

        # Draw eye position (estimated)
        eye_px = center_x + int(eye_pos[0] * scale)
        eye_py = center_y + int(eye_pos[2] * scale)
        cv2.circle(frame, (eye_px, eye_py), 5, (255, 100, 100), -1)
        cv2.putText(
            frame, "Eye", (eye_px - 10, eye_py - 8),
            cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 100, 100), 1, cv2.LINE_AA
        )

        # Draw gaze line from eye to camera
        cv2.line(frame, (eye_px, eye_py), (cam_px, cam_py), (100, 255, 100), 1)

    def handle_calibration_key(self, key: int) -> bool:
        """
        Handle calibration mode key presses.

        Args:
            key: Key code

        Returns:
            True if key was handled
        """
        step_xy = self.calib_cfg.step_xy
        step_z = self.calib_cfg.step_z

        if key == self.KEY_LEFT:
            self.cursor_gaze.adjust_camera_offset(dx=-step_xy)
            return True
        elif key == self.KEY_RIGHT:
            self.cursor_gaze.adjust_camera_offset(dx=step_xy)
            return True
        elif key == self.KEY_UP:
            self.cursor_gaze.adjust_camera_offset(dy=-step_xy)
            return True
        elif key == self.KEY_DOWN:
            self.cursor_gaze.adjust_camera_offset(dy=step_xy)
            return True
        elif key == ord("+") or key == ord("="):
            self.cursor_gaze.adjust_camera_offset(dz=step_z)
            return True
        elif key == ord("-") or key == ord("_"):
            self.cursor_gaze.adjust_camera_offset(dz=-step_z)
            return True
        elif key == ord("["):
            self.cursor_gaze.adjust_focal_length(-self.calib_cfg.step_focal)
            return True
        elif key == ord("]"):
            self.cursor_gaze.adjust_focal_length(self.calib_cfg.step_focal)
            return True
        elif key == ord("r"):
            x, y, z = self.default_camera_offset
            self.cursor_gaze.set_camera_offset(x, y, z)
            self.cursor_gaze.set_focal_length(self.default_focal_length)
            self.logger.log("Camera offset and focal length reset to default")
            return True

        return False

    def process_frame(self, frame):
        """
        Process a single frame with Cursor Gaze.

        Args:
            frame: BGR video frame

        Returns:
            Processed frame with Cursor Gaze applied
        """
        display_frame = frame.copy()
        
        # Determine current window size for drawing
        # If we are only running virtual cam, we might not have a window
        # But this method assumes we return a frame to be displayed/sent.
        
        # Pass video size from config
        video_size = self.display_cfg.video_size

        # Get eye data for all detected faces
        face_data_list = self.face_predictor.list_eye_data(frame, self.eye_config)

        # Get cursor position for gaze targeting
        try:
            cursor_pos = self._get_cursor_position()
        except Exception as e:
            self.logger.log(f"Error getting cursor: {e}")
            cursor_pos = None

        # Process first detected face
        # TODO: Add logic to toggle cursor tracking vs camera tracking if needed
        # For now, always track cursor if available
        
        for face_data in face_data_list:
            try:
                # Apply Cursor Gaze (pass video_size and cursor info)
                display_frame = self.cursor_gaze.apply_correction(
                    display_frame,
                    face_data, 
                    video_size,
                    target_point=cursor_pos,
                    screen_size=self.screen_res
                )
            except Exception as e:
                self.logger.log(f"Error: {e}")
            break  # Only process first face

        return display_frame

    def run(self):
        """Main application loop."""
        self.logger.log(f"Starting camera {self.camera_id}...")
        cap = cv2.VideoCapture(self.camera_id)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.display_cfg.video_size[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.display_cfg.video_size[1])

        cv2.namedWindow(self.display_cfg.window_name)
        self.logger.log("Press 'g' to toggle gaze, 'c' for calibration, 'q' to quit")

        while True:
            ret, frame = cap.read()
            if not ret:
                self.logger.log("Failed to read frame")
                break

            if self.cursor_gaze_enabled:
                display_frame = self.process_frame(frame)
            else:
                display_frame = frame.copy()

            # Send to Virtual Camera (Clean frame, no overlay)
            if self.vcam:
                # Convert BGR to RGB
                try:
                    frame_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
                    self.vcam.send(frame_rgb)
                except Exception as e:
                     self.logger.log(f"VM Error: {e}")

            # Draw UI overlays on the windowed frame only
            self.draw_status(display_frame)

            # Draw calibration overlay if enabled
            if self.calibration_mode:
                self.draw_calibration_overlay(display_frame)

            cv2.imshow(self.display_cfg.window_name, display_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("g"):
                self.cursor_gaze_enabled = not self.cursor_gaze_enabled
                self.logger.log(
                    f"Cursor Gaze: {'enabled' if self.cursor_gaze_enabled else 'disabled'}"
                )
            elif key == ord("c"):
                self.calibration_mode = not self.calibration_mode
                self.logger.log(
                    f"Calibration mode: {'enabled' if self.calibration_mode else 'disabled'}"
                )
            elif self.calibration_mode:
                self.handle_calibration_key(key)

        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        self.cursor_gaze.close()
        if self.vcam:
            self.vcam.close()
        self.logger.log("Shutdown complete")
