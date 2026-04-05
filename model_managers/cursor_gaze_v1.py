"""
Cursor Gaze Engine V1

This module provides the Cursor Gaze core engine, which includes the 
deep learning model wrapper and gaze redirection logic.
"""

import math
import yaml
import numpy as np
import tensorflow as tf
import cv2
from dataclasses import dataclass

from tf_models.cursor_gaze_v1 import gaze_warp_model
from utils.logger import Logger
from model_managers.user_settings_db import UserSettingsDB


################################################################################
# Configuration Classes
################################################################################


@dataclass
class GazeWarpModelConfig:
    """Hyperparameters for the gaze warp model."""
    height: int = 48
    width: int = 64
    encoded_angle_dim: int = 16


@dataclass
class GazeModelConfig:
    """Configuration for the Cursor Gaze model."""
    
    model_dir: str = "./weights/cursor_gaze_v1/flx/12/"
    eye_input_size: tuple[int, int] = (48, 64)  # (height, width)
    ef_dim: int = 12
    channel: int = 3
    gaze_warp_model: GazeWarpModelConfig = None
    
    def __post_init__(self):
        if self.gaze_warp_model is None:
            self.gaze_warp_model = GazeWarpModelConfig()
        elif isinstance(self.gaze_warp_model, dict):
            self.gaze_warp_model = GazeWarpModelConfig(**self.gaze_warp_model)
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> "GazeModelConfig":
        """Load configuration from YAML file."""
        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)
        
        # Convert eye_input_size from list to tuple
        if 'eye_input_size' in data and isinstance(data['eye_input_size'], list):
            data['eye_input_size'] = tuple(data['eye_input_size'])
        
        return cls(**data)


@dataclass
class CameraUserSetting:
    """User-adjustable camera and screen geometry settings."""
    
    focal_length: float = 650.0
    ipd: float = 6.3  # Inter-pupillary distance in cm
    camera_offset: tuple[float, float, float] = (0, -21, -1)  # relative to screen center
    
    def to_dict(self) -> dict:
        """Convert to dictionary for database storage."""
        return {
            'focal_length': self.focal_length,
            'ipd': self.ipd,
            'camera_offset': list(self.camera_offset),
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "CameraUserSetting":
        """Load from dictionary."""
        if 'camera_offset' in data and isinstance(data['camera_offset'], list):
            data['camera_offset'] = tuple(data['camera_offset'])
        return cls(**data)


################################################################################
# Cursor Gaze Model
################################################################################


class GazeModel:
    """TensorFlow model wrapper for eye gaze redirection in Cursor Gaze."""

    def __init__(self, config: GazeModelConfig):
        self.cfg = config
        self.logger = Logger("GazeModel")
        self._load_models()

    def _load_models(self):
        """Load left and right eye models."""
        # Build ModelConfig for gaze_warp_model
        model_cfg = gaze_warp_model.ModelConfig(
            height=self.cfg.gaze_warp_model.height,
            width=self.cfg.gaze_warp_model.width,
            encoded_angle_dim=self.cfg.gaze_warp_model.encoded_angle_dim,
        )

        # Left eye model
        self.logger.log("Loading left eye model...")
        with tf.Graph().as_default() as g_left:
            with tf.name_scope("inputs"):
                self.le_img = tf.compat.v1.placeholder(
                    tf.float32,
                    [None, self.cfg.eye_input_size[0], self.cfg.eye_input_size[1], self.cfg.channel],
                )
                self.le_fp = tf.compat.v1.placeholder(
                    tf.float32,
                    [None, self.cfg.eye_input_size[0], self.cfg.eye_input_size[1], self.cfg.ef_dim],
                )
                self.le_ang = tf.compat.v1.placeholder(tf.float32, [None, 2])

            self.le_pred, _, _ = gaze_warp_model.build_inference_graph(
                self.le_img, self.le_fp, self.le_ang, False, model_cfg
            )
            self.l_sess = tf.compat.v1.Session(
                config=tf.compat.v1.ConfigProto(allow_soft_placement=True),
                graph=g_left,
            )
            self._restore_checkpoint(self.l_sess, self.cfg.model_dir + "L/")

        # Right eye model
        self.logger.log("Loading right eye model...")
        with tf.Graph().as_default() as g_right:
            with tf.name_scope("inputs"):
                self.re_img = tf.compat.v1.placeholder(
                    tf.float32,
                    [None, self.cfg.eye_input_size[0], self.cfg.eye_input_size[1], self.cfg.channel],
                )
                self.re_fp = tf.compat.v1.placeholder(
                    tf.float32,
                    [None, self.cfg.eye_input_size[0], self.cfg.eye_input_size[1], self.cfg.ef_dim],
                )
                self.re_ang = tf.compat.v1.placeholder(tf.float32, [None, 2])

            self.re_pred, _, _ = gaze_warp_model.build_inference_graph(
                self.re_img, self.re_fp, self.re_ang, False, model_cfg
            )
            self.r_sess = tf.compat.v1.Session(
                config=tf.compat.v1.ConfigProto(allow_soft_placement=True),
                graph=g_right,
            )
            self._restore_checkpoint(self.r_sess, self.cfg.model_dir + "R/")

        self.logger.log("Models loaded successfully")

    def _restore_checkpoint(self, sess, model_dir: str):
        """Restore model from checkpoint."""
        saver = tf.compat.v1.train.Saver(tf.compat.v1.global_variables())
        ckpt = tf.compat.v1.train.get_checkpoint_state(model_dir)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            self.logger.log(f"Warning: No checkpoint found in {model_dir}")

    def infer_eye(
        self, eye: str, img: np.ndarray, anchor_map: np.ndarray, angle: list
    ) -> np.ndarray:
        """
        Run inference for a single eye.

        Args:
            eye: "L" or "R"
            img: Eye image normalized to [0, 1], shape (H, W, 3)
            anchor_map: Feature point map, shape (H, W, ef_dim)
            angle: [vertical, horizontal] correction angles

        Returns:
            Corrected eye image, shape (H, W, 3)
        """
        if eye == "L":
            result = self.l_sess.run(
                self.le_pred,
                feed_dict={
                    self.le_img: np.expand_dims(img, axis=0),
                    self.le_fp: np.expand_dims(anchor_map, axis=0),
                    self.le_ang: np.expand_dims(angle, axis=0),
                },
            )
        else:
            result = self.r_sess.run(
                self.re_pred,
                feed_dict={
                    self.re_img: np.expand_dims(img, axis=0),
                    self.re_fp: np.expand_dims(anchor_map, axis=0),
                    self.re_ang: np.expand_dims(angle, axis=0),
                },
            )
        return result.reshape(self.cfg.eye_input_size[0], self.cfg.eye_input_size[1], 3)

    def close(self):
        """Close TensorFlow sessions."""
        self.l_sess.close()
        self.r_sess.close()


from model_managers.cursor_gaze_tflite import GazeModelTFLite
try:
    from model_managers.cursor_gaze_coreml import GazeModelCoreML
    COREML_AVAILABLE = True
except ImportError:
    COREML_AVAILABLE = False

class CursorGazeCorrector:
    """
    High-level Cursor Gaze interface with database-backed user settings.
    
    Takes FaceData from a FacePredictor and applies eye gaze redirection.
    video_size is passed from outside via apply_correction.
    """

    def __init__(
        self,
        config_path: str = "./model_managers/cursor_gaze_v1_01.yaml",
        db_path: str = "./user_settings.db",
        setting_name: str = "camera_default",
        use_tflite: bool = True,
        use_coreml: bool = True,
    ):
        """
        Initialize Cursor Gaze engine.
        
        Args:
            config_path: Path to YAML configuration file
            db_path: Path to SQLite database
            setting_name: Name of camera setting to load from database
            use_tflite: Whether to use TFLite model (default: True)
            use_coreml: Whether to use CoreML model (default: True, prioritizes over TFLite)
        """
        self.logger = Logger("CursorGazeCorrector")
        
        # Load model configuration from YAML
        self.model_cfg = GazeModelConfig.from_yaml(config_path)
        self.logger.log(f"Loaded model config from: {config_path}")
        
        # Initialize database
        self.db = UserSettingsDB(db_path)
        self.setting_name = setting_name
        
        # Load camera settings from database or use defaults
        self.camera_settings = self._load_camera_settings()
        
        # Initialize model (Priority: CoreML > TFLite > TF1)
        self.model = None
        
        if use_coreml and COREML_AVAILABLE:
            self.logger.log("Attempting to load CoreML Inference Engine...")
            try:
                self.model = GazeModelCoreML(self.model_cfg)
                self.logger.log("[OK] Using CoreML Backend")
            except Exception as e:
                self.logger.log(f"Failed to load CoreML model: {e}")
        
        if self.model is None and use_tflite:
            self.logger.log("Attempting to load TFLite Inference Engine...")
            try:
                self.model = GazeModelTFLite(self.model_cfg)
                self.logger.log("[OK] Using TFLite Backend")
            except Exception as e:
                self.logger.log(f"Failed to load TFLite model: {e}")

        if self.model is None:
            self.logger.log("[WARN] Falling back to TF1.x Graph Backend")
            self.model = GazeModel(self.model_cfg)

        # Pixel border to cut when replacing eyes (reduces edge artifacts)
        self.pixel_cut = (3, 4)
        
        # Smoothing state
        self.last_angles = None

        # Last estimated eye position (for visualization)
        self.last_eye_position: list[float] = [0, 0, -60]

    def _load_camera_settings(self) -> CameraUserSetting:
        """Load camera settings from database or return defaults."""
        saved = self.db.get_setting(self.setting_name)
        if saved:
            self.logger.log(f"Loaded camera settings from database: {self.setting_name}")
            return CameraUserSetting.from_dict(saved)
        else:
            self.logger.log("Using default camera settings")
            settings = CameraUserSetting()
            # Save defaults to database
            self.db.save_setting(self.setting_name, settings.to_dict())
            return settings

    def save_camera_settings(self):
        """Save current camera settings to database."""
        self.db.save_setting(self.setting_name, self.camera_settings.to_dict())
        self.logger.log(f"Saved camera settings to database: {self.setting_name}")

    ############################################################################
    # Camera Offset Adjustment API
    ############################################################################

    def get_camera_offset(self) -> tuple[float, float, float]:
        """Get current camera offset (x, y, z) in cm."""
        return self.camera_settings.camera_offset

    def set_camera_offset(self, x: float, y: float, z: float) -> None:
        """
        Set camera offset relative to screen center.

        Args:
            x: Horizontal offset in cm (positive = right)
            y: Vertical offset in cm (positive = down)
            z: Depth offset in cm (negative = behind screen)
        """
        self.camera_settings.camera_offset = (x, y, z)
        self.save_camera_settings()
        self.logger.log(f"Camera offset set to: ({x:.1f}, {y:.1f}, {z:.1f})")

    def adjust_camera_offset(self, dx: float = 0, dy: float = 0, dz: float = 0) -> tuple[float, float, float]:
        """
        Adjust camera offset by delta values.

        Args:
            dx: Change in X (horizontal)
            dy: Change in Y (vertical)
            dz: Change in Z (depth)

        Returns:
            New camera offset tuple
        """
        x, y, z = self.camera_settings.camera_offset
        self.camera_settings.camera_offset = (x + dx, y + dy, z + dz)
        self.save_camera_settings()
        return self.camera_settings.camera_offset

    def get_last_eye_position(self) -> list[float]:
        """Get the last estimated eye position [x, y, z] in cm."""
        return self.last_eye_position

    ############################################################################
    # Focal Length Adjustment API
    ############################################################################

    def get_focal_length(self) -> float:
        """Get current focal length in pixels."""
        return self.camera_settings.focal_length

    def set_focal_length(self, focal_length: float) -> None:
        """
        Set focal length.

        Args:
            focal_length: Focal length in pixels (typically 500-1000)
        """
        self.camera_settings.focal_length = focal_length
        self.save_camera_settings()
        self.logger.log(f"Focal length set to: {focal_length:.1f}")

    def adjust_focal_length(self, delta: float) -> float:
        """
        Adjust focal length by delta value.

        Args:
            delta: Change in focal length (pixels)

        Returns:
            New focal length
        """
        self.camera_settings.focal_length += delta
        self.save_camera_settings()
        return self.camera_settings.focal_length

    ############################################################################
    # IPD Adjustment API
    ############################################################################

    def get_ipd(self) -> float:
        """Get inter-pupillary distance in cm."""
        return self.camera_settings.ipd

    def set_ipd(self, ipd: float) -> None:
        """
        Set inter-pupillary distance.

        Args:
            ipd: IPD in cm (typically 5.5-7.0)
        """
        self.camera_settings.ipd = ipd
        self.save_camera_settings()
        self.logger.log(f"IPD set to: {ipd:.1f} cm")

    ############################################################################
    # Gaze Estimation and Correction
    ############################################################################

    def estimate_gaze_angle(
        self, 
        le_center: tuple[float, float], 
        re_center: tuple[float, float],
        video_size: tuple[int, int],
        target_point: tuple[int, int] = None,
        screen_size: tuple[int, int] = None,
    ) -> tuple[list[int], list[float]]:
        """
        Estimate gaze redirection angles based on eye positions.

        Args:
            le_center: Left eye center (x, y) in pixels
            re_center: Right eye center (x, y) in pixels
            video_size: (width, height) of video frame
            target_point: Optional (x, y) screen pixels to look at
            screen_size: Optional (width, height) screen resolution

        Returns:
            (alpha [vertical, horizontal], eye_position [x, y, z])
        """
        settings = self.camera_settings

        # Estimate eye depth from inter-pupillary distance
        ipd_pixels = np.sqrt(
            (le_center[0] - re_center[0]) ** 2 + (le_center[1] - re_center[1]) ** 2
        )
        eye_z = -(settings.focal_length * settings.ipd) / ipd_pixels

        # Estimate eye position in 3D (camera coordinates, cm)
        eye_x = (
            -abs(eye_z)
            * (le_center[0] + re_center[0] - video_size[0])
            / (2 * settings.focal_length)
            + settings.camera_offset[0]
        )
        eye_y = (
            abs(eye_z)
            * (le_center[1] + re_center[1] - video_size[1])
            / (2 * settings.focal_length)
            + settings.camera_offset[1]
        )

        eye_position = [eye_x, eye_y, eye_z]

        # Store for visualization
        self.last_eye_position = eye_position

        # Target gaze point relative to camera
        if target_point is not None and screen_size is not None:
            # Map screen pixel to physical coordinates (cm) relative to Camera
            sx, sy = screen_size
            tx, ty = target_point
            
            # Screen physical width assumption (approx 13" laptop)
            SCREEN_WIDTH_CM = 30.0
            screen_height_cm = SCREEN_WIDTH_CM * (sy / sx)
            
            # Normalized coordinates (-0.5 to 0.5) from center
            # Note: Screen Y is 0 at top, increasing downwards.
            # We want +Y to be UP in 3D space, so we invert Y relative to center.
            # Center is 0.5. Top (0) should be +0.5 (UP). Bottom (1) should be -0.5 (DOWN).
            # Formally: (0.5 - normalized_y)
            nx = (tx / sx) - 0.5
            ny = 0.5 - (ty / sy) 
            
            # Target in Screen Space (Origin at Screen Center)
            t_x_screen = nx * SCREEN_WIDTH_CM
            t_y_screen = ny * screen_height_cm
            t_z_screen = 0
            
            # Transform to Camera Space (Camera is at settings.camera_offset relative to Screen Center)
            # Vector(Cam -> Target) = Vector(Cam -> ScreenCenter) + Vector(ScreenCenter -> Target)
            # Vector(Cam -> ScreenCenter) = -settings.camera_offset
            target = (
                t_x_screen - settings.camera_offset[0],
                t_y_screen - settings.camera_offset[1],
                t_z_screen - settings.camera_offset[2]
            )
        else:
            # Default: Look at camera (Origin)
            target = (0, 0, 0)

        # Calculate angles
        a_v = math.degrees(math.atan((target[1] - eye_y) / (target[2] - eye_z)))
        a_h = math.degrees(math.atan((target[0] - eye_x) / (target[2] - eye_z)))

        # Add camera offset compensation angles if NOT tracking cursor
        # (If tracking cursor, the geometry above handles it, but the model trained to look at camera 
        # might expect angles relative to camera axis. The formula below was adding offset compensation
        # for a fixed camera target. If we have explicit target, do we need this?)
        
        # The original code:
        # a_v = atan( (0 - eye_y) / (0 - eye_z) )  <-- vector from eye to origin
        # plus compensation: atan( (eye_y - offset_y) / (offset_z - eye_z) )
        # (eye_y - offset_y) is eye position relative to SCREEN CENTER (Screen Space Y).
        # (offset_z - eye_z) is Screen Center Z - Eye Z.
        
        # It seems the model takes inputs relative to "Screen Normal"? 
        # Or maybe the "compensation" was essentially doing what I did above but in angles?
        # Let's trust my geometric vector calculation above (Vector Eye -> Target).
        # But wait, does the model expect "Gaze Angle relative to Camera Axis" or "Gaze Angle relative to Head"?
        # Usually DeepWarp models take (Pitch, Yaw) relative to camera axis.
        # My calculation `atan(dy/dz)` gives angle relative to Z axis.
        # So it should be correct.
        
        # However, check the original compensation logic:
        # It adds an angle.
        # `atan((eye_y - offset_y) / ...)` is angle of the Eye -> ScreenCenter vector??
        # If the original code works for looking at camera, it calculates:
        # Angle(Eye->Cam) + Angle(???).
        
        # Let's assume my new geometric calculation `target - eye` is sufficient and encompasses the full geometry.
        # So I will ONLY use my calculated a_v, a_h. 
        # BUT, if target is (0,0,0), my code gives `atan(-eye_y / -eye_z)`.
        # Original code gave `atan(-eye_y/-eye_z) + compensation`.
        # Why? Maybe the model was trained with a bias?
        # Or maybe `eye_position` calc is weird.
        
        # Let's preserve original logic for (0,0,0) case to be safe, 
        # and use my logic for cursor case.
        
        if target_point is None:
             a_v += math.degrees(
                math.atan((eye_y - settings.camera_offset[1]) / (settings.camera_offset[2] - eye_z))
            )
             a_h += math.degrees(
                math.atan((eye_x - settings.camera_offset[0]) / (settings.camera_offset[2] - eye_z))
            )

        # Apply Smoothing (EMA) if we have history
        if hasattr(self, 'last_angles') and self.last_angles is not None:
             # Alpha = smoothing factor (0.0 - 1.0). Lower = smoother but more lag.
             # 0.12 provides stronger smoothing to fix "wobbly" pupil look.
             alpha_smooth = 0.12
             
             prev_v, prev_h = self.last_angles
             a_v = alpha_smooth * a_v + (1 - alpha_smooth) * prev_v
             a_h = alpha_smooth * a_h + (1 - alpha_smooth) * prev_h
        
        # Store for next frame
        self.last_angles = (a_v, a_h)

        return [int(a_v), int(a_h)], eye_position

    def correct_eye(
        self, eye_data, eye_side: str, angle: list[int]
    ) -> np.ndarray:
        """
        Apply Cursor Gaze to a single eye.

        Args:
            eye_data: Eye extraction data (EyeData from face_predictor)
            eye_side: "L" or "R"
            angle: [vertical, horizontal] correction angles

        Returns:
            Corrected eye image resized to original size
        """
        result = self.model.infer_eye(
            eye_side, eye_data.image, eye_data.anchor_map, angle
        )
        # Resize back to original size using Bicubic interpolation for better sharpness
        return cv2.resize(
            result, 
            (eye_data.original_size[1], eye_data.original_size[0]),
            interpolation=cv2.INTER_CUBIC
        )

    def _match_histograms(self, source, template):
        """
        Match the histogram of the source image to the template.
        Helps in blending the synthetic eye with the original lighting.
        """
        # Simple mean/std adjustment for speed and stability
        # Convert to LAB for better color preservation
        src_lab = cv2.cvtColor(source, cv2.COLOR_BGR2LAB).astype(np.float32)
        tpl_lab = cv2.cvtColor(template, cv2.COLOR_BGR2LAB).astype(np.float32)
        
        corrected = src_lab.copy()
        for i in range(3):
            src_mean, src_std = np.mean(src_lab[..., i]), np.std(src_lab[..., i])
            tpl_mean, tpl_std = np.mean(tpl_lab[..., i]), np.std(tpl_lab[..., i])
            
            # Avoid division by zero
            if src_std < 1e-6: src_std = 1.0
            
            # Transfer statistics: (x - mu_src) * (std_tpl / std_src) + mu_tpl
            corrected[..., i] = (src_lab[..., i] - src_mean) * (tpl_std / src_std) + tpl_mean
            
        corrected = np.clip(corrected, 0, 255).astype(np.uint8)
        return cv2.cvtColor(corrected, cv2.COLOR_LAB2BGR)

    def _add_noise(self, image, strength=0.05):
        """Add Gaussian noise to mimic camera grain."""
        row, col, ch = image.shape
        mean = 0
        # Scale sigma based on pixel intensity range (0-255)
        sigma = strength * 255
        gauss = np.random.normal(mean, sigma, (row, col, ch))
        gauss = gauss.reshape(row, col, ch)
        noisy = image + gauss
        return np.clip(noisy, 0, 255).astype(np.uint8)

    def _blend_eye(self, frame, eye_data, corrected_eye):
        """
        Blend the corrected eye into the frame with lighting adjustment and noise.
        """
        pc = self.pixel_cut
        
        # Define the region to replace (removing the border pixels)
        h, w = eye_data.original_size
        cw = w - 2 * pc[1]
        ch = h - 2 * pc[0]
        
        if cw <= 0 or ch <= 0:
            return

        # Coordinates in the full frame
        y_start = eye_data.top_left[0] + pc[0]
        x_start = eye_data.top_left[1] + pc[1]
        y_end = y_start + ch
        x_end = x_start + cw
        
        # Ensure we are within frame bounds
        if y_start < 0 or x_start < 0 or y_end > frame.shape[0] or x_end > frame.shape[1]:
            return

        # Destination ROI (Original Eye)
        dest_roi = frame[y_start:y_end, x_start:x_end]

        # Source image (corrected eye crop)
        src_raw = corrected_eye[pc[0] : -pc[0], pc[1] : -pc[1]] * 255
        src_raw = np.clip(src_raw, 0, 255).astype(np.uint8)
        
        # 1. Color/Lighting Matching (Histogram Transfer)
        src = self._match_histograms(src_raw, dest_roi)

        # 2. Gentle Gamma (Highlight Dampening)
        # Prevents glowing whites without crushing darks
        src_lab = cv2.cvtColor(src, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(src_lab)
        l = (l.astype(np.float32) / 255.0) ** 1.15 * 255.0
        src_lab = cv2.merge([l.astype(np.uint8), a, b])
        src = cv2.cvtColor(src_lab, cv2.COLOR_LAB2BGR)

        # 3. Add Subtle Camera Noise
        # Reduced strength to avoid "deep fried" look
        src = self._add_noise(src, strength=0.02)

        # 4. Minimal Sharpening
        # Just enough to define the iris, not the noise
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        sharpened = cv2.filter2D(src, -1, kernel)
        src = cv2.addWeighted(src, 0.90, sharpened, 0.10, 0)
        
        # Create a Feathered Mask
        # We want the center to be opaque (1.0) and edges to fade to transparent (0.0)
        mask = np.zeros((ch, cw), dtype=np.float32)
        
        # Define a center rectangle that is fully opaque
        # Fade out over 'fade_pixels'
        fade = 6
        cv2.rectangle(
            mask, 
            (fade, fade), 
            (cw - fade, ch - fade), 
            1.0, 
            -1
        )
        # Blur the mask to create the gradient
        mask = cv2.GaussianBlur(mask, (2*fade + 1, 2*fade + 1), 0)
        
        # Expand mask to 3 channels
        mask = np.stack([mask] * 3, axis=2)
        
        # Alpha Blend: src * alpha + dest * (1 - alpha)
        blended = (src * mask + dest_roi * (1.0 - mask)).astype(np.uint8)
        
        # Assign back
        frame[y_start:y_end, x_start:x_end] = blended

    def apply_correction(
        self, 
        frame: np.ndarray, 
        face_data, 
        video_size: tuple[int, int],
        target_point: tuple[int, int] = None,
        screen_size: tuple[int, int] = None,
    ) -> np.ndarray:
        """
        Apply Cursor Gaze redirection to a frame using extracted face data.

        Args:
            frame: BGR video frame to modify
            face_data: Extracted face/eye data from FacePredictor (FaceData)
            video_size: (width, height) of video frame
            target_point: Optional (x, y) screen pixels to look at
            screen_size: Optional (width, height) screen resolution

        Returns:
            Frame with corrected gaze
        """
        if face_data.left_eye is None or face_data.right_eye is None:
            return frame

        le = face_data.left_eye
        re = face_data.right_eye

        # Estimate gaze angle (video_size passed from outside)
        alpha, _ = self.estimate_gaze_angle(le.center, re.center, video_size, target_point, screen_size)

        # Correct both eyes
        le_corrected = self.correct_eye(le, "L", alpha)
        re_corrected = self.correct_eye(re, "R", alpha)

        # Blend corrected eyes into frame
        self._blend_eye(frame, le, le_corrected)
        self._blend_eye(frame, re, re_corrected)

        return frame

    def close(self):
        """Release model resources."""
        self.model.close()
