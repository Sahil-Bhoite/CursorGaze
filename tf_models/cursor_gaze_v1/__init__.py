"""
TensorFlow Cursor Gaze V1 Package

This package provides neural network components for real-time gaze redirection
in video, based on the DeepWarp approach.

Modules:
    - gaze_warp_model: Main Cursor Gaze model architecture
    - layers: Reusable neural network layer building blocks
    - spatial_transform: Differentiable image warping operations
"""

from tf_models.cursor_gaze_v1 import gaze_warp_model
from tf_models.cursor_gaze_v1 import layers
from tf_models.cursor_gaze_v1 import spatial_transform

# Model configuration
from tf_models.cursor_gaze_v1.gaze_warp_model import ModelConfig

# Key functions for inference
from tf_models.cursor_gaze_v1.gaze_warp_model import build_inference_graph

__all__ = [
    "gaze_warp_model",
    "layers", 
    "spatial_transform",
    "ModelConfig",
    "build_inference_graph",
]
