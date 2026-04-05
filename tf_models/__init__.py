"""
TensorFlow Models Package for Cursor Gaze

This package provides versioned models for the Cursor Gaze system.

Versions:
    - cursor_gaze_v1: Initial DeepWarp-based Cursor Gaze model
"""

# Import versioned models
from tf_models import cursor_gaze_v1

# Backward compatibility: expose v1 as default
from tf_models.cursor_gaze_v1 import gaze_warp_model
from tf_models.cursor_gaze_v1 import layers
from tf_models.cursor_gaze_v1 import spatial_transform
from tf_models.cursor_gaze_v1 import ModelConfig
from tf_models.cursor_gaze_v1 import build_inference_graph

# Additional backward compatibility aliases
from tf_models.cursor_gaze_v1 import gaze_warp_model as flx
from tf_models.cursor_gaze_v1 import layers as tf_utils
from tf_models.cursor_gaze_v1 import spatial_transform as transformation

__all__ = [
    "cursor_gaze_v1",
    "gaze_warp_model",
    "layers", 
    "spatial_transform",
    "ModelConfig",
    "build_inference_graph",
    # Backward compatibility
    "flx",
    "tf_utils",
    "transformation",
]
