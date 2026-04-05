import numpy as np
import tensorflow as tf
from dataclasses import dataclass
from utils.logger import Logger
from model_managers.cursor_gaze_v1 import GazeModelConfig

class GazeModelTFLite:
    """
    TensorFlow Lite model wrapper for Cursor Gaze.
    Offers significantly better performance on CPU/M1 compared to TF 1.x Session.
    """

    def __init__(self, config: GazeModelConfig, model_path_l="gaze_model_L.tflite", model_path_r="gaze_model_R.tflite"):
        self.cfg = config
        self.logger = Logger("GazeModelTFLite")
        
        self.logger.log(f"Loading TFLite models: {model_path_l}, {model_path_r}")
        
        # Load Interpreters
        self.interpreter_l = tf.lite.Interpreter(model_path=model_path_l)
        self.interpreter_l.allocate_tensors()
        
        self.interpreter_r = tf.lite.Interpreter(model_path=model_path_r)
        self.interpreter_r.allocate_tensors()
        
        # Get input/output details
        self.input_details_l = self.interpreter_l.get_input_details()
        self.output_details_l = self.interpreter_l.get_output_details()
        
        self.input_details_r = self.interpreter_r.get_input_details()
        self.output_details_r = self.interpreter_r.get_output_details()
        
        self.logger.log("TFLite models loaded successfully")

    def _set_inputs(self, interpreter, input_details, img, anchor_map, angle):
        """Helper to set inputs for an interpreter."""
        # We need to match inputs by their expected shape or name
        # In convert_to_tflite.py we passed [img, fp, ang] order
        # Let's verify by shape
        
        for input_detail in input_details:
            shape = input_detail['shape']
            # shape format: [batch, H, W, C] or [batch, 2]
            
            if len(shape) == 4:
                if shape[3] == 3: # Image inputs/Placeholder
                    # self.le_img shape: (None, 48, 64, 3)
                    input_data = np.expand_dims(img, axis=0).astype(np.float32)
                    interpreter.set_tensor(input_detail['index'], input_data)
                elif shape[3] == 12: # Anchor map inputs/Placeholder_1
                    # self.le_fp shape: (None, 48, 64, 12)
                    input_data = np.expand_dims(anchor_map, axis=0).astype(np.float32)
                    interpreter.set_tensor(input_detail['index'], input_data)
            elif len(shape) == 2:
                if shape[1] == 2: # Angle inputs/Placeholder_2
                    # self.le_ang shape: (None, 2)
                    input_data = np.expand_dims(angle, axis=0).astype(np.float32)
                    interpreter.set_tensor(input_detail['index'], input_data)

    def infer_eye(
        self, eye: str, img: np.ndarray, anchor_map: np.ndarray, angle: list
    ) -> np.ndarray:
        """
        Run inference for a single eye using TFLite.

        Args:
            eye: "L" or "R"
            img: Eye image normalized to [0, 1], shape (H, W, 3)
            anchor_map: Feature point map, shape (H, W, ef_dim)
            angle: [vertical, horizontal] correction angles

        Returns:
            Corrected eye image, shape (H, W, 3)
        """
        if eye == "L":
            interpreter = self.interpreter_l
            input_details = self.input_details_l
            output_details = self.output_details_l
        else:
            interpreter = self.interpreter_r
            input_details = self.input_details_r
            output_details = self.output_details_r
            
        # Set inputs
        self._set_inputs(interpreter, input_details, img, anchor_map, angle)
        
        # Run inference
        interpreter.invoke()
        
        # Get output
        # Assuming output[0] is the predicted image (which depends on outputs list in conversion)
        # We only exported [pred], so index 0 is correct.
        output_data = interpreter.get_tensor(output_details[0]['index'])
        
        # Output shape is [1, 48, 64, 3], reshape to [48, 64, 3]
        return output_data.reshape(self.cfg.eye_input_size[0], self.cfg.eye_input_size[1], 3)

    def close(self):
        """No explicit close needed for TFLite interpreter, but keeping api consistent."""
        pass
