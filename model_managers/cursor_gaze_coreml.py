import numpy as np
import coremltools as ct
from utils.logger import Logger
from model_managers.cursor_gaze_v1 import GazeModelConfig

class GazeModelCoreML:
    """
    CoreML model wrapper for Cursor Gaze.
    Uses Apple Neural Engine (ANE) on M1/M2/M3 chips for maximum performance.
    """

    def __init__(self, config: GazeModelConfig, model_path_l="GazeModelL.mlpackage", model_path_r="GazeModelR.mlpackage"):
        self.cfg = config
        self.logger = Logger("GazeModelCoreML")
        
        self.logger.log(f"Loading CoreML models: {model_path_l}, {model_path_r}")
        
        try:
            # Load models using the compute units that include ANE
            # compute_units=ct.ComputeUnit.ALL is default
            self.model_l = ct.models.MLModel(model_path_l)
            self.model_r = ct.models.MLModel(model_path_r)
            self.logger.log("CoreML models loaded successfully")
        except Exception as e:
            self.logger.log(f"Failed to load CoreML models: {e}")
            raise e

    def infer_eye(
        self, eye: str, img: np.ndarray, anchor_map: np.ndarray, angle: list
    ) -> np.ndarray:
        """
        Run inference for a single eye using CoreML.

        Args:
            eye: "L" or "R"
            img: Eye image normalized to [0, 1], shape (H, W, 3)
            anchor_map: Feature point map, shape (H, W, ef_dim)
            angle: [vertical, horizontal] correction angles

        Returns:
            Corrected eye image, shape (H, W, 3)
        """
        model = self.model_l if eye == "L" else self.model_r
            
        # Prepare inputs dictionary
        # CoreML expects specific input names as found in inspection
        inputs = {
            "inputs_placeholder": np.expand_dims(img, axis=0).astype(np.float32),
            "inputs_placeholder_1": np.expand_dims(anchor_map, axis=0).astype(np.float32),
            "inputs_placeholder_2": np.expand_dims(angle, axis=0).astype(np.float32)
        }
        
        # Run prediction
        # The output dictionary keys depend on the model. We saw "Identity" in inspection.
        prediction = model.predict(inputs)
        
        # Get output
        # Verify output name or just get the first value if name is unknown/variable
        if "Identity" in prediction:
            output_data = prediction["Identity"]
        else:
            # Fallback: get first value
            output_data = next(iter(prediction.values()))
            
        # Output shape is [1, 48, 64, 3], reshape to [48, 64, 3]
        return output_data.reshape(self.cfg.eye_input_size[0], self.cfg.eye_input_size[1], 3)

    def close(self):
        """No explicit close needed for CoreML models."""
        pass
