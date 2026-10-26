import numpy as np
from tensorflow.keras.models import load_model

# FER2013 directory loaders commonly return classes in alphabetical order.
EMOTION_LABELS = [
    "Angry",
    "Disgust",
    "Fear",
    "Happy",
    "Neutral",
    "Sad",
    "Surprise",
]

class EmotionModel:
    def __init__(self, model_path: str):
        self.model = load_model(model_path, compile=False)

        input_shape = self.model.input_shape
        if isinstance(input_shape, list):
            raise ValueError("Expected a single-input emotion model")
        if len(input_shape) != 4 or input_shape[-1] != 1:
            raise ValueError(
                f"Expected a grayscale NHWC model input, got {input_shape}"
            )
        if input_shape[1] is None or input_shape[2] is None:
            raise ValueError(f"Model input dimensions must be fixed, got {input_shape}")

        # OpenCV resize takes (width, height); Keras input_shape is (height, width).
        self.input_size = (int(input_shape[2]), int(input_shape[1]))

        output_shape = self.model.output_shape
        if isinstance(output_shape, list) or output_shape[-1] != len(EMOTION_LABELS):
            raise ValueError(
                f"Expected {len(EMOTION_LABELS)} emotion outputs, got {output_shape}"
            )

    def predict(self, face_tensor):
        """
        face_tensor shape: (1, height, width, 1), matching the loaded model.
        """
        preds = self.model.predict(face_tensor, verbose=0)
        idx = int(np.argmax(preds))
        return EMOTION_LABELS[idx], float(preds[0][idx])
