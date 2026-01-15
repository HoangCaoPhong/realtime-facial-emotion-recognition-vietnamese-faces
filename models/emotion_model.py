import numpy as np
from tensorflow.keras.models import load_model

INPUT_SIZE = (64, 64)
EMOTION_LABELS = [
    "Angry",
    "Disgust",
    "Fear",
    "Happy",
    "Sad",
    "Surprise",
    "Neutral"
]

class EmotionModel:
    def __init__(self, model_path: str):
        self.model = load_model(model_path, compile=False)

    def predict(self, face_tensor):
        """
        face_tensor shape: (1, 64, 64, 1)
        """
        preds = self.model.predict(face_tensor, verbose=0)
        idx = int(np.argmax(preds))
        return EMOTION_LABELS[idx], float(preds[0][idx])
