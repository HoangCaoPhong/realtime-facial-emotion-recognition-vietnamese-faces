import cv2
import numpy as np
from models.emotion_model import INPUT_SIZE

class WebcamApp:
    def __init__(self, emotion_model):
        self.emotion_model = emotion_model
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        self.cap = cv2.VideoCapture(0)

    def preprocess_face(self, gray_frame, x, y, w, h):
        face = gray_frame[y:y+h, x:x+w]
        face = cv2.resize(face, INPUT_SIZE)
        face = face.astype("float32") / 255.0
        face = np.reshape(face, (1, INPUT_SIZE[0], INPUT_SIZE[1], 1))
        return face

    def run(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.3,
                minNeighbors=5
            )

            for (x, y, w, h) in faces:
                face_tensor = self.preprocess_face(gray, x, y, w, h)
                emotion, conf = self.emotion_model.predict(face_tensor)

                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(
                    frame,
                    f"{emotion} ({conf:.2f})",
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2
                )

            cv2.imshow(
                "Realtime Facial Emotion Recognition (Vietnamese Faces)",
                frame
            )

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        self.cap.release()
        cv2.destroyAllWindows()
