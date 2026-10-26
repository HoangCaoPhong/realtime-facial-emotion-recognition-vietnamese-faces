import cv2
import numpy as np

class WebcamApp:
    def __init__(self, emotion_model):
        self.emotion_model = emotion_model
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        if self.face_cascade.empty():
            raise RuntimeError("OpenCV could not load the Haar face cascade")

        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            self.cap.release()
            raise RuntimeError("Could not open webcam 0; check camera access or index")

    def preprocess_face(self, gray_frame, x, y, w, h):
        face = gray_frame[y:y+h, x:x+w]
        face = cv2.resize(face, self.emotion_model.input_size)
        face = face.astype("float32") / 255.0
        height, width = self.emotion_model.input_size[1], self.emotion_model.input_size[0]
        face = np.reshape(face, (1, height, width, 1))
        return face

    def run(self):
        try:
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
        finally:
            self.cap.release()
            cv2.destroyAllWindows()
