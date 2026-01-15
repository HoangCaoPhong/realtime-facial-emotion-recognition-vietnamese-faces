from models.emotion_model import EmotionModel
from app.webcam_app import WebcamApp

def main():
    model = EmotionModel("model/emotion_model.h5")
    app = WebcamApp(model)
    app.run()

if __name__ == "__main__":
    main()
