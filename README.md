Model file is not included due to size. Please download separately.

## Run the webcam demo

Place a compatible Keras checkpoint at `model/emotion_model.h5`, then install
the project dependencies and start the demo:

```bash
python -m pip install -r requirements.txt
python -m demo.run_webcam
```

The webcam preprocessing reads the model's fixed grayscale input dimensions
from the checkpoint. For FER2013 models trained with the usual alphabetical
class-folder order, the output labels are Angry, Disgust, Fear, Happy, Neutral,
Sad, Surprise. Keep the label order in `models/emotion_model.py` aligned with
the checkpoint's training class mapping.
