import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import base64
import numpy as np
import threading
from io import BytesIO
from PIL import Image
from keras.models import load_model

_model = None
_model_lock = threading.Lock()
ANSWERS = ['bag', 'bed', 'bucket', 'chair', 'clock', 'curtain', 'hat']

def get_model():
    global _model
    with _model_lock:
        if _model is None:
            print("[INFO] Loading CAPTCHA model...")
            _model = load_model('./models/1.keras')
            print("[INFO] Model loaded successfully!")
    return _model

def preprocess_image(image_b64: str) -> np.ndarray:
    image_io = BytesIO(base64.b64decode(image_b64))
    img = Image.open(image_io).convert("RGB").resize((100, 100))
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return arr

def get_solutions(images: list[str], target: str):
    model = get_model()
    imgs = [preprocess_image(img) for img in images]
    batch = np.stack(imgs, axis=0)

    with _model_lock:
        preds = model(batch).numpy()

    solutions = [
        i for i, pred in enumerate(preds)
        if ANSWERS[np.argmax(pred)].lower() == target.lower()
    ]

    return solutions
