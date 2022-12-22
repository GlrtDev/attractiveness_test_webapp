import numpy as np
import cv2
class Preprocessor():
    def __init__(self) -> None:
        pass

    def load_image(self, img_bytes):
        arr = np.fromstring(img_bytes, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        return img

preprocessor = Preprocessor()