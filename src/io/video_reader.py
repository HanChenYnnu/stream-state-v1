from typing import Generator, Tuple

import cv2
import numpy as np


class VideoReader:
    def __init__(self, path: str):
        self.path = path

    def read(self) -> Generator[Tuple[int, np.ndarray, float], None, None]:
        cap = cv2.VideoCapture(self.path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            ts = idx / fps
            yield idx, frame, ts
            idx += 1
        cap.release()
