from pathlib import Path
from typing import Iterable

import cv2
import pandas as pd


class VideoWriter:
    def __init__(self, path: str, width: int, height: int, fps: int):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.writer = cv2.VideoWriter(path, fourcc, fps, (width, height))

    def write(self, frame):
        self.writer.write(frame)

    def close(self):
        self.writer.release()


def write_csv(rows: Iterable[dict], path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(list(rows)).to_csv(path, index=False)
