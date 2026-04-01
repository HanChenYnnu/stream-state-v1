from dataclasses import dataclass

import cv2
import mediapipe as mp
import numpy as np


@dataclass
class FaceDetectionResult:
    present: bool
    confidence: float
    bbox: tuple[int, int, int, int] | None


class FaceDetector:
    def __init__(self, min_detection_confidence: float = 0.5):
        self.detector = mp.solutions.face_detection.FaceDetection(
            model_selection=0,
            min_detection_confidence=min_detection_confidence,
        )

    def predict(self, frame_bgr: np.ndarray) -> FaceDetectionResult:
        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        res = self.detector.process(rgb)
        if not res.detections:
            return FaceDetectionResult(False, 0.0, None)
        det = res.detections[0]
        score = float(det.score[0])
        bb = det.location_data.relative_bounding_box
        x1 = max(0, int(bb.xmin * w))
        y1 = max(0, int(bb.ymin * h))
        x2 = min(w - 1, int((bb.xmin + bb.width) * w))
        y2 = min(h - 1, int((bb.ymin + bb.height) * h))
        return FaceDetectionResult(True, score, (x1, y1, x2, y2))
