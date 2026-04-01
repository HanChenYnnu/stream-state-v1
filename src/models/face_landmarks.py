from dataclasses import dataclass

import cv2
import mediapipe as mp
import numpy as np


@dataclass
class FaceProxyFeatures:
    gaze_forward: float
    brow_tension: float
    mouth_downturn: float
    eye_squint: float


class FaceLandmarksEstimator:
    def __init__(self):
        self.mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def estimate(self, frame_bgr: np.ndarray) -> FaceProxyFeatures:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        res = self.mesh.process(rgb)
        if not res.multi_face_landmarks:
            return FaceProxyFeatures(0.0, 0.0, 0.0, 0.0)
        lms = res.multi_face_landmarks[0].landmark

        left_eye = lms[33]
        right_eye = lms[263]
        nose = lms[1]
        eye_mid_x = (left_eye.x + right_eye.x) / 2
        gaze_forward = max(0.0, 1.0 - min(1.0, abs(nose.x - eye_mid_x) * 5))

        brow = (lms[70].y + lms[300].y) / 2
        eye = (lms[159].y + lms[386].y) / 2
        brow_tension = float(max(0.0, min(1.0, (eye - brow) * 8)))

        mouth_corner = (lms[61].y + lms[291].y) / 2
        mouth_center = lms[13].y
        mouth_downturn = float(max(0.0, min(1.0, (mouth_corner - mouth_center) * 10)))

        lid_gap = abs(lms[159].y - lms[145].y) + abs(lms[386].y - lms[374].y)
        eye_squint = float(max(0.0, min(1.0, 1.0 - lid_gap * 30)))
        return FaceProxyFeatures(gaze_forward, brow_tension, mouth_downturn, eye_squint)
