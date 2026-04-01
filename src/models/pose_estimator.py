from dataclasses import dataclass

import cv2
import mediapipe as mp
import numpy as np


@dataclass
class PoseProxyFeatures:
    posture_engaged: float


class PoseEstimator:
    def __init__(self):
        self.pose = mp.solutions.pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def estimate(self, frame_bgr: np.ndarray) -> PoseProxyFeatures:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        res = self.pose.process(rgb)
        if not res.pose_landmarks:
            return PoseProxyFeatures(0.2)
        lms = res.pose_landmarks.landmark
        left_shoulder = lms[11]
        right_shoulder = lms[12]
        left_hip = lms[23]
        right_hip = lms[24]

        shoulder_y = (left_shoulder.y + right_shoulder.y) / 2
        hip_y = (left_hip.y + right_hip.y) / 2
        torso = max(1e-5, hip_y - shoulder_y)
        engaged = float(max(0.0, min(1.0, 1.2 - abs(torso - 0.28) * 3.5)))
        return PoseProxyFeatures(engaged)
