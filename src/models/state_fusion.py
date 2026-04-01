from dataclasses import dataclass


@dataclass
class StateScores:
    attention_score: float
    affect_score: float
    blocked_score: float


def fuse_scores(face_present: float, gaze_forward: float, posture_engaged: float, affect_score: float) -> StateScores:
    attention = 0.35 * face_present + 0.45 * gaze_forward + 0.20 * posture_engaged
    blocked = max(0.0, min(1.0, (1.0 - attention) * 0.6 + affect_score * 0.7))
    return StateScores(float(attention), float(affect_score), float(blocked))
