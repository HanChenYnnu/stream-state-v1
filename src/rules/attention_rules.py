
def attention_risk(attention_score: float) -> float:
    return max(0.0, min(1.0, 1.0 - attention_score))


def away_flag(face_presence_ratio: float) -> bool:
    return face_presence_ratio < 0.2
