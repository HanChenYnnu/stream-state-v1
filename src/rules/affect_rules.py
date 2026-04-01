
def frustration_flag(affect_score: float) -> bool:
    return affect_score > 0.62


def blocked_flag(attention_score: float, affect_score: float, blocked_score: float) -> bool:
    return blocked_score > 0.66 and attention_score < 0.48 and affect_score > 0.5
