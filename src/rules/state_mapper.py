from src.rules.affect_rules import blocked_flag, frustration_flag
from src.rules.attention_rules import away_flag


STATE_TO_ZH = {
    "S0": "正常推进",
    "S1": "轻度分心",
    "S2": "高风险走神",
    "S3": "情绪受挫",
    "S4": "疑似认知受阻",
    "S5": "暂时离开",
}


def map_state(attention_score: float, affect_score: float, blocked_score: float, face_presence_ratio: float) -> tuple[str, float]:
    if away_flag(face_presence_ratio):
        return "S5", 0.9
    if blocked_flag(attention_score, affect_score, blocked_score):
        return "S4", max(0.5, blocked_score)
    if frustration_flag(affect_score) and attention_score >= 0.5:
        return "S3", max(0.5, affect_score)
    if attention_score < 0.33:
        return "S2", 0.75
    if attention_score < 0.55:
        return "S1", 0.65
    return "S0", 0.8
