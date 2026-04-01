import cv2

from src.rules.state_mapper import STATE_TO_ZH


def draw_overlay(frame, state_code: str, confidence: float, scores: dict, features: dict):
    text = f"{state_code} {STATE_TO_ZH[state_code]} conf={confidence:.2f}"
    cv2.putText(frame, text, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (40, 255, 40), 2)
    y = 60
    for k, v in {**scores, **features}.items():
        cv2.putText(frame, f"{k}: {v:.2f}", (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        y += 22
    return frame
