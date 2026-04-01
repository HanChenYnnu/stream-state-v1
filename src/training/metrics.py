from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from src.models.temporal_head import HEADS


def compute_head_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    result = {}
    macro_scores = []
    for i, h in enumerate(HEADS):
        yt = y_true[:, i]
        yp = y_pred[:, i]
        macro = f1_score(yt, yp, average="macro", zero_division=0)
        weighted = f1_score(yt, yp, average="weighted", zero_division=0)
        acc = accuracy_score(yt, yp)
        macro_scores.append(macro)
        result[h] = {
            "macro_f1": float(macro),
            "weighted_f1": float(weighted),
            "accuracy": float(acc),
            "confusion_matrix": confusion_matrix(yt, yp, labels=[0, 1, 2, 3]).tolist(),
            "classification_report": classification_report(yt, yp, labels=[0, 1, 2, 3], output_dict=True, zero_division=0),
        }
    result["overall"] = {"macro_f1": float(np.mean(macro_scores))}
    return result


def save_metrics_json(metrics: dict, path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
