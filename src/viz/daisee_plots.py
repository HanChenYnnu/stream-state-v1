from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.models.temporal_head import HEADS


def plot_confusion_matrices(metrics: dict, out_dir: str):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for h in HEADS:
        cm = np.array(metrics[h]["confusion_matrix"])
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.imshow(cm, cmap="Blues")
        ax.set_title(f"{h} confusion matrix")
        ax.set_xlabel("pred")
        ax.set_ylabel("true")
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center")
        plt.tight_layout()
        plt.savefig(out / f"cm_{h}.png")
        plt.close(fig)


def plot_classwise_f1(metrics: dict, out_png: str):
    heads = []
    vals = []
    for h in HEADS:
        heads.append(h)
        vals.append(metrics[h]["macro_f1"])
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(heads, vals, color="#4CAF50")
    ax.set_ylim(0, 1)
    ax.set_ylabel("macro F1")
    ax.set_title("DAiSEE per-head macro F1")
    plt.tight_layout()
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png)
    plt.close(fig)


def plot_mapped_state_timeline(pred_df: pd.DataFrame, out_png: str):
    mapping = {"S0": 0, "S1": 1, "S2": 2, "S3": 3, "S4": 4}
    y = [mapping[s] for s in pred_df["mapped_state"].tolist()]
    fig, ax = plt.subplots(figsize=(12, 3))
    ax.plot(np.arange(len(y)), y, marker="o")
    ax.set_yticks(list(mapping.values()))
    ax.set_yticklabels(list(mapping.keys()))
    ax.set_xlabel("clip index")
    ax.set_title("Mapped educational state timeline (S0-S4)")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png)
    plt.close(fig)
