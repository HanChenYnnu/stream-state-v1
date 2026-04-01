from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_timeline(csv_path: str, out_path: str):
    df = pd.read_csv(csv_path)
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    axes[0].plot(df["timestamp"], df["attention_score"], label="attention")
    axes[1].plot(df["timestamp"], df["affect_score"], label="affect", color="orange")
    axes[2].plot(df["timestamp"], df["blocked_score"], label="blocked", color="red")
    for ax in axes:
        ax.legend()
        ax.grid(alpha=0.3)
    axes[2].set_xlabel("time (sec)")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close(fig)
