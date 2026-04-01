from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.core.utils import load_yaml
from src.datasets.collate import collate_daisee
from src.datasets.daisee_dataset import DAiSEEDataset
from src.mapping.daisee_to_learning_state import map_row_to_learning_state
from src.models.temporal_head import HEADS, build_daisee_model
from src.training.metrics import compute_head_metrics, save_metrics_json
from src.viz.daisee_plots import plot_classwise_f1, plot_confusion_matrices, plot_mapped_state_timeline


@torch.no_grad()
def run_eval(model, loader, device, model_type: str):
    model.eval()
    rows = []
    y_true, y_pred = [], []
    for batch in tqdm(loader, desc="eval"):
        labels = batch["labels"].to(device)
        if "frames" in batch:
            batch["frames"] = batch["frames"].to(device)
        if "proxy_seq" in batch:
            batch["proxy_seq"] = batch["proxy_seq"].to(device)

        if model_type == "frame_baseline":
            logits = model(frames=batch["frames"])
        elif model_type == "temporal_proxy":
            logits = model(proxy_seq=batch["proxy_seq"])
        else:
            logits = model(frames=batch["frames"], proxy_seq=batch["proxy_seq"])

        pred = torch.stack([logits[h].argmax(dim=1) for h in HEADS], dim=1)
        y_true.append(labels.cpu().numpy())
        y_pred.append(pred.cpu().numpy())

        for i, cid in enumerate(batch["clip_id"]):
            r = {"clip_id": cid}
            for j, h in enumerate(HEADS):
                r[f"true_{h}"] = int(labels[i, j].item())
                r[f"pred_{h}"] = int(pred[i, j].item())
            r["mapped_state"] = map_row_to_learning_state(r)
            rows.append(r)
    return pd.DataFrame(rows), np.concatenate(y_true, axis=0), np.concatenate(y_pred, axis=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/daisee.yaml")
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_type = cfg["model"]["model_type"]
    mode = "raw" if model_type == "frame_baseline" else ("proxy" if model_type == "temporal_proxy" else "hybrid")

    ds = DAiSEEDataset(
        manifest_csv=cfg["dataset"]["manifest_csv"],
        dataset_root=cfg["dataset"]["root"],
        split=cfg["eval"]["split"],
        mode=mode,
        num_frames=cfg["sampling"]["num_frames"],
        image_size=(cfg["sampling"]["image_size"], cfg["sampling"]["image_size"]),
    )
    loader = DataLoader(ds, batch_size=cfg["eval"]["batch_size"], shuffle=False, num_workers=cfg["eval"]["num_workers"], collate_fn=collate_daisee)

    model = build_daisee_model(model_type).to(device)
    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state["model_state"], strict=True)

    pred_df, y_true, y_pred = run_eval(model, loader, device, model_type)
    metrics = compute_head_metrics(y_true, y_pred)

    out_dir = Path(cfg["output"]["eval_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_csv = out_dir / "predictions.csv"
    pred_df.to_csv(pred_csv, index=False)
    save_metrics_json(metrics, str(out_dir / "classification_report.json"))

    plot_confusion_matrices(metrics, str(out_dir / "figures"))
    plot_classwise_f1(metrics, str(out_dir / "figures" / "classwise_f1.png"))
    plot_mapped_state_timeline(pred_df, str(out_dir / "figures" / "mapped_state_timeline.png"))
    print(f"saved predictions: {pred_csv}")


if __name__ == "__main__":
    main()
