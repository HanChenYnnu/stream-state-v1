from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.core.utils import load_yaml
from src.datasets.collate import collate_daisee
from src.datasets.daisee_dataset import DAiSEEDataset
from src.models.temporal_head import HEADS, build_daisee_model
from src.training.losses import multi_head_ce_loss
from src.training.metrics import compute_head_metrics, save_metrics_json


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _to_device(batch: dict, device: torch.device) -> dict:
    out = {k: v for k, v in batch.items()}
    out["labels"] = out["labels"].to(device)
    if "frames" in out:
        out["frames"] = out["frames"].to(device)
    if "proxy_seq" in out:
        out["proxy_seq"] = out["proxy_seq"].to(device)
    return out


def train_one_epoch(model, loader, optimizer, device, model_type: str):
    model.train()
    running = 0.0
    for batch in tqdm(loader, desc="train", leave=False):
        batch = _to_device(batch, device)
        optimizer.zero_grad()
        if model_type == "frame_baseline":
            logits = model(frames=batch["frames"])
        elif model_type == "temporal_proxy":
            logits = model(proxy_seq=batch["proxy_seq"])
        else:
            logits = model(frames=batch["frames"], proxy_seq=batch["proxy_seq"])
        loss, _ = multi_head_ce_loss(logits, batch["labels"])
        loss.backward()
        optimizer.step()
        running += float(loss.detach().cpu())
    return running / max(1, len(loader))


@torch.no_grad()
def evaluate(model, loader, device, model_type: str):
    model.eval()
    ys, ps = [], []
    val_loss = 0.0
    for batch in tqdm(loader, desc="val", leave=False):
        batch = _to_device(batch, device)
        if model_type == "frame_baseline":
            logits = model(frames=batch["frames"])
        elif model_type == "temporal_proxy":
            logits = model(proxy_seq=batch["proxy_seq"])
        else:
            logits = model(frames=batch["frames"], proxy_seq=batch["proxy_seq"])
        loss, _ = multi_head_ce_loss(logits, batch["labels"])
        val_loss += float(loss.detach().cpu())
        pred = torch.stack([logits[h].argmax(dim=1) for h in HEADS], dim=1)
        ys.append(batch["labels"].cpu().numpy())
        ps.append(pred.cpu().numpy())
    y_true = np.concatenate(ys, axis=0)
    y_pred = np.concatenate(ps, axis=0)
    metrics = compute_head_metrics(y_true, y_pred)
    return val_loss / max(1, len(loader)), metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/daisee.yaml")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    set_seed(int(cfg["seed"]))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_type = cfg["model"]["model_type"]
    mode = "raw" if model_type == "frame_baseline" else ("proxy" if model_type == "temporal_proxy" else "hybrid")

    train_ds = DAiSEEDataset(
        manifest_csv=cfg["dataset"]["manifest_csv"],
        dataset_root=cfg["dataset"]["root"],
        split="train",
        mode=mode,
        num_frames=cfg["sampling"]["num_frames"],
        image_size=(cfg["sampling"]["image_size"], cfg["sampling"]["image_size"]),
    )
    val_ds = DAiSEEDataset(
        manifest_csv=cfg["dataset"]["manifest_csv"],
        dataset_root=cfg["dataset"]["root"],
        split="val",
        mode=mode,
        num_frames=cfg["sampling"]["num_frames"],
        image_size=(cfg["sampling"]["image_size"], cfg["sampling"]["image_size"]),
    )

    train_loader = DataLoader(train_ds, batch_size=cfg["train"]["batch_size"], shuffle=True, num_workers=cfg["train"]["num_workers"], collate_fn=collate_daisee)
    val_loader = DataLoader(val_ds, batch_size=cfg["train"]["batch_size"], shuffle=False, num_workers=cfg["train"]["num_workers"], collate_fn=collate_daisee)

    model = build_daisee_model(model_type).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["train"]["lr"], weight_decay=cfg["train"]["weight_decay"])

    out_dir = Path(cfg["output"]["run_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    log_csv = out_dir / "train_log.csv"
    ckpt_path = out_dir / "best.pt"
    best = -1.0

    with open(log_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "train_loss", "val_loss", "val_macro_f1"])
        for epoch in range(1, cfg["train"]["epochs"] + 1):
            tr = train_one_epoch(model, train_loader, optimizer, device, model_type)
            vl, m = evaluate(model, val_loader, device, model_type)
            macro = m["overall"]["macro_f1"]
            w.writerow([epoch, tr, vl, macro])
            if macro > best:
                best = macro
                torch.save({"model_state": model.state_dict(), "config": cfg}, ckpt_path)
                save_metrics_json(m, str(out_dir / "best_val_metrics.json"))
            print(f"epoch={epoch} train={tr:.4f} val={vl:.4f} macro_f1={macro:.4f}")

    print(f"saved best checkpoint: {ckpt_path}")


if __name__ == "__main__":
    main()
