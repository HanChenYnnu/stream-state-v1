from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

HEADS = ["boredom", "confusion", "engagement", "frustration"]


def _uniform_indices(total: int, num: int) -> np.ndarray:
    if total <= 0:
        return np.zeros((num,), dtype=np.int64)
    if total < num:
        return np.linspace(0, max(0, total - 1), num=num, dtype=np.int64)
    return np.linspace(0, total - 1, num=num, dtype=np.int64)


def _read_video_frames(video_path: str, num_frames: int, image_size: tuple[int, int]) -> np.ndarray:
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    idxs = set(_uniform_indices(total, num_frames).tolist())
    frames = []
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if i in idxs:
            frame = cv2.resize(frame, image_size)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        i += 1
    cap.release()
    if not frames:
        frames = [np.zeros((image_size[1], image_size[0], 3), dtype=np.uint8) for _ in range(num_frames)]
    while len(frames) < num_frames:
        frames.append(frames[-1])
    frames = np.stack(frames[:num_frames], axis=0)
    return frames


class DAiSEEDataset(Dataset):
    def __init__(
        self,
        manifest_csv: str,
        dataset_root: str,
        split: str,
        mode: str = "raw",
        num_frames: int = 16,
        image_size: tuple[int, int] = (224, 224),
    ):
        self.root = Path(dataset_root)
        self.df = pd.read_csv(manifest_csv)
        self.df = self.df[self.df["split"] == split].reset_index(drop=True)
        self.mode = mode
        self.num_frames = num_frames
        self.image_size = image_size

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        y = torch.tensor([int(max(0, row[h])) for h in HEADS], dtype=torch.long)
        sample = {
            "clip_id": row["clip_id"],
            "video_path": str(self.root / row["video_path"]),
            "labels": y,
        }
        if self.mode in {"proxy", "hybrid"}:
            feat_path = row.get("feature_path", "")
            if not feat_path:
                raise ValueError("feature_path is required for proxy/hybrid mode")
            arr = np.load(feat_path)["features"].astype(np.float32)
            sample["proxy_seq"] = torch.from_numpy(arr)
        if self.mode in {"raw", "hybrid"}:
            frames = _read_video_frames(sample["video_path"], self.num_frames, self.image_size)
            frames = torch.from_numpy(frames).float().permute(0, 3, 1, 2) / 255.0
            sample["frames"] = frames
        return sample
