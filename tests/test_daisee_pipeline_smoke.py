import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.datasets.collate import collate_daisee
from src.datasets.daisee_dataset import DAiSEEDataset
from src.models.temporal_head import build_daisee_model
from src.training.losses import multi_head_ce_loss


def test_tiny_proxy_pipeline(tmp_path):
    feat = np.random.rand(8, 9).astype("float32")
    feat_path = tmp_path / "c1.npz"
    np.savez_compressed(feat_path, features=feat)

    df = pd.DataFrame([
        {
            "video_path": "dummy.mp4",
            "clip_id": "c1",
            "subject_id": "s1",
            "split": "train",
            "boredom": 1,
            "confusion": 0,
            "engagement": 2,
            "frustration": 1,
            "feature_path": str(feat_path),
        }
    ])
    manifest = tmp_path / "m.csv"
    df.to_csv(manifest, index=False)

    ds = DAiSEEDataset(str(manifest), dataset_root=str(tmp_path), split="train", mode="proxy")
    batch = next(iter(DataLoader(ds, batch_size=1, collate_fn=collate_daisee)))

    model = build_daisee_model("temporal_proxy")
    logits = model(proxy_seq=batch["proxy_seq"])
    loss, _ = multi_head_ce_loss(logits, batch["labels"])
    assert torch.isfinite(loss)
