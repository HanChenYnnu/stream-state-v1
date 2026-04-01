import torch


def collate_daisee(batch: list[dict]) -> dict:
    out = {
        "clip_id": [x["clip_id"] for x in batch],
        "video_path": [x["video_path"] for x in batch],
        "labels": torch.stack([x["labels"] for x in batch], dim=0),
    }
    if "frames" in batch[0]:
        out["frames"] = torch.stack([x["frames"] for x in batch], dim=0)
    if "proxy_seq" in batch[0]:
        out["proxy_seq"] = torch.stack([x["proxy_seq"] for x in batch], dim=0)
    return out
