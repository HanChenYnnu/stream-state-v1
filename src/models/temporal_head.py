from __future__ import annotations

import torch
import torch.nn as nn

HEADS = ["boredom", "confusion", "engagement", "frustration"]
NUM_CLASSES = 4


class MultiHeadClassifier(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.ReLU(), nn.Dropout(0.1))
        self.heads = nn.ModuleDict({h: nn.Linear(hidden_dim, NUM_CLASSES) for h in HEADS})

    def forward_heads(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        z = self.proj(x)
        return {h: self.heads[h](z) for h in HEADS}


class FrameBaseline(nn.Module):
    def __init__(self, hidden_dim: int = 128):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.cls = MultiHeadClassifier(in_dim=64, hidden_dim=hidden_dim)

    def forward(self, frames: torch.Tensor, **kwargs) -> dict[str, torch.Tensor]:
        # frames: B, T, C, H, W
        b, t, c, h, w = frames.shape
        x = frames.reshape(b * t, c, h, w)
        feat = self.backbone(x).flatten(1).reshape(b, t, -1).mean(dim=1)
        return self.cls.forward_heads(feat)


class TemporalProxyGRU(nn.Module):
    def __init__(self, proxy_dim: int = 9, hidden_dim: int = 128):
        super().__init__()
        self.gru = nn.GRU(input_size=proxy_dim, hidden_size=hidden_dim, num_layers=1, batch_first=True, bidirectional=True)
        self.cls = MultiHeadClassifier(in_dim=hidden_dim * 2, hidden_dim=hidden_dim)

    def forward(self, proxy_seq: torch.Tensor, **kwargs) -> dict[str, torch.Tensor]:
        out, _ = self.gru(proxy_seq)
        pooled = out.mean(dim=1)
        return self.cls.forward_heads(pooled)


class LightweightFusionModel(nn.Module):
    def __init__(self, proxy_dim: int = 9, hidden_dim: int = 128):
        super().__init__()
        self.frame = FrameBaseline(hidden_dim=hidden_dim)
        self.proxy_encoder = nn.GRU(proxy_dim, hidden_dim, batch_first=True)
        self.cls = MultiHeadClassifier(in_dim=hidden_dim + 64, hidden_dim=hidden_dim)

    def forward(self, frames: torch.Tensor, proxy_seq: torch.Tensor, **kwargs) -> dict[str, torch.Tensor]:
        b, t, c, h, w = frames.shape
        x = frames.reshape(b * t, c, h, w)
        frame_feat = self.frame.backbone(x).flatten(1).reshape(b, t, -1).mean(dim=1)
        seq_out, _ = self.proxy_encoder(proxy_seq)
        proxy_feat = seq_out.mean(dim=1)
        fused = torch.cat([frame_feat, proxy_feat], dim=1)
        return self.cls.forward_heads(fused)


def build_daisee_model(model_type: str) -> nn.Module:
    if model_type == "frame_baseline":
        return FrameBaseline()
    if model_type == "temporal_proxy":
        return TemporalProxyGRU()
    if model_type == "lightweight_fusion":
        return LightweightFusionModel()
    raise ValueError(f"Unsupported model_type={model_type}")
