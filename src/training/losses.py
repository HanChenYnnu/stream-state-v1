from __future__ import annotations

import torch
import torch.nn.functional as F

from src.models.temporal_head import HEADS


def multi_head_ce_loss(logits: dict[str, torch.Tensor], targets: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
    losses = {}
    total = 0.0
    for i, h in enumerate(HEADS):
        l = F.cross_entropy(logits[h], targets[:, i])
        total = total + l
        losses[h] = float(l.detach().cpu())
    total = total / len(HEADS)
    return total, losses
