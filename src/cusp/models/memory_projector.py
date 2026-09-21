"""Projection from posterior statistics to fixed-size user memory."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from .posterior import PosteriorState


class PosteriorMemoryProjector(nn.Module):
    """Compress posterior mean and log-variance into a fixed memory vector."""

    def __init__(self, latent_dim: int, memory_dim: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(latent_dim * 2, memory_dim),
            nn.GELU(),
            nn.LayerNorm(memory_dim),
            nn.Linear(memory_dim, memory_dim),
        )

    def forward(self, posterior: PosteriorState) -> Tensor:
        """Project `[mean; log_variance]` to `[B,Dm]`."""
        statistics = torch.cat(
            [posterior.mean.float(), posterior.log_variance.float()], dim=-1
        )
        return self.network(statistics)
