"""Dimension-wise observability for user-state evidence."""

from __future__ import annotations

from torch import Tensor, nn


class EvidenceObservabilityNetwork(nn.Module):
    """Predict which latent user-state dimensions a support context can reveal."""

    def __init__(self, hidden_size: int, latent_dim: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, latent_dim),
        )

    def forward(self, context_hidden: Tensor, support_mask: Tensor) -> Tensor:
        if context_hidden.ndim != 3:
            raise ValueError("context_hidden must have shape [B,S,H]")
        if support_mask.shape != context_hidden.shape[:2]:
            raise ValueError("support_mask must have shape [B,S]")
        context_hidden = context_hidden.to(dtype=self.network[0].weight.dtype)
        observability = self.network(context_hidden).sigmoid()
        return observability.masked_fill(~support_mask.unsqueeze(-1), 0.0)
