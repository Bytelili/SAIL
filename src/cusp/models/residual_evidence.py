"""Counterfactual residual evidence mean encoder."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass
class ResidualEvidenceFeatures:
    """Intermediate counterfactual evidence features."""

    mean: Tensor
    raw_residual: Tensor
    normalized_residual: Tensor


class ResidualEvidenceEncoder(nn.Module):
    """Map observed-minus-context representations into latent evidence means."""

    def __init__(self, hidden_size: int, latent_dim: int) -> None:
        super().__init__()
        self.direction_projector = nn.Sequential(
            nn.Linear(hidden_size, hidden_size, bias=False),
            nn.GELU(),
            nn.Linear(hidden_size, latent_dim, bias=False),
        )
        self.epsilon = 1e-8

    def forward(
        self,
        context_hidden: Tensor,
        observed_hidden: Tensor,
        support_mask: Tensor,
    ) -> ResidualEvidenceFeatures:
        """Compute `observed - stop_gradient(context)` and its latent mean."""
        if context_hidden.shape != observed_hidden.shape or context_hidden.ndim != 3:
            raise ValueError("Support hidden tensors must share shape [B,S,H]")
        if support_mask.shape != context_hidden.shape[:2]:
            raise ValueError("support_mask must have shape [B,S]")
        raw = observed_hidden - context_hidden.detach()
        raw_float = raw.float()
        magnitude = torch.linalg.vector_norm(raw_float, dim=-1, keepdim=True)
        direction = raw_float / magnitude.clamp_min(self.epsilon)
        projected_direction = self.direction_projector(
            direction.to(self.direction_projector[0].weight.dtype)
        )
        # For a fixed direction, evidence magnitude is monotone in ||r||.
        # At r=0 both direction and magnitude are exactly zero, so delta=0.
        mean = projected_direction * magnitude.to(projected_direction.dtype)
        mask = support_mask.unsqueeze(-1)
        return ResidualEvidenceFeatures(
            mean=mean.masked_fill(~mask, 0),
            raw_residual=raw.masked_fill(~mask, 0),
            normalized_residual=direction.masked_fill(~mask, 0),
        )
