"""Learned bounded precision for each counterfactual support item."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn


class EvidencePrecisionNetwork(nn.Module):
    """Predict continuous per-latent precision in a configured finite range."""

    def __init__(
        self,
        hidden_size: int,
        latent_dim: int,
        min_precision: float = 0.001,
        max_precision: float = 100.0,
        initial_precision: float = 0.05,
    ) -> None:
        super().__init__()
        if not 0 < min_precision < max_precision:
            raise ValueError("Precision bounds must satisfy 0 < min < max")
        self.min_precision = float(min_precision)
        self.max_precision = float(max_precision)
        if not math.isfinite(initial_precision):
            raise ValueError("initial_precision must be finite")
        self.network = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, latent_dim),
        )
        probability = (float(initial_precision) - self.min_precision) / (
            self.max_precision - self.min_precision
        )
        epsilon = 1e-7
        probability = min(max(probability, epsilon), 1.0 - epsilon)
        self.initial_precision = self.min_precision + probability * (
            self.max_precision - self.min_precision
        )
        nn.init.zeros_(self.network[0].bias)
        final = self.network[-1]
        nn.init.constant_(final.bias, math.log(probability / (1.0 - probability)))

    def forward(self, normalized_residual: Tensor, support_mask: Tensor) -> Tensor:
        """Return float32 precision, with exactly zero at padding positions."""
        if normalized_residual.ndim != 3:
            raise ValueError("normalized_residual must have shape [B,S,H]")
        if support_mask.shape != normalized_residual.shape[:2]:
            raise ValueError("support_mask must have shape [B,S]")
        inputs = torch.cat(
            [normalized_residual.float(), normalized_residual.float().abs()], dim=-1
        )
        raw = self.network(inputs)
        precision = self.min_precision + (
            self.max_precision - self.min_precision
        ) * torch.sigmoid(raw.float())
        return precision.masked_fill(~support_mask.unsqueeze(-1), 0.0)
