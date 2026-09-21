"""Structured bounded evidence precision from the complete SAIL evidence tuple."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn


class StructuredEvidencePrecisionNetwork(nn.Module):
    """Predict precision from context, observed, Population reference, and residual."""

    def __init__(
        self,
        hidden_size: int,
        latent_dim: int,
        min_precision: float,
        max_precision: float,
        initial_precision: float,
    ) -> None:
        super().__init__()
        if not 0 < min_precision < max_precision:
            raise ValueError("Precision bounds must satisfy 0 < min < max")
        self.min_precision = float(min_precision)
        self.max_precision = float(max_precision)
        probability = (float(initial_precision) - min_precision) / (
            max_precision - min_precision
        )
        probability = min(max(probability, 1e-7), 1.0 - 1e-7)
        self.network = nn.Sequential(
            nn.Linear(hidden_size * 4, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, latent_dim),
        )
        nn.init.zeros_(self.network[0].bias)
        nn.init.constant_(
            self.network[-1].bias, math.log(probability / (1.0 - probability))
        )

    def forward(
        self,
        context_hidden: Tensor,
        observed_hidden: Tensor,
        population_reference: Tensor,
        raw_residual: Tensor,
        support_mask: Tensor,
    ) -> Tensor:
        """Return bounded dimension-wise precision, zeroed at padding positions."""
        tensors = (context_hidden, observed_hidden, population_reference, raw_residual)
        if any(tensor.ndim != 3 for tensor in tensors):
            raise ValueError("Structured precision inputs must have shape [B,S,H]")
        if any(tensor.shape != context_hidden.shape for tensor in tensors[1:]):
            raise ValueError("Structured precision inputs must share shape")
        if support_mask.shape != context_hidden.shape[:2]:
            raise ValueError("support_mask must have shape [B,S]")
        inputs = torch.cat(
            [
                context_hidden.float(),
                observed_hidden.float(),
                population_reference.float(),
                raw_residual.float().abs(),
            ],
            dim=-1,
        )
        raw = self.network(inputs)
        precision = self.min_precision + (
            self.max_precision - self.min_precision
        ) * torch.sigmoid(raw.float())
        return precision.masked_fill(~support_mask.unsqueeze(-1), 0.0)
