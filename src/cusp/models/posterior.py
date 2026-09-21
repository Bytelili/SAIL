"""Closed-form diagonal Gaussian user-state posterior."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass
class PosteriorState:
    """Sufficient statistics and diagnostics for a user-state posterior."""

    mean: Tensor
    variance: Tensor
    log_variance: Tensor
    precision: Tensor
    information_gain: Tensor
    evidence_count: Tensor


class DiagonalGaussianPosterior(nn.Module):
    """Aggregate independent diagonal Gaussian evidence in float32."""

    def __init__(self, latent_dim: int, prior_precision: float = 1.0) -> None:
        super().__init__()
        if prior_precision <= 0:
            raise ValueError("prior_precision must be positive")
        self.latent_dim = latent_dim
        self.prior_precision = float(prior_precision)

    def forward(
        self,
        evidence_mean: Tensor,
        evidence_precision: Tensor,
        support_mask: Tensor,
        observability: Tensor | None = None,
    ) -> PosteriorState:
        """Compute the order-invariant batch posterior."""
        if evidence_mean.shape != evidence_precision.shape or evidence_mean.ndim != 3:
            raise ValueError("Evidence tensors must share shape [B,S,Dz]")
        if evidence_mean.shape[-1] != self.latent_dim:
            raise ValueError("Evidence latent dimension does not match posterior")
        if support_mask.shape != evidence_mean.shape[:2]:
            raise ValueError("support_mask must have shape [B,S]")
        mean = evidence_mean.float()
        precision = evidence_precision.float() * support_mask.unsqueeze(-1).float()
        if observability is None:
            observability = torch.ones_like(mean)
        if observability.shape != mean.shape:
            raise ValueError("observability must share shape [B,S,Dz]")
        observability = (
            observability.float() * support_mask.unsqueeze(-1).float()
        )
        posterior_precision = self.prior_precision + (
            precision * observability.square()
        ).sum(dim=1)
        natural = (precision * observability * mean).sum(dim=1)
        posterior_mean = natural / posterior_precision
        variance = posterior_precision.reciprocal()
        information_gain = (
            (posterior_precision / self.prior_precision).log().sum(dim=-1)
            / (2.0 * self.latent_dim)
        )
        return PosteriorState(
            mean=posterior_mean,
            variance=variance,
            log_variance=variance.log(),
            precision=posterior_precision,
            information_gain=information_gain,
            evidence_count=support_mask.sum(dim=1).to(torch.int64),
        )

    def prior(self, batch_size: int, device: torch.device | str) -> PosteriorState:
        """Construct an explicit zero-evidence prior."""
        mean = torch.zeros((batch_size, self.latent_dim), dtype=torch.float32, device=device)
        precision = torch.full_like(mean, self.prior_precision)
        variance = precision.reciprocal()
        return PosteriorState(
            mean=mean,
            variance=variance,
            log_variance=variance.log(),
            precision=precision,
            information_gain=torch.zeros(batch_size, dtype=torch.float32, device=device),
            evidence_count=torch.zeros(batch_size, dtype=torch.int64, device=device),
        )

    def update(
        self,
        previous_state: PosteriorState,
        evidence_mean: Tensor,
        evidence_precision: Tensor,
        evidence_mask: Tensor | None = None,
        observability: Tensor | None = None,
    ) -> PosteriorState:
        """Apply one or more evidence items to an existing posterior state."""
        if evidence_mean.ndim == 2:
            evidence_mean = evidence_mean.unsqueeze(1)
            evidence_precision = evidence_precision.unsqueeze(1)
            if observability is not None and observability.ndim == 2:
                observability = observability.unsqueeze(1)
        if evidence_mean.ndim != 3 or evidence_mean.shape != evidence_precision.shape:
            raise ValueError("Online evidence must have shape [B,S,Dz] or [B,Dz]")
        batch, count, _ = evidence_mean.shape
        if evidence_mask is None:
            evidence_mask = torch.ones(
                (batch, count), dtype=torch.bool, device=evidence_mean.device
            )
        precision = evidence_precision.float() * evidence_mask.unsqueeze(-1).float()
        if observability is None:
            observability = torch.ones_like(evidence_mean)
        if observability.shape != evidence_mean.shape:
            raise ValueError("online observability must share evidence shape")
        observability = (
            observability.float() * evidence_mask.unsqueeze(-1).float()
        )
        old_precision = previous_state.precision.float()
        old_natural = old_precision * previous_state.mean.float()
        new_precision = old_precision + (
            precision * observability.square()
        ).sum(dim=1)
        new_natural = old_natural + (
            precision * observability * evidence_mean.float()
        ).sum(dim=1)
        mean = new_natural / new_precision
        variance = new_precision.reciprocal()
        information_gain = (
            (new_precision / self.prior_precision).log().sum(dim=-1)
            / (2.0 * self.latent_dim)
        )
        return PosteriorState(
            mean=mean,
            variance=variance,
            log_variance=variance.log(),
            precision=new_precision,
            information_gain=information_gain,
            evidence_count=previous_state.evidence_count
            + evidence_mask.sum(dim=1).to(torch.int64),
        )
