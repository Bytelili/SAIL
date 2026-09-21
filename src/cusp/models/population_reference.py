"""Compatible multimodal Population reference for CUSP support evidence."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass
class PopulationReferenceOutput:
    """Population reference and normalized candidate responsibilities."""

    reference: Tensor
    responsibilities: Tensor
    entropy: Tensor


class CompatiblePopulationReference(nn.Module):
    """Mix Population candidates by likelihood and observed-behavior compatibility."""

    def __init__(
        self,
        *,
        probability_temperature: float = 1.0,
        compatibility_temperature: float = 1.0,
    ) -> None:
        super().__init__()
        if probability_temperature <= 0 or compatibility_temperature <= 0:
            raise ValueError("Population reference temperatures must be positive")
        self.probability_temperature = float(probability_temperature)
        self.compatibility_temperature = float(compatibility_temperature)

    def forward(
        self,
        candidate_hidden: Tensor,
        candidate_log_probability: Tensor,
        observed_hidden: Tensor,
        candidate_mask: Tensor,
        support_mask: Tensor,
    ) -> PopulationReferenceOutput:
        """Return the compatible mixture reference for every support interaction."""
        if candidate_hidden.ndim != 4:
            raise ValueError("candidate_hidden must have shape [B,S,M,H]")
        if candidate_log_probability.shape != candidate_hidden.shape[:3]:
            raise ValueError("candidate_log_probability must have shape [B,S,M]")
        if candidate_mask.shape != candidate_hidden.shape[:3]:
            raise ValueError("candidate_mask must have shape [B,S,M]")
        if observed_hidden.shape != (
            candidate_hidden.shape[0],
            candidate_hidden.shape[1],
            candidate_hidden.shape[3],
        ):
            raise ValueError("observed_hidden must have shape [B,S,H]")
        if support_mask.shape != candidate_hidden.shape[:2]:
            raise ValueError("support_mask must have shape [B,S]")
        valid = candidate_mask & support_mask.unsqueeze(-1)
        if torch.any(support_mask & ~valid.any(dim=-1)):
            raise ValueError("Every valid support item needs a Population candidate")

        normalized_log_probability = (
            candidate_log_probability.float() / self.probability_temperature
        )
        normalized_candidate = torch.nn.functional.normalize(
            candidate_hidden.float(), p=2, dim=-1, eps=1e-8
        )
        normalized_observed = torch.nn.functional.normalize(
            observed_hidden.float(), p=2, dim=-1, eps=1e-8
        )
        cosine_distance = 1.0 - (
            normalized_candidate * normalized_observed.unsqueeze(2)
        ).sum(dim=-1)
        responsibility_logits = (
            normalized_log_probability
            - cosine_distance / self.compatibility_temperature
        ).masked_fill(~valid, float("-inf"))
        safe_logits = torch.where(
            valid.any(dim=-1, keepdim=True),
            responsibility_logits,
            torch.zeros_like(responsibility_logits),
        )
        responsibilities = torch.softmax(safe_logits, dim=-1)
        responsibilities = torch.where(
            valid, responsibilities, torch.zeros_like(responsibilities)
        )
        reference = (
            responsibilities.unsqueeze(-1) * candidate_hidden.float()
        ).sum(dim=2)
        reference = reference.masked_fill(~support_mask.unsqueeze(-1), 0.0)
        entropy = -(
            responsibilities
            * responsibilities.clamp_min(1e-12).log()
        ).sum(dim=-1)
        entropy = entropy.masked_fill(~support_mask, 0.0)
        if not torch.isfinite(reference).all() or not torch.isfinite(entropy).all():
            raise FloatingPointError("Population candidate responsibilities are non-finite")
        return PopulationReferenceOutput(reference, responsibilities, entropy)
