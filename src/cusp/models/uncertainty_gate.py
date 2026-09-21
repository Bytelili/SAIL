"""Information-gain constrained, query-conditioned personalization gate."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass
class GateOutput:
    """Base confidence, learned context confidence, and their product."""

    alpha: Tensor
    information_confidence: Tensor
    context_confidence: Tensor


class UncertaintyAwareGate(nn.Module):
    """Guarantee exact zero personalization when information gain is zero."""

    def __init__(
        self, memory_dim: int, context_dim: int, *, compatibility_mode: str = "none"
    ) -> None:
        super().__init__()
        if compatibility_mode not in {"none", "cosine"}:
            raise ValueError("compatibility_mode must be none or cosine")
        self.compatibility_mode = compatibility_mode
        hidden = max(memory_dim, 32)
        self.network = nn.Sequential(
            nn.Linear(memory_dim + context_dim + 1, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )
        # Kept absent in the legacy mode so prior formal checkpoints remain
        # loadable.  In cosine mode this turns the gate into a true
        # query-conditioned posterior applicability test.
        self.query_to_memory = (
            nn.Linear(context_dim, memory_dim, bias=False)
            if compatibility_mode == "cosine"
            else None
        )

    def forward(
        self, memory: Tensor, query_context: Tensor, information_gain: Tensor
    ) -> GateOutput:
        """Compute `alpha=(1-exp(-I))*sigmoid(f(memory,context,I))`."""
        information_gain = information_gain.float()
        base = 1.0 - torch.exp(-information_gain.clamp_min(0))
        learned = torch.sigmoid(
            self.network(
                torch.cat(
                    [
                        memory.float(),
                        query_context.float(),
                        information_gain.unsqueeze(-1),
                    ],
                    dim=-1,
                ).to(self.network[0].weight.dtype)
            )
        ).squeeze(-1)
        compatibility = torch.ones_like(learned)
        if self.query_to_memory is not None:
            projected_query = self.query_to_memory(query_context.float())
            cosine = torch.nn.functional.cosine_similarity(
                memory.float(), projected_query.float(), dim=-1
            )
            # Values below orthogonality are treated as unsupported rather
            # than allowing a user posterior to rewrite an unrelated app.
            compatibility = cosine.clamp_min(0.0)
        context_confidence = learned * compatibility
        return GateOutput(
            alpha=base * context_confidence,
            information_confidence=base,
            context_confidence=context_confidence,
        )
