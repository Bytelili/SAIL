"""Low-rank personalized hidden/logit residual without a new vocabulary matrix."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass
class PersonalizationOutput:
    """Personalized residual values for selected token positions."""

    residual_logits: Tensor
    delta_hidden: Tensor
    token_gate: Tensor


class PersonalizationResidualHead(nn.Module):
    """Reuse frozen LM output embeddings to construct vocabulary residual logits."""

    def __init__(
        self, hidden_size: int, memory_dim: int, residual_rank: int
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.residual_rank = residual_rank
        self.query_projection = nn.Linear(hidden_size, residual_rank)
        self.memory_projection = nn.Linear(memory_dim, residual_rank)
        self.gate_projection = nn.Linear(residual_rank * 3, residual_rank)
        self.output_projection = nn.Linear(residual_rank, hidden_size)

    def forward(
        self,
        query_hidden: Tensor,
        memory: Tensor,
        output_embedding_weight: Tensor,
    ) -> PersonalizationOutput:
        """Compute low-rank token residuals and project with frozen LM embeddings."""
        if query_hidden.ndim not in {2, 3}:
            raise ValueError("query_hidden must have shape [N,H] or [B,T,H]")
        squeeze = query_hidden.ndim == 2
        if squeeze:
            query_hidden = query_hidden.unsqueeze(1)
        if memory.ndim != 2 or memory.shape[0] != query_hidden.shape[0]:
            raise ValueError("memory must have shape [B,Dm] aligned with query_hidden")
        if output_embedding_weight.ndim != 2 or output_embedding_weight.shape[1] != self.hidden_size:
            raise ValueError("output_embedding_weight must have shape [V,H]")
        query_hidden = query_hidden.to(self.query_projection.weight.dtype)
        memory = memory.to(self.memory_projection.weight.dtype)
        query_rank = torch.tanh(self.query_projection(query_hidden))
        memory_rank = torch.tanh(self.memory_projection(memory)).unsqueeze(1)
        memory_rank = memory_rank.expand(-1, query_hidden.shape[1], -1)
        interaction = query_rank * memory_rank
        token_gate = torch.sigmoid(
            self.gate_projection(
                torch.cat([interaction, query_rank, memory_rank], dim=-1)
            )
        )
        delta_hidden = self.output_projection(token_gate * interaction)
        frozen_embedding = output_embedding_weight.detach()
        residual_logits = (
            delta_hidden.to(frozen_embedding.dtype)
            @ frozen_embedding.transpose(0, 1)
        ) / math.sqrt(self.hidden_size)
        if squeeze:
            residual_logits = residual_logits.squeeze(1)
            delta_hidden = delta_hidden.squeeze(1)
            token_gate = token_gate.squeeze(1)
        return PersonalizationOutput(residual_logits, delta_hidden, token_gate)
