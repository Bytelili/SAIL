"""Padding-aware representation pooling."""

from __future__ import annotations

import torch
from torch import Tensor


def _validate(hidden_states: Tensor, attention_mask: Tensor) -> None:
    if hidden_states.ndim != 3:
        raise ValueError("hidden_states must have shape [batch, sequence, hidden]")
    if attention_mask.ndim != 2 or attention_mask.shape != hidden_states.shape[:2]:
        raise ValueError("attention_mask must have shape [batch, sequence]")
    if hidden_states.shape[1] == 0:
        raise ValueError("Cannot pool an empty sequence")
    if torch.any(attention_mask.sum(dim=1) == 0):
        raise ValueError("Every sequence must contain at least one non-padding token")


def pool_last_non_padding_token(hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    """Return the hidden state at the greatest non-padding index per row."""
    _validate(hidden_states, attention_mask)
    positions = torch.arange(
        attention_mask.shape[1], device=attention_mask.device
    ).unsqueeze(0)
    last_indices = positions.masked_fill(attention_mask == 0, -1).max(dim=1).values
    batch_indices = torch.arange(hidden_states.shape[0], device=hidden_states.device)
    return hidden_states[batch_indices, last_indices.to(hidden_states.device)]


def masked_mean_pool(hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    """Mean-pool non-padding hidden states."""
    _validate(hidden_states, attention_mask)
    weights = attention_mask.to(hidden_states.dtype).unsqueeze(-1)
    return (hidden_states * weights).sum(dim=1) / weights.sum(dim=1)
