"""Target-position, KL, and user-swap losses."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor

from cusp.models.posterior import PosteriorState


@dataclass
class GatheredTargets:
    """Flattened prediction positions corresponding only to supervised target tokens."""

    base_logits: Tensor
    hidden: Tensor
    target_ids: Tensor
    batch_indices: Tensor
    time_indices: Tensor


def gather_target_positions(
    logits: Tensor, hidden_states: Tensor, labels: Tensor
) -> GatheredTargets:
    """Gather causal prediction positions whose next-token label is supervised."""
    if logits.shape[:2] != labels.shape or hidden_states.shape[:2] != labels.shape:
        raise ValueError("logits, hidden_states, and labels must align on [B,L]")
    target_mask = labels[:, 1:] != -100
    batch_indices, time_indices = target_mask.nonzero(as_tuple=True)
    if batch_indices.numel() == 0:
        raise ValueError("No supervised target positions were found")
    return GatheredTargets(
        base_logits=logits[batch_indices, time_indices],
        hidden=hidden_states[batch_indices, time_indices],
        target_ids=labels[batch_indices, time_indices + 1],
        batch_indices=batch_indices,
        time_indices=time_indices,
    )


def causal_lm_target_loss(logits: Tensor, labels: Tensor) -> Tensor:
    """Compute shifted causal LM loss while ignoring prompt and padding labels."""
    return F.cross_entropy(
        logits[:, :-1].reshape(-1, logits.shape[-1]).float(),
        labels[:, 1:].reshape(-1),
        ignore_index=-100,
    )


def diagonal_gaussian_kl(posterior: PosteriorState) -> Tensor:
    """KL to N(0,I), averaged by latent size and support-bearing samples."""
    per_sample = 0.5 * (
        posterior.mean.square()
        + posterior.variance
        - posterior.log_variance
        - 1.0
    ).mean(dim=-1)
    valid = posterior.evidence_count > 0
    if not torch.any(valid):
        return per_sample.sum() * 0.0
    return per_sample[valid].mean()


def symmetric_diagonal_gaussian_kl(
    first: PosteriorState,
    second: PosteriorState,
    valid: Tensor,
) -> Tensor:
    """Symmetric KL between two diagonal user-state posteriors."""
    if first.mean.shape != second.mean.shape:
        raise ValueError("Posterior means must share shape")
    if valid.shape != first.mean.shape[:1]:
        raise ValueError("valid must have shape [B]")
    first_to_second = 0.5 * (
        second.log_variance
        - first.log_variance
        + (first.variance + (first.mean - second.mean).square())
        / second.variance.clamp_min(1e-8)
        - 1.0
    ).mean(dim=-1)
    second_to_first = 0.5 * (
        first.log_variance
        - second.log_variance
        + (second.variance + (second.mean - first.mean).square())
        / first.variance.clamp_min(1e-8)
        - 1.0
    ).mean(dim=-1)
    symmetric = 0.5 * (first_to_second + second_to_first)
    if not torch.any(valid):
        return symmetric.sum() * 0.0
    return symmetric[valid].mean()


def per_example_target_log_probability(
    selected_logits: Tensor,
    target_ids: Tensor,
    batch_indices: Tensor,
    batch_size: int,
) -> Tensor:
    """Average selected-token log probability within each batch example."""
    token_scores = F.log_softmax(selected_logits.float(), dim=-1).gather(
        1, target_ids.unsqueeze(1)
    ).squeeze(1)
    sums = torch.zeros(batch_size, dtype=token_scores.dtype, device=token_scores.device)
    counts = torch.zeros_like(sums)
    sums.scatter_add_(0, batch_indices, token_scores)
    counts.scatter_add_(0, batch_indices, torch.ones_like(token_scores))
    return sums / counts.clamp_min(1)


def swap_margin_loss(
    same_user_scores: Tensor, swapped_user_scores: Tensor, margin: float
) -> Tensor:
    """Hinge loss requiring same-user state to outperform a swapped user."""
    if same_user_scores.shape != swapped_user_scores.shape:
        raise ValueError("Same and swapped scores must share shape")
    return torch.relu(margin - (same_user_scores - swapped_user_scores)).mean()


def directional_target_margin_loss(
    final_logits: Tensor,
    base_logits: Tensor,
    target_ids: Tensor,
    margin: float,
) -> Tensor:
    """Make personalization beat the frozen model's top-1 token when it is wrong."""
    base_top1 = base_logits.detach().argmax(dim=-1)
    active = base_top1 != target_ids
    if not torch.any(active):
        return final_logits.sum() * 0.0
    rows = torch.arange(final_logits.shape[0], device=final_logits.device)
    target_score = final_logits[rows, target_ids]
    competitor_score = final_logits[rows, base_top1]
    return torch.relu(margin - (target_score - competitor_score))[active].mean()


def preserve_correct_margin_loss(
    final_logits: Tensor,
    base_logits: Tensor,
    target_ids: Tensor,
    margin_cap: float,
) -> Tensor:
    """Prevent personalization from erasing a correct frozen-model decision."""
    if margin_cap <= 0:
        return final_logits.sum() * 0.0
    base_top2 = base_logits.detach().topk(k=2, dim=-1)
    base_top1 = base_top2.indices[:, 0]
    active = base_top1 == target_ids
    if not torch.any(active):
        return final_logits.sum() * 0.0
    runner_up = base_top2.indices[:, 1]
    rows = torch.arange(final_logits.shape[0], device=final_logits.device)
    final_margin = (
        final_logits[rows, target_ids] - final_logits[rows, runner_up]
    )
    base_margin = (
        base_top2.values[:, 0] - base_top2.values[:, 1]
    ).clamp(max=margin_cap)
    return torch.relu(base_margin - final_margin)[active].mean()
