"""Runtime contract for the candidate-free Generative SAIL line."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


FORBIDDEN_INTENT_CANDIDATE_KEYS = frozenset(
    {
        "candidate_intents",
        "candidate_logits",
        "candidate_pool",
        "candidate_scores",
        "query_candidates",
        "target_candidate_index",
    }
)


def assert_candidate_free_batch(batch: Mapping[str, Any]) -> None:
    """Reject intent-candidate tensors while allowing history retrieval metadata."""
    forbidden = sorted(FORBIDDEN_INTENT_CANDIDATE_KEYS.intersection(batch))
    if forbidden:
        raise ValueError(
            "Generative SAIL does not accept intent candidates: "
            + ", ".join(forbidden)
        )
