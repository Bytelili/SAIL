"""Explicit candidate-free facade over the direct-logit CUSP model."""

from __future__ import annotations

from typing import Any

from torch import Tensor

from cusp.models.cusp_model import CUSPModel, CUSPOutput

from .contract import assert_candidate_free_batch


class GenerativeCUSPModel(CUSPModel):
    """CUSP that personalizes vocabulary logits and generates autoregressively.

    The Population model remains frozen.  Historical interactions produce
    counterfactual residual evidence, the evidence forms a user posterior, and
    the posterior memory creates a residual over the full LM vocabulary.  No
    final-intent candidate set, classifier, or reranker is part of this graph.
    """

    method_family = "candidate_free_autoregressive_generation"

    def forward_candidate_free(
        self,
        *,
        query_input_ids: Tensor,
        query_attention_mask: Tensor,
        query_labels: Tensor,
        support_context_hidden: Tensor,
        support_observed_hidden: Tensor,
        support_mask: Tensor,
        support_population_candidate_hidden: Tensor | None = None,
        support_population_candidate_log_probability: Tensor | None = None,
        support_population_candidate_mask: Tensor | None = None,
        negative_support: dict[str, Tensor] | None = None,
        **unexpected: Any,
    ) -> CUSPOutput:
        assert_candidate_free_batch(unexpected)
        if unexpected:
            raise TypeError(
                "Unexpected Generative CUSP inputs: "
                + ", ".join(sorted(unexpected))
            )
        return super().forward(
            query_input_ids=query_input_ids,
            query_attention_mask=query_attention_mask,
            query_labels=query_labels,
            support_context_hidden=support_context_hidden,
            support_observed_hidden=support_observed_hidden,
            support_mask=support_mask,
            support_population_candidate_hidden=(
                support_population_candidate_hidden
            ),
            support_population_candidate_log_probability=(
                support_population_candidate_log_probability
            ),
            support_population_candidate_mask=support_population_candidate_mask,
            negative_support=negative_support,
        )
