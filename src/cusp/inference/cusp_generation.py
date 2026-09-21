"""Greedy autoregressive CUSP decoding with per-step personalized residuals."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch
from torch import Tensor

from cusp.models.cusp_model import CUSPModel
from cusp.models.representation_pooling import pool_last_non_padding_token
from cusp.models.posterior import PosteriorState


@dataclass(frozen=True)
class CachedPosteriorMemory:
    """Reusable query-independent state for one user/support-prefix pair."""

    posterior: PosteriorState
    memory: Tensor


@dataclass(frozen=True)
class CUSPGenerationOutput:
    """Generated tokens plus compact, CPU-resident personalization diagnostics."""

    tokens: Tensor
    gate_alpha: Tensor
    information_confidence: Tensor
    context_confidence: Tensor
    information_gain: Tensor
    evidence_count: Tensor
    posterior_precision_mean: Tensor
    posterior_variance_mean: Tensor
    residual_scale: float


class PosteriorRuntimeCache:
    """In-memory cache keyed by (user_id, S, CUSP checkpoint hash)."""

    def __init__(self) -> None:
        self._values: dict[tuple[object, ...], CachedPosteriorMemory] = {}
        self.hits = 0
        self.misses = 0

    def get_or_compute(
        self,
        key: tuple[object, ...],
        factory: Callable[[], CachedPosteriorMemory],
    ) -> CachedPosteriorMemory:
        if key in self._values:
            self.hits += 1
            return self._values[key]
        value = factory()
        self._values[key] = value
        self.misses += 1
        return value

    @property
    def unique_user_s_pairs(self) -> int:
        return len(self._values)


def compute_posterior_memory(
    model: CUSPModel,
    support_context_hidden: Tensor,
    support_observed_hidden: Tensor,
    support_mask: Tensor,
    support_population_candidate_hidden: Tensor | None = None,
    support_population_candidate_log_probability: Tensor | None = None,
    support_population_candidate_mask: Tensor | None = None,
) -> CachedPosteriorMemory:
    """Compute query-independent posterior and memory once."""
    population_reference = support_context_hidden
    if model.population_reference_mode == "candidates" and torch.any(support_mask):
        if (
            support_population_candidate_hidden is None
            or support_population_candidate_log_probability is None
            or support_population_candidate_mask is None
        ):
            raise ValueError(
                "Candidate Population reference mode requires candidate tensors"
            )
        population_reference = model.population_reference(
            support_population_candidate_hidden,
            support_population_candidate_log_probability,
            support_observed_hidden,
            support_population_candidate_mask,
            support_mask,
        ).reference
    evidence_context = (
        model.counterfactual_context_scale * population_reference
        if model.evidence_mode == "counterfactual"
        else torch.zeros_like(support_context_hidden)
    )
    features = model.residual_evidence(
        evidence_context, support_observed_hidden, support_mask
    )
    if model.precision_mode == "learned":
        precision = (
            model.structured_evidence_precision(
                support_context_hidden,
                support_observed_hidden,
                population_reference,
                features.raw_residual,
                support_mask,
            )
            if model.precision_feature_mode == "structured"
            else model.evidence_precision(features.normalized_residual, support_mask)
        )
    else:
        precision = support_mask.unsqueeze(-1).to(
            features.mean.dtype
        ).expand_as(features.mean)
    observability = model.evidence_observability(
        support_context_hidden, support_mask
    )
    posterior = (
        model.posterior(
            features.mean,
            precision,
            support_mask,
            observability=observability,
        )
        if model.posterior_mode == "gaussian"
        else model._mean_posterior(features.mean, support_mask)
    )
    return CachedPosteriorMemory(
        posterior=posterior,
        memory=model.memory_projector(posterior),
    )


def cusp_greedy_generate(
    model: CUSPModel,
    input_ids: Tensor,
    attention_mask: Tensor,
    support_context_hidden: Tensor,
    support_observed_hidden: Tensor,
    support_mask: Tensor,
    *,
    support_population_candidate_hidden: Tensor | None = None,
    support_population_candidate_log_probability: Tensor | None = None,
    support_population_candidate_mask: Tensor | None = None,
    max_new_tokens: int,
    eos_token_id: int,
    do_sample: bool = False,
    num_beams: int = 1,
    cached_posterior_memory: CachedPosteriorMemory | None = None,
    query_model_inputs: dict[str, Tensor] | None = None,
    return_diagnostics: bool = False,
) -> Tensor | CUSPGenerationOutput:
    """Decode CUSP greedily; sampling and beam search are intentionally unsupported."""
    if do_sample:
        raise ValueError("CUSP generation does not support sampling")
    if num_beams != 1:
        raise ValueError("CUSP generation does not support beam search")
    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be positive")
    model.eval()
    with torch.inference_mode():
        cached = cached_posterior_memory or compute_posterior_memory(
            model,
            support_context_hidden,
            support_observed_hidden,
            support_mask,
            support_population_candidate_hidden,
            support_population_candidate_log_probability,
            support_population_candidate_mask,
        )
        posterior, memory = cached.posterior, cached.memory

        base_model = model.population_model.base_model
        generation_kwargs = {
            "attention_mask": attention_mask,
            "use_cache": True,
            **(query_model_inputs or {}),
        }
        uses_generation_helpers = all(
            hasattr(base_model, name)
            for name in (
                "prepare_inputs_for_generation",
                "_get_initial_cache_position",
                "_update_model_kwargs_for_generation",
            )
        )
        if uses_generation_helpers:
            generation_kwargs = base_model._get_initial_cache_position(
                input_ids.shape[1],
                input_ids.device,
                generation_kwargs,
            )
            initial_inputs = base_model.prepare_inputs_for_generation(
                input_ids,
                **generation_kwargs,
            )
            initial = base_model(
                **initial_inputs,
                output_hidden_states=True,
                return_dict=True,
            )
        else:
            initial = base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True,
                use_cache=True,
                past_key_values=None,
                **(query_model_inputs or {}),
            )
        query_context = pool_last_non_padding_token(
            initial.hidden_states[-1], attention_mask
        )
        conditioned_memory, _ = model.condition_memory(memory, query_context)
        gate = model.compute_gate(conditioned_memory, query_context, posterior)
        gate = model.safety_calibrate_gate(gate)
        outputs = initial
        current_mask = attention_mask
        generated_sequence = input_ids
        if uses_generation_helpers:
            generation_kwargs = base_model._update_model_kwargs_for_generation(
                initial,
                generation_kwargs,
                is_encoder_decoder=False,
            )
        generated: list[Tensor] = []
        finished = torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)
        for step in range(max_new_tokens):
            if step == 0:
                positions = torch.arange(
                    attention_mask.shape[1], device=attention_mask.device
                ).unsqueeze(0)
                last = positions.masked_fill(attention_mask == 0, -1).max(dim=1).values
                batch = torch.arange(input_ids.shape[0], device=input_ids.device)
                base_logits = outputs.logits[batch, last]
                last_hidden = outputs.hidden_states[-1][batch, last]
            else:
                base_logits = outputs.logits[:, -1]
                last_hidden = outputs.hidden_states[-1][:, -1]
            residual = model.personalization_head(
                last_hidden,
                conditioned_memory,
                model.population_model.output_embedding_weight(),
            ).residual_logits
            residual = model.calibrate_residual_logits(residual)
            final_logits = (
                base_logits
                + model.residual_scale.to(base_logits.dtype)
                * gate.alpha.unsqueeze(-1).to(base_logits.dtype)
                * residual.to(base_logits.dtype)
            )
            next_token = final_logits.argmax(dim=-1)
            generated.append(next_token)
            generated_sequence = torch.cat(
                [generated_sequence, next_token.unsqueeze(1)], dim=1
            )
            finished |= next_token == eos_token_id
            if torch.all(finished) or step + 1 == max_new_tokens:
                break
            if uses_generation_helpers:
                model_inputs = (
                    base_model.prepare_inputs_for_generation(
                        generated_sequence,
                        **generation_kwargs,
                    )
                )
                outputs = base_model(
                    **model_inputs,
                    output_hidden_states=True,
                    return_dict=True,
                )
                generation_kwargs = base_model._update_model_kwargs_for_generation(
                    outputs,
                    generation_kwargs,
                    is_encoder_decoder=False,
                )
            else:
                current_mask = torch.cat(
                    [
                        current_mask,
                        torch.ones(
                            (current_mask.shape[0], 1),
                            dtype=current_mask.dtype,
                            device=current_mask.device,
                        ),
                    ],
                    dim=1,
                )
                outputs = base_model(
                    input_ids=next_token.unsqueeze(1),
                    attention_mask=current_mask,
                    output_hidden_states=True,
                    return_dict=True,
                    use_cache=True,
                    past_key_values=outputs.past_key_values,
                )
        tokens = torch.stack(generated, dim=1)
        if not return_diagnostics:
            return tokens
        return CUSPGenerationOutput(
            tokens=tokens,
            gate_alpha=gate.alpha.detach().float().cpu(),
            information_confidence=(
                gate.information_confidence.detach().float().cpu()
            ),
            context_confidence=gate.context_confidence.detach().float().cpu(),
            information_gain=posterior.information_gain.detach().float().cpu(),
            evidence_count=posterior.evidence_count.detach().cpu(),
            posterior_precision_mean=posterior.precision.detach().float().mean(-1).cpu(),
            posterior_variance_mean=posterior.variance.detach().float().mean(-1).cpu(),
            residual_scale=float(model.residual_scale.detach().float().cpu()),
        )
