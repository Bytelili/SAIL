"""Complete differentiable SAIL forward graph over a frozen Population Model."""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from sail.training.losses import (
    GatheredTargets,
    diagonal_gaussian_kl,
    directional_target_margin_loss,
    gather_target_positions,
    per_example_target_log_probability,
    preserve_correct_margin_loss,
    symmetric_diagonal_gaussian_kl,
    swap_margin_loss,
)

from .evidence_precision import EvidencePrecisionNetwork
from .memory_projector import PosteriorMemoryProjector
from .observability import EvidenceObservabilityNetwork
from .personalization_head import PersonalizationResidualHead
from .population_model import PopulationModel
from .posterior import DiagonalGaussianPosterior, PosteriorState
from .population_reference import CompatiblePopulationReference
from .residual_evidence import ResidualEvidenceEncoder, ResidualEvidenceFeatures
from .structured_precision import StructuredEvidencePrecisionNetwork
from .uncertainty_gate import GateOutput, UncertaintyAwareGate


@dataclass
class EvidenceOutput:
    """Mean, precision, and original counterfactual residual."""

    mean: Tensor
    precision: Tensor
    observability: Tensor
    raw_residual: Tensor


@dataclass
class SAILOutput:
    """Losses and diagnostics from a target-position SAIL forward pass."""

    loss: Tensor
    query_loss: Tensor
    kl_loss: Tensor
    evidence_loss: Tensor
    consistency_loss: Tensor
    swap_loss: Tensor
    precision_regularization: Tensor
    directional_loss: Tensor
    stability_loss: Tensor
    final_logits: Tensor
    base_logits: Tensor
    posterior: PosteriorState
    gate: GateOutput
    dimension_gate: Tensor
    evidence: EvidenceOutput
    swap_skipped: bool
    same_user_scores: Tensor | None
    swapped_user_scores: Tensor | None
    top1_flip_rate: Tensor
    beneficial_flip_rate: Tensor
    harmful_flip_rate: Tensor


class SAILModel(nn.Module):
    """Counterfactual User-State Posterior Memory with frozen population logits."""

    def __init__(
        self,
        population_model: PopulationModel,
        *,
        latent_dim: int = 256,
        memory_dim: int = 256,
        residual_rank: int = 256,
        min_precision: float = 0.001,
        max_precision: float = 100.0,
        initial_precision: float = 0.05,
        prior_precision: float = 1.0,
        query_weight: float = 1.0,
        kl_weight: float = 0.0,
        swap_weight: float = 0.0,
        precision_weight: float = 0.0,
        evidence_weight: float = 0.1,
        consistency_weight: float = 0.0,
        swap_margin: float = 0.1,
        directional_weight: float = 1.0,
        directional_margin: float = 0.5,
        stability_weight: float = 0.0,
        stability_margin: float = 1.0,
        initial_residual_scale: float | None = None,
        residual_normalization: str = "rms",
        residual_target_rms: float = 1.0,
        evidence_mode: str = "counterfactual",
        counterfactual_context_scale: float = 1.0,
        precision_mode: str = "learned",
        posterior_mode: str = "gaussian",
        gate_mode: str = "information",
        max_gate_alpha: float = 1.0,
        compatibility_mode: str = "none",
        population_reference_mode: str = "candidates",
        population_probability_temperature: float = 1.0,
        population_compatibility_temperature: float = 1.0,
        precision_feature_mode: str = "structured",
        dimension_gate_mode: str = "learned",
    ) -> None:
        super().__init__()
        population_model.freeze_for_sail()
        self.population_model = population_model
        hidden_size = population_model.hidden_size
        self.residual_evidence = ResidualEvidenceEncoder(hidden_size, latent_dim)
        self.evidence_precision = EvidencePrecisionNetwork(
            hidden_size,
            latent_dim,
            min_precision,
            max_precision,
            initial_precision,
        )
        self.structured_evidence_precision = StructuredEvidencePrecisionNetwork(
            hidden_size,
            latent_dim,
            min_precision,
            max_precision,
            initial_precision,
        )
        self.population_reference = CompatiblePopulationReference(
            probability_temperature=population_probability_temperature,
            compatibility_temperature=population_compatibility_temperature,
        )
        self.evidence_observability = EvidenceObservabilityNetwork(
            hidden_size, latent_dim
        )
        self.posterior = DiagonalGaussianPosterior(latent_dim, prior_precision)
        self.memory_projector = PosteriorMemoryProjector(latent_dim, memory_dim)
        self.personalization_head = PersonalizationResidualHead(
            hidden_size, memory_dim, residual_rank
        )
        self.uncertainty_gate = UncertaintyAwareGate(
            memory_dim, hidden_size, compatibility_mode=compatibility_mode
        )
        if dimension_gate_mode not in {"none", "learned"}:
            raise ValueError("dimension_gate_mode must be none or learned")
        self.dimension_gate_mode = dimension_gate_mode
        self.memory_dimension_gate = (
            nn.Sequential(
                nn.Linear(memory_dim + hidden_size, memory_dim),
                nn.Sigmoid(),
            )
            if dimension_gate_mode == "learned"
            else None
        )
        if self.memory_dimension_gate is not None:
            nn.init.zeros_(self.memory_dimension_gate[0].weight)
            nn.init.constant_(self.memory_dimension_gate[0].bias, 2.0)
        # Low initial evidence precision already keeps early personalization small.
        # This scale preserves usable gradients without weakening the S=0 guarantee.
        if initial_residual_scale is None:
            residual_raw = -2.3
        else:
            if initial_residual_scale <= 0:
                raise ValueError("initial_residual_scale must be positive")
            residual_raw = math.log(math.expm1(initial_residual_scale))
        self.residual_scale_raw = nn.Parameter(torch.tensor(residual_raw))
        self.query_weight = query_weight
        self.kl_weight = kl_weight
        self.swap_weight = swap_weight
        self.precision_weight = precision_weight
        self.evidence_weight = evidence_weight
        self.consistency_weight = consistency_weight
        self.swap_margin = swap_margin
        self.directional_weight = directional_weight
        self.directional_margin = directional_margin
        self.stability_weight = stability_weight
        self.stability_margin = stability_margin
        if residual_normalization not in {"none", "rms"}:
            raise ValueError("residual_normalization must be none or rms")
        if residual_target_rms <= 0:
            raise ValueError("residual_target_rms must be positive")
        self.residual_normalization = residual_normalization
        self.residual_target_rms = residual_target_rms
        if evidence_mode not in {"counterfactual", "observed"}:
            raise ValueError("evidence_mode must be counterfactual or observed")
        if not 0.0 <= counterfactual_context_scale <= 1.0:
            raise ValueError("counterfactual_context_scale must lie in [0, 1]")
        if precision_mode not in {"learned", "unit"}:
            raise ValueError("precision_mode must be learned or unit")
        if posterior_mode not in {"gaussian", "mean"}:
            raise ValueError("posterior_mode must be gaussian or mean")
        if gate_mode not in {"learned", "information", "binary"}:
            raise ValueError("gate_mode must be learned, information, or binary")
        if not 0.0 < max_gate_alpha <= 1.0:
            raise ValueError("max_gate_alpha must lie in (0, 1]")
        self.evidence_mode = evidence_mode
        self.counterfactual_context_scale = float(counterfactual_context_scale)
        self.precision_mode = precision_mode
        self.posterior_mode = posterior_mode
        self.gate_mode = gate_mode
        self.compatibility_mode = compatibility_mode
        if population_reference_mode not in {"context", "candidates"}:
            raise ValueError(
                "population_reference_mode must be context or candidates"
            )
        if precision_feature_mode not in {"residual", "structured"}:
            raise ValueError(
                "precision_feature_mode must be residual or structured"
            )
        self.population_reference_mode = population_reference_mode
        self.precision_feature_mode = precision_feature_mode
        # Constructor defaults select the canonical SAIL execution
        # graph.  Alternative modules remain available for inspecting the
        # ablations, but inactive alternatives are not counted as trainable
        # parameters in the selected graph.
        if self.precision_mode != "learned":
            self.evidence_precision.requires_grad_(False)
            self.structured_evidence_precision.requires_grad_(False)
        elif self.precision_feature_mode == "structured":
            self.evidence_precision.requires_grad_(False)
        else:
            self.structured_evidence_precision.requires_grad_(False)
        if self.gate_mode != "learned":
            self.uncertainty_gate.requires_grad_(False)
        # A learned gate can otherwise become near-binary when the posterior
        # precision saturates.  This is a hard safety envelope, not a learned
        # rescaling: it keeps personalization a residual correction.
        self.max_gate_alpha = float(max_gate_alpha)

    def safety_calibrate_gate(self, gate: GateOutput) -> GateOutput:
        """Cap personalization strength while preserving gate diagnostics."""
        if self.max_gate_alpha >= 1.0:
            return gate
        alpha = gate.alpha.clamp(max=self.max_gate_alpha)
        return GateOutput(
            alpha=alpha,
            information_confidence=gate.information_confidence,
            context_confidence=gate.context_confidence,
        )

    @property
    def residual_scale(self) -> Tensor:
        """Positive, small-at-initialization residual scale."""
        return F.softplus(self.residual_scale_raw)

    def train(self, mode: bool = True) -> "SAILModel":
        """Keep the frozen population path in eval mode while training SAIL modules."""
        super().train(mode)
        self.population_model.eval()
        return self

    def move_sail_modules(self, device: torch.device | str) -> None:
        """Move only external trainable modules, preserving Population device_map."""
        for module in (
            self.residual_evidence,
            self.evidence_precision,
            self.structured_evidence_precision,
            self.population_reference,
            self.evidence_observability,
            self.posterior,
            self.memory_projector,
            self.personalization_head,
            self.uncertainty_gate,
        ):
            module.to(device)
        if self.memory_dimension_gate is not None:
            self.memory_dimension_gate.to(device)
        self.residual_scale_raw.data = self.residual_scale_raw.data.to(device)

    def _personalized_selected_logits(
        self,
        *,
        base_selected_logits: Tensor,
        selected_hidden: Tensor,
        batch_indices: Tensor,
        memory: Tensor,
        gate: GateOutput,
    ) -> Tensor:
        personalized = self.personalization_head(
            selected_hidden,
            memory[batch_indices],
            self.population_model.output_embedding_weight(),
        )
        residual_logits = self.calibrate_residual_logits(
            personalized.residual_logits
        )
        alpha = gate.alpha[batch_indices].unsqueeze(-1).to(base_selected_logits.dtype)
        return (
            base_selected_logits
            + self.residual_scale.to(base_selected_logits.dtype)
            * alpha
            * residual_logits.to(base_selected_logits.dtype)
        )

    def condition_memory(
        self,
        memory: Tensor,
        query_context: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Apply the query-conditioned gate over memory dimensions."""
        if self.memory_dimension_gate is None:
            gate = torch.ones_like(memory, dtype=torch.float32)
            return memory, gate
        gate = self.memory_dimension_gate(
            torch.cat([memory.float(), query_context.float()], dim=-1).to(
                self.memory_dimension_gate[0].weight.dtype
            )
        ).float()
        return memory * gate.to(memory.dtype), gate

    def calibrate_residual_logits(self, residual_logits: Tensor) -> Tensor:
        """Optionally give every token-position residual a controlled RMS."""
        if self.residual_normalization == "none":
            return residual_logits
        centered = residual_logits.float() - residual_logits.float().mean(
            dim=-1, keepdim=True
        )
        rms = centered.square().mean(dim=-1, keepdim=True).sqrt().clamp_min(1e-6)
        return (
            centered / rms * self.residual_target_rms
        ).to(residual_logits.dtype)

    def forward(
        self,
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
        query_model_inputs: dict[str, Tensor] | None = None,
    ) -> SAILOutput:
        """Run evidence, posterior, memory, gated target logits, and all losses."""
        population_reference = support_context_hidden
        has_any_support = bool(torch.any(support_mask))
        if self.population_reference_mode == "candidates" and has_any_support:
            if (
                support_population_candidate_hidden is None
                or support_population_candidate_log_probability is None
                or support_population_candidate_mask is None
            ):
                raise ValueError(
                    "Candidate Population reference mode requires candidate hidden "
                    "states, log probabilities, and masks"
                )
            population_reference = self.population_reference(
                support_population_candidate_hidden,
                support_population_candidate_log_probability,
                support_observed_hidden,
                support_population_candidate_mask,
                support_mask,
            ).reference
        evidence_context = (
            self.counterfactual_context_scale * population_reference
            if self.evidence_mode == "counterfactual"
            else torch.zeros_like(support_context_hidden)
        )
        features: ResidualEvidenceFeatures = self.residual_evidence(
            evidence_context, support_observed_hidden, support_mask
        )
        if self.precision_mode == "learned":
            precision = (
                self.structured_evidence_precision(
                    support_context_hidden,
                    support_observed_hidden,
                    population_reference,
                    features.raw_residual,
                    support_mask,
                )
                if self.precision_feature_mode == "structured"
                else self.evidence_precision(
                    features.normalized_residual, support_mask
                )
            )
        else:
            precision = support_mask.unsqueeze(-1).to(
                features.mean.dtype
            ).expand_as(features.mean)
        observability = self.evidence_observability(
            support_context_hidden, support_mask
        )
        posterior = (
            self.posterior(
                features.mean,
                precision,
                support_mask,
                observability=observability,
            )
            if self.posterior_mode == "gaussian"
            else self._mean_posterior(features.mean, support_mask)
        )
        latent = posterior.mean
        if self.training and self.posterior_mode == "gaussian":
            latent = posterior.mean + posterior.variance.sqrt() * torch.randn_like(
                posterior.mean
            )
        memory = self.memory_projector(replace(posterior, mean=latent))

        with torch.no_grad():
            native_selected = self.population_model.forward_selected_teacher_forced(
                query_input_ids,
                query_attention_mask,
                query_labels,
                **(query_model_inputs or {}),
            )
            if native_selected is None:
                population = self.population_model.forward_teacher_forced(
                    query_input_ids,
                    query_attention_mask,
                    **(query_model_inputs or {}),
                )
                base_full_logits = population.logits
                hidden = population.hidden_states[-1]
                selected = gather_target_positions(
                    base_full_logits, hidden, query_labels
                )
            else:
                hidden = native_selected["full_hidden"]
                selected = GatheredTargets(
                    base_logits=native_selected["base_logits"],
                    hidden=native_selected["selected_hidden"],
                    target_ids=native_selected["target_ids"],
                    batch_indices=native_selected["batch_indices"],
                    time_indices=native_selected["time_indices"],
                )
        query_context = _first_target_prediction_hidden(hidden, query_labels)
        conditioned_memory, dimension_gate = self.condition_memory(
            memory, query_context
        )
        gate = self.compute_gate(conditioned_memory, query_context, posterior)
        gate = self.safety_calibrate_gate(gate)
        final_logits = self._personalized_selected_logits(
            base_selected_logits=selected.base_logits,
            selected_hidden=selected.hidden,
            batch_indices=selected.batch_indices,
            memory=conditioned_memory,
            gate=gate,
        )
        query_loss = F.cross_entropy(final_logits.float(), selected.target_ids)
        directional_loss = directional_target_margin_loss(
            final_logits.float(),
            selected.base_logits.float(),
            selected.target_ids,
            self.directional_margin,
        )
        stability_loss = preserve_correct_margin_loss(
            final_logits.float(),
            selected.base_logits.float(),
            selected.target_ids,
            self.stability_margin,
        )
        kl_loss = diagonal_gaussian_kl(posterior)
        evidence_error = (
            features.mean.float()
            - observability.float() * latent.float().unsqueeze(1)
        )
        evidence_terms = 0.5 * (
            precision.float() * evidence_error.square()
            - precision.float().clamp_min(1e-8).log()
        )
        evidence_loss = (
            evidence_terms[support_mask].mean()
            if torch.any(support_mask)
            else evidence_terms.sum() * 0.0
        )
        consistency_loss = evidence_loss * 0.0
        if support_mask.shape[1] >= 2:
            positions = torch.arange(
                support_mask.shape[1], device=support_mask.device
            ).unsqueeze(0)
            offsets = (
                torch.randint(
                    0,
                    2,
                    (support_mask.shape[0], 1),
                    device=support_mask.device,
                )
                if self.training
                else torch.zeros(
                    (support_mask.shape[0], 1),
                    dtype=torch.long,
                    device=support_mask.device,
                )
            )
            first_mask = support_mask & ((positions + offsets) % 2 == 0)
            second_mask = support_mask & ~first_mask
            both_nonempty = first_mask.any(dim=1) & second_mask.any(dim=1)
            first_posterior = self.posterior(
                features.mean,
                precision,
                first_mask,
                observability=observability,
            )
            second_posterior = self.posterior(
                features.mean,
                precision,
                second_mask,
                observability=observability,
            )
            consistency_loss = symmetric_diagonal_gaussian_kl(
                first_posterior,
                second_posterior,
                both_nonempty,
            )
        valid_precision = precision[support_mask]
        precision_regularization = (
            valid_precision.log().square().mean()
            if valid_precision.numel()
            else precision.sum() * 0.0
        )

        swap_loss = query_loss * 0.0
        swap_skipped = True
        same_scores = None
        swapped_scores = None
        if negative_support is not None:
            negative_indices = negative_support["indices"].to(memory.device)
            if negative_indices.shape != (memory.shape[0],):
                raise ValueError("negative support indices must have shape [B]")
            if memory.shape[0] > 1:
                if torch.any(
                    negative_indices
                    == torch.arange(memory.shape[0], device=memory.device)
                ):
                    raise ValueError("Swap negatives must use a different user")
                swapped_memory = memory[negative_indices]
                swapped_memory, _ = self.condition_memory(
                    swapped_memory, query_context
                )
                swapped_information = posterior.information_gain[negative_indices]
                swapped_gate = self._gate_from_statistics(
                    swapped_memory,
                    query_context,
                    swapped_information,
                    posterior.evidence_count[negative_indices],
                )
                swapped_gate = self.safety_calibrate_gate(swapped_gate)
                swapped_logits = self._personalized_selected_logits(
                    base_selected_logits=selected.base_logits,
                    selected_hidden=selected.hidden,
                    batch_indices=selected.batch_indices,
                    memory=swapped_memory,
                    gate=swapped_gate,
                )
                same_scores = per_example_target_log_probability(
                    final_logits, selected.target_ids, selected.batch_indices, memory.shape[0]
                )
                swapped_scores = per_example_target_log_probability(
                    swapped_logits, selected.target_ids, selected.batch_indices, memory.shape[0]
                )
                swap_loss = swap_margin_loss(
                    same_scores, swapped_scores, self.swap_margin
                )
                swap_skipped = False

        loss = (
            self.query_weight * query_loss
            + self.kl_weight * kl_loss
            + self.swap_weight * swap_loss
            + self.precision_weight * precision_regularization
            + self.evidence_weight * evidence_loss
            + self.consistency_weight * consistency_loss
            + self.directional_weight * directional_loss
            + self.stability_weight * stability_loss
        )
        base_top1 = selected.base_logits.detach().argmax(dim=-1)
        final_top1 = final_logits.detach().argmax(dim=-1)
        top1_flip_rate = (base_top1 != final_top1).float().mean()
        base_wrong = base_top1 != selected.target_ids
        base_correct = ~base_wrong
        beneficial_flip_rate = (
            ((final_top1 == selected.target_ids) & base_wrong).float().sum()
            / base_wrong.float().sum().clamp_min(1.0)
        )
        harmful_flip_rate = (
            ((final_top1 != selected.target_ids) & base_correct).float().sum()
            / base_correct.float().sum().clamp_min(1.0)
        )
        if not torch.isfinite(loss):
            raise FloatingPointError("SAIL loss became NaN or Inf")
        return SAILOutput(
            loss=loss,
            query_loss=query_loss,
            kl_loss=kl_loss,
            evidence_loss=evidence_loss,
            consistency_loss=consistency_loss,
            swap_loss=swap_loss,
            precision_regularization=precision_regularization,
            directional_loss=directional_loss,
            stability_loss=stability_loss,
            final_logits=final_logits,
            base_logits=selected.base_logits,
            posterior=posterior,
            gate=gate,
            dimension_gate=dimension_gate,
            evidence=EvidenceOutput(
                features.mean,
                precision,
                observability,
                features.raw_residual,
            ),
            swap_skipped=swap_skipped,
            same_user_scores=same_scores,
            swapped_user_scores=swapped_scores,
            top1_flip_rate=top1_flip_rate,
            beneficial_flip_rate=beneficial_flip_rate,
            harmful_flip_rate=harmful_flip_rate,
        )

    def _mean_posterior(
        self, evidence_mean: Tensor, support_mask: Tensor
    ) -> PosteriorState:
        mask = support_mask.unsqueeze(-1).float()
        count = mask.sum(dim=1)
        mean = (evidence_mean.float() * mask).sum(dim=1) / count.clamp_min(1.0)
        mean = mean * (count > 0).float()
        precision = torch.full_like(mean, self.posterior.prior_precision)
        variance = precision.reciprocal()
        information = torch.log1p(count.squeeze(-1))
        return PosteriorState(
            mean=mean,
            variance=variance,
            log_variance=variance.log(),
            precision=precision,
            information_gain=information,
            evidence_count=support_mask.sum(dim=1).to(torch.int64),
        )

    @staticmethod
    def _binary_gate(memory: Tensor, posterior: PosteriorState) -> GateOutput:
        alpha = (posterior.evidence_count > 0).float()
        return GateOutput(
            alpha=alpha,
            information_confidence=alpha,
            context_confidence=torch.ones_like(alpha),
        )

    def compute_gate(
        self,
        memory: Tensor,
        query_context: Tensor,
        posterior: PosteriorState,
    ) -> GateOutput:
        """Compute the configured gate identically in training and generation."""
        return self._gate_from_statistics(
            memory,
            query_context,
            posterior.information_gain,
            posterior.evidence_count,
        )

    def _gate_from_statistics(
        self,
        memory: Tensor,
        query_context: Tensor,
        information_gain: Tensor,
        evidence_count: Tensor,
    ) -> GateOutput:
        if self.gate_mode == "learned":
            learned = self.uncertainty_gate(
                memory,
                query_context,
                information_gain,
            )
            # S=0 is the exact frozen population policy, not merely a learned
            # approximation to it. A learned gate may have a non-zero bias,
            # so enforce the no-evidence invariant explicitly in both training
            # and autoregressive generation.
            observed = (evidence_count > 0).to(learned.alpha.dtype)
            return GateOutput(
                alpha=learned.alpha * observed,
                information_confidence=learned.information_confidence * observed,
                context_confidence=learned.context_confidence,
            )
        if self.gate_mode == "information":
            confidence = 1.0 - torch.exp(-information_gain.float().clamp_min(0.0))
            confidence = confidence * (evidence_count > 0).float()
            return GateOutput(
                alpha=confidence,
                information_confidence=confidence,
                context_confidence=torch.ones_like(confidence),
            )
        alpha = (evidence_count > 0).float()
        return GateOutput(
            alpha=alpha,
            information_confidence=alpha,
            context_confidence=torch.ones_like(alpha),
        )


def _first_target_prediction_hidden(hidden: Tensor, labels: Tensor) -> Tensor:
    shifted_target_mask = labels[:, 1:] != -100
    if torch.any(shifted_target_mask.sum(dim=1) == 0):
        raise ValueError("Every query must contain at least one target token")
    first_prediction = shifted_target_mask.to(torch.int64).argmax(dim=1)
    batch = torch.arange(hidden.shape[0], device=hidden.device)
    return hidden[batch, first_prediction]
