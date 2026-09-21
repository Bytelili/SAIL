"""Population intent model interface over a Qwen2-VL causal LM."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn

from .representation_pooling import masked_mean_pool, pool_last_non_padding_token


class PopulationModel(nn.Module):
    """Expose generation, teacher forcing, and frozen representation extraction."""

    def __init__(self, base_model: nn.Module, pooling: str = "last_non_padding") -> None:
        super().__init__()
        if pooling not in {"last_non_padding", "masked_mean"}:
            raise ValueError(f"Unsupported representation pooling: {pooling}")
        self.base_model = base_model
        self.pooling = pooling

    def generate(self, **kwargs: Any) -> Tensor:
        """Delegate autoregressive generation to the population model."""
        return self.base_model.generate(**kwargs)

    def forward_teacher_forced(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        labels: Tensor | None = None,
        **model_inputs: Tensor,
    ) -> Any:
        """Return population logits and hidden states under teacher forcing."""
        return self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            output_hidden_states=True,
            return_dict=True,
            use_cache=False,
            **model_inputs,
        )

    def forward_logits_only(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        **model_inputs: Tensor,
    ) -> Any:
        """Return raw frozen Population logits without retaining every hidden layer."""
        return self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=False,
            return_dict=True,
            use_cache=False,
            **model_inputs,
        )

    def _native_qwen2vl_final_hidden(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        **model_inputs: Tensor,
    ) -> Tensor | None:
        """Return the native Qwen2-VL final state without LM logits/layer history."""

        native_backbone = getattr(self.base_model, "model", None)
        model_type = getattr(
            getattr(self.base_model, "config", None), "model_type", None
        )
        if not isinstance(native_backbone, nn.Module) or model_type != "qwen2_vl":
            return None
        outputs = native_backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=False,
            return_dict=True,
            use_cache=False,
            **model_inputs,
        )
        hidden = getattr(outputs, "last_hidden_state", None)
        return outputs[0] if hidden is None else hidden

    def forward_selected_teacher_forced(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        labels: Tensor,
        **model_inputs: Tensor,
    ) -> dict[str, Tensor] | None:
        """Compute only supervised Qwen2-VL logits, preserving exact LM semantics.

        Action-level training supervises a short assistant suffix.  Materializing
        ``[batch, sequence, vocabulary]`` logits for the image/prompt prefix is
        unnecessary and was the dominant GPU-memory allocation.  This native
        path runs the unchanged checkpoint backbone, gathers the same causal
        prediction states as ``gather_target_positions``, and applies the
        checkpoint's unchanged LM head only to those states.  Non-Qwen models
        return ``None`` and retain the general legacy path.
        """

        hidden = self._native_qwen2vl_final_hidden(
            input_ids,
            attention_mask,
            **model_inputs,
        )
        if hidden is None:
            return None
        if hidden.shape[:2] != labels.shape:
            raise ValueError("native hidden states and labels must align on [B,L]")
        target_mask = labels[:, 1:] != -100
        batch_indices, time_indices = target_mask.nonzero(as_tuple=True)
        if batch_indices.numel() == 0:
            raise ValueError("No supervised target positions were found")
        selected_hidden = hidden[batch_indices, time_indices]
        output_embeddings = self.base_model.get_output_embeddings()
        if output_embeddings is None:
            raise RuntimeError("Qwen2-VL does not expose its LM output head")
        selected_logits = output_embeddings(selected_hidden)
        return {
            "base_logits": selected_logits,
            "selected_hidden": selected_hidden,
            "full_hidden": hidden,
            "target_ids": labels[batch_indices, time_indices + 1],
            "batch_indices": batch_indices,
            "time_indices": time_indices,
        }

    def _encode(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        **model_inputs: Tensor,
    ) -> Tensor:
        with torch.no_grad():
            # Qwen2-VL's causal-LM wrapper always materializes full-vocabulary
            # logits and, when requested, every layer's hidden state.  Neither
            # is needed for representation caching.  Calling the checkpoint's
            # native multimodal backbone produces the identical final hidden
            # state while avoiding both large allocations.  Keep the legacy
            # path for small test doubles and non-Qwen wrappers that do not
            # expose a native backbone.
            hidden = self._native_qwen2vl_final_hidden(
                input_ids,
                attention_mask,
                **model_inputs,
            )
            if hidden is None:
                outputs = self.forward_teacher_forced(
                    input_ids,
                    attention_mask,
                    **model_inputs,
                )
                hidden = outputs.hidden_states[-1]
            if self.pooling == "last_non_padding":
                return pool_last_non_padding_token(hidden, attention_mask)
            return masked_mean_pool(hidden, attention_mask)

    def encode_context(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        **model_inputs: Tensor,
    ) -> Tensor:
        """Encode profile, time, scenario, and current_app without intent."""
        return self._encode(input_ids, attention_mask, **model_inputs)

    def encode_observed_interaction(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        **model_inputs: Tensor,
    ) -> Tensor:
        """Encode a completed support interaction including observed intent."""
        return self._encode(input_ids, attention_mask, **model_inputs)

    def freeze_for_cusp(self) -> None:
        """Freeze every population parameter and switch to evaluation mode."""
        self.eval()
        for parameter in self.parameters():
            parameter.requires_grad_(False)

    @property
    def hidden_size(self) -> int:
        """Read hidden size from the loaded model config."""
        config = getattr(self.base_model, "config", None)
        for name in ("hidden_size", "text_config"):
            value = getattr(config, name, None)
            if name == "text_config" and value is not None:
                value = getattr(value, "hidden_size", None)
            if isinstance(value, int):
                return value
        raise AttributeError("Unable to resolve population hidden_size from model config")

    def output_embedding_weight(self) -> Tensor:
        """Return the frozen LM output embedding used by residual logits."""
        output_embeddings = self.base_model.get_output_embeddings()
        if output_embeddings is None or not hasattr(output_embeddings, "weight"):
            raise RuntimeError("Population model does not expose output embedding weights")
        return output_embeddings.weight
