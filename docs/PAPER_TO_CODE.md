# Paper-to-code map

| Paper object | Public implementation |
|---|---|
| Frozen Population model | models/population_model.py |
| Compatible Population reference | models/population_reference.py |
| Residual evidence | models/residual_evidence.py |
| Dimension-wise observability | models/observability.py |
| Bounded evidence precision | models/evidence_precision.py and models/structured_precision.py |
| Diagonal Gaussian posterior | models/posterior.py |
| Posterior memory tokens | models/memory_projector.py |
| Uncertainty-aware gate | models/uncertainty_gate.py |
| Low-rank residual logits | models/personalization_head.py |
| Integrated SAIL/CUSP graph | models/cusp_model.py |
| Direct autoregressive decoding | inference/cusp_generation.py |
| Learning objectives | training/losses.py |

For a batch of historical evidence, the public core represents each interaction
with a mean and diagonal precision. The posterior adds natural parameters across
valid history items, projects the resulting mean and log-variance into memory
tokens, and conditions a bounded residual correction on the current query.

The final response is generated autoregressively. Candidate tensors are accepted
only for estimating the frozen Population reference; the public generation
contract rejects final-answer candidate pools and reranking fields.
