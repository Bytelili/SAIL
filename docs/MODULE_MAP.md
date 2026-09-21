# SAIL module map

| Method component | Implementation |
|---|---|
| Frozen Population model | `src/sail/models/population_model.py` |
| Compatible Population reference | `src/sail/models/population_reference.py` |
| Residual evidence | `src/sail/models/residual_evidence.py` |
| Dimension-wise observability | `src/sail/models/observability.py` |
| Bounded evidence precision | `src/sail/models/evidence_precision.py`, `src/sail/models/structured_precision.py` |
| Diagonal Gaussian posterior | `src/sail/models/posterior.py` |
| Posterior memory tokens | `src/sail/models/memory_projector.py` |
| Query-conditioned gate | `src/sail/models/uncertainty_gate.py` |
| Low-rank residual logits | `src/sail/models/personalization_head.py` |
| Integrated SAIL model | `src/sail/models/sail_model.py` |
| Direct autoregressive decoding | `src/sail/inference/sail_generation.py` |
| Learning objectives | `src/sail/training/losses.py` |

For historical evidence, each interaction contributes a mean and diagonal
precision. The posterior sums natural parameters across valid interactions,
projects its mean and log-variance into memory tokens, and conditions a bounded
residual correction on the current query. Candidate tensors estimate the
Population reference; final responses are generated autoregressively without
candidate reranking.
