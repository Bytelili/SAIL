# SAIL

SAIL personalizes a frozen Population language model from a short sequence of
user interactions. It estimates a Population reference for each observed
interaction, encodes the difference as user evidence, and combines that
evidence in a diagonal-Gaussian posterior. A query-conditioned gate then
applies a bounded residual correction during autoregressive generation.

## Components

- Population-reference aggregation over precomputed candidates
- Residual evidence encoding and dimension-wise observability
- Bounded evidence precision and online posterior updates
- Posterior-to-memory projection and query-conditioned gating
- Low-rank residual logits and direct autoregressive decoding
- Query, support, and directional learning objectives

Population candidates are used to estimate support-side evidence, not to
rerank final answers.

## Layout

- `src/sail/models/sail_model.py`: integrated SAIL model and objectives
- `src/sail/models/population_reference.py`: Population reference
- `src/sail/models/posterior.py`: user-state posterior
- `src/sail/models/structured_precision.py`: evidence reliability
- `src/sail/models/memory_projector.py`: memory tokens
- `src/sail/models/uncertainty_gate.py`: personalization gate
- `src/sail/models/personalization_head.py`: residual generation head
- `src/sail/inference/sail_generation.py`: autoregressive decoding
- `src/sail/training/losses.py`: learning objectives
- `configs/sail.yaml`: model defaults
- `docs/METHOD_DETAILS.md` and `docs/MODULE_MAP.md`: architecture and module map

## Interface

The package requires Python 3.10+ and PyTorch 2.1+. Install the modules with
`python -m pip install -e .`. A frozen Population model and precomputed
candidate tensors are supplied by the caller. This repository does not contain
datasets, checkpoints, or end-to-end training and benchmark runners.
