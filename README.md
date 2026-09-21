# SAIL / Generative CUSP — public paper core

This directory contains the compact, paper-aligned implementation of the
personalization mechanism used by SAIL (Generative CUSP). It is intentionally
separated from the internal experiment repository so that it can be reviewed
and uploaded to GitHub without datasets, checkpoints, machine paths, credentials,
run outputs, or private experiment infrastructure.

## What is included

- a frozen Population-model wrapper;
- compatible Population-reference aggregation over precomputed candidates;
- residual evidence encoding and dimension-wise observability;
- bounded structured evidence precision;
- diagonal-Gaussian user posterior and online posterior update;
- posterior-to-memory projection;
- uncertainty-aware personalization gate;
- low-rank residual generation head;
- direct autoregressive generation;
- the paper-defined loss functions and candidate-free generation contract.

The implementation preserves the paper-level computation:

1. estimate a Population reference for each observed interaction;
2. encode the observed-minus-Population residual as user evidence;
3. aggregate evidence in precision space to form a user posterior;
4. project the posterior into memory tokens;
5. apply a bounded, query-conditioned residual to direct generation.

Population candidates are internal evidence-estimation variables. They are not
a final-answer candidate set, and inference does not perform final reranking.

## Canonical method defaults

The `CUSPModel` constructor defaults describe the canonical method graph in the
paper: candidate-conditioned Population references, structured bounded
precision, a learned dimension-wise query gate, the analytical
information-concentration gate, RMS-calibrated residual logits, and objective
weights `(query, support, directional) = (1.0, 0.1, 1.0)`.  The initial
precision is `0.05`; the latent, memory, and residual interaction dimensions
default to `256`.

These defaults specify the method, not an executable reproduction recipe.  The
repository deliberately provides no dataset loader, candidate producer,
training runner, evaluation entry point, checkpoint resolver, or command that
reproduces the reported experiments.

## Deliberately not included

This is a paper-core release, not the complete internal reproduction bundle.
The following remain outside this directory:

- datasets, screenshots, user records, checkpoints, and generated artifacts;
- official validation/test adapters and all evaluation scripts;
- exact final-run hyperparameters, sweep configurations, and ablation recipes;
- internal candidate-generation, semantic-deduplication, and production cache
  scheduling heuristics;
- cluster/device orchestration, private paths, manifests, and experiment ledgers;
- unpublished extensions for personalized GUI execution.

These exclusions protect unpublished engineering details and private resources.
They must be disclosed as exclusions when the repository is released; this
directory should not be described as a one-command reproduction package.

## Installation

    python -m pip install -e .

The public core requires Python 3.10+ and PyTorch 2.1+. A base causal language
model is supplied by the caller through the PopulationModel wrapper.

## Code map

- src/cusp/models/cusp_model.py: integrated paper computation and objectives.
- src/cusp/models/population_reference.py: candidate-compatible reference.
- src/cusp/models/posterior.py: precision-space posterior aggregation.
- src/cusp/models/*precision.py: evidence reliability modules.
- src/cusp/models/memory_projector.py: posterior memory tokens.
- src/cusp/models/uncertainty_gate.py: bounded personalization gate.
- src/cusp/models/personalization_head.py: low-rank residual generation.
- src/cusp/inference/cusp_generation.py: direct autoregressive decoding.
- src/cusp/training/losses.py: paper-level learning objectives.
- docs/PAPER_TO_CODE.md: formula-to-code correspondence.
- docs/CANONICAL_ALIGNMENT.md: canonical defaults and the non-runnable boundary.
- PUBLIC_RELEASE_SCOPE.md: exact publication boundary.

## Release note

Before creating a public GitHub repository, choose and add a license. Until then,
the absence of a license means reuse rights have not been granted.
