# Public release scope

## Included and scientifically attributable

The files under src/cusp/models, src/cusp/inference, and
src/cusp/training/losses.py implement the method components described in the
paper. Source files were copied from the active server implementation rather
than reconstructed from figures or result tables.

## Withheld by design

The release omits exact experiment recipes, evaluation/test code, dataset
adapters, candidate-production heuristics, checkpoints, results, and deployment
orchestration. No omitted file is silently replaced by a fake implementation.
Where an external component is required, the public code expects tensors or a
base-model module through an explicit interface.

## Claims this directory supports

- inspection of the paper-level model computation;
- inspection of the canonical paper-method defaults and active module graph;
- reuse of the posterior, evidence, gate, and residual-generation modules;
- integration with a caller-provided causal language model and candidate cache.

## Claims this directory does not support

- exact reproduction of reported benchmark numbers;
- official FingerTip-20K evaluation;
- reproduction of private training schedules or unpublished extensions;
- equivalence to the full internal repository.

The constructor defaults align the public method graph with the canonical SAIL
definition, but they are not a self-contained experiment recipe.  In
particular, this directory intentionally contains no runnable data, training,
candidate-production, or evaluation entry point.
