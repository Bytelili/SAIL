# Canonical paper-method alignment

This note records the public constructor defaults that define the canonical
SAIL method graph.  It is a method specification, not an experiment recipe or
an executable configuration.

| Component | Canonical public default |
|---|---|
| Population reference | candidate-conditioned compatible mixture |
| Population-reference temperatures | probability 1.0; compatibility 1.0 |
| Evidence precision | structured bounded precision |
| Precision bounds and initialization | 0.001, 100.0, and 0.05 |
| User-state, memory, and residual dimensions | 256, 256, and 256 |
| Prior precision | 1.0 |
| Dimension gate | learned query-conditioned sigmoid gate |
| Global strength | analytical information-concentration gate |
| Residual calibration | centered unit-RMS calibration |
| Query, support, and directional weights | 1.0, 0.1, and 1.0 |
| KL and auxiliary loss weights | 0.0 unless an ablation explicitly enables them |

With a Population backbone hidden size of 3584, the selected canonical graph
contains 83,112,961 active trainable parameters.  Alternative networks used to
inspect ablations remain in the source tree, but inactive alternatives have
`requires_grad=False` for the selected constructor configuration.

## Deliberate non-runnable boundary

The public repository does not include candidate sampling, dataset adapters,
the training loop, optimizer construction, checkpoint resolution, metric
evaluation, or a command-line entry point.  Candidate tensors and the caller's
frozen causal language model enter through explicit interfaces.  Consequently,
the code exposes the paper method without constituting a one-command
reproduction package.

