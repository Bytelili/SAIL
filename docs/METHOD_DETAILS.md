# SAIL method details

The default model definition uses the following settings.

| Component | Default |
|---|---|
| Population reference | Candidate-conditioned compatible mixture |
| Population-reference temperatures | Probability 1.0; compatibility 1.0 |
| Evidence precision | Structured bounded precision |
| Precision bounds and initialization | 0.001, 100.0, and 0.05 |
| User-state, memory, and residual dimensions | 256, 256, and 256 |
| Prior precision | 1.0 |
| Dimension gate | Learned query-conditioned sigmoid gate |
| Global strength | Analytical information-concentration gate |
| Residual calibration | Centered unit-RMS calibration |
| Query, support, and directional weights | 1.0, 0.1, and 1.0 |
| KL and auxiliary loss weights | 0.0 unless enabled explicitly |

With a Population backbone hidden size of 3584, this model definition has
83,112,961 active trainable parameters. Alternative networks retained for
ablations have `requires_grad=False` in the selected configuration.

The package accepts a caller-provided frozen Population model and candidate
tensors through explicit interfaces. Dataset preparation, candidate sampling,
checkpoint loading, and benchmark evaluation are separate from these modules.
