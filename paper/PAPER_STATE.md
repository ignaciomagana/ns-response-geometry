# Paper state

The manuscript is a live derivation, not a post-processing step.

**Updated:** 2026-09-18

## Drafted

- Introduction and conservative novelty positioning.
- Differential EOS-to-observable map.
- Metric-aware induced response \(G=JC_{\rm EOS}J^T\).
- Stellar-sequence quotient and cotangent-space normalization.
- Local versus global/integrability distinction.
- Second-order curvature expansion.
- Exact enthalpy-coordinate TOV, tidal, and slow-rotation equations.
- Nodal latent sound-speed EOS coordinates and functional covariance.
- Stage 1 validation table and independent-solver comparison.
- Restricted Gaussian-mode pilot response hierarchy, explicitly labeled as
  a development diagnostic.

## Numerical statements currently allowed in the paper

- Exact incompressible background agreement at \(10^{-6}\) scale.
- Independent radius/enthalpy cross-check maximum discrepancies
  \(4.71\times10^{-7}\) and \(8.10\times10^{-7}\) for the two validation
  cases.
- Original pilot I--Love/C--Love transverse RMS ratio \(0.093\)--\(0.133\)
  over \(C\simeq0.067\)--\(0.171\), with explicit caveat that it uses one
  reference EOS and the legacy Gaussian coordinates.
- Original pilot maximum autodiff/finite-difference Jacobian discrepancy
  \(8.2\times10^{-8}\).

## Pending before the central result can be written

- nodal grid convergence;
- covariance-kernel and correlation-length robustness;
- reference-EOS robustness;
- production autodiff/finite-difference check;
- numerical-resolution check;
- broad nonparametric EOS ensemble.

The current \`response-robustness\` workflow addresses the first five items.
