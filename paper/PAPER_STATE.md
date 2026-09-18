# Paper state

The manuscript is a live derivation, not a post-processing step.

**Updated:** 2026-09-18

## Drafted

- Introduction rewritten around the completed response-geometry analysis.
- Differential EOS-to-observable map.
- Metric-aware induced response \(G=JC_{\rm EOS}J^T\).
- Stellar-sequence quotient and cotangent normalization.
- Local versus global/integrability distinction.
- Exact enthalpy-coordinate TOV, tidal, and slow-rotation equations.
- Nodal latent sound-speed EOS coordinates and functional covariance.
- Stage 1 validation and independent-solver comparison.
- Production nodal response robustness.
- Broad functional-background ensemble.
- Full three-observable normal spectrum and I--Love alignment.
- High-resolution Frobenius integrability.
- Cross-validated gradient-only scalar potential reconstruction.
- Second-order curvature and finite perturbative radius.
- Controlled localized-softening breakdown.
- Systematic transition center/width/depth screen.
- High-resolution transition-location refinement in enthalpy and physical
  stellar coordinates.
- Discussion and conclusions through H5.
- Production figures for normal spectrum, integrability, curvature, and
  transition location.

## Numerical statements currently allowed

All manuscript numbers must trace to versioned result JSON files. Core claims:
- independent stellar-solver discrepancies below \(10^{-6}\);
- broad-ensemble median soft/hard RMS ratio \(0.0204\);
- broad-ensemble median I--Love alignment \(0.9977\);
- Frobenius median \(1.23\times10^{-3}\);
- held-out cubic soft-surface scatter \(8.48\times10^{-3}\);
- second-order relative RMSE \(0.0062,0.024,0.156,0.553\) at amplitudes
  \(0.05,0.10,0.25,0.50\);
- refined transition variance enhancement \(85.06\) with \(90.24\%\)
  positive excess sensitivity localized on the imposed layer;
- scaled transition-location maximum at \(h_{\rm tr}/h_c=0.45\) for the
  current \(\Gamma=2\), \(C\ge0.1\) controlled experiment.

## Active manuscript target

Test whether the transition-vulnerability location survives changes in the
smooth reference EOS at matched baseline compactness. If it moves
substantially, retain the result only as a demonstration of localizable
response vulnerability. If a common shell emerges, report it cautiously
without claiming a universal microphysical transition radius.

## Remaining before final paper pass

- matched-compactness reference-EOS transition-location audit;
- final targeted literature/novelty re-check for response-operator geometry;
- numerical/claim provenance audit;
- final figure readability and manuscript build check;
- clean-room paper review.
