# Paper state

The manuscript is a live derivation, not a post-processing step.

**Updated:** 2026-09-24

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

The matched-reference and fixed-final-compactness controls have been run.
They do not support promoting the original fixed-depth
\(h_{\rm tr}/h_c\simeq0.45\)--0.50 ridge to a reference-EOS-independent
vulnerable shell. A finer fixed-final-compactness refinement also completed
successfully in GitHub Actions, but its full result currently exists only in
the workflow log/artifact and has not yet been promoted to a committed
versioned JSON or incorporated into the manuscript.

The immediate manuscript task is therefore to recover that refinement
artifact, commit its result with provenance, fold the controlled comparison
into the Results/Discussion, and then perform the final claim/figure audit.

## Remaining before final paper pass

- promote the fixed-final-compactness refinement artifact to a committed,
  versioned result file and manuscript text;
- final targeted literature/novelty re-check for response-operator geometry;
- numerical/claim provenance audit;
- final figure readability and rendered-page inspection;
- clean-room paper review.
