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
- Matched-reference null test and fixed-final-compactness control/refinement.
- Discussion and conclusions through H5.
- Production figures for normal spectrum, integrability, curvature,
  fixed-depth transition location, and fixed-final-compactness location.

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
  original \(\Gamma=2\), \(C\ge0.1\) fixed-depth experiment;
- fixed-final-compactness stable maxima span
  \(h_{\rm tr}/h_c=0.35\)--0.65 and
  \(m_{\rm tr}/M=0.340\)--0.852;
- seven of eight fixed-final maxima lie at
  \(0.426\le r_{\rm tr}/R\le0.457\), with one at \(0.533\);
- the fixed-final scan does not support a reference-EOS-independent
  vulnerable shell.

## Active manuscript target

The transition-location controls are now complete and incorporated.  The
manuscript explicitly rejects a universal-shell interpretation of the
fixed-depth ridge and reports the tighter fractional-radius behavior only as
a diagnostic observation.

## Final render status

- Current manuscript compiles successfully.
- All 19 rendered pages inspected.
- No clipped text, overlapping elements, or broken glyphs found.
- The rendered PDF is tracked in the repository at `paper/main.pdf`.
- The paper workflow re-renders and updates the tracked PDF after successful pushes to `main`.

## Remaining before final paper pass

- final targeted literature/novelty re-check for response-operator geometry;
- numerical/claim provenance audit against every quantitative manuscript
  statement;
- final figure readability and rendered-page inspection;
- clean-room code/results/manuscript consistency review;
- decide whether any additional physics is needed before submission.


## 2026-09-24 correction in progress

The first fixed-final-compactness refinement has been superseded after a
branch/stability audit. Its manuscript subsection and figure are provisional
until the branch-safe replacement completes. Do not treat the current
fixed-final numerical values as submission-ready.
