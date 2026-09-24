# Paper state

The manuscript is a live derivation, not a post-processing step.

**Updated:** 2026-09-24

## Current authoritative status

Paper-1 analysis through controlled breakdown is complete for the declared
static/slow-rotation observable set
\[
(C,\bar I,\Lambda_2).
\]

The live manuscript uses the branch-safe cross-EOS transition control.
Earlier matched-reference and fixed-final controls are retained only as
historical provenance and must not be used for scientific claims.

## Drafted and validated

- Introduction and conservative novelty positioning.
- Differential EOS-to-observable map.
- Metric-aware induced response \(G=JC_{\rm EOS}J^T\).
- Stellar-sequence quotient and cotangent normalization.
- Exact enthalpy-coordinate TOV, tidal, and slow-rotation equations.
- Nodal latent sound-speed EOS coordinates and functional covariance.
- Stage-1 solver validation and independent radius-coordinate cross-check.
- Production nodal-response robustness.
- Broad causal functional-background ensemble.
- Full three-observable normal spectrum and I--Love alignment.
- High-resolution Frobenius integrability.
- Cross-validated gradient-only scalar-potential reconstruction.
- Second-order curvature and finite perturbative radius.
- Controlled localized-softening breakdown.
- Fixed-depth transition-location screen/refinement.
- Branch-safe cross-EOS fixed-final-compactness control.
- Discussion and conclusions.
- Quantitative claim-provenance ledger.
- Clean-room code/results/manuscript consistency review.
- Production figures and final PDF render.

## Quantitative statements currently allowed

All quantitative claims must map to `paper/CLAIM_PROVENANCE.md`.

Core live claims include:
- independent stellar-solver discrepancies below \(10^{-6}\);
- broad-ensemble median soft/hard RMS ratio \(0.0204\);
- broad-ensemble median I--Love alignment \(0.9977\);
- Frobenius median \(1.23\times10^{-3}\);
- held-out cubic soft-surface scatter \(8.48\times10^{-3}\);
- quadratic-response median relative RMSE
  \(0.0062,0.024,0.156,0.553\) at amplitudes
  \(0.05,0.10,0.25,0.50\);
- refined controlled-softening variance enhancement \(85.06\), with
  \(90.24\%\) of positive excess sensitivity localized on the imposed layer;
- branch-safe cross-EOS selected points spanning I--Love/C--Love RMS
  \(0.229\)--\(0.398\), I--Love alignment \(0.939\)--\(0.979\), and
  soft/hard RMS \(0.046\)--\(0.139\);
- branch-safe layer centers spanning \(r/R=0.368\)--0.476 and
  \(m/M=0.235\)--0.675, with no reference-EOS-independent shell identified.

## Explicitly superseded analyses

Do not use for current scientific claims:
- `results/matched_compactness_transition_20260918.json`;
- `results/fixed_final_compactness_transition_20260918.json`;
- `results/fixed_final_compactness_refinement_20260918.json`.

The first two are precursors built on the same baseline/branch assumptions
later found unsafe; the third is explicitly superseded by
`results/fixed_final_compactness_branchsafe_20260924.json`.

Their scripts/results remain for provenance only. See `LEGACY.md` and
`paper/CLAIM_PROVENANCE.md`.

## Domain of the current claims

The broad ensemble is a causal latent-sound-speed function-space stress test,
not a nuclear-theory posterior or a catalog of realistic named EOSs. The
paper states this explicitly. A realistic nuclear-EOS ensemble is a useful
future validation/extension, not evidence already supplied by this repository.

## Final render status

- Current manuscript compiles successfully.
- All 19 rendered pages inspected.
- No clipped text, overlapping elements, or broken glyphs found.
- Rendered PDF is tracked at `paper/main.pdf`.
- The paper workflow also uploads PDF/page renders as an Actions artifact.

## Remaining work

No missing analysis is required to support the manuscript as currently
worded. Remaining work is optional/additive:
- author-level editorial changes and affiliation/coauthor metadata;
- an optional named/realistic nuclear-EOS validation set;
- Stage 7 observable extensions (\(\bar Q,\Lambda_3,f\)-modes, dynamical
  response), which should be treated as follow-up work and trigger a fresh
  novelty audit.
