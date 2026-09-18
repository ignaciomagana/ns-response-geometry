# Quantitative claim provenance

**Updated:** 2026-09-18

This file is the paper-facing numerical provenance ledger.  Every quantitative
claim in the manuscript should map to a versioned result file and a committed
generating script.  If a result is superseded, update the manuscript and this
ledger in the same commit.

## Stellar-structure validation

| Manuscript claim | Versioned source | Generator |
|---|---|---|
| Exact incompressible background agreement at roughly \(10^{-6}\) | Stage-1 validation workflow/logs and tests | scripts/stage1_validation.py |
| Independent enthalpy/radius solver max differences \(4.71\times10^{-7}\), \(8.10\times10^{-7}\) | Stage-1 validation workflow/logs | scripts/stage1_validation.py |

## Local response hierarchy

| Manuscript claim | Versioned source | Generator |
|---|---|---|
| Nodal 17-to-33 response-ratio change \(3.6\times10^{-5}\) fractionally | results/response_robustness_20260918.json | scripts/response_robustness.py |
| SE/Matérn-3/2/exponential baseline I--Love/C--Love RMS ratio \(0.1024\)--\(0.1034\) | results/response_robustness_20260918.json | scripts/response_robustness.py |
| Broad stable ensemble median I--Love/C--Love RMS ratio \(0.0697\), \(q_{0.90}=0.205\) | results/background_ensemble_fine_20260918.json | scripts/background_ensemble.py |
| Broad stable ensemble median soft/hard RMS ratio \(0.0204\), \(q_{0.90}=0.0469\), max \(0.0883\) | results/normal_spectrum_20260918.json | scripts/normal_spectrum.py |
| Broad stable ensemble median soft/I--Love alignment \(0.9977\) | results/normal_spectrum_20260918.json | scripts/normal_spectrum.py |

## Integrability

| Manuscript claim | Versioned source | Generator |
|---|---|---|
| Frobenius median \(1.23\times10^{-3}\), \(q_{0.90}=3.48\times10^{-3}\), max \(7.36\times10^{-3}\) | results/integrability_frobenius_20260918.json | scripts/integrability_frobenius.py |
| Cubic gradient-only held-out \(\sigma(F)=8.48\times10^{-3}\) | results/potential_reconstruction_20260918.json | scripts/potential_reconstruction.py |
| One-dimensional I--Love gradient reconstruction \(\sigma(F)=1.98\times10^{-2}\) | results/potential_reconstruction_20260918.json | scripts/potential_reconstruction.py |
| Smooth-to-rough scatter reduction \(66\%\), rough-to-smooth \(51\%\) | results/potential_reconstruction_20260918.json | scripts/potential_reconstruction.py |

## Curvature

| Manuscript claim | Versioned source | Generator |
|---|---|---|
| Quadratic median relative RMSE \(0.0062,0.024,0.156,0.553\) at amplitudes \(0.05,0.10,0.25,0.50\) | results/curvature_scatter_20260918.json | scripts/curvature_scatter.py |
| At \(s=0.25\), median curvature-only fraction approximately \(0.63\) | results/curvature_scatter_20260918.json | scripts/curvature_scatter.py |
| Curvature finite-difference radius stability approximately \(2.5\times10^{-3}\) | results/curvature_scatter_20260918.json | scripts/curvature_scatter.py |

## Controlled localized softening

| Manuscript claim | Versioned source | Generator |
|---|---|---|
| Refined transition A: alignment \(0.90245\), I--Love/C--Love ratio \(0.49559\), soft/hard ratio \(0.09619\) | results/transition_breakdown_20260918.json | scripts/transition_breakdown.py |
| I--Love variance enhancement \(85.06\) | results/transition_breakdown_20260918.json | scripts/transition_breakdown.py |
| \(90.24\%\) of positive excess sensitivity inside \(h=0.14\)--0.18, peak \(h=0.155\) | results/transition_breakdown_20260918.json | scripts/transition_breakdown.py |
| 33-node transition screen: 69 physical/stable points of 96 | results/transition_scan_screen_20260918.json | scripts/transition_scan.py |
| Moving ridge near \(h_{\rm tr}/h_c=0.45\)--0.50 | results/transition_scan_screen_20260918.json | scripts/transition_scan.py |
| 65-node scaled scan strongest \(C\ge0.1\) case at \(h_{\rm tr}/h_c=0.45\) | results/transition_location_refinement_20260918.json | scripts/transition_location_refinement.py |
| That case: \(C=0.113\), alignment \(0.922\), I--Love/C--Love \(0.447\), soft/hard \(0.135\), variance factor \(56.7\), localized fraction \(93.3\%\) | results/transition_location_refinement_20260918.json | scripts/transition_location_refinement.py |
| Deformed shell center \(r/R=0.453\), \(m/M=0.759\) | results/transition_location_refinement_20260918.json | scripts/transition_location_refinement.py |

## Pending

The matched-compactness reference-EOS transition-location audit is running.
Its results must be added here before any new shell-location statement enters
the manuscript.

## Reproduction policy

- The ordinary unit-test workflow guards the solver and geometry utilities.
- The paper workflow guards LaTeX compilation.
- The science-reproduction workflow reruns the broad ensemble, normal
  spectrum, Frobenius, global potential, and curvature calculations.
- Specialized transition workflows rerun the localized-softening analyses.
- Local-VM calculations are acceptable for development, but paper-level
  quantitative claims should be reproduced in GitHub Actions when feasible.


## Matched-reference transition-location null test

| Manuscript claim | Versioned source | Generator |
|---|---|---|
| Unconstrained fixed-depth maxima across \(\Gamma=1.7,1.85,2.0\), baseline \(C=0.12,0.16\), occur at \(h_{\rm tr}/h_c=0.40\)--0.55 but have deformed \(C=0.045\)--0.090 | results/matched_compactness_transition_20260918.json | scripts/matched_compactness_transition.py |
| With post-softening \(C\ge0.10\), only three baseline-\(C=0.16\) cases survive and their maxima shift to \(h_{\rm tr}/h_c=0.25,0.30,0.35\) with alignments \(0.9998,0.9989,0.9999\) | results/matched_compactness_transition_20260918.json | scripts/matched_compactness_transition.py |
| Fixed-depth \(h_{\rm tr}/h_c\simeq0.45\)--0.50 is therefore not promoted as a universal vulnerable shell | results/matched_compactness_transition_20260918.json | scripts/matched_compactness_transition.py |

The next fixed-final-compactness experiment is designed to remove this global-expansion confound.
