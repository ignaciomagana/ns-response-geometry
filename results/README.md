# Results manifest

This directory contains both current paper results and historical analyses.
For quantitative manuscript claims, the authority is
`paper/CLAIM_PROVENANCE.md`.

## Current paper results

| Result | Generator | Status |
|---|---|---|
| `stage1_validation_20260918.json` | `scripts/stage1_validation.py` | current validation |
| `preliminary_response_20260918.json` | `scripts/preliminary_response.py` | current historical/development diagnostic; not production evidence |
| `response_robustness_20260918.json` | `scripts/response_robustness.py` | current |
| `background_ensemble_fine_20260918.json` | `scripts/background_ensemble.py` | current |
| `normal_spectrum_20260918.json` | `scripts/normal_spectrum.py` | current |
| `integrability_frobenius_20260918.json` | `scripts/integrability_frobenius.py` | current |
| `potential_reconstruction_20260918.json` | `scripts/potential_reconstruction.py` | current |
| `curvature_scatter_20260918.json` | `scripts/curvature_scatter.py` | current |
| `transition_breakdown_20260918.json` | `scripts/transition_breakdown.py` | current controlled example |
| `transition_scan_screen_20260918.json` | `scripts/transition_scan.py` | current fixed-depth screen |
| `transition_location_refinement_20260918.json` | `scripts/transition_location_refinement.py` | current fixed-depth refinement |
| `fixed_final_compactness_branchsafe_20260924.json` | `scripts/fixed_final_compactness_branchsafe.py` | **authoritative current cross-EOS control** |

## Superseded results — provenance only

Do not use these for current scientific claims:

| Result | Generator | Reason |
|---|---|---|
| `matched_compactness_transition_20260918.json` | `scripts/matched_compactness_transition.py` | includes unsafe Gamma=1.70 baseline in cross-EOS summary |
| `fixed_final_compactness_transition_20260918.json` | `scripts/fixed_final_compactness_transition.py` | unsafe baseline/global monotonic root assumption |
| `fixed_final_compactness_refinement_20260918.json` | `scripts/fixed_final_compactness_refinement.py` | unsafe baseline/global monotonic root assumption |

The branch-safe 2026-09-24 result supersedes all three as cross-EOS evidence.

## Reproduction

Paper-level CPU workflows use `constraints-paper-20260924.txt`; see
`ENVIRONMENT.md`. Historical analyses are retained rather than deleted so
the correction path remains auditable.

See also `LEGACY.md`.
