# Legacy and superseded analyses

This file prevents historical transition-control calculations from being
mistaken for current science after a disconnected work session.

## Authoritative current cross-EOS control

Use only:

- `scripts/fixed_final_compactness_branchsafe.py`
- `results/fixed_final_compactness_branchsafe_20260924.json`
- `paper/figures/branchsafe_fixed_final.tex`
- `paper/data/branchsafe_gamma_*.dat`

This calculation starts all reference EOSs at a positive-mass-slope
baseline, samples the full compactness-versus-depth curve, follows the first
root connected to the undeformed star, and checks the mass slope along the
deformation path.

## Superseded cross-EOS calculations

The following are retained for provenance only and must not support current
scientific claims:

- `scripts/matched_compactness_transition.py`
- `results/matched_compactness_transition_20260918.json`
- `.github/workflows/matched-compactness-transition.yml`

- `scripts/fixed_final_compactness_transition.py`
- `results/fixed_final_compactness_transition_20260918.json`
- `.github/workflows/fixed-final-compactness-transition.yml`

- `scripts/fixed_final_compactness_refinement.py`
- `results/fixed_final_compactness_refinement_20260918.json`
- `.github/workflows/fixed-final-compactness-refinement.yml`

The latter two families used a baseline that includes a negative-mass-slope
Gamma=1.70 configuration and/or a globally monotonic depth-root assumption
later shown to fail. The matched-reference control shares the unsafe
Gamma=1.70 baseline.

Their workflows are manual-only to preserve reproducibility without
automatically regenerating superseded results after core-code changes.

## Superseded paper assets

These files are not referenced by the live manuscript and correspond to the
superseded fixed-final interpretation:

- `paper/figures/fixed_final_location.tex`
- `paper/data/fixed_final_gamma_1p7.dat`
- `paper/data/fixed_final_gamma_1p85.dat`
- `paper/data/fixed_final_gamma_2.dat`

They remain in place solely for historical traceability.

## Still-valid precursor analyses

The original Gamma=2 fixed-depth transition experiments remain valid as
controlled demonstrations of localization:

- `results/transition_breakdown_20260918.json`
- `results/transition_scan_screen_20260918.json`
- `results/transition_location_refinement_20260918.json`

The branch-safe cross-EOS analysis changes only the interpretation of a
putative universal transition shell; it does not invalidate the fixed-depth
localization demonstration.

See `paper/CLAIM_PROVENANCE.md` for the authoritative claim-by-claim map.
