# Current state

**Last updated:** 2026-09-18

## Stage

Stage 0 — repository initialization and mathematical manuscript scaffold.

## Completed

- Repository initialized.
- Core scientific question fixed: characterize quasi-universal relations using the differential geometry of the EOS-to-observable map.
- Literature novelty boundary recorded in `MEMORY.md`.
- Basis dependence of a raw Jacobian SVD identified; the primary finite-dimensional object is instead (G=JC_{m EOS}J^T).
- Stellar-sequence motion identified as a nuisance/tangent direction that must be quotiented before defining EOS universality.
- Paper will be drafted continuously at every stage.

## Current scientific target

Demonstrate, without inserting an empirical I--Love fit, that the EOS-induced response transverse to the stellar sequence is much smaller for the ((lnar I,lnLambda_2)) plane than for a weaker control relation such as ((ln C,lnLambda_2)).

## Immediate next actions

1. Write the initial mathematical paper draft with definitions, novelty positioning, and explicit local-to-global integrability problem.
2. Implement EOS interfaces and a validated enthalpy-coordinate TOV solver.
3. Add the (l=2) tidal perturbation and slow-rotation frame-dragging equations.
4. Validate (M,R,Lambda_2,I) against independent benchmarks before enabling autodiff claims.
5. Build finite-basis response kernels and finite-difference derivative checks.

## Gate to Stage 1 completion

Do not proceed to response geometry until:
- TOV mass/radius convergence is demonstrated;
- tidal deformability agrees with an independent implementation/reference benchmark;
- moment of inertia agrees with an independent implementation/reference benchmark;
- gradients are numerically stable under integration/grid tolerances.

## Manuscript status

Initial manuscript scaffold is being created now. It must already contain the formal response-geometry definitions and explicit relationship to prior stationarity work. No numerical results are to be stated until produced.

## Blockers

None at repository level. The main technical risk is differentiable treatment of the surface/EOS interpolation without contaminating functional derivatives.

## Session protocol

At the beginning of every future work session:
1. read `STATE.md`;
2. read `MEMORY.md`;
3. read `SCIENCE_CONTRACT.md`;
4. inspect the latest commits / changed files.

At the end:
1. update `STATE.md`;
2. record durable decisions in `MEMORY.md`;
3. update the paper for whatever changed;
4. leave the repository in a reproducible state.
