# Current state

**Last updated:** 2026-09-18

## Stage

Stage 0 — repository initialization and mathematical manuscript scaffold.

## Completed

- Repository initialized.
- Core question fixed: characterize quasi-universal relations using the differential geometry of the EOS-to-observable map.
- Literature novelty boundary recorded in \`MEMORY.md\`.
- Basis dependence of a raw Jacobian SVD identified; primary finite-dimensional object is \(G=JC_{\rm EOS}J^T\).
- Stellar-sequence motion identified as a tangent direction that must be quotiented before defining EOS universality.
- Continuous-paper policy fixed.

## Current scientific target

Demonstrate, without inserting an empirical I--Love fit, that EOS-induced response transverse to the stellar sequence is much smaller for \((\ln\bar I,\ln\Lambda_2)\) than for a weaker control such as \((\ln C,\ln\Lambda_2)\).

## Immediate next actions

1. Complete initial mathematical paper draft.
2. Implement EOS interfaces and a validated enthalpy-coordinate TOV solver.
3. Add \(l=2\) static tides and first-order frame dragging.
4. Validate \(M,R,\Lambda_2,I\) independently before response-geometry claims.
5. Build finite-basis response kernels and finite-difference derivative checks.

## Gate to Stage 1 completion

Do not proceed to response geometry until mass/radius convergence is demonstrated; \(\Lambda_2\) and \(I\) agree with independent references; and gradients are stable under numerical tolerances.

## Manuscript status

Initial manuscript is being created with the formal response-geometry definitions and explicit relationship to prior stationarity work. No numerical results are to be stated until produced.

## Blockers

None at repository level. Main technical risk: differentiable surface/EOS interpolation without contaminating functional derivatives.

## Session protocol

At the beginning of every session read \`STATE.md\`, \`MEMORY.md\`, \`SCIENCE_CONTRACT.md\`, and inspect current commits. At the end update \`STATE.md\`, record durable decisions in \`MEMORY.md\`, update the paper, and leave the repository reproducible.
