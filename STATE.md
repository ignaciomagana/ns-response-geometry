# Current state

**Last updated:** 2026-09-18

## Stage

Stage 2/3 transition — stellar structure is validated; production nodal EOS
response kernels are implemented; robustness of the local response hierarchy
is being audited.

## Completed

- Mathematical manuscript scaffold and novelty boundaries.
- Differentiable enthalpy-coordinate TOV solver.
- Static \(l=2\) tidal response including self-bound surface correction.
- First-order Hartle moment of inertia.
- Exact constant-density background benchmark.
- Newtonian homogeneous and \(n=1\) polytrope limits.
- Independent adaptive radius-coordinate DOP853 cross-check.
- Stage 1 maximum cross-solver discrepancy below \(10^{-6}\).
- Metric-aware observable response \(G=JC_{\rm EOS}J^T\).
- Cotangent-space normalization of universal-relation normals.
- EOS basis-change and observable-coordinate invariance unit tests.
- Thermodynamically consistent latent sound-speed EOS reconstruction.
- Production nodal latent-field coordinates for a fixed functional EOS metric.
- Legacy Gaussian pilot: strong I--Love suppression relative to C--Love,
  retained as a diagnostic only.
- Manuscript updated with the exact stellar equations and Stage 1 validation
  numbers.

## Current scientific target

Establish whether the I--Love transverse-response suppression survives:
1. refinement of the nodal EOS field;
2. squared-exponential, Mat\'ern-3/2, and exponential EOS covariances;
3. multiple correlation lengths;
4. multiple smooth causal reference EOSs;
5. consistent observable-coordinate transformations and alternative
   normalization metrics;
6. stellar/EOS reconstruction resolution.

The workflow \`response-robustness\` is executing this audit.

## Immediate next actions

1. Read and commit the robustness result as a versioned result file.
2. Update the Results section with only the tests that pass.
3. If H1/H2 survive, move from controlled polytropic references to a broad
   nonparametric EOS ensemble.
4. Build the full three-observable normal spectrum; do not call 2D
   reconstruction an integrability test because the 2D normal is unique from
   the sequence tangent alone.
5. Only after the full response field is sampled in a sufficiently
   multidimensional domain, implement a genuine Frobenius/path-dependence
   test.
6. Then evaluate second-order directional curvature and finite-amplitude
   scatter.

## Validation status

Stage 1: passed.

Production response derivatives: nodal autodiff/finite-difference audit is
part of the current robustness workflow.

## Manuscript status

- Introduction: drafted.
- Response geometry: drafted.
- Stellar equations: synchronized with production solver.
- Numerical validation: Stage 1 results written.
- Results: original Gaussian pilot written explicitly as a restricted
  development diagnostic.
- Production robustness results: pending current workflow.
- Integrability/curvature/breakdown: structured placeholders only.

## Blockers / risks

- The broad nonparametric EOS ensemble has not yet been defined; do not call
  the controlled polytropic robustness study a population-independent result.
- Observable-metric dependence must be separated from mere coordinate
  dependence.
- A genuine integrability test requires a response covector field sampled in
  more than the one-dimensional stellar sequence.

## Session protocol

At the beginning of every session read \`STATE.md\`, \`MEMORY.md\`,
\`SCIENCE_CONTRACT.md\`, inspect recent commits, and inspect CI. At the end
update \`STATE.md\`, record durable decisions in \`MEMORY.md\`, update the
paper, and leave all reported numerical results reproducible from scripts or
versioned result files.
