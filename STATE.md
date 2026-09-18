# Current state

**Last updated:** 2026-09-18

## Stage

Stage 4 — local-to-global geometry. Stages 1--3 are complete for the current
static/slow-rotation observable set \((C,\bar I,\Lambda_2)\).

## Completed

- Mathematical formulation and conservative novelty boundary.
- Differentiable enthalpy-coordinate TOV solver.
- Static \(l=2\) tidal response and first-order Hartle moment of inertia.
- Exact constant-density benchmark and Newtonian limiting checks.
- Independent adaptive radius-coordinate DOP853 cross-check.
- Stage 1 maximum cross-solver discrepancy below \(10^{-6}\).
- Thermodynamically consistent latent sound-speed EOS reconstruction.
- Production nodal EOS coordinates that refine a fixed functional covariance.
- Metric-aware response \(G=JC_{\rm EOS}J^T\).
- Cotangent-space normalization and observable-coordinate invariance tests.
- Nodal-grid, covariance-kernel, correlation-length, reference-EOS,
  observable-metric, derivative, and numerical-resolution robustness audit.
- Fine broad background ensemble: 44 stable configurations from 16 large
  smooth/rough causal latent-field backgrounds.
- Full three-observable normal-spectrum calculation.

## Central result so far

After quotienting motion along a stellar sequence, the full
\((\ln C,\ln\bar I,\ln\Lambda_2)\) EOS response contains one strongly
suppressed normal mode.

For the broad 44-point stable background ensemble:
- median soft/hard normal RMS ratio: \(0.0204\);
- 90th percentile: \(0.0469\);
- maximum: \(0.0883\);
- median cosine alignment of the soft covector with the embedded I--Love
  normal: \(0.9977\).

For \(C\ge0.05\), the maximum normal-mode RMS ratio is \(0.0797\), the
minimum I--Love alignment is \(0.9756\), and the median alignment is
\(0.9979\).

The simpler I--Love/C--Love plane comparison also survives large functional
EOS deformations. For the same broad stable ensemble its median RMS ratio is
\(0.0697\), with 90th percentile \(0.205\). For all stable points with
\(C\ge0.05\), the ratio is below \(0.231\).

## Current scientific target

Test H3: whether the soft normal covector field is approximately integrable,
rather than merely locally soft.

The first nontrivial calculation uses three-dimensional domain patches
\[
(h_c,\alpha_i,\alpha_j)
\]
where \(\alpha_i,\alpha_j\) move the EOS along covariance principal
directions.  The map into
\((\ln C,\ln\bar I,\ln\Lambda_2)\) is locally inverted to obtain
\(\partial n_A/\partial y^B\), after which the Frobenius quantity
\[
n\cdot(\nabla_y\times n)
\]
is evaluated.  Multiple EOS-direction pairs and finite-amplitude background
points are required.

A 2D I--Love reconstruction alone is not an integrability test and must not
be presented as one.

## Immediate next actions

1. Sample the soft covector on several 3D domain patches built from leading
   EOS covariance eigenmodes.
2. Evaluate the Frobenius obstruction and its normalized form.
3. Check sign/gauge continuity of the eigen-covector and repeat under finite
   background displacement.
4. If locally integrable, reconstruct a scalar quasi-invariant \(F(y)\) and
   test closed-loop/path dependence directly.
5. Update the manuscript immediately with either the positive or negative
   result.
6. Only then proceed to Hessian/directional-curvature tests of finite scatter.
7. Follow curvature with sharp-transition/rapid-sound-speed-variation stress
   tests.

## Validation / execution status

Stage 1: passed.

Stages 2--3: passed for the declared response metrics and tested broad
functional backgrounds. Versioned numerical summaries:
- \`results/response_robustness_20260918.json\`
- \`results/background_ensemble_fine_20260918.json\`
- \`results/normal_spectrum_20260918.json\`

GitHub Actions entered a runner-start failure mode on 2026-09-18: failed
jobs contained no executed steps, including the unchanged pytest workflow
that had passed earlier. Broad-ensemble and normal-spectrum calculations
were therefore reproduced on a local mirror of the exact GitHub numerical
core and committed back with scripts and provenance. Re-run CI when Actions
resumes; do not reinterpret runner-start failures as physics failures.

## Manuscript status

- Introduction: drafted.
- Response geometry: drafted.
- Stellar equations: synchronized with production solver.
- Numerical validation: Stage 1 results written.
- Production robustness: written.
- Broad functional-background result: written.
- Full normal-spectrum hierarchy and I--Love alignment: written.
- Integrability: next active section.
- Curvature/breakdown: placeholders only.

## Session protocol

At the beginning of every session read \`STATE.md\`, \`MEMORY.md\`,
\`SCIENCE_CONTRACT.md\`, inspect recent commits/results, and inspect CI.
At the end update \`STATE.md\`, record durable mathematical/implementation
decisions in \`MEMORY.md\`, update the paper, and leave every reported
numerical result reproducible from a committed script plus a versioned result
file.
