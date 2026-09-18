# ns-response-geometry

Differential geometry of neutron-star quasi-universal relations.

The central object is the relativistic map
\[
\Phi:(\mathcal E,h_c)\mapsto \mathbf y,
\]
from an equation of state (EOS) and stellar-sequence coordinate to dimensionless observables such as
\[
\mathbf y=(\ln C,\ln \bar I,\ln \Lambda_2,\ln \bar Q,\ldots).
\]

The project asks whether quasi-universal relations arise because the EOS-to-observable response has anomalously small transverse directions after motion along a stellar sequence is quotiented out.

The initial paper is deliberately narrow: static/slow-rotation GR; \(C,\bar I,\Lambda_2\); recovery of I--Love as a validation target; metric-aware EOS response geometry; local-to-global integrability; and controlled breakdown for broad/phase-transition EOS ensembles. Higher multipoles, \(f\)-modes, and dynamical tides are extensions only after the core claim survives.

## Working rule

This repository is the source of truth. Every work session begins by reading \`STATE.md\`, \`MEMORY.md\`, and \`SCIENCE_CONTRACT.md\`, and ends by updating \`STATE.md\`. The manuscript in \`paper/\` is updated at every scientific stage rather than written after the analysis.

## Layout

- \`STATE.md\` — current status, last completed action, next action, blockers.
- \`MEMORY.md\` — durable scientific and implementation decisions.
- \`SCIENCE_CONTRACT.md\` — claims the project is allowed to make and validation requirements.
- \`ROADMAP.md\` — staged research plan and paper deliverables.
- \`paper/\` — continuously updated manuscript.
- \`src/ns_response_geometry/\` — implementation.
- \`tests/\` — numerical/unit tests.

## Core mathematical objects

For EOS coordinates \(a_i\),
\[
J_{Ai}=\frac{\partial y_A}{\partial a_i}.
\]
A raw SVD of \(J\) is basis dependent, so the primary observable-space response metric is
\[
G = J\,C_{\rm EOS}\,J^T,
\]
where \(C_{\rm EOS}\) is a covariance/metric on physically allowed EOS perturbations.

If
\[
t_A=\frac{\partial y_A}{\partial h_c}
\]
is the tangent to a stellar sequence, quasi-universality is characterized by the transverse response
\[
G_\perp=P_\perp G P_\perp^T.
\]

A small-eigenvalue covector defines a local approximately EOS-insensitive direction. A global universal relation exists only if that local distribution is approximately integrable.

See \`SCIENCE_CONTRACT.md\` and \`paper/main.tex\` for the precise formulation.
