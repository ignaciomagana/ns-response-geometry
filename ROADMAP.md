# Roadmap

The paper is developed in lockstep with the mathematics/code. Every stage has a manuscript deliverable.

## Stage 0 — formulation and prior-art boundary
Define the EOS-to-observable map, EOS metric/covariance, stellar-sequence quotient, local universality, global integrability, and second-order curvature program.

**Paper:** Introduction and mathematical formulation; conservative novelty positioning; Results structure with no invented outcomes.

## Stage 1 — validated stellar structure
Implement EOS protocol, enthalpy-coordinate TOV, \(l=2\) static tides, Hartle frame dragging and \(I\), with convergence and independent benchmark tests.

**Paper:** stellar-structure equations, boundary conditions, numerical validation subsection, validation table/figure.

## Stage 2 — finite-dimensional response kernels
Implement bounded sound-speed latent field, basis expansion, JAX Jacobians, symmetric finite-difference checks, and density/enthalpy-resolved sensitivity diagnostics.

**Paper:** EOS perturbations and functional derivatives; kernel figures; explicit connection to Chan/Yip--Leung stationarity.

## Stage 3 — metric-aware transverse geometry
Construct \(C_{\rm EOS}\), \(G_{\rm EOS}\), sequence tangent/quotient, eigenmodes, and invariance tests.

**Paper:** central Results section; I--Love versus C--Love response hierarchy.

## Stage 4 — local-to-global integrability
Transport local low-response covectors, measure path dependence, test Frobenius/integrability where applicable, and reconstruct a global relation without inserting the standard I--Love polynomial.

**Paper:** integrability section and derived relation.

## Stage 5 — curvature and finite scatter
Compute Hessian-vector products/directional second derivatives and compare predicted finite perturbation scatter with nonlinear EOS ensembles.

**Paper:** second-order origin of residual scatter.

## Stage 6 — controlled breakdown
Use broad nonparametric EOS ensembles and sharp/phase-transition structure. Identify density-resolved modes responsible for loss of universality.

**Paper:** domain of validity and breakdown mechanism.

## Stage 7 — discovery extension
Only after stages 1--6 work, add candidates such as
\[
\bar Q,\quad \Lambda_3,\quad \Lambda_4,\quad M\omega_f
\]
and dynamical-response coefficients. Any new claimed relation triggers a fresh literature audit.
