# Science contract

This document fixes the initial scientific claims, validation standards, and scope. Changes require an explicit entry in \`MEMORY.md\` and \`STATE.md\`.

## Question

Can neutron-star quasi-universal relations be characterized and discovered as approximately integrable low-response directions of the relativistic map from EOS function space to observable space?

## Definitions

Let
\[
\Phi:(\mathcal E,h_c)\mapsto y^A
\]
be the relativistic stellar-structure map.

For EOS coordinates \(a^i\),
\[
J^A{}_i=\frac{\partial y^A}{\partial a^i}.
\]

Given a positive EOS covariance/metric \(C_{\rm EOS}^{ij}\), define
\[
G^{AB}_{\rm EOS}=J^A{}_i C_{\rm EOS}^{ij}J^B{}_j.
\]

In the functional limit,
\[
G^{AB}_{\rm EOS}
=
\int dh\,dh'\,
K^A(h)C_{\rm EOS}(h,h')K^B(h'),
\qquad
K^A(h)=\frac{\delta y^A}{\delta s(h)}.
\]

The sequence tangent is
\[
t^A=\frac{\partial y^A}{\partial h_c}.
\]

A local quasi-universal direction is a covector \(n_A\) with small EOS response after the sequence direction is quotiented,
\[
n_A\,\delta y^A_{\rm EOS}\simeq0,
\]
subject to normalization in a declared observable-space metric.

A global relation \(F(y)\simeq{\rm const}\) additionally requires approximate integrability,
\[
n_A\propto\partial_A F .
\]

## Primary hypotheses

H1. I--Love appears as a suppressed transverse response mode without using an empirical I--Love fit to construct the mode.

H2. The suppression is stronger than for a control relation such as C--Love over the same EOS/configuration domain.

H3. The suppressed local covectors are approximately integrable over the ordinary cold-hadronic domain.

H4. Leading finite EOS scatter is predicted by second-order response/curvature after first-order transverse sensitivity is suppressed.

H5. Strong phase-transition-like structure produces identifiable changes in transverse response and/or integrability/curvature.

## Falsification conditions

The interpretation fails or must be weakened if the hierarchy disappears under reasonable EOS metrics, is primarily an observable-coordinate artifact, cannot recover known I--Love strength without inserting it explicitly, is strongly non-integrable, has no useful second-order connection to finite scatter, or adds no information about controlled breakdown.

Negative results must be reported rather than tuned away.

## Scope for paper 1

Included: cold barotropic GR; stable nonrotating \(M,R,\Lambda_2\) sequences; first-order slow rotation for \(I\); broad/nonparametric EOS perturbations; I--Love and C--Love; tangent quotient, integrability, curvature, phase-transition stress tests.

Excluded initially: rapid rotation, magnetic fields, finite temperature/composition, modified gravity, full \(\bar Q\), dynamical tides and mode spectra.

## Numerical validation contract

No response-geometry conclusion is publishable until observables converge numerically; independent benchmarks agree; autodiff agrees with symmetric finite differences; results are stable to EOS basis refinement; at least two reasonable EOS covariance/metric choices are tested; and observable-coordinate dependence is explicitly investigated.

## Literature/novelty contract

Prior stationarity and sensitivity work are foundations, not novelty targets. Novelty is the geometric synthesis and its demonstrated consequences.
