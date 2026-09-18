# Science contract

This document fixes the initial scientific claims, validation standards, and scope. Changes require an explicit entry in `MEMORY.md` and `STATE.md`.

## Question

Can neutron-star quasi-universal relations be characterized and discovered as approximately integrable low-response directions of the relativistic map from EOS function space to observable space?

## Definitions

Let
[
Phi:(mathcal E,h_c)mapsto y^A
]
be the relativistic stellar-structure map.

For EOS coordinates (a^i), define
[
J^A{}_i=rac{partial y^A}{partial a^i}.
]

Given a positive EOS covariance/metric (C_{m EOS}^{ij}), define the induced observable response
[
G^{AB}_{m EOS}=J^A{}_i C_{m EOS}^{ij}J^B{}_j.
]

In the functional limit this becomes
[
G^{AB}_{m EOS}
=
int dh,dh',
K^A(h)C_{m EOS}(h,h')K^B(h'),
]
where (K^A(h)=delta y^A/delta s(h)).

The sequence tangent is
[
t^A=rac{partial y^A}{partial h_c}.
]

A quasi-universal relation is not defined merely by small scatter in a sampled EOS catalog. Locally, it corresponds to a covector (n_A) with small EOS response after the sequence direction is quotiented:
[
n_A,delta y^A_{m EOS}simeq 0,
]
subject to normalization in a declared observable-space metric.

A global relation (F(y)simeq {m const}) additionally requires approximate integrability,
[
n_A propto partial_A F.
]

## Primary hypotheses

H1. Known I--Love universality appears as a suppressed transverse response mode without using an empirical I--Love fit in constructing the mode.

H2. The suppression is stronger than for a control relation such as C--Love over the same EOS/configuration domain.

H3. The suppressed local covectors are approximately integrable over the ordinary cold-hadronic domain, allowing reconstruction of a global relation.

H4. The leading finite EOS scatter is predicted by second-order response/curvature after first-order transverse sensitivity is suppressed.

H5. Strong phase-transition-like EOS structure produces identifiable changes in transverse response and/or integrability/curvature, explaining degradation of universality.

## Falsification conditions

The central interpretation fails or must be weakened if:
- the response hierarchy disappears under reasonable changes of EOS metric/prior;
- the hierarchy is primarily an artifact of chosen observable coordinates;
- known I--Love strength cannot be recovered without inserting the relation explicitly;
- local low-response modes are strongly non-integrable;
- second-order response has no useful connection to finite-ensemble scatter;
- phase-transition breakdown cannot be connected to response geometry more informatively than ordinary scatter measurements.

Negative results remain scientifically useful and must be reported rather than tuned away.

## Scope for paper 1

Included:
- cold barotropic GR neutron stars;
- stable nonrotating sequences for (M,R,Lambda_2);
- first-order slow rotation for (I);
- nonparametric/broad EOS perturbations;
- I--Love and C--Love as primary validation/control;
- response metric, tangent quotient, integrability, and curvature;
- phase-transition stress tests.

Excluded initially:
- rapid rotation;
- magnetic fields;
- temperature/composition evolution;
- modified gravity;
- full (ar Q) calculation;
- dynamical tides and mode spectra.

## Numerical validation contract

No response-geometry conclusion is publishable until:
- stellar observables converge with numerical resolution/tolerance;
- independent reference calculations agree to a pre-declared tolerance;
- autodiff derivatives agree with symmetric finite differences;
- results are stable to EOS basis refinement;
- results are repeated for more than one reasonable EOS covariance/metric;
- coordinate dependence in observable space is explicitly investigated.

## Literature/novelty contract

Prior stationarity and sensitivity work must be treated as foundations, not displaced by novelty language. The paper's novelty is the geometric synthesis and its consequences, if demonstrated.
