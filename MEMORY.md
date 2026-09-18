# Project memory

This file is durable context for future work sessions. Read it before changing code or manuscript.

## Scientific objective

Explain neutron-star quasi-universal relations as properties of the differential geometry of the relativistic EOS-to-observable map, rather than as empirical polynomial fits.

The object of interest is
[
Phi:(mathcal E,h_c)ightarrow mathbf y,
]
where (mathcal E) is an EOS function, (h_c) labels a stellar configuration, and (mathbf y) contains dimensionless observables.

The central hypotheses are:

1. EOS perturbations map into an effectively low-dimensional subspace of observable space.
2. After quotienting the stellar-sequence tangent, known strong relations such as I--Love correspond to unusually small transverse response modes.
3. Local low-response covectors need not define global relations; approximate universal relations require approximate integrability.
4. Residual scatter is controlled at next order by curvature/Hessian terms.
5. Phase transitions or other non-smooth EOS structure should appear as additional transverse response directions or increased curvature.

## Novelty boundaries established by literature audit

Do **not** claim:
- first I--Love--Q relation;
- first explanation via incompressibility/self-similarity;
- first stationarity of I--Love under EOS perturbations;
- first arbitrary density perturbation calculation;
- first automatic differentiation through TOV;
- first PCA/ML/SBI search for universal relations;
- first sensitivity matrix of neutron-star observables to EOS/model parameters;
- first dynamical-tide universal relation.

The intended novelty is the combined, basis-aware construction:
- nonparametric EOS-to-multiobservable differential operator;
- physically metric-weighted observable response (G=JC_{m EOS}J^T);
- explicit removal of the stellar-sequence tangent;
- universal relations as approximately integrable low-transverse-response distributions;
- higher-order curvature as a predictor of residual scatter;
- breakdown characterized by changes in response rank/eigenstructure.

Closest prior-art anchors to cite centrally:
Yagi & Yunes (2013); Yagi et al. (2014); Sham et al. (2015); Chan et al. (2016); Yip & Leung (2017); Soma et al. (automatic differentiation); Legred et al. (2024 nonparametric universality tests); Manoharan & Kokkotas (2024 data-driven universal-relation discovery); Krüger & Völkel (2026 SBI discovery); Cruz-Camacho et al. (2026 sensitivity/PCA); recent dynamical-response work only as an extension.

## Mathematical choices

A raw SVD of a Jacobian in arbitrary EOS coordinates is not a physical statement because basis rescaling changes singular values. Use an EOS metric/covariance.

For finite EOS coordinates (a_i),
[
J_{Ai}=partial y_A/partial a_i,
qquad
G_{AB}=J_{Ai}C^{ij}_{m EOS}J_{Bj}.
]

In the functional limit,
[
G_{AB}=int dh,dh',
K_A(h)C_{m EOS}(h,h')K_B(h'),
]
with
[
K_A(h)=delta y_A/delta s(h).
]

The stellar-sequence direction must be separated from EOS dependence. Let
[
t_A=partial y_A/partial h_c.
]
The transverse response is formed with an appropriate projector/quotient metric. The exact projector must be defined carefully once the observable-space metric is fixed; do not silently assume Euclidean geometry in log-observable coordinates without testing coordinate dependence.

The initial EOS field should be a bounded sound-speed representation, likely a latent unconstrained field mapped to (0<c_s^2<1). Finite-basis derivatives are an implementation device, not the definition of the science object.

## Initial observable set

Paper 1 starts with:
- compactness (C=M/R),
- dimensionless moment of inertia (ar I=I/M^3),
- quadrupolar dimensionless tidal deformability (Lambda_2).

The I--Love relation is a validation target, not an assumed model.

Add (ar Q,Lambda_3,f)-modes, or dynamical response only after the base construction is numerically validated.

## Numerical principles

- GR throughout paper 1.
- Prefer enthalpy-coordinate stellar structure.
- JAX/differentiable implementation.
- Every automatic derivative must be checked against symmetric finite differences at representative EOS/configurations.
- Surface handling and EOS interpolation must be differentiable and tested.
- No claims from an unvalidated solver.
- Known analytic/benchmark solutions should be used where possible.
- Separate local differential statements from global finite-perturbation tests.

## Manuscript policy

The paper is a live research document. Every stage must update `paper/main.tex`:
- mathematical definitions are written before numerical implementation;
- known results/limitations go into the text immediately;
- results sections may contain explicit TODO placeholders until data exist, but no invented numbers;
- failed hypotheses are retained in the research history and the paper is revised accordingly.

## Style

Physics/mathematics first. No rhetorical AI-style filler. State assumptions explicitly. Distinguish theorem/identity, approximation, numerical observation, and conjecture.
