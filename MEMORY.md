# Project memory

Read this file before changing code or manuscript.

## Scientific objective

Explain neutron-star quasi-universal relations as properties of the differential geometry of the relativistic EOS-to-observable map, rather than as empirical polynomial fits.

\[
\Phi:(\mathcal E,h_c)\rightarrow \mathbf y ,
\]
where \(\mathcal E\) is an EOS function, \(h_c\) labels a stellar configuration, and \(\mathbf y\) contains dimensionless observables.

The central hypotheses are:

1. EOS perturbations map into an effectively low-dimensional subspace of observable space.
2. After quotienting the stellar-sequence tangent, strong relations such as I--Love correspond to unusually small transverse response modes.
3. Local low-response covectors need not define global relations; global quasi-universality additionally requires approximate integrability.
4. Residual finite EOS scatter is controlled at next order by curvature/Hessian terms.
5. Phase transitions or other sharp EOS structure should appear as additional transverse response directions and/or increased curvature.

## Novelty boundaries established by literature audit

Do **not** claim first:
- I--Love--Q relation;
- explanation via incompressibility or self-similarity;
- stationarity of I--Love under EOS perturbations;
- arbitrary density-perturbation calculation;
- automatic differentiation through TOV;
- PCA/ML/SBI search for universal relations;
- sensitivity matrix of neutron-star observables to EOS/model parameters;
- dynamical-tide universal relation.

Intended novelty:
- nonparametric EOS-to-multiobservable differential operator;
- physically metric-weighted observable response \(G=JC_{\rm EOS}J^T\);
- explicit removal of the stellar-sequence tangent;
- universal relations as approximately integrable low-transverse-response distributions;
- higher-order curvature as a predictor of residual scatter;
- breakdown characterized by changes in response rank/eigenstructure.

Closest prior-art anchors: Yagi & Yunes (2013); Yagi et al. (2014); Sham et al. (2015); Chan et al. (2016); Yip & Leung (2017); Soma et al. (2023); Legred et al. (2024); Manoharan & Kokkotas (2024); Krüger & Völkel (2026); Cruz-Camacho et al. (2026).

## Mathematical choices

A raw SVD of a Jacobian in arbitrary EOS coordinates is not physical because a basis rescaling changes its singular values. Introduce an EOS covariance/metric.

For finite EOS coordinates \(a_i\),
\[
J_{Ai}=\frac{\partial y_A}{\partial a_i},
\qquad
G_{AB}=J_{Ai}C^{ij}_{\rm EOS}J_{Bj}.
\]

In the functional limit,
\[
G_{AB}=\int dh\,dh'\,
K_A(h)C_{\rm EOS}(h,h')K_B(h'),
\qquad
K_A(h)=\frac{\delta y_A}{\delta s(h)}.
\]

The stellar-sequence direction must be separated from EOS dependence:
\[
t_A=\frac{\partial y_A}{\partial h_c}.
\]

The transverse response is formed with an appropriate projector/quotient metric. Do not silently assume Euclidean geometry in log-observable coordinates; observable-coordinate dependence is itself a required robustness test.

The initial EOS field should be a bounded sound-speed representation, likely an unconstrained latent field mapped to \(0<c_s^2<1\). Finite-basis derivatives are an implementation device, not the definition of the science object.

## Initial observable set

Paper 1 starts with
\[
C=M/R,\qquad \bar I=I/M^3,\qquad \Lambda_2 .
\]

I--Love is a validation target, not an assumed model. Add \(\bar Q,\Lambda_3,f\)-modes or dynamical response only after the base construction is validated.

## Numerical principles

- GR throughout paper 1.
- Prefer enthalpy-coordinate stellar structure.
- JAX/differentiable implementation.
- Every automatic derivative must be checked against symmetric finite differences.
- Surface handling and EOS interpolation must be differentiable and tested.
- No scientific claims from an unvalidated solver.
- Separate local differential statements from global finite-perturbation tests.

## Manuscript policy

The paper is a live research document. Every stage updates \`paper/main.tex\`. Mathematical definitions are written before implementation; known limitations are recorded immediately; result placeholders remain explicit until numbers exist; failed hypotheses are retained in project history.

## Style

Physics/mathematics first. State assumptions explicitly. Distinguish identity, approximation, numerical observation, and conjecture.


## 2026-09-18 implementation decisions and first diagnostics

Stage 1 stellar structure is validated. The production fixed-step enthalpy
solver agrees with an independent adaptive radius-coordinate DOP853 solver
to maximum absolute relative differences of \(4.71\times10^{-7}\) for the
constant-density test case and \(8.10\times10^{-7}\) for the
\(\Gamma=2\) polytrope when comparing
\(M,R,C,k_2,\Lambda_2,I,\bar I,I/(MR^2)\). The exact incompressible
background is reproduced at the \(10^{-6}\) level.

The original five-Gaussian-mode pilot found an I--Love/C--Love transverse RMS
response ratio of \(0.093\)--\(0.133\) over
\(C\simeq0.067\)--\(0.171\), with the Jacobian agreeing with symmetric
finite differences to \(8.2\times10^{-8}\) maximum relative error. This is a
development diagnostic only.

Important correction: a Gaussian basis with a covariance placed directly on
its coefficients does not represent a fixed function-space prior as the
number/width of basis functions changes. It must not be used to claim basis
convergence.

Production response coordinates are therefore nodal values
\(a_i=\delta u(h_i)\) of the latent sound-speed perturbation. A target
functional covariance \(C_{\rm EOS}(h,h')\) is evaluated at those nodes.
Increasing node density then refines one fixed process. Baseline covariance:
squared exponential; robustness kernels: Mat\'ern-3/2 and exponential.
The legacy Gaussian builder remains only to reproduce the first pilot.

Universal directions are covectors, not displacement eigenvectors. The
primary local scalar in a two-observable plane is
\[
n_A G^{AB}n_B,\qquad n_A t^A=0,\qquad n_A g^{AB}n_B=1.
\]
The implementation tests invariance under consistent EOS-basis and
observable-coordinate transformations.


## 2026-09-18 integrability, curvature, and transition-stage decisions

H3 passed in the tested domain.  A genuine three-dimensional Frobenius audit,
not the automatic 2D I--Love integrability, gives normalized obstruction
\(\eta_F\) with median \(1.23\times10^{-3}\), 90th percentile
\(3.48\times10^{-3}\), and maximum \(7.36\times10^{-3}\) across nine
high-resolution zero/smooth/rough patches.  A cubic scalar potential trained
only on soft-covector gradients generalizes across held-out EOS backgrounds
with median \(\sigma(F)=8.48\times10^{-3}\), versus
\(1.98\times10^{-2}\) for the one-dimensional I--Love gradient
reconstruction.

H4 passed only within a finite perturbative radius.  Across nine background
configurations and random GP EOS directions, median relative RMSE for the
quadratic response is 0.0062, 0.024, 0.156, and 0.553 at latent amplitudes
0.05, 0.10, 0.25, and 0.50.  Do not imply second order explains arbitrary
finite EOS excursions.

Controlled localized sound-speed softening shows a specific H5 mechanism:
ordinary I--Love can degrade strongly while the full soft/hard normal
hierarchy survives.  In the refined center=0.16, width=0.04, depth=2 case at
h_c=0.30, the I--Love alignment is 0.90245 and I--Love/C--Love RMS ratio
0.49559, while the full soft/hard RMS ratio remains 0.09619.  The
covariance-weighted I--Love variance increases by 85.06, with 90.24% of the
positive excess direct sensitivity inside the imposed h=0.14--0.18 layer
and a peak at h=0.155.

Novelty re-check specific to this stage:
- Jiang & Yagi (Phys. Rev. D 101, 124006, 2020) already derive analytic
  I--Love--C relations.  Do not claim the inclusion of compactness or a
  C-I-Love relation/surface as first.
- Han & Steiner (Phys. Rev. D 99, 083014, 2019) already show that sharp
  phase transitions can limit the accuracy of tidal universal relations.
  Do not claim first phase-transition breakdown of a universal relation.
- Hybrid/exotic-star studies, including recent elastic hybrid-star work,
  show that useful universal relations can persist with exotic cores.
- The candidate new contribution is narrower: the metric-aware differential
  response operator distinguishes breakdown of a chosen projection from
  breakdown of low-dimensional response itself, identifies rotation of the
  soft covector, and localizes the added sensitivity to the responsible EOS
  layer.  This wording remains provisional until the systematic
  transition-location refinement is complete.

The 33-node transition screen contains 69 stable/physical points out of 96.
Its strongest stable degradation follows a moving ridge:
\((h_c,h_{\rm tr})=(0.20,0.10),(0.30,0.14),(0.40,0.18)\), suggesting
\(h_{\rm tr}/h_c\sim0.45\)--0.50.  This must be checked at 65 nodes and
mapped into \(r/R\) and \(m/M\) before physical interpretation.


## 2026-09-18 final terminology/novelty re-check

A targeted search through September 18, 2026 still finds no neutron-star
universal-relation paper that combines the nonparametric EOS response
operator, an explicit EOS metric/covariance, quotient by the stellar-sequence
direction, the full normal-response spectrum, Frobenius integrability, and
curvature-controlled finite scatter.

Terminology caveat: Siffert, "Response Geometry for Einstein metrics"
(arXiv:2608.08777, 2026) independently uses the phrase "response geometry"
for a general differential observation-map construction and response tensor
in the mathematical study of Einstein metrics. This is not neutron-star
universal-relation prior art and does not overlap the present calculations,
but the manuscript must not imply that the generic phrase "response
geometry" is coined here.

Recent 2026 neutron-star work found in this re-check continues to focus on
empirical/data-driven universal relations, large EOS databases, rotating-star
relations, sensitivity to microscopic EOS parameters, or neural surrogates of
the EOS-to-observable map. None found performs the normal-spectrum and
integrability construction used here.
