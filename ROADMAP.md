# Roadmap

The paper is developed in lockstep with the mathematics/code. Every stage has a manuscript deliverable.

## Stage 0 — formulation and prior-art boundary

**Science**
- Define EOS-to-observable map.
- Define EOS metric/covariance and induced response metric.
- Separate stellar-sequence tangent from EOS directions.
- Define local universality and global integrability.
- State second-order curvature program.

**Paper**
- Draft Introduction.
- Draft Mathematical formulation.
- Write prior-art distinctions conservatively.
- Create Results section structure with no invented outcomes.

**Exit**
Definitions are internally coherent and coordinate/basis caveats are explicit.

## Stage 1 — validated stellar structure

**Code**
- EOS protocol.
- Enthalpy-coordinate TOV.
- (l=2) static tidal perturbation and (k_2,Lambda_2).
- Hartle frame dragging and (I).
- convergence/benchmark tests.

**Paper**
- Complete Stellar structure subsection with equations and boundary conditions.
- Add numerical validation subsection.
- Add first validation table/figure when generated.

**Exit**
(M,R,Lambda_2,I) independently validated.

## Stage 2 — finite-dimensional response kernels

**Code**
- bounded sound-speed latent field;
- basis expansion;
- JAX Jacobians;
- symmetric finite-difference checks;
- density/enthalpy-resolved sensitivity diagnostics.

**Science**
- inspect (K_I,K_Lambda,K_C);
- establish basis convergence.

**Paper**
- write EOS perturbation and functional-derivative section;
- first kernel figures;
- explicitly compare to Chan/Yip-Leung stationarity.

**Exit**
Gradients are trustworthy.

## Stage 3 — metric-aware transverse geometry

**Code**
- (C_{m EOS});
- induced (G_{m EOS});
- sequence tangent;
- quotient/projector;
- eigenmodes and invariance tests.

**Science**
- I--Love vs C--Love response hierarchy;
- dependence on mass and EOS metric.

**Paper**
- this is the central Results section;
- response spectra and transverse-mode figures.

**Exit**
Either H1/H2 survive robustly or the project is reframed.

## Stage 4 — local-to-global integrability

**Code/math**
- transport local low-response covectors;
- path-dependence diagnostics;
- Frobenius/integrability tests where applicable;
- reconstruct global relation without fitting the standard I--Love polynomial.

**Paper**
- integrability subsection;
- reconstructed relation compared to empirical I--Love only after derivation.

**Exit**
Quantify whether local geometry explains a global relation.

## Stage 5 — curvature and finite scatter

**Code/math**
- Hessian-vector products/directional second derivatives;
- predicted finite perturbation scatter;
- nonlinear EOS ensemble validation.

**Paper**
- connect curvature to percent-level residuals;
- compare predicted and measured scatter.

## Stage 6 — controlled breakdown

**Code/science**
- broad nonparametric EOS ensemble;
- sharp sound-speed structure/phase transitions;
- identify density-resolved modes causing loss of universality.

**Paper**
- breakdown section;
- clarify domain of validity rather than claiming universality everywhere.

## Stage 7 — discovery extension

Only after stages 1--6 work.

Candidate observables:
[
ar Q,quad Lambda_3,quad Lambda_4,quad Momega_f,quad 	ext{dynamic response coefficients}.
]

Search for new approximately integrable low-response combinations. Any claimed new universal relation requires an explicit literature re-check at the time of discovery.

## Final paper logic

1. Existing quasi-universal relations are known empirically and have partial analytic explanations.
2. We define a differential, metric-aware notion of EOS sensitivity.
3. Quotienting the stellar-sequence direction exposes transverse EOS sensitivity.
4. Known universality becomes a stringent validation of the construction.
5. Integrability determines when local insensitivity produces a global relation.
6. Curvature controls finite residual scatter.
7. Breakdown reveals which EOS directions destroy universality.
8. Only then use the framework for relation discovery.
