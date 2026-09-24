# Current state

**Last updated:** 2026-09-24

## Stage

Stage 6 — controlled breakdown and physical localization. Stages 1--5 are
complete for the current static/slow-rotation observable set
\((C,\bar I,\Lambda_2)\).

## Completed

- Differential EOS-to-observable formulation and conservative novelty audit.
- Validated differentiable enthalpy-coordinate TOV solver.
- Static \(l=2\) tides and first-order Hartle moment of inertia.
- Exact constant-density and Newtonian limiting checks.
- Independent adaptive radius-coordinate DOP853 cross-check.
- Thermodynamically consistent latent sound-speed EOS reconstruction.
- Production nodal EOS coordinates representing a fixed functional metric.
- Metric-aware response \(G=JC_{\rm EOS}J^T\).
- Cotangent-space normalization and coordinate-invariance tests.
- Nodal, covariance-kernel, correlation-length, reference-EOS,
  observable-metric, derivative, and resolution robustness.
- Broad finite-background EOS ensemble.
- Full three-observable normal spectrum.
- High-resolution Frobenius integrability audit.
- Cross-validated scalar potential reconstructed from covector gradients only.
- Second-order curvature versus finite EOS displacement.
- Controlled localized sound-speed softening.
- Systematic transition center/width/depth screen.
- 65-node transition-location refinement and mapping into \(r/R\) and \(m/M\).
- Matched-reference transition-location null test across \(\Gamma=1.70,1.85,2.00\).
- Fixed-final-compactness control that removes the dominant global-expansion confound.
- Finer fixed-final-compactness refinement completed in GitHub Actions and promoted to `results/fixed_final_compactness_refinement_20260918.json` with workflow provenance.
- Production figures and manuscript introduction/results/discussion updates.
- Core science reproduction and paper-build workflows passing in GitHub Actions.
- Manuscript workflow now compiles and renders every PDF page on every push/pull request.

## Central results

### Soft normal mode

Across the broad 44-point stable EOS-background ensemble:
- median soft/hard normal RMS ratio \(=0.0204\);
- 90th percentile \(=0.0469\);
- maximum \(=0.0883\);
- median soft-covector/I--Love alignment \(=0.9977\).

For \(C\ge0.05\), the maximum normal-mode ratio is \(0.0797\) and the minimum
I--Love alignment is \(0.9756\).

### Integrability

The high-resolution Frobenius diagnostic gives
\[
{\rm median}(\eta_F)=1.23\times10^{-3},\quad
q_{0.90}=3.48\times10^{-3},\quad
\max\eta_F=7.36\times10^{-3}.
\]

A cubic scalar potential reconstructed only from soft-covector gradients
generalizes across held-out EOS backgrounds with median
\(\sigma(F)=8.48\times10^{-3}\), versus
\(1.98\times10^{-2}\) for the one-dimensional I--Love gradient
reconstruction.

### Curvature

Across zero/smooth/rough backgrounds, median relative RMSE of the quadratic
finite-response prediction is \(0.0062,0.024,0.156,0.553\) at latent
amplitudes \(0.05,0.10,0.25,0.50\), respectively. Second order is
quantitative only over a finite neighborhood.

### Controlled breakdown

A localized causal sound-speed softening layer can strongly degrade the
I--Love projection while the full soft/hard hierarchy survives.

For the refined center \(h=0.16\), width \(0.04\), depth \(2\) case at
\(h_c=0.30\):
- I--Love alignment \(=0.90245\);
- I--Love/C--Love RMS ratio \(=0.49559\);
- full soft/hard RMS ratio \(=0.09619\);
- covariance-weighted I--Love variance increase \(=85.06\);
- \(90.24\%\) of the positive excess direct sensitivity lies inside the
  imposed \(h=0.14\)--0.18 layer;
- excess peaks at \(h=0.155\).

The 33-node systematic screen contains 69 stable/physical points out of 96.
The strongest degradation follows a moving enthalpy ridge near
\(h_{\rm tr}/h_c\simeq0.45\)--0.50.

The 65-node refinement confirms the ridge. For the scaled-location scan with
width \(0.25h_c\), depth \(1.5\), and \(C\ge0.1\), the strongest case occurs
at \(h_{\rm tr}/h_c=0.45\). At \(h_c=0.40\):
- \(C=0.113\);
- I--Love alignment \(=0.922\);
- I--Love/C--Love RMS ratio \(=0.447\);
- full soft/hard RMS ratio \(=0.135\);
- I--Love variance enhancement \(=56.7\);
- \(93.3\%\) of positive excess sensitivity lies on the imposed layer.
The deformed layer is centered at \(r/R=0.453\), \(m/M=0.759\).

## Current scientific target

The transition-location program is complete for the present controlled
polytropic family.  After fixing the final compactness, the stable maximizing
locations span \(h_{\rm tr}/h_c=0.35\)--0.65 and
\(m_{\rm tr}/M=0.340\)--0.852.  Seven of eight maxima fall in the narrower
range \(r_{\rm tr}/R=0.426\)--0.457\), but one lies at \(0.533\), and the
\(\Gamma=1.70\) sequence has no stable point at
\(C_{\rm final}=0.14\).  The result therefore does not support a universal
vulnerable shell; the fractional-radius clustering is retained only as a
diagnostic observation.

The current target is the final paper-quality pass:
1. targeted literature/novelty re-check through the current date;
2. numerical/claim provenance audit against all committed result files;
3. figure and rendered-page readability audit;
4. clean-room consistency review of the manuscript against the code/results;
5. only then decide whether additional physics is required before submission.

## Validation / execution status

GitHub Actions are operational again after the repository was made public.
Recent tests, paper, science-reproduction, transition-breakdown, and
transition-location workflows are green. The paper workflow now runs on every
push and pull request, compiles `paper/main.tex`, renders all pages to PNG,
and uploads the PDF plus page renders as the `manuscript-render` artifact.

Versioned science results include:
- results/response_robustness_20260918.json
- results/background_ensemble_fine_20260918.json
- results/normal_spectrum_20260918.json
- results/integrability_frobenius_20260918.json
- results/potential_reconstruction_20260918.json
- results/curvature_scatter_20260918.json
- results/transition_breakdown_20260918.json
- results/transition_scan_screen_20260918.json
- results/transition_location_refinement_20260918.json
- results/matched_compactness_transition_20260918.json
- results/fixed_final_compactness_transition_20260918.json
- results/fixed_final_compactness_refinement_20260918.json

## Manuscript status

- Introduction: rewritten around completed analysis.
- Response geometry: drafted and synchronized.
- Stellar structure/numerics: validated and written.
- Local hierarchy/full normal spectrum: written.
- Integrability/global reconstruction: written.
- Curvature: written.
- Controlled breakdown, matched-reference null test, and fixed-final-
  compactness refinement: written.
- Production figures: inserted, including the fixed-final-compactness
  location summary.
- Discussion/conclusions: updated through the final H5 interpretation.
- Remaining work before submission-level review: final novelty/literature
  check, claim-provenance audit, rendered figure/readability audit, and
  clean-room numerical consistency pass.

## Session protocol

At the beginning of every session read STATE.md, MEMORY.md,
SCIENCE_CONTRACT.md, inspect recent commits/results, and inspect CI.
At the end update STATE.md, record durable decisions in MEMORY.md,
update the paper, leave every reported numerical result reproducible from a
committed script plus a versioned result file, and verify the per-round paper
workflow produced `main.pdf` plus rendered page images. Inspect the rendered
pages rather than treating a successful LaTeX compile as sufficient.


## 2026-09-24 fixed-final control audit failure

The first fixed-final-compactness refinement is superseded and must not be
used for paper claims. An explicit audit found that its
\(\Gamma=1.70\), baseline-\(C=0.16\) reference has
\(dM/dh_c=-1.903<0\), and that \(C(D)\) is non-monotonic at 18 of the 51
tested transition locations. The original depth solver assumed global
monotonicity, so several roots were not branch-safe.

A replacement branch-safe audit is running with a common baseline
\(C=0.14\), targets \(C_{\rm final}=0.10,0.12\), explicit positive
baseline mass-slope checks, sampled depth curves, and selection of the
smallest-depth root continuously connected to \(D=0\). Until that result is
complete, the fixed-final subsection/figure are provisional. The original \(\Gamma=2\) fixed-depth scans are unaffected. The cross-EOS matched-reference control is also partially superseded because it includes the same invalid \(\Gamma=1.70\), baseline-\(C=0.16\) reference.
