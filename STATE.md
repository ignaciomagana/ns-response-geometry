# Current state

**Last updated:** 2026-09-24

## Stage

Paper-1 analysis complete through Stage 6 for the current static/slow-rotation
observable set \((C,\bar I,\Lambda_2)\). Stage 7 discovery extensions are
deferred.

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
- Preliminary cross-EOS transition controls audited and superseded after a
  branch/stability failure was found.
- Branch-safe fixed-final-compactness control across
  \(\Gamma=1.70,1.85,2.00\), using positive-mass-slope baselines,
  first depth roots connected to \(D=0\), and pathwise mass-slope checks.
- Branch-safe result promoted to
  `results/fixed_final_compactness_branchsafe_20260924.json` with GitHub
  Actions provenance.
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

The controlled transition-location program is complete for the present
polytropic family.  The original \(\Gamma=2\) fixed-depth experiment
demonstrates that response vulnerability can be localized for a specified
configuration.  The branch-safe cross-EOS control shows that this location
does not define a universal shell: for all six
\((\Gamma,C_{\rm final})\) cases, the largest retained response occurs at
the deepest transition location that still passes the positive-mass-slope
path screen.  Those boundary-adjacent locations span
\[
h_{\rm tr}/h_c=0.475\text{--}0.725,\qquad
r_{\rm tr}/R=0.368\text{--}0.476,\qquad
m_{\rm tr}/M=0.235\text{--}0.675 .
\]
The cross-EOS result is therefore a negative localization result: vulnerability
grows toward the screened branch boundary rather than selecting a
reference-EOS-independent interior shell.

The literature/novelty sweep, quantitative provenance audit, and clean-room
code/manuscript consistency pass are complete.  The final manuscript build and rendered-page inspection at current HEAD are complete. All 19 rendered pages were inspected with no clipping, overlap, or broken glyphs. The paper workflow now commits the rendered PDF to `paper/main.pdf` on successful pushes in addition to uploading the manuscript artifact. The current round is complete pending only any optional additional physics or author-level editorial changes.

## Validation / execution status

GitHub Actions are operational again after the repository was made public.
Recent tests, paper, science-reproduction, transition-breakdown, and
transition-location workflows are green. The paper workflow now runs on every
push and pull request, compiles `paper/main.tex`, renders all pages to PNG,
and uploads the PDF plus page renders as the `manuscript-render` artifact.

Versioned science results include:
- results/stage1_validation_20260918.json
- results/preliminary_response_20260918.json
- results/response_robustness_20260918.json
- results/background_ensemble_fine_20260918.json
- results/normal_spectrum_20260918.json
- results/integrability_frobenius_20260918.json
- results/potential_reconstruction_20260918.json
- results/curvature_scatter_20260918.json
- results/transition_breakdown_20260918.json
- results/transition_scan_screen_20260918.json
- results/transition_location_refinement_20260918.json
- results/matched_compactness_transition_20260918.json  [superseded; provenance only]
- results/fixed_final_compactness_transition_20260918.json  [superseded; provenance only]
- results/fixed_final_compactness_refinement_20260918.json  [superseded; provenance only]
- results/fixed_final_compactness_branchsafe_20260924.json

## Manuscript status

- Introduction: rewritten around completed analysis.
- Response geometry: drafted and synchronized.
- Stellar structure/numerics: validated and written.
- Local hierarchy/full normal spectrum: written.
- Integrability/global reconstruction: written.
- Curvature: written.
- Controlled breakdown and fixed-depth localization: written.
- Superseded cross-EOS controls removed from live scientific claims.
- Branch-safe cross-EOS control and negative universal-shell result: written.
- Production figures: inserted, including branch-safe response curves.
- Discussion/conclusions: synchronized with the final H5 interpretation.
- Literature/novelty, claim-provenance, and clean-room consistency audits:
  complete.
- Final rendered-page inspection: complete (19/19 pages inspected).
- Rendered manuscript PDF: committed at `paper/main.pdf`.

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

The correction is complete.  The branch-safe replacement uses a common
baseline \(C=0.14\), targets \(C_{\rm final}=0.10,0.12\), explicit
positive baseline mass-slope checks, sampled depth curves, first roots
connected to \(D=0\), and positive-mass-slope checks along the deformation
path.  Its result is versioned in
`results/fixed_final_compactness_branchsafe_20260924.json`.  The original
\(\Gamma=2\) fixed-depth scans remain valid; the older cross-EOS controls
remain in the repository only as superseded provenance.


## Continuity guardrails

- `results/fixed_final_compactness_branchsafe_20260924.json` is the only
  current cross-EOS fixed-final transition control.
- The matched-reference, original fixed-final, and fixed-final refinement
  analyses are historical provenance only; their automatic workflows are
  disabled.
- The broad EOS ensemble is a causal function-space stress test, not a
  realistic nuclear-EOS posterior.
- `paper/CLAIM_PROVENANCE.md` is the authority for quantitative manuscript
  claims; `LEGACY.md` is the authority for superseded artifacts.
- Stage 7 is deferred. Do not add new observables to paper 1 without an
  explicit scope change and fresh literature audit.


## 2026-09-24 full continuity audit

A full repository pass was completed after multiple disconnected sessions.

### Recovery order for a new session

Read these in order before doing new work:

1. `STATE.md` — current scientific state and next-scope boundary.
2. `SCIENCE_CONTRACT.md` — exact claims/scope/falsification contract.
3. `paper/PAPER_STATE.md` — authoritative manuscript status.
4. `paper/CLAIM_PROVENANCE.md` — quantitative claim-to-result map.
5. `MEMORY.md` — chronology, mathematical decisions, and corrections.
6. `LEGACY.md` — analyses that must not be revived as current evidence.
7. `results/README.md` — current versus superseded result manifest.
8. `ENVIRONMENT.md` and `constraints-paper-20260924.txt` —
   frozen numerical reproduction environment.
9. Inspect recent commits and GitHub Actions before changing code or paper.

### Audit findings and fixes

- No open PRs or issues.
- All committed result JSON files parse successfully.
- Live TeX has no TODO/FIXME placeholders.
- All 18 manuscript citation keys resolve to the 18 bibliography entries.
- All checked `\ref`/`\eqref` targets resolve.
- Stale `PAPER_STATE.md` fixed-final claims were replaced by the
  branch-safe result.
- The earlier matched-reference and fixed-final cross-EOS analyses are
  explicitly marked superseded and their workflows are manual-only.
- The broad-EOS claim domain was corrected from an overbroad hadronic wording
  to the tested cold, causal latent-EOS domain.
- The Python reproduction environment is frozen in
  `constraints-paper-20260924.txt`.
- The paper PDF auto-publish workflow was made race-safe for concurrent pushes.

### Frozen-environment verification

Using the frozen 2026-09-24 constraints, the following GitHub Actions
workflows completed successfully:

- unit tests;
- Stage-1 validation report;
- response robustness;
- core science reproduction (broad ensemble, normal spectrum, Frobenius,
  potential reconstruction, curvature);
- transition breakdown;
- transition-location refinement;
- transition scan;
- branch-safe fixed-final-compactness control;
- manuscript compile/render.

### Not missing, but optional strengthening

The current manuscript does not require the following for its stated claims,
and explicitly limits its scope accordingly:

- a named/realistic nuclear-EOS ensemble or nuclear-theory posterior;
- a radial-mode eigenvalue stability calculation beyond the
  positive-(dM/dh_c) screen;
- microphysical first-order phase-transition models rather than controlled
  sound-speed softening profiles;
- Stage-7 observables such as (ar Q,Lambda_3,f)-modes, or dynamical
  tides.

These would strengthen or extend the work, but are not evidence already
claimed in paper 1.

### Author/repository choices still open

- repository license: none chosen; do not add one without an author decision;
- manuscript affiliation/coauthor metadata: not populated beyond the current
  author entry; do not invent it;
- release/tag/DOI/CITATION metadata: not yet created.

### Bottom line

There is no known lost analysis or unresolved session state required to
support the current paper. Paper-1 science through Stage 6 is complete under
the declared scope. New work should either be an explicit strengthening pass
(realistic EOS/radial stability) or a separately scoped Stage-7 extension.
