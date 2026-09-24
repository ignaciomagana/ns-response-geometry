# Reproducibility environment

**Frozen:** 2026-09-24

The paper-level numerical results have been checked with the following CPU
Python environment in GitHub Actions:

- Python 3.11
- JAX 0.10.2
- JAXLIB 0.10.2
- NumPy 2.4.6
- SciPy 1.17.1
- pytest 9.1.1
- ml_dtypes 0.6.0
- opt_einsum 3.4.0
- `JAX_ENABLE_X64=true`
- `XLA_PYTHON_CLIENT_PREALLOCATE=false` for science workflows

The exact Python constraints are stored in
`constraints-paper-20260924.txt`.

The package metadata in `pyproject.toml` intentionally retains broad lower
bounds for normal development. Paper-reproduction workflows install through
the frozen constraints file so future dependency resolution does not silently
change the numerical environment.

LaTeX is built in GitHub Actions with `xu-cheng/latex-action@v3`. The paper
workflow renders every page with Poppler and tracks `paper/main.pdf`.

If the frozen environment must be updated, rerun the unit tests, core science
reproduction, branch-safe transition control, manuscript build, and numerical
claim audit before changing this file.
