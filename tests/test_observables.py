import math

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)

from ns_response_geometry.eos import IncompressibleEOS, RelativisticPolytrope
from ns_response_geometry.observables import solve_observables
from ns_response_geometry.tov import incompressible_exact


@pytest.mark.parametrize("h_c", [0.05, 0.15, 0.30])
def test_extended_solver_preserves_exact_incompressible_background(h_c):
    eos = IncompressibleEOS(epsilon0=1.0)
    numeric = solve_observables(eos, h_c, n_steps=8192)
    exact = incompressible_exact(1.0, h_c)

    np.testing.assert_allclose(numeric.mass, exact.mass, rtol=3e-5, atol=1e-8)
    np.testing.assert_allclose(numeric.radius, exact.radius, rtol=3e-5, atol=1e-8)


def test_incompressible_low_compactness_limits():
    eos = IncompressibleEOS(epsilon0=1.0)
    result = solve_observables(eos, 0.0025, n_steps=8192)

    # Newtonian homogeneous-fluid limits: k2 -> 3/4, I/(MR^2) -> 2/5.
    assert 0.70 < float(result.k2) < 0.75
    assert abs(float(result.i_mr2) - 0.4) < 0.005


def test_n1_polytrope_low_compactness_limits():
    eos = RelativisticPolytrope(K=100.0, gamma=2.0)
    result = solve_observables(eos, 0.0025, n_steps=16384)

    k2_newton = (15.0 - math.pi**2) / (2.0 * math.pi**2)
    i_mr2_newton = (2.0 / 3.0) * (math.pi**2 - 6.0) / math.pi**2

    assert float(result.compactness) < 0.003
    assert abs(float(result.k2) - k2_newton) < 0.01
    assert abs(float(result.i_mr2) - i_mr2_newton) < 0.005


def test_extended_observables_are_differentiable():
    eos = IncompressibleEOS(epsilon0=1.0)

    grad_lambda = jax.grad(
        lambda h: jax.numpy.log(solve_observables(eos, h, n_steps=2048).lambda2)
    )(0.15)
    grad_ibar = jax.grad(
        lambda h: jax.numpy.log(solve_observables(eos, h, n_steps=2048).ibar)
    )(0.15)

    assert np.isfinite(float(grad_lambda))
    assert np.isfinite(float(grad_ibar))
