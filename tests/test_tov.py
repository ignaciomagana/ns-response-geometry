import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)

from ns_response_geometry.eos import IncompressibleEOS
from ns_response_geometry.tov import incompressible_exact, solve_star


@pytest.mark.parametrize("h_c", [0.05, 0.15, 0.30])
def test_incompressible_star_matches_exact_solution(h_c):
    eos = IncompressibleEOS(epsilon0=1.0)
    numeric = solve_star(eos, h_c, n_steps=8192)
    exact = incompressible_exact(1.0, h_c)

    np.testing.assert_allclose(
        np.asarray(numeric.compactness),
        np.asarray(exact.compactness),
        rtol=3e-5,
        atol=1e-8,
    )
    np.testing.assert_allclose(
        np.asarray(numeric.radius),
        np.asarray(exact.radius),
        rtol=3e-5,
        atol=1e-8,
    )
    np.testing.assert_allclose(
        np.asarray(numeric.mass),
        np.asarray(exact.mass),
        rtol=5e-5,
        atol=1e-8,
    )


def test_background_map_is_differentiable_in_central_enthalpy():
    eos = IncompressibleEOS(epsilon0=1.0)

    def mass_of_h(h):
        return solve_star(eos, h, n_steps=2048).mass

    grad = jax.grad(mass_of_h)(0.15)
    assert np.isfinite(float(grad))
    assert float(grad) > 0.0


def test_profile_endpoint_matches_solve_star():
    eos = IncompressibleEOS(epsilon0=1.0)
    star = solve_star(eos, 0.15, n_steps=2048)

    from ns_response_geometry.tov import solve_star_profile
    profile = solve_star_profile(eos, 0.15, n_steps=2048)

    np.testing.assert_allclose(profile.radius[-1], star.radius, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(profile.mass[-1], star.mass, rtol=1e-12, atol=1e-12)
    assert np.all(np.diff(np.asarray(profile.enthalpy)) < 0.0)
    assert np.all(np.diff(np.asarray(profile.radius)) > 0.0)
    assert np.all(np.diff(np.asarray(profile.mass)) > 0.0)
