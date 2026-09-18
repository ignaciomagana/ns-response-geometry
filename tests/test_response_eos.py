import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from ns_response_geometry.eos import RelativisticPolytrope
from ns_response_geometry.observables import solve_observables
from ns_response_geometry.response import response_jacobian, sequence_tangent
from ns_response_geometry.response_eos import build_sound_speed_eos


def test_zero_modes_reconstruct_reference_polytrope():
    reference = RelativisticPolytrope(K=100.0, gamma=2.0)
    coeffs = jnp.zeros(5)
    eos = build_sound_speed_eos(
        reference,
        coeffs,
        h_match=0.02,
        h_max=0.5,
        n_high=2048,
    )

    h = jnp.linspace(0.025, 0.45, 20)
    np.testing.assert_allclose(
        np.asarray(eos.pressure(h)),
        np.asarray(reference.pressure(h)),
        rtol=3.0e-5,
        atol=1.0e-10,
    )
    np.testing.assert_allclose(
        np.asarray(eos.energy_density(h)),
        np.asarray(reference.energy_density(h)),
        rtol=3.0e-5,
        atol=1.0e-10,
    )


def test_zero_modes_reproduce_reference_stellar_observables():
    reference = RelativisticPolytrope(K=100.0, gamma=2.0)
    eos = build_sound_speed_eos(
        reference,
        jnp.zeros(5),
        h_match=0.02,
        h_max=0.5,
        n_high=2048,
    )

    direct = solve_observables(reference, 0.15, n_steps=4096)
    rebuilt = solve_observables(eos, 0.15, n_steps=4096)

    np.testing.assert_allclose(
        np.asarray(rebuilt[:8]),
        np.asarray(direct[:8]),
        rtol=2.0e-4,
        atol=1.0e-9,
    )


def test_response_jacobian_and_sequence_tangent_are_finite():
    reference = RelativisticPolytrope(K=100.0, gamma=2.0)
    coeffs = jnp.zeros(3)
    kwargs = dict(h_match=0.02, h_max=0.4, n_high=512)

    J = response_jacobian(
        reference,
        coeffs,
        0.15,
        n_steps=1024,
        eos_kwargs=kwargs,
    )
    tangent = sequence_tangent(
        reference,
        coeffs,
        0.15,
        n_steps=1024,
        eos_kwargs=kwargs,
    )

    assert J.shape == (3, 3)
    assert tangent.shape == (3,)
    assert np.all(np.isfinite(np.asarray(J)))
    assert np.all(np.isfinite(np.asarray(tangent)))
    assert np.linalg.norm(np.asarray(J)) > 0.0
