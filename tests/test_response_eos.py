import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from ns_response_geometry.eos import RelativisticPolytrope
from ns_response_geometry.observables import solve_observables
from ns_response_geometry.response import response_jacobian, sequence_tangent
from ns_response_geometry.response_eos import build_nodal_sound_speed_eos


def test_zero_nodal_field_reconstructs_reference_polytrope():
    reference = RelativisticPolytrope(K=100.0, gamma=2.0)
    nodes = jnp.linspace(0.02, 0.5, 9)
    eos = build_nodal_sound_speed_eos(
        reference,
        jnp.zeros(nodes.shape[0]),
        h_match=0.02,
        h_max=0.5,
        h_nodes=nodes,
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


def test_zero_nodal_field_reproduces_reference_stellar_observables():
    reference = RelativisticPolytrope(K=100.0, gamma=2.0)
    nodes = jnp.linspace(0.02, 0.5, 9)
    eos = build_nodal_sound_speed_eos(
        reference,
        jnp.zeros(nodes.shape[0]),
        h_match=0.02,
        h_max=0.5,
        h_nodes=nodes,
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
    nodes = jnp.linspace(0.02, 0.4, 5)
    values = jnp.zeros(nodes.shape[0])
    kwargs = dict(h_match=0.02, h_max=0.4, h_nodes=nodes, n_high=512)

    J = response_jacobian(
        reference,
        values,
        0.15,
        n_steps=1024,
        eos_kwargs=kwargs,
    )
    tangent = sequence_tangent(
        reference,
        values,
        0.15,
        n_steps=1024,
        eos_kwargs=kwargs,
    )

    assert J.shape == (3, 5)
    assert tangent.shape == (3,)
    assert np.all(np.isfinite(np.asarray(J)))
    assert np.all(np.isfinite(np.asarray(tangent)))
    assert np.linalg.norm(np.asarray(J)) > 0.0


def test_nodal_response_converges_with_grid_refinement_at_zero_field():
    reference = RelativisticPolytrope(K=100.0, gamma=2.0)
    h_c = 0.16

    norms = []
    for n_nodes in (5, 9, 17):
        nodes = jnp.linspace(0.02, 0.5, n_nodes)
        J = response_jacobian(
            reference,
            jnp.zeros(n_nodes),
            h_c,
            n_steps=1024,
            eos_kwargs=dict(
                h_match=0.02,
                h_max=0.5,
                h_nodes=nodes,
                n_high=512,
            ),
        )
        norms.append(float(jnp.linalg.norm(J)))

    assert all(np.isfinite(norms))
    assert norms[-1] > 0.0
