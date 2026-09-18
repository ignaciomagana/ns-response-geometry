"""EOS response kernels and local observable geometry."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from .geometry import induced_response, transverse_response
from .observables import solve_observables
from .response_eos import build_nodal_sound_speed_eos


def log_observable_vector(eos, h_c, *, n_steps=4096):
    """Return (ln C, ln Ibar, ln Lambda_2)."""
    result = solve_observables(eos, h_c, n_steps=n_steps)
    return jnp.log(
        jnp.stack((result.compactness, result.ibar, result.lambda2))
    )


def response_jacobian(
    reference_eos,
    coefficients,
    h_c,
    *,
    n_steps=4096,
    eos_kwargs=None,
    eos_builder=build_nodal_sound_speed_eos,
):
    """Jacobian d(ln C,ln Ibar,ln Lambda)/d EOS coordinates."""
    eos_kwargs = {} if eos_kwargs is None else dict(eos_kwargs)

    def observable_map(a):
        eos = eos_builder(reference_eos, a, **eos_kwargs)
        return log_observable_vector(eos, h_c, n_steps=n_steps)

    return jax.jacfwd(observable_map)(jnp.asarray(coefficients))


def sequence_tangent(
    reference_eos,
    coefficients,
    h_c,
    *,
    n_steps=4096,
    eos_kwargs=None,
    eos_builder=build_nodal_sound_speed_eos,
):
    """d(ln C,ln Ibar,ln Lambda)/d h_c at fixed EOS."""
    eos_kwargs = {} if eos_kwargs is None else dict(eos_kwargs)
    eos = eos_builder(reference_eos, coefficients, **eos_kwargs)

    def observable_map(h):
        return log_observable_vector(eos, h, n_steps=n_steps)

    return jax.jacfwd(observable_map)(jnp.asarray(h_c))


def local_response_geometry(
    reference_eos,
    coefficients,
    h_c,
    eos_covariance,
    *,
    n_steps=4096,
    eos_kwargs=None,
    observable_metric=None,
    eos_builder=build_nodal_sound_speed_eos,
):
    """Return J, sequence tangent, G and projected transverse response."""
    J = response_jacobian(
        reference_eos,
        coefficients,
        h_c,
        n_steps=n_steps,
        eos_kwargs=eos_kwargs,
        eos_builder=eos_builder,
    )
    tangent = sequence_tangent(
        reference_eos,
        coefficients,
        h_c,
        n_steps=n_steps,
        eos_kwargs=eos_kwargs,
        eos_builder=eos_builder,
    )
    G = induced_response(J, eos_covariance)
    G_perp = transverse_response(G, tangent, observable_metric)
    return J, tangent, G, G_perp
