"""EOS response kernels and local observable geometry."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from .geometry import induced_response, transverse_response
from .observables import solve_observables
from .response_eos import build_sound_speed_eos


def log_observable_vector(eos, h_c, *, n_steps=4096):
    """Return (ln C, ln Ibar, ln Lambda_2)."""
    result = solve_observables(eos, h_c, n_steps=n_steps)
    return jnp.log(
        jnp.stack((result.compactness, result.ibar, result.lambda2))
    )


def response_jacobian(
    reference_eos,
    coeffs,
    h_c,
    *,
    n_steps=4096,
    eos_kwargs=None,
):
    """Jacobian d(ln C,ln Ibar,ln Lambda)/d coefficients."""
    eos_kwargs = {} if eos_kwargs is None else dict(eos_kwargs)

    def observable_map(a):
        eos = build_sound_speed_eos(reference_eos, a, **eos_kwargs)
        return log_observable_vector(eos, h_c, n_steps=n_steps)

    return jax.jacfwd(observable_map)(jnp.asarray(coeffs))


def sequence_tangent(
    reference_eos,
    coeffs,
    h_c,
    *,
    n_steps=4096,
    eos_kwargs=None,
):
    """d(ln C,ln Ibar,ln Lambda)/d h_c at fixed EOS."""
    eos_kwargs = {} if eos_kwargs is None else dict(eos_kwargs)
    eos = build_sound_speed_eos(reference_eos, coeffs, **eos_kwargs)

    def observable_map(h):
        return log_observable_vector(eos, h, n_steps=n_steps)

    return jax.jacfwd(observable_map)(jnp.asarray(h_c))


def local_response_geometry(
    reference_eos,
    coeffs,
    h_c,
    eos_covariance,
    *,
    n_steps=4096,
    eos_kwargs=None,
    observable_metric=None,
):
    """Return J, sequence tangent, G and the projected transverse response."""
    J = response_jacobian(
        reference_eos,
        coeffs,
        h_c,
        n_steps=n_steps,
        eos_kwargs=eos_kwargs,
    )
    tangent = sequence_tangent(
        reference_eos,
        coeffs,
        h_c,
        n_steps=n_steps,
        eos_kwargs=eos_kwargs,
    )
    G = induced_response(J, eos_covariance)
    G_perp = transverse_response(G, tangent, observable_metric)
    return J, tangent, G, G_perp
