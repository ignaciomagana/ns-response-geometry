"""Differentiable TOV background solver in relativistic enthalpy.

The integration uses a fixed-step RK4 scheme so that the map from EOS
parameters to stellar observables has a simple differentiable control flow.
Adaptive/independent solvers are reserved for validation.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp


class Star(NamedTuple):
    mass: jnp.ndarray
    radius: jnp.ndarray
    compactness: jnp.ndarray
    central_enthalpy: jnp.ndarray


def _rhs(h, y, eos):
    r, m = y
    p = eos.pressure(h)
    eps = eos.energy_density(h)

    denom = m + 4.0 * jnp.pi * r**3 * p
    dr_dh = -r * (r - 2.0 * m) / denom
    dm_dh = 4.0 * jnp.pi * r**2 * eps * dr_dh
    return jnp.stack((dr_dh, dm_dh))


def _rk4_step(h, y, dh, eos):
    k1 = _rhs(h, y, eos)
    k2 = _rhs(h + 0.5 * dh, y + 0.5 * dh * k1, eos)
    k3 = _rhs(h + 0.5 * dh, y + 0.5 * dh * k2, eos)
    k4 = _rhs(h + dh, y + dh * k3, eos)
    return y + (dh / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def _central_start(h_c, eos, fractional_offset):
    """Regular series start a small enthalpy interval below the center."""
    p_c = eos.pressure(h_c)
    eps_c = eos.energy_density(h_c)

    delta_h = jnp.maximum(jnp.abs(h_c) * fractional_offset, 1.0e-10)
    delta_h = jnp.minimum(delta_h, 0.1 * h_c)
    h0 = h_c - delta_h

    # From dh/dr = -(4 pi / 3) (epsilon_c + 3 p_c) r + O(r^3).
    r0 = jnp.sqrt(
        3.0 * delta_h / (2.0 * jnp.pi * (eps_c + 3.0 * p_c))
    )
    m0 = (4.0 * jnp.pi / 3.0) * eps_c * r0**3
    return h0, jnp.stack((r0, m0))


def solve_star(
    eos,
    h_c,
    *,
    n_steps: int = 4096,
    central_fractional_offset: float = 1.0e-6,
) -> Star:
    """Integrate one stable or unstable spherical configuration.

    Stability is not imposed here; callers construct sequences and select the
    desired branch.  h_c must be positive.
    """
    h_c = jnp.asarray(h_c)
    h0, y0 = _central_start(h_c, eos, central_fractional_offset)
    dh = -h0 / float(n_steps)

    def body(carry, _):
        h, y = carry
        y_new = _rk4_step(h, y, dh, eos)
        return (h + dh, y_new), y_new

    (_, y_final), _ = jax.lax.scan(body, (h0, y0), xs=None, length=n_steps)
    radius, mass = y_final
    return Star(
        mass=mass,
        radius=radius,
        compactness=mass / radius,
        central_enthalpy=h_c,
    )


def incompressible_exact(epsilon0, h_c):
    """Exact Schwarzschild-interior M,R,C for the benchmark EOS."""
    y = jnp.expm1(h_c)  # p_c / epsilon0
    s = (1.0 + y) / (1.0 + 3.0 * y)  # sqrt(1 - 2 C)
    compactness = 0.5 * (1.0 - s**2)
    radius = jnp.sqrt(3.0 * compactness / (4.0 * jnp.pi * epsilon0))
    mass = compactness * radius
    return Star(
        mass=mass,
        radius=radius,
        compactness=compactness,
        central_enthalpy=jnp.asarray(h_c),
    )
