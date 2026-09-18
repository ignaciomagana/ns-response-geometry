"""Static quadrupolar tides and first-order slow rotation.

The equations are integrated together with the TOV background in relativistic
enthalpy using the same fixed-step RK4 strategy as the background solver.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp


class StellarObservables(NamedTuple):
    mass: jnp.ndarray
    radius: jnp.ndarray
    compactness: jnp.ndarray
    k2: jnp.ndarray
    lambda2: jnp.ndarray
    moment_of_inertia: jnp.ndarray
    ibar: jnp.ndarray
    i_mr2: jnp.ndarray
    central_enthalpy: jnp.ndarray


def _tidal_compressibility_term(eos, h, p, eps):
    """Return (epsilon+p)/c_s^2 with safe endpoint handling."""
    cs2 = eos.sound_speed_squared(h)
    finite_positive = jnp.isfinite(cs2) & (cs2 > 0.0)
    safe_cs2 = jnp.where(finite_positive, cs2, 1.0)
    return jnp.where(finite_positive, (eps + p) / safe_cs2, 0.0)


def _rhs(h, state, eos):
    r, m, ytidal, wbar, dwbar_dr = state
    p = eos.pressure(h)
    eps = eos.energy_density(h)

    background_denom = m + 4.0 * jnp.pi * r**3 * p
    dr_dh = -r * (r - 2.0 * m) / background_denom
    dm_dh = 4.0 * jnp.pi * r**2 * eps * dr_dh

    one_minus_2c = 1.0 - 2.0 * m / r
    compressibility = _tidal_compressibility_term(eos, h, p, eps)
    F = (1.0 - 4.0 * jnp.pi * r**2 * (eps - p)) / one_minus_2c
    Q = (
        4.0
        * jnp.pi
        * (5.0 * eps + 9.0 * p + compressibility - 6.0 / (4.0 * jnp.pi * r**2))
        / one_minus_2c
        - 4.0
        * (m + 4.0 * jnp.pi * r**3 * p) ** 2
        / (r**4 * one_minus_2c**2)
    )
    dy_dr = -(ytidal**2 + F * ytidal + r**2 * Q) / r

    dm_dr = 4.0 * jnp.pi * r**2 * eps
    dnu_dr = (m + 4.0 * jnp.pi * r**3 * p) / (r * (r - 2.0 * m))
    dln_one_minus_2c_dr = (
        -2.0 * dm_dr / r + 2.0 * m / r**2
    ) / one_minus_2c
    dlnj_dr = -dnu_dr + 0.5 * dln_one_minus_2c_dr

    d2wbar_dr2 = (
        -(4.0 / r + dlnj_dr) * dwbar_dr
        - 4.0 * dlnj_dr * wbar / r
    )

    return jnp.stack(
        (
            dr_dh,
            dm_dh,
            dy_dr * dr_dh,
            dwbar_dr * dr_dh,
            d2wbar_dr2 * dr_dh,
        )
    )


def _rk4_step(h, state, dh, eos):
    k1 = _rhs(h, state, eos)
    k2 = _rhs(h + 0.5 * dh, state + 0.5 * dh * k1, eos)
    k3 = _rhs(h + 0.5 * dh, state + 0.5 * dh * k2, eos)
    k4 = _rhs(h + dh, state + dh * k3, eos)
    return state + (dh / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def _central_start(h_c, eos, fractional_offset):
    p_c = eos.pressure(h_c)
    eps_c = eos.energy_density(h_c)

    delta_h = jnp.maximum(jnp.abs(h_c) * fractional_offset, 1.0e-10)
    delta_h = jnp.minimum(delta_h, 0.1 * h_c)
    h0 = h_c - delta_h

    r0 = jnp.sqrt(
        3.0 * delta_h / (2.0 * jnp.pi * (eps_c + 3.0 * p_c))
    )
    m0 = (4.0 * jnp.pi / 3.0) * eps_c * r0**3

    # Regular l=2 tidal solution.
    y0 = jnp.asarray(2.0)

    # Regular slow-rotation expansion wbar = 1 + a r^2 + O(r^4).
    a = (8.0 * jnp.pi / 5.0) * (eps_c + p_c)
    w0 = 1.0 + a * r0**2
    z0 = 2.0 * a * r0

    return h0, jnp.stack((r0, m0, y0, w0, z0))


def _love_number_k2(compactness, y_surface):
    C = compactness
    y = y_surface
    common = 2.0 + 2.0 * C * (y - 1.0) - y

    numerator = (8.0 / 5.0) * C**5 * (1.0 - 2.0 * C) ** 2 * common
    denominator = (
        2.0 * C * (6.0 - 3.0 * y + 3.0 * C * (5.0 * y - 8.0))
        + 4.0
        * C**3
        * (
            13.0
            - 11.0 * y
            + C * (3.0 * y - 2.0)
            + 2.0 * C**2 * (1.0 + y)
        )
        + 3.0
        * (1.0 - 2.0 * C) ** 2
        * common
        * jnp.log(1.0 - 2.0 * C)
    )
    return numerator / denominator


def solve_observables(
    eos,
    h_c,
    *,
    n_steps: int = 8192,
    central_fractional_offset: float = 1.0e-4,
) -> StellarObservables:
    """Return M, R, Lambda_2 and I for one stellar configuration."""
    h_c = jnp.asarray(h_c)
    h0, state0 = _central_start(h_c, eos, central_fractional_offset)
    dh = -h0 / float(n_steps)

    def body(carry, _):
        h, state = carry
        state_new = _rk4_step(h, state, dh, eos)
        return (h + dh, state_new), state_new

    (_, final), _ = jax.lax.scan(body, (h0, state0), xs=None, length=n_steps)
    radius, mass, y_inside, w_surface, z_surface = final
    compactness = mass / radius

    # Density-discontinuity correction for self-bound stars.
    y_surface = (
        y_inside
        - 4.0 * jnp.pi * radius**3 * eos.surface_energy_density / mass
    )
    k2 = _love_number_k2(compactness, y_surface)
    lambda2 = (2.0 / 3.0) * k2 / compactness**5

    # Exterior solution wbar = Omega - 2 J/r^3.
    angular_momentum = radius**4 * z_surface / 6.0
    omega = w_surface + radius * z_surface / 3.0
    moment_of_inertia = angular_momentum / omega

    return StellarObservables(
        mass=mass,
        radius=radius,
        compactness=compactness,
        k2=k2,
        lambda2=lambda2,
        moment_of_inertia=moment_of_inertia,
        ibar=moment_of_inertia / mass**3,
        i_mr2=moment_of_inertia / (mass * radius**2),
        central_enthalpy=h_c,
    )
