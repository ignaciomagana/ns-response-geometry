"""Independent adaptive radius-coordinate reference solver for tests only."""

from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp


def _thermo(eos, h):
    p = float(eos.pressure(h))
    eps = float(eos.energy_density(h))
    cs2 = float(eos.sound_speed_squared(h))
    compressibility = 0.0 if (not np.isfinite(cs2) or cs2 <= 0.0) else (eps + p) / cs2
    return p, eps, compressibility


def _rhs(r, state, eos):
    h, m, ytidal, wbar, dwbar_dr = state
    p, eps, compressibility = _thermo(eos, h)

    one_minus_2c = 1.0 - 2.0 * m / r
    dm_dr = 4.0 * np.pi * r**2 * eps
    dh_dr = -(m + 4.0 * np.pi * r**3 * p) / (r * (r - 2.0 * m))

    F = (1.0 - 4.0 * np.pi * r**2 * (eps - p)) / one_minus_2c
    Q = (
        4.0
        * np.pi
        * (5.0 * eps + 9.0 * p + compressibility - 6.0 / (4.0 * np.pi * r**2))
        / one_minus_2c
        - 4.0
        * (m + 4.0 * np.pi * r**3 * p) ** 2
        / (r**4 * one_minus_2c**2)
    )
    dy_dr = -(ytidal**2 + F * ytidal + r**2 * Q) / r

    dnu_dr = (m + 4.0 * np.pi * r**3 * p) / (r * (r - 2.0 * m))
    dln_one_minus_2c_dr = (
        -2.0 * dm_dr / r + 2.0 * m / r**2
    ) / one_minus_2c
    dlnj_dr = -dnu_dr + 0.5 * dln_one_minus_2c_dr
    d2wbar_dr2 = (
        -(4.0 / r + dlnj_dr) * dwbar_dr
        - 4.0 * dlnj_dr * wbar / r
    )

    return np.array((dh_dr, dm_dr, dy_dr, dwbar_dr, d2wbar_dr2))


def _k2(compactness, y_surface):
    C = compactness
    y = y_surface
    common = 2.0 + 2.0 * C * (y - 1.0) - y
    denominator = (
        2.0 * C * (6.0 - 3.0 * y + 3.0 * C * (5.0 * y - 8.0))
        + 4.0
        * C**3
        * (13.0 - 11.0 * y + C * (3.0 * y - 2.0) + 2.0 * C**2 * (1.0 + y))
        + 3.0 * (1.0 - 2.0 * C) ** 2 * common * np.log(1.0 - 2.0 * C)
    )
    return (8.0 / 5.0) * C**5 * (1.0 - 2.0 * C) ** 2 * common / denominator


def solve_reference(eos, h_c):
    """Adaptive r-coordinate integration, intentionally separate from production."""
    p_c, eps_c, _ = _thermo(eos, h_c)
    r0 = 1.0e-6

    h0 = h_c - (2.0 * np.pi / 3.0) * (eps_c + 3.0 * p_c) * r0**2
    m0 = (4.0 * np.pi / 3.0) * eps_c * r0**3
    a = (8.0 * np.pi / 5.0) * (eps_c + p_c)

    state0 = np.array((h0, m0, 2.0, 1.0 + a * r0**2, 2.0 * a * r0))

    def surface_event(r, state):
        return state[0]

    surface_event.terminal = True
    surface_event.direction = -1

    sol = solve_ivp(
        lambda r, state: _rhs(r, state, eos),
        (r0, 100.0),
        state0,
        method="DOP853",
        rtol=1.0e-10,
        atol=1.0e-12,
        max_step=1.0e-2,
        events=surface_event,
    )
    if len(sol.t_events[0]) != 1:
        raise RuntimeError("Reference solver did not locate a unique h=0 surface.")

    radius = sol.t_events[0][0]
    _, mass, y_inside, w_surface, z_surface = sol.y_events[0][0]
    compactness = mass / radius

    y_surface = y_inside - 4.0 * np.pi * radius**3 * eos.surface_energy_density / mass
    k2 = _k2(compactness, y_surface)
    lambda2 = (2.0 / 3.0) * k2 / compactness**5

    angular_momentum = radius**4 * z_surface / 6.0
    omega = w_surface + radius * z_surface / 3.0
    moment_of_inertia = angular_momentum / omega

    return np.array(
        (
            mass,
            radius,
            compactness,
            k2,
            lambda2,
            moment_of_inertia,
            moment_of_inertia / mass**3,
            moment_of_inertia / (mass * radius**2),
        )
    )
