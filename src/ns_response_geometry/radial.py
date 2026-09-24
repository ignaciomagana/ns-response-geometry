"""Relativistic radial l=0 pulsations in the Gondek et al. form.

This module is intentionally independent of the differentiable response code.
It solves the two first-order radial-pulsation equations on a fixed TOV
background and locates the fundamental eigenvalue by shooting.

Metric convention:
    ds^2 = -exp(2 nu) dt^2 + exp(2 lambda) dr^2 + r^2 dOmega^2,
    exp(2 lambda) = (1 - 2m/r)^-1.

The perturbation variables are
    zeta = Delta r / r,
    Delta_p = Lagrangian pressure perturbation.

For a cold barotrope, Gamma p = (epsilon + p) c_s^2.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

from .tov import solve_star_profile


@dataclass(frozen=True)
class RadialModeResult:
    omega2: float
    omega2_scaled: float
    nodes: int
    residual: float
    radius: float
    mass: float
    compactness: float
    h_c: float


def _background(eos, h_c, n_steps):
    profile = solve_star_profile(eos, h_c, n_steps=n_steps)
    h_desc = np.asarray(profile.enthalpy, dtype=float)
    r_desc = np.asarray(profile.radius, dtype=float)
    m_desc = np.asarray(profile.mass, dtype=float)

    h = h_desc[::-1]
    r = r_desc[::-1]
    m = m_desc[::-1]

    radius = float(r_desc[-1])
    mass = float(m_desc[-1])
    compactness = mass / radius

    return h, r, m, radius, mass, compactness


def _interp_background(h, h_grid, r_grid, m_grid):
    r = float(np.interp(h, h_grid, r_grid))
    m = float(np.interp(h, h_grid, m_grid))
    return r, m


def _thermo(eos, h):
    p = float(eos.pressure(h))
    eps = float(eos.energy_density(h))
    cs2 = float(eos.sound_speed_squared(h))
    return p, eps, cs2


def _mode_rhs(
    h,
    state,
    *,
    eos,
    omega2,
    h_grid,
    r_grid,
    m_grid,
    radius,
    mass,
):
    zeta, delta_p = state
    r, m = _interp_background(h, h_grid, r_grid, m_grid)
    p, eps, cs2 = _thermo(eos, h)

    one_minus_2c = 1.0 - 2.0 * m / r
    e2lambda = 1.0 / one_minus_2c

    # Hydrostatic relation in enthalpy coordinates.
    dr_dh = -r * (r - 2.0 * m) / (m + 4.0 * np.pi * r**3 * p)
    dp_dr = (eps + p) / dr_dh

    # nu(h) = nu(R) - h exactly for a barotropic TOV background.
    e2nu = (1.0 - 2.0 * mass / radius) * np.exp(-2.0 * h)
    e2lambda_minus_2nu = e2lambda / e2nu

    gamma_p = (eps + p) * cs2
    if gamma_p <= 0.0:
        gamma_p = np.finfo(float).tiny

    dzeta_dr = (
        -(3.0 * zeta + delta_p / gamma_p) / r
        - dp_dr * zeta / (eps + p)
    )

    ddelta_dr = zeta * (
        omega2 * e2lambda_minus_2nu * (eps + p) * r
        - 4.0 * dp_dr
        - 8.0 * np.pi * e2lambda * (eps + p) * r * p
        + r * dp_dr**2 / (eps + p)
    ) + delta_p * (
        dp_dr / (eps + p)
        - 4.0 * np.pi * (eps + p) * r * e2lambda
    )

    return np.asarray((dzeta_dr * dr_dh, ddelta_dr * dr_dh))


def shoot_radial_mode(
    eos,
    h_c,
    omega2_scaled,
    *,
    background_steps=4096,
    h_floor=1.0e-6,
    rtol=3.0e-9,
    atol=1.0e-11,
    return_solution=False,
):
    """Shoot one trial radial mode.

    omega2_scaled = omega^2 R^3 / M.
    The surface residual uses the regular eta = Delta p / p condition at a
    small positive enthalpy h_floor.
    """
    h_grid, r_grid, m_grid, radius, mass, compactness = _background(
        eos, h_c, background_steps
    )
    h_start = float(h_grid[-1])
    h_floor = min(float(h_floor), 0.1 * h_start)
    if h_floor <= h_grid[0]:
        h_floor = max(10.0 * h_grid[0], 1.0e-8)

    p0, eps0, cs20 = _thermo(eos, h_start)
    zeta0 = 1.0
    delta0 = -3.0 * (eps0 + p0) * cs20 * zeta0

    omega2 = float(omega2_scaled) * mass / radius**3

    t_eval = np.linspace(h_start, h_floor, 900)
    sol = solve_ivp(
        lambda h, y: _mode_rhs(
            h,
            y,
            eos=eos,
            omega2=omega2,
            h_grid=h_grid,
            r_grid=r_grid,
            m_grid=m_grid,
            radius=radius,
            mass=mass,
        ),
        (h_start, h_floor),
        np.asarray((zeta0, delta0)),
        method="DOP853",
        rtol=rtol,
        atol=atol,
        t_eval=t_eval,
    )
    if not sol.success:
        raise RuntimeError(f"Radial-mode integration failed: {sol.message}")

    zeta_s = float(sol.y[0, -1])
    delta_s = float(sol.y[1, -1])
    p_s, _, _ = _thermo(eos, h_floor)

    eta_numeric = delta_s / p_s
    eta_surface = -(
        (float(omega2_scaled) + compactness) / (1.0 - 2.0 * compactness)
        + 4.0
    ) * zeta_s
    residual = eta_numeric - eta_surface

    zeta = np.asarray(sol.y[0], dtype=float)
    scale = np.max(np.abs(zeta))
    if scale == 0.0:
        nodes = 0
    else:
        zz = zeta / scale
        mask = np.abs(zz) > 1.0e-5
        signs = np.sign(zz[mask])
        nodes = int(np.sum(signs[1:] * signs[:-1] < 0.0))

    result = RadialModeResult(
        omega2=omega2,
        omega2_scaled=float(omega2_scaled),
        nodes=nodes,
        residual=float(residual),
        radius=radius,
        mass=mass,
        compactness=compactness,
        h_c=float(h_c),
    )
    if return_solution:
        return result, sol
    return result


def fundamental_radial_mode(
    eos,
    h_c,
    *,
    background_steps=4096,
    h_floor=1.0e-6,
    scan_min=-8.0,
    scan_max=30.0,
    scan_points=161,
):
    """Locate the node-free fundamental radial eigenvalue.

    Candidate roots are obtained from sign changes of the shooting residual.
    The smallest root with zero interior nodes is returned.
    """
    grid = np.linspace(scan_min, scan_max, scan_points)
    residuals = []
    for value in grid:
        try:
            residuals.append(
                shoot_radial_mode(
                    eos,
                    h_c,
                    value,
                    background_steps=background_steps,
                    h_floor=h_floor,
                ).residual
            )
        except Exception:
            residuals.append(np.nan)
    residuals = np.asarray(residuals)

    brackets = []
    for i in range(len(grid) - 1):
        a, b = residuals[i], residuals[i + 1]
        if not (np.isfinite(a) and np.isfinite(b)):
            continue
        if a == 0.0:
            brackets.append((grid[i], grid[i]))
        elif a * b < 0.0:
            brackets.append((grid[i], grid[i + 1]))

    candidates = []
    for lo, hi in brackets:
        if lo == hi:
            root = lo
        else:
            root = brentq(
                lambda x: shoot_radial_mode(
                    eos,
                    h_c,
                    x,
                    background_steps=background_steps,
                    h_floor=h_floor,
                ).residual,
                lo,
                hi,
                xtol=2.0e-8,
                rtol=2.0e-8,
                maxiter=80,
            )
        result = shoot_radial_mode(
            eos,
            h_c,
            root,
            background_steps=background_steps,
            h_floor=h_floor,
        )
        candidates.append(result)

    node_free = [r for r in candidates if r.nodes == 0]
    if not node_free:
        raise RuntimeError(
            "No node-free radial eigenmode found in the requested omega^2 scan."
        )
    return min(node_free, key=lambda r: r.omega2_scaled)
