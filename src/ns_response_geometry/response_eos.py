"""Differentiable sound-speed perturbations of a reference EOS.

Production response coordinates are nodal values of a latent perturbation
delta u(h), with

    c_s^2(h) = sigmoid(logit(c_{s,0}^2(h)) + delta u(h)).

Below h_match the reference EOS is retained. Above h_match, pressure and
energy density are reconstructed from

    dp/dh         = epsilon + p,
    d epsilon/dh = (epsilon + p) / c_s^2.

Nodal coordinates are preferred for production because a GP covariance
evaluated at the nodes approximates one fixed function-space metric as the
node grid is refined. The older Gaussian-mode builder is retained only for
diagnostics and backward compatibility.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp


class TabulatedSoundSpeedEOS(NamedTuple):
    h_grid: jnp.ndarray
    p_grid: jnp.ndarray
    eps_grid: jnp.ndarray
    cs2_grid: jnp.ndarray

    def pressure(self, h):
        h = jnp.asarray(h)
        return jnp.interp(
            jnp.maximum(h, 0.0),
            self.h_grid,
            self.p_grid,
            left=0.0,
            right=self.p_grid[-1],
        )

    def energy_density(self, h):
        h = jnp.asarray(h)
        return jnp.interp(
            jnp.maximum(h, 0.0),
            self.h_grid,
            self.eps_grid,
            left=0.0,
            right=self.eps_grid[-1],
        )

    def sound_speed_squared(self, h):
        h = jnp.asarray(h)
        return jnp.interp(
            jnp.maximum(h, 0.0),
            self.h_grid,
            self.cs2_grid,
            left=0.0,
            right=self.cs2_grid[-1],
        )

    @property
    def surface_energy_density(self):
        return 0.0


def squared_exponential_covariance(points, *, amplitude=1.0, length_scale=0.08):
    """Squared-exponential covariance evaluated at latent-field nodes."""
    points = jnp.asarray(points)
    delta = points[:, None] - points[None, :]
    return amplitude**2 * jnp.exp(-0.5 * (delta / length_scale) ** 2)


def matern32_covariance(points, *, amplitude=1.0, length_scale=0.08):
    """Matern-3/2 covariance evaluated at latent-field nodes."""
    points = jnp.asarray(points)
    distance = jnp.abs(points[:, None] - points[None, :])
    x = jnp.sqrt(3.0) * distance / length_scale
    return amplitude**2 * (1.0 + x) * jnp.exp(-x)


def exponential_covariance(points, *, amplitude=1.0, length_scale=0.08):
    """Exponential covariance with rougher latent-field realizations."""
    points = jnp.asarray(points)
    distance = jnp.abs(points[:, None] - points[None, :])
    return amplitude**2 * jnp.exp(-distance / length_scale)


def _logit(x):
    return jnp.log(x) - jnp.log1p(-x)


def _bounded_cs2_from_delta(reference_eos, h, delta_u):
    base_cs2 = jnp.clip(reference_eos.sound_speed_squared(h), 1.0e-8, 1.0 - 1.0e-8)
    return jax.nn.sigmoid(_logit(base_cs2) + delta_u)


def _thermo_rhs_with_delta(h, state, delta_u, reference_eos):
    p, eps = state
    cs2 = _bounded_cs2_from_delta(reference_eos, h, delta_u)
    enthalpy_density = eps + p
    return jnp.stack((enthalpy_density, enthalpy_density / cs2))


def _thermo_rk4_nodal(h, state, dh, nodes, values, reference_eos):
    def delta(x):
        return jnp.interp(x, nodes, values)

    k1 = _thermo_rhs_with_delta(h, state, delta(h), reference_eos)
    k2 = _thermo_rhs_with_delta(
        h + 0.5 * dh, state + 0.5 * dh * k1, delta(h + 0.5 * dh), reference_eos
    )
    k3 = _thermo_rhs_with_delta(
        h + 0.5 * dh, state + 0.5 * dh * k2, delta(h + 0.5 * dh), reference_eos
    )
    k4 = _thermo_rhs_with_delta(
        h + dh, state + dh * k3, delta(h + dh), reference_eos
    )
    return state + (dh / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def build_nodal_sound_speed_eos(
    reference_eos,
    delta_u_nodes,
    *,
    h_match=0.02,
    h_max=0.6,
    h_nodes=None,
    n_low=128,
    n_high=2048,
):
    """Construct a causal EOS from nodal latent sound-speed perturbations."""
    values = jnp.asarray(delta_u_nodes)
    if h_nodes is None:
        nodes = jnp.linspace(h_match, h_max, values.shape[0])
    else:
        nodes = jnp.asarray(h_nodes)

    low_h = jnp.linspace(0.0, h_match, n_low, endpoint=False)
    low_p = reference_eos.pressure(low_h)
    low_eps = reference_eos.energy_density(low_h)
    low_cs2 = reference_eos.sound_speed_squared(low_h)

    high_h = jnp.linspace(h_match, h_max, n_high)
    dh = high_h[1] - high_h[0]
    state0 = jnp.stack(
        (
            reference_eos.pressure(h_match),
            reference_eos.energy_density(h_match),
        )
    )

    def body(state, h):
        new_state = _thermo_rk4_nodal(
            h, state, dh, nodes, values, reference_eos
        )
        return new_state, new_state

    _, states = jax.lax.scan(body, state0, high_h[:-1])
    high_states = jnp.concatenate((state0[None, :], states), axis=0)
    high_p = high_states[:, 0]
    high_eps = high_states[:, 1]
    high_delta = jnp.interp(high_h, nodes, values)
    high_cs2 = _bounded_cs2_from_delta(reference_eos, high_h, high_delta)

    return TabulatedSoundSpeedEOS(
        h_grid=jnp.concatenate((low_h, high_h)),
        p_grid=jnp.concatenate((low_p, high_p)),
        eps_grid=jnp.concatenate((low_eps, high_eps)),
        cs2_grid=jnp.concatenate((low_cs2, high_cs2)),
    )


def gaussian_basis(h, centers, width):
    h = jnp.asarray(h)
    centers = jnp.asarray(centers)
    return jnp.exp(-0.5 * ((h[..., None] - centers) / width) ** 2)


def _latent_sound_speed_gaussian(h, coeffs, centers, width, reference_eos):
    base_cs2 = reference_eos.sound_speed_squared(h)
    base_cs2 = jnp.clip(base_cs2, 1.0e-8, 1.0 - 1.0e-8)
    latent = _logit(base_cs2) + gaussian_basis(h, centers, width) @ coeffs
    return jax.nn.sigmoid(latent)


def _thermo_rhs_gaussian(h, state, coeffs, centers, width, reference_eos):
    p, eps = state
    cs2 = _latent_sound_speed_gaussian(h, coeffs, centers, width, reference_eos)
    enthalpy_density = eps + p
    return jnp.stack((enthalpy_density, enthalpy_density / cs2))


def _thermo_rk4_gaussian(h, state, dh, coeffs, centers, width, reference_eos):
    args = (coeffs, centers, width, reference_eos)
    k1 = _thermo_rhs_gaussian(h, state, *args)
    k2 = _thermo_rhs_gaussian(h + 0.5 * dh, state + 0.5 * dh * k1, *args)
    k3 = _thermo_rhs_gaussian(h + 0.5 * dh, state + 0.5 * dh * k2, *args)
    k4 = _thermo_rhs_gaussian(h + dh, state + dh * k3, *args)
    return state + (dh / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def build_sound_speed_eos(
    reference_eos,
    coeffs,
    *,
    h_match=0.02,
    h_max=0.6,
    n_low=128,
    n_high=2048,
    centers=None,
    width=None,
):
    """Legacy Gaussian-mode builder retained for the original pilot."""
    coeffs = jnp.asarray(coeffs)
    if centers is None:
        centers = jnp.linspace(h_match, h_max, coeffs.shape[0])
    else:
        centers = jnp.asarray(centers)
    if width is None:
        width = (h_max - h_match) / jnp.maximum(coeffs.shape[0] - 1, 1)
    width = jnp.asarray(width)

    low_h = jnp.linspace(0.0, h_match, n_low, endpoint=False)
    low_p = reference_eos.pressure(low_h)
    low_eps = reference_eos.energy_density(low_h)
    low_cs2 = reference_eos.sound_speed_squared(low_h)

    high_h = jnp.linspace(h_match, h_max, n_high)
    dh = high_h[1] - high_h[0]
    state0 = jnp.stack(
        (
            reference_eos.pressure(h_match),
            reference_eos.energy_density(h_match),
        )
    )

    def body(state, h):
        new_state = _thermo_rk4_gaussian(
            h, state, dh, coeffs, centers, width, reference_eos
        )
        return new_state, new_state

    _, states = jax.lax.scan(body, state0, high_h[:-1])
    high_states = jnp.concatenate((state0[None, :], states), axis=0)
    high_p = high_states[:, 0]
    high_eps = high_states[:, 1]
    high_cs2 = _latent_sound_speed_gaussian(
        high_h, coeffs, centers, width, reference_eos
    )

    return TabulatedSoundSpeedEOS(
        h_grid=jnp.concatenate((low_h, high_h)),
        p_grid=jnp.concatenate((low_p, high_p)),
        eps_grid=jnp.concatenate((low_eps, high_eps)),
        cs2_grid=jnp.concatenate((low_cs2, high_cs2)),
    )
