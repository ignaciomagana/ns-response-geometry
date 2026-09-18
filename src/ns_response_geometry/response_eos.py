"""Differentiable sound-speed perturbations of a reference EOS.

The response coordinates live in a latent field u(h), with
c_s^2 = sigmoid(u).  Below h_match we retain an analytic reference EOS as a
low-density anchor.  Above h_match p(h) and epsilon(h) are reconstructed from

    dp/dh       = epsilon + p,
    d epsilon/dh = (epsilon + p) / c_s^2.

This avoids perturbing thermodynamically inconsistent p/epsilon tables.
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


def gaussian_basis(h, centers, width):
    """Gaussian basis functions evaluated at h.

    The basis is intentionally simple.  Physical conclusions are required to
    converge under basis refinement/change rather than depend on this choice.
    """
    h = jnp.asarray(h)
    centers = jnp.asarray(centers)
    return jnp.exp(-0.5 * ((h[..., None] - centers) / width) ** 2)


def squared_exponential_covariance(centers, *, amplitude=1.0, length_scale=0.08):
    """Coefficient-space covariance used as one EOS metric choice."""
    centers = jnp.asarray(centers)
    delta = centers[:, None] - centers[None, :]
    return amplitude**2 * jnp.exp(-0.5 * (delta / length_scale) ** 2)


def _logit(x):
    return jnp.log(x) - jnp.log1p(-x)


def _latent_sound_speed(h, coeffs, centers, width, reference_eos):
    base_cs2 = reference_eos.sound_speed_squared(h)
    base_cs2 = jnp.clip(base_cs2, 1.0e-8, 1.0 - 1.0e-8)
    latent = _logit(base_cs2) + gaussian_basis(h, centers, width) @ coeffs
    return jax.nn.sigmoid(latent)


def _thermo_rhs(h, state, coeffs, centers, width, reference_eos):
    p, eps = state
    cs2 = _latent_sound_speed(h, coeffs, centers, width, reference_eos)
    enthalpy_density = eps + p
    return jnp.stack((enthalpy_density, enthalpy_density / cs2))


def _thermo_rk4(h, state, dh, coeffs, centers, width, reference_eos):
    args = (coeffs, centers, width, reference_eos)
    k1 = _thermo_rhs(h, state, *args)
    k2 = _thermo_rhs(h + 0.5 * dh, state + 0.5 * dh * k1, *args)
    k3 = _thermo_rhs(h + 0.5 * dh, state + 0.5 * dh * k2, *args)
    k4 = _thermo_rhs(h + dh, state + dh * k3, *args)
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
    """Construct a differentiable causal EOS from latent sound-speed modes.

    The reference EOS supplies the low-density anchor and the zero-perturbation
    sound speed.  The intended reference for initial validation is a causal
    relativistic polytrope; later analyses will replace it with the declared
    broad EOS ensemble.
    """
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
        new_state = _thermo_rk4(
            h, state, dh, coeffs, centers, width, reference_eos
        )
        return new_state, new_state

    _, states = jax.lax.scan(body, state0, high_h[:-1])
    high_states = jnp.concatenate((state0[None, :], states), axis=0)
    high_p = high_states[:, 0]
    high_eps = high_states[:, 1]
    high_cs2 = _latent_sound_speed(
        high_h, coeffs, centers, width, reference_eos
    )

    return TabulatedSoundSpeedEOS(
        h_grid=jnp.concatenate((low_h, high_h)),
        p_grid=jnp.concatenate((low_p, high_p)),
        eps_grid=jnp.concatenate((low_eps, high_eps)),
        cs2_grid=jnp.concatenate((low_cs2, high_cs2)),
    )
