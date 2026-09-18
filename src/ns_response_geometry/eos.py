"""Equation-of-state interfaces used by the stellar solver.

The production response analysis will use a bounded sound-speed field. These
simple EOS classes exist first to validate the stellar integrators.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp


@dataclass(frozen=True)
class IncompressibleEOS:
    """Constant-energy-density star used only as an analytic benchmark."""

    epsilon0: float = 1.0

    def pressure(self, h):
        h = jnp.asarray(h)
        return self.epsilon0 * jnp.expm1(jnp.maximum(h, 0.0))

    def energy_density(self, h):
        h = jnp.asarray(h)
        return jnp.ones_like(h) * self.epsilon0

    def sound_speed_squared(self, h):
        h = jnp.asarray(h)
        return jnp.ones_like(h) * jnp.inf

    @property
    def surface_energy_density(self):
        return self.epsilon0


@dataclass(frozen=True)
class RelativisticPolytrope:
    """Cold relativistic polytrope in units G=c=1.

    p = K rho^Gamma,
    epsilon = rho + p/(Gamma-1),

    with exp(h) = 1 + Gamma K rho^(Gamma-1)/(Gamma-1).
    """

    K: float
    gamma: float

    def rest_mass_density(self, h):
        h = jnp.asarray(h)
        gm1 = self.gamma - 1.0
        x = (gm1 / (self.gamma * self.K)) * jnp.expm1(jnp.maximum(h, 0.0))
        return jnp.maximum(x, 0.0) ** (1.0 / gm1)

    def pressure(self, h):
        rho = self.rest_mass_density(h)
        return self.K * rho**self.gamma

    def energy_density(self, h):
        rho = self.rest_mass_density(h)
        p = self.K * rho**self.gamma
        return rho + p / (self.gamma - 1.0)

    def sound_speed_squared(self, h):
        rho = self.rest_mass_density(h)
        p = self.K * rho**self.gamma
        eps = rho + p / (self.gamma - 1.0)
        denom = jnp.where(eps + p > 0.0, eps + p, 1.0)
        return jnp.where(p > 0.0, self.gamma * p / denom, 0.0)

    @property
    def surface_energy_density(self):
        return 0.0
