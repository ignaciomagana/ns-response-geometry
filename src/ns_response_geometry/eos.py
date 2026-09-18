"""Equation-of-state interfaces used by the stellar solver.

The production response analysis will use a bounded sound-speed field.  These
simple EOS classes exist first to validate the background integrator.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp


@dataclass(frozen=True)
class IncompressibleEOS:
    """Constant-energy-density star.

    This is not a causal material EOS and is used only as an analytic TOV
    benchmark.  With h = integral dp/(epsilon+p),

        p(h) = epsilon_0 [exp(h) - 1].
    """

    epsilon0: float = 1.0

    def pressure(self, h):
        h = jnp.asarray(h)
        return self.epsilon0 * jnp.expm1(jnp.maximum(h, 0.0))

    def energy_density(self, h):
        h = jnp.asarray(h)
        return jnp.ones_like(h) * self.epsilon0


@dataclass(frozen=True)
class RelativisticPolytrope:
    """Cold relativistic polytrope in units G=c=1.

    p = K rho^Gamma,
    epsilon = rho + p/(Gamma-1).

    The relativistic enthalpy has the closed form

        exp(h) = 1 + Gamma/(Gamma-1) K rho^(Gamma-1).
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
