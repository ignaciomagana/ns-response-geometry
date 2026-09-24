"""Published piecewise-polytropic fits to named cold neutron-star EOSs.

The core parameters are the four-parameter fits of Read et al. as tabulated
by Lackey & Wade (2015). The low-density sector is the four-piece analytic
SLy crust used by Read et al.

All public methods use the same relativistic enthalpy coordinate as the
production stellar solver and return geometrized quantities in km^{-2}.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import jax.numpy as jnp


G_CGS = 6.67430e-8
C_CGS = 2.99792458e10
RHO_CGS_TO_KM2 = G_CGS / C_CGS**2 * 1.0e10
M_SUN_KM = 1.4766250385

# Read et al. low-density SLy crust.  The tabulated constants are K_i / c^2,
# so p/c^2 = Kbar_i rho^Gamma in g cm^-3.
_CRUST_KBAR = (
    6.80110e-9,
    1.06186e-6,
    5.32697e1,
    3.99874e-8,
)
_CRUST_GAMMA = (1.58425, 1.28733, 0.62223, 1.35692)
_CRUST_UPPER_RHO = (2.44034e7, 3.78358e11, 2.62780e12)

_CORE_RHO1 = 10.0**14.7
_CORE_RHO2 = 10.0**15.0


NAMED_EOS_FITS = {
    "SLy": (34.384, 3.005, 2.988, 2.851),
    "ENG": (34.437, 3.514, 3.130, 3.168),
    "MPA1": (34.495, 3.446, 3.572, 2.887),
    "MS1": (34.858, 3.224, 3.033, 1.325),
    "MS1b": (34.855, 3.456, 3.011, 1.425),
    "H4": (34.669, 2.909, 2.246, 2.144),
    "ALF2": (34.616, 4.070, 2.411, 1.890),
}


def _continuity_offset(a_left, pbar, rho, gamma_left, gamma_right):
    return (
        a_left
        + pbar / rho
        * (1.0 / (gamma_left - 1.0) - 1.0 / (gamma_right - 1.0))
    )


@dataclass(frozen=True)
class NamedPiecewisePolytrope:
    """Seven-piece crust+core fit to a named cold EOS.

    Rest-mass density rho is handled internally in cgs g/cm^3.  Pressure is
    represented as p/c^2 in the same mass-density units before conversion to
    geometrized km^-2.  Energy-density continuity constants are fixed by the
    first law exactly as in the Read et al. parameterization.
    """

    name: str
    log10_p1: float
    gamma1: float
    gamma2: float
    gamma3: float

    @classmethod
    def from_name(cls, name: str) -> "NamedPiecewisePolytrope":
        if name not in NAMED_EOS_FITS:
            raise KeyError(f"Unknown named EOS fit: {name}")
        return cls(name, *NAMED_EOS_FITS[name])

    def _segment_data(self):
        # Pressure in mass-density units, p/c^2 [g cm^-3].
        pbar1 = 10.0**self.log10_p1 / C_CGS**2

        k_core1 = pbar1 / _CORE_RHO1**self.gamma1
        k_core2 = pbar1 / _CORE_RHO1**self.gamma2
        pbar2 = k_core2 * _CORE_RHO2**self.gamma2
        k_core3 = pbar2 / _CORE_RHO2**self.gamma3

        # Match the last crust polytrope to core piece 1.
        gamma_crust = _CRUST_GAMMA[-1]
        k_crust = _CRUST_KBAR[-1]
        rho_match = (k_crust / k_core1) ** (1.0 / (self.gamma1 - gamma_crust))

        gammas = (
            *_CRUST_GAMMA,
            self.gamma1,
            self.gamma2,
            self.gamma3,
        )
        kbars = (
            *_CRUST_KBAR,
            k_core1,
            k_core2,
            k_core3,
        )
        upper_rho = (
            *_CRUST_UPPER_RHO,
            rho_match,
            _CORE_RHO1,
            _CORE_RHO2,
        )

        offsets = [0.0]
        boundaries = upper_rho
        for i, rho_b in enumerate(boundaries):
            pbar = kbars[i] * rho_b**gammas[i]
            offsets.append(
                _continuity_offset(
                    offsets[-1], pbar, rho_b, gammas[i], gammas[i + 1]
                )
            )

        # Enthalpy h = log[(epsilon+p)/rho].
        upper_h = []
        for i, rho_b in enumerate(upper_rho):
            H = (
                1.0
                + offsets[i]
                + gammas[i] / (gammas[i] - 1.0)
                * kbars[i]
                * rho_b ** (gammas[i] - 1.0)
            )
            upper_h.append(math.log(H))

        return (
            jnp.asarray(gammas),
            jnp.asarray(kbars),
            jnp.asarray(offsets),
            jnp.asarray(upper_h),
            float(rho_match),
        )

    @property
    def match_density_cgs(self):
        return self._segment_data()[-1]

    def rest_mass_density_cgs(self, h):
        h = jnp.asarray(h)
        gammas, kbars, offsets, upper_h, _ = self._segment_data()
        index = jnp.sum(jnp.maximum(h, 0.0)[..., None] >= upper_h, axis=-1)
        gamma = gammas[index]
        kbar = kbars[index]
        offset = offsets[index]

        x = (
            (gamma - 1.0)
            / (gamma * kbar)
            * (jnp.exp(jnp.maximum(h, 0.0)) - 1.0 - offset)
        )
        # The neutron-drip crust piece has Gamma < 1, so the sign of the
        # prefactor matters: clip only after forming the full positive
        # inversion variable.
        x = jnp.maximum(x, 1.0e-300)
        rho = x ** (1.0 / (gamma - 1.0))
        return jnp.where(h > 0.0, rho, 0.0)

    def pressure(self, h):
        h = jnp.asarray(h)
        rho = self.rest_mass_density_cgs(h)
        gammas, kbars, _, upper_h, _ = self._segment_data()
        index = jnp.sum(jnp.maximum(h, 0.0)[..., None] >= upper_h, axis=-1)
        gamma = gammas[index]
        kbar = kbars[index]
        pbar = kbar * rho**gamma
        return pbar * RHO_CGS_TO_KM2

    def energy_density(self, h):
        h = jnp.asarray(h)
        rho = self.rest_mass_density_cgs(h)
        gammas, kbars, offsets, upper_h, _ = self._segment_data()
        index = jnp.sum(jnp.maximum(h, 0.0)[..., None] >= upper_h, axis=-1)
        gamma = gammas[index]
        kbar = kbars[index]
        offset = offsets[index]
        pbar = kbar * rho**gamma
        epsbar = (1.0 + offset) * rho + pbar / (gamma - 1.0)
        return epsbar * RHO_CGS_TO_KM2

    def sound_speed_squared(self, h):
        h = jnp.asarray(h)
        p = self.pressure(h)
        eps = self.energy_density(h)
        gammas, _, _, upper_h, _ = self._segment_data()
        index = jnp.sum(jnp.maximum(h, 0.0)[..., None] >= upper_h, axis=-1)
        gamma = gammas[index]
        denom = jnp.where(eps + p > 0.0, eps + p, 1.0)
        return jnp.where(p > 0.0, gamma * p / denom, 0.0)

    @property
    def surface_energy_density(self):
        return 0.0


def named_eos(name: str) -> NamedPiecewisePolytrope:
    return NamedPiecewisePolytrope.from_name(name)
