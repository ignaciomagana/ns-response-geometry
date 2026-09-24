"""Fast benchmark of named piecewise-polytropic EOS implementation."""

import json

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
from scipy.optimize import brentq, minimize_scalar

from ns_response_geometry.realistic_eos import M_SUN_KM, NamedPiecewisePolytrope
from ns_response_geometry.tov import solve_star


EOS_NAMES = ("SLy", "ENG", "MPA1", "MS1", "MS1b", "H4", "ALF2")
PUBLISHED = {
    "SLy":  (2.049, 0.02, 11.736, -0.21),
    "ENG":  (2.240, -0.05, 12.059, -0.69),
    "MPA1": (2.461, -0.16, 12.473, -0.26),
    "MS1":  (2.767, -0.54, 14.918, 0.06),
    "MS1b": (2.776, -1.03, 14.583, -0.32),
    "H4":   (2.032, -0.85, 13.774, 1.34),
    "ALF2": (2.086, -5.26, 13.188, -3.66),
}


def expected(name):
    m, me, r, re = PUBLISHED[name]
    return m * (1.0 + me / 100.0), r * (1.0 + re / 100.0)


def benchmark(name):
    eos = NamedPiecewisePolytrope.from_name(name)

    @jax.jit
    def star(h):
        return solve_star(eos, h, n_steps=2048)

    star(jnp.asarray(0.2)).mass.block_until_ready()

    def vals(h):
        s = star(jnp.asarray(h))
        return float(s.mass / M_SUN_KM), float(s.radius), float(s.compactness)

    hs = np.linspace(0.03, 0.9, 72)
    rows = np.asarray([(h, *vals(h)) for h in hs])
    i = int(np.argmax(rows[:, 1]))
    if i < 2 or i >= len(rows) - 2:
        raise RuntimeError(f"Maximum-mass bracket failed for {name}")

    opt = minimize_scalar(
        lambda h: -vals(float(h))[0],
        bounds=(rows[i - 2, 0], rows[i + 2, 0]),
        method="bounded",
        options={"xatol": 2.0e-7},
    )
    hmax = float(opt.x)
    mmax, rmax, cmax = vals(hmax)

    stable = rows[: i + 1]
    masses = stable[:, 1]
    cross = np.where((masses[:-1] - 1.4) * (masses[1:] - 1.4) <= 0)[0]
    if len(cross) == 0:
        raise RuntimeError(f"1.4 Msun bracket failed for {name}")
    j = int(cross[-1])
    h14 = brentq(
        lambda h: vals(float(h))[0] - 1.4,
        stable[j, 0],
        stable[j + 1, 0],
        xtol=1.0e-10,
    )
    _, r14, c14 = vals(h14)

    mref, rref = expected(name)
    return dict(
        h_max=hmax,
        mmax_msun=mmax,
        r_at_mmax_km=rmax,
        c_at_mmax=cmax,
        h_1p4=float(h14),
        r_1p4_km=r14,
        c_1p4=c14,
        published_fit_mmax_msun=mref,
        published_fit_r_1p4_km=rref,
        mmax_fractional_error=(mmax - mref) / mref,
        r14_fractional_error=(r14 - rref) / rref,
    )


def main():
    rows = {name: benchmark(name) for name in EOS_NAMES}
    payload = dict(
        source="Lackey & Wade 2015 Table I / Read et al. fits",
        results=rows,
        max_abs_mmax_fractional_error=max(
            abs(v["mmax_fractional_error"]) for v in rows.values()
        ),
        max_abs_r14_fractional_error=max(
            abs(v["r14_fractional_error"]) for v in rows.values()
        ),
    )
    print("REALISTIC_EOS_BENCHMARK_JSON=" + json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
