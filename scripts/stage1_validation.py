"""Emit reproducible Stage-1 stellar-solver validation numbers."""

import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import jax
jax.config.update("jax_enable_x64", True)
import numpy as np

from ns_response_geometry.eos import IncompressibleEOS, RelativisticPolytrope
from ns_response_geometry.observables import solve_observables
from ns_response_geometry.tov import incompressible_exact
from tests.reference_scipy import solve_reference


NAMES = [
    "mass",
    "radius",
    "compactness",
    "k2",
    "lambda2",
    "moment_of_inertia",
    "ibar",
    "i_mr2",
]


def relative_errors(production, reference):
    production = np.asarray(production, dtype=float)
    reference = np.asarray(reference, dtype=float)
    return {
        name: float((p - r) / r)
        for name, p, r in zip(NAMES, production, reference)
    }


def main():
    exact_rows = []
    eos = IncompressibleEOS(epsilon0=1.0)
    for h_c in (0.05, 0.15, 0.30):
        numeric = solve_observables(eos, h_c, n_steps=8192)
        exact = incompressible_exact(1.0, h_c)
        exact_rows.append(
            dict(
                h_c=h_c,
                mass_relative_error=float((numeric.mass - exact.mass) / exact.mass),
                radius_relative_error=float((numeric.radius - exact.radius) / exact.radius),
                compactness_relative_error=float(
                    (numeric.compactness - exact.compactness) / exact.compactness
                ),
            )
        )

    cross_rows = []
    cases = [
        ("incompressible", IncompressibleEOS(epsilon0=1.0), 0.15),
        ("gamma2_polytrope", RelativisticPolytrope(K=100.0, gamma=2.0), 0.10),
    ]
    for name, eos, h_c in cases:
        production = solve_observables(eos, h_c, n_steps=16384)
        reference = solve_reference(eos, h_c)
        cross_rows.append(
            dict(
                case=name,
                h_c=h_c,
                relative_errors=relative_errors(production[:8], reference),
                max_absolute_relative_error=float(
                    np.max(np.abs(list(relative_errors(production[:8], reference).values())))
                ),
            )
        )

    incompressible_low = solve_observables(
        IncompressibleEOS(epsilon0=1.0), 0.0025, n_steps=8192
    )
    n1_low = solve_observables(
        RelativisticPolytrope(K=100.0, gamma=2.0), 0.0025, n_steps=16384
    )
    limits = dict(
        incompressible=dict(
            compactness=float(incompressible_low.compactness),
            k2=float(incompressible_low.k2),
            k2_newtonian=0.75,
            i_mr2=float(incompressible_low.i_mr2),
            i_mr2_newtonian=0.4,
        ),
        n1_polytrope=dict(
            compactness=float(n1_low.compactness),
            k2=float(n1_low.k2),
            k2_newtonian=(15.0 - math.pi**2) / (2.0 * math.pi**2),
            i_mr2=float(n1_low.i_mr2),
            i_mr2_newtonian=(2.0 / 3.0) * (math.pi**2 - 6.0) / math.pi**2,
        ),
    )

    payload = dict(
        exact_incompressible=exact_rows,
        independent_radius_crosscheck=cross_rows,
        newtonian_limits=limits,
    )
    print("VALIDATION_JSON=" + json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
