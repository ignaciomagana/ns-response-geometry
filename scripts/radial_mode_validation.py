"""Validate the radial l=0 solver against the turning-point theorem."""

import json

import jax
jax.config.update("jax_enable_x64", True)
import numpy as np
from scipy.optimize import minimize_scalar

from ns_response_geometry.eos import RelativisticPolytrope
from ns_response_geometry.radial import fundamental_radial_mode
from ns_response_geometry.realistic_eos import M_SUN_KM, NamedPiecewisePolytrope
from ns_response_geometry.tov import solve_star


def turning_point(eos, lo, hi):
    opt = minimize_scalar(
        lambda h: -float(solve_star(eos, h, n_steps=4096).mass),
        bounds=(lo, hi),
        method="bounded",
        options={"xatol": 2.0e-7},
    )
    h = float(opt.x)
    star = solve_star(eos, h, n_steps=8192)
    return dict(
        h_c=h,
        mass=float(star.mass),
        mass_msun=float(star.mass / M_SUN_KM),
        radius=float(star.radius),
        compactness=float(star.compactness),
    )


def mode(eos, h):
    r = fundamental_radial_mode(
        eos,
        h,
        background_steps=4096,
        h_floor=2.0e-6,
        scan_min=-4.0,
        scan_max=8.0,
        scan_points=61,
    )
    return dict(
        h_c=float(h),
        omega2=float(r.omega2),
        omega2_scaled=float(r.omega2_scaled),
        nodes=int(r.nodes),
        compactness=float(r.compactness),
        residual=float(r.residual),
    )


def audit(name, eos, bracket):
    tp = turning_point(eos, *bracket)
    h = tp["h_c"]
    points = {
        "below": mode(eos, 0.90 * h),
        "near": mode(eos, h),
        "above": mode(eos, 1.10 * h),
    }
    return dict(name=name, turning_point=tp, modes=points)


def main():
    payload = dict(
        gamma2_polytrope=audit(
            "Gamma2_K100",
            RelativisticPolytrope(K=100.0, gamma=2.0),
            (0.15, 0.9),
        ),
        sly=audit(
            "SLy",
            NamedPiecewisePolytrope.from_name("SLy"),
            (0.15, 0.9),
        ),
    )
    print("RADIAL_VALIDATION_JSON=" + json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
