"""True radial-mode stability audit of the branch-safe transition controls.

The input is the authoritative branch-safe cross-EOS result.  For each of
the six selected (Gamma, C_final) deformation paths, evaluate the fundamental
l=0 mode at:
  - undeformed baseline;
  - half the selected transition depth;
  - selected endpoint;
and at the first deeper x location that is reachable but rejected by the
positive-dM/dh_c path screen.

This turns the earlier mass-slope proxy into a direct dynamical-stability
check without changing the response-geometry analysis.
"""

import json
from pathlib import Path

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

from ns_response_geometry.eos import RelativisticPolytrope
from ns_response_geometry.observables import solve_observables
from ns_response_geometry.radial import fundamental_radial_mode
from ns_response_geometry.response_eos import build_nodal_sound_speed_eos


RESULT_PATH = Path("results/fixed_final_compactness_branchsafe_20260924.json")

N_NODES = 65
N_STEPS = 2048
N_HIGH = 2048
EDGE = 0.008
nodes = jnp.linspace(0.02, 0.50, N_NODES)


def transition_profile(center, width, depth):
    left = center - width / 2.0
    right = center + width / 2.0
    return -0.5 * depth * (
        jnp.tanh((nodes - left) / EDGE)
        - jnp.tanh((nodes - right) / EDGE)
    )


def build_deformed_eos(reference, center, width, depth):
    return build_nodal_sound_speed_eos(
        reference,
        transition_profile(center, width, depth),
        h_match=0.02,
        h_max=0.50,
        h_nodes=nodes,
        n_low=128,
        n_high=N_HIGH,
    )


def mass_slope(eos, h_c):
    dh = 1.0e-3
    mp = solve_observables(eos, h_c + dh, n_steps=N_STEPS).mass
    mm = solve_observables(eos, h_c - dh, n_steps=N_STEPS).mass
    return float((mp - mm) / (2.0 * dh))


def mode_result(eos, h_c):
    star = solve_observables(eos, h_c, n_steps=N_STEPS)
    mode = fundamental_radial_mode(
        eos,
        h_c,
        background_steps=4096,
        h_floor=2.0e-6,
        scan_min=-3.0,
        scan_max=6.0,
        scan_points=46,
    )
    return dict(
        compactness=float(star.compactness),
        dM_dh=mass_slope(eos, h_c),
        omega2=float(mode.omega2),
        omega2_scaled=float(mode.omega2_scaled),
        radial_nodes=int(mode.nodes),
        shooting_residual=float(mode.residual),
        dynamically_stable=bool(mode.omega2_scaled > 0.0),
    )


def first_rejected_after(payload, selected):
    candidates = [
        r for r in payload["rows"]
        if abs(r["gamma"] - selected["gamma"]) < 1.0e-12
        and abs(
            r["target_compactness"] - selected["target_compactness"]
        ) < 1.0e-12
        and r["x"] > selected["x"]
        and r.get("reachable", False)
        and not r.get("retained", False)
        and r.get("depth") is not None
    ]
    if not candidates:
        return None
    return min(candidates, key=lambda r: r["x"])


def main():
    payload = json.loads(RESULT_PATH.read_text())
    rows = []

    for selected in payload["best_by_gamma_and_target"]:
        gamma = float(selected["gamma"])
        h_c = float(selected["h_c"])
        reference = RelativisticPolytrope(K=100.0, gamma=gamma)

        path = []
        for fraction in (0.0, 0.5, 1.0):
            depth = fraction * float(selected["depth"])
            eos = build_deformed_eos(
                reference,
                float(selected["center"]),
                float(selected["width"]),
                depth,
            )
            out = mode_result(eos, h_c)
            out.update(
                path_fraction=fraction,
                depth=depth,
            )
            path.append(out)

        rejected = first_rejected_after(payload, selected)
        rejected_result = None
        if rejected is not None:
            eos_rejected = build_deformed_eos(
                reference,
                float(rejected["center"]),
                float(rejected["width"]),
                float(rejected["depth"]),
            )
            rejected_result = mode_result(eos_rejected, h_c)
            rejected_result.update(
                x=float(rejected["x"]),
                depth=float(rejected["depth"]),
                prior_proxy_dM_dh=float(rejected["dM_dh"]),
                prior_proxy_path_min_dM_dh=float(
                    rejected["path_min_dM_dh"]
                ) if rejected.get("path_min_dM_dh") is not None else None,
            )

        rows.append(
            dict(
                gamma=gamma,
                target_compactness=float(selected["target_compactness"]),
                h_c=h_c,
                selected_x=float(selected["x"]),
                selected_depth=float(selected["depth"]),
                selected_proxy_dM_dh=float(selected["dM_dh"]),
                selected_proxy_path_min_dM_dh=float(
                    selected["path_min_dM_dh"]
                ),
                selected_path=path,
                first_rejected=rejected_result,
            )
        )

    selected_endpoints = [r["selected_path"][-1] for r in rows]
    selected_all = [
        point for row in rows for point in row["selected_path"]
    ]
    rejected = [
        r["first_rejected"] for r in rows
        if r["first_rejected"] is not None
    ]

    output = dict(
        source_result=str(RESULT_PATH),
        n_selected_paths=len(rows),
        selected_all_dynamically_stable=all(
            p["dynamically_stable"] for p in selected_all
        ),
        selected_endpoint_omega2_scaled_min=float(
            min(p["omega2_scaled"] for p in selected_endpoints)
        ),
        selected_endpoint_omega2_scaled_max=float(
            max(p["omega2_scaled"] for p in selected_endpoints)
        ),
        n_first_rejected=len(rejected),
        n_first_rejected_dynamically_unstable=sum(
            not p["dynamically_stable"] for p in rejected
        ),
        rejected_omega2_scaled_min=float(
            min(p["omega2_scaled"] for p in rejected)
        ) if rejected else None,
        rejected_omega2_scaled_max=float(
            max(p["omega2_scaled"] for p in rejected)
        ) if rejected else None,
        rows=rows,
    )

    print(
        "RADIAL_TRANSITION_STABILITY_JSON="
        + json.dumps(output, sort_keys=True)
    )


if __name__ == "__main__":
    main()
