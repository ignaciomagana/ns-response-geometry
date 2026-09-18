"""High-resolution transition-location refinement in physical stellar coordinates.

Two analyses are run with 65 EOS nodes:
1. Refine the three strongest stable "moving ridge" cases selected by the
   33-node screen at fixed width 0.08 and depth 1.5.
2. Scan transition location at fixed fractional width w/h_c = 0.25 and depth
   1.5 for h_transition/h_c in [0.25, ..., 0.75].

For each configuration the transition center is mapped to r/R and m/M using
the same enthalpy-coordinate TOV solver. This tests whether the breakdown
tracks a common physical shell of the star rather than an absolute enthalpy.
"""

import json
import numpy as np

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from ns_response_geometry.eos import RelativisticPolytrope
from ns_response_geometry.geometry import induced_response, plane_normal_variance
from ns_response_geometry.observables import solve_observables
from ns_response_geometry.response_eos import (
    build_nodal_sound_speed_eos,
    squared_exponential_covariance,
)
from ns_response_geometry.tov import solve_star_profile

N_NODES = 65
N_STEPS = 2048
N_HIGH = 2048
EDGE = 0.008
DEPTH = 1.5

H_C_VALUES = (0.20, 0.30, 0.40)
X_VALUES = (0.25, 0.35, 0.45, 0.55, 0.65, 0.75)
WIDTH_FRACTION = 0.25

nodes = jnp.linspace(0.02, 0.50, N_NODES)
reference = RelativisticPolytrope(K=100.0, gamma=2.0)
eos_kwargs = dict(
    h_match=0.02,
    h_max=0.50,
    h_nodes=nodes,
    n_low=128,
    n_high=N_HIGH,
)
response_covariance = squared_exponential_covariance(
    nodes, amplitude=0.10, length_scale=0.08
)


def transition_profile(center, width, depth=DEPTH):
    left = center - width / 2.0
    right = center + width / 2.0
    return -0.5 * depth * (
        jnp.tanh((nodes - left) / EDGE)
        - jnp.tanh((nodes - right) / EDGE)
    )


def log_observables(values, h_c):
    eos = build_nodal_sound_speed_eos(reference, values, **eos_kwargs)
    r = solve_observables(eos, h_c, n_steps=N_STEPS)
    return jnp.log(jnp.stack((r.compactness, r.ibar, r.lambda2)))


def mass(values, h_c):
    eos = build_nodal_sound_speed_eos(reference, values, **eos_kwargs)
    return solve_observables(eos, h_c, n_steps=N_STEPS).mass


@jax.jit
def response_state(values, h_c):
    y = log_observables(values, h_c)
    J = jax.jacfwd(lambda a: log_observables(a, h_c))(values)
    tangent = jax.jacfwd(lambda h: log_observables(values, h))(h_c)
    G = induced_response(J, response_covariance)

    that = tangent / jnp.linalg.norm(tangent)
    projector = jnp.eye(3) - jnp.outer(that, that)
    eigenvalues, eigenvectors = jnp.linalg.eigh(projector @ G @ projector)
    soft = eigenvectors[:, 1]

    ilove = jnp.array([0.0, -tangent[2], tangent[1]])
    ilove = ilove / jnp.linalg.norm(ilove)
    soft = jnp.where(jnp.dot(soft, ilove) < 0.0, -soft, soft)

    ilove_var, _ = plane_normal_variance(
        G, tangent, [1, 2], jnp.eye(3)
    )
    clove_var, _ = plane_normal_variance(
        G, tangent, [0, 2], jnp.eye(3)
    )

    dh = 1.0e-3
    dM_dh = (
        mass(values, h_c + dh) - mass(values, h_c - dh)
    ) / (2.0 * dh)

    eos = build_nodal_sound_speed_eos(reference, values, **eos_kwargs)

    return (
        y,
        J,
        soft,
        ilove,
        eigenvalues,
        ilove_var,
        clove_var,
        dM_dh,
        jnp.min(eos.cs2_grid[128:]),
    )


def physical_shell(eos, h_c, center, width):
    profile = solve_star_profile(eos, h_c, n_steps=N_STEPS)
    h = np.asarray(profile.enthalpy)[::-1]
    radius = np.asarray(profile.radius)[::-1]
    mass_profile = np.asarray(profile.mass)[::-1]

    R = float(radius[-1])
    M = float(mass_profile[-1])

    def at(target):
        r = float(np.interp(target, h, radius))
        m = float(np.interp(target, h, mass_profile))
        return dict(r_over_R=r / R, m_over_M=m / M)

    left = max(center - width / 2.0, 0.0)
    right = min(center + width / 2.0, h_c)

    return dict(
        center=at(min(center, h_c)),
        inner_edge=at(min(right, h_c)),
        outer_edge=at(min(left, h_c)),
    )


def evaluate(center, width, depth, h_c):
    values = transition_profile(center, width, depth)
    zero = jnp.zeros(N_NODES)

    transition = response_state(values, jnp.asarray(h_c))
    baseline = response_state(zero, jnp.asarray(h_c))

    (
        y,
        J,
        soft,
        ilove,
        eigenvalues,
        ilove_var,
        clove_var,
        dM_dh,
        min_cs2,
    ) = transition
    (
        y0,
        J0,
        _,
        ilove0,
        evals0,
        ilove_var0,
        clove_var0,
        _,
        _,
    ) = baseline

    eos = build_nodal_sound_speed_eos(reference, values, **eos_kwargs)
    eos0 = build_nodal_sound_speed_eos(reference, zero, **eos_kwargs)

    transition_response = np.asarray(ilove @ J)
    baseline_response = np.asarray(ilove0 @ J0)
    direct = transition_response**2
    direct0 = baseline_response**2
    excess = np.maximum(direct - direct0, 0.0)

    hnodes = np.asarray(nodes)
    left = center - width / 2.0
    right = center + width / 2.0
    window = (hnodes >= left) & (hnodes <= right)
    excess_sum = float(excess.sum())

    Cmat = np.asarray(response_covariance)
    var = float(transition_response @ Cmat @ transition_response)
    var0 = float(baseline_response @ Cmat @ baseline_response)

    compactness = float(jnp.exp(y[0]))
    finite = (
        np.all(np.isfinite(np.asarray(y)))
        and np.all(np.isfinite(np.asarray(eigenvalues)))
        and np.isfinite(float(dM_dh))
    )
    stable = finite and float(dM_dh) > 0.0
    physical = stable and 0.0 < compactness < 0.35

    return dict(
        center=float(center),
        width=float(width),
        depth=float(depth),
        h_c=float(h_c),
        center_over_h_c=float(center / h_c),
        compactness=compactness,
        baseline_compactness=float(jnp.exp(y0[0])),
        stable=bool(stable),
        physical_screen=bool(physical),
        dM_dh=float(dM_dh),
        min_cs2=float(min_cs2),
        ilove_alignment=float(jnp.abs(jnp.dot(soft, ilove))),
        ilove_clove_rms_ratio=float(jnp.sqrt(ilove_var / clove_var)),
        normal_rms_ratio=float(jnp.sqrt(eigenvalues[1] / eigenvalues[2])),
        baseline_normal_rms_ratio=float(
            jnp.sqrt(evals0[1] / evals0[2])
        ),
        ilove_variance_factor=float(var / var0),
        localized_positive_excess_fraction=(
            float(excess[window].sum() / excess_sum)
            if excess_sum > 0.0 else 0.0
        ),
        excess_peak_enthalpy=(
            float(hnodes[np.argmax(excess)])
            if excess_sum > 0.0 else None
        ),
        transition_shell=physical_shell(eos, h_c, center, width),
        baseline_shell=physical_shell(eos0, h_c, center, width),
    )


def best(rows):
    good = [row for row in rows if row["physical_screen"]]
    return dict(
        n=len(good),
        weakest_alignment=min(good, key=lambda r: r["ilove_alignment"]),
        largest_ilove_clove=max(
            good, key=lambda r: r["ilove_clove_rms_ratio"]
        ),
        largest_normal_ratio=max(
            good, key=lambda r: r["normal_rms_ratio"]
        ),
        largest_variance_factor=max(
            good, key=lambda r: r["ilove_variance_factor"]
        ),
    )


def main():
    response_state(
        jnp.zeros(N_NODES), jnp.asarray(0.30)
    )[0].block_until_ready()

    ridge_cases = (
        (0.20, 0.10, 0.08, 1.5),
        (0.30, 0.14, 0.08, 1.5),
        (0.40, 0.18, 0.08, 1.5),
    )
    ridge = [
        evaluate(center, width, depth, h_c)
        for h_c, center, width, depth in ridge_cases
    ]

    scaled = []
    for h_c in H_C_VALUES:
        width = WIDTH_FRACTION * h_c
        for x in X_VALUES:
            center = x * h_c
            scaled.append(evaluate(center, width, DEPTH, h_c))

    best_by_hc = {}
    for h_c in H_C_VALUES:
        subset = [r for r in scaled if r["h_c"] == h_c]
        best_by_hc[str(h_c)] = best(subset)

    payload = dict(
        refined=True,
        n_nodes=N_NODES,
        n_steps=N_STEPS,
        n_high=N_HIGH,
        edge=EDGE,
        ridge_cases=ridge,
        scaled_location=dict(
            depth=DEPTH,
            width_fraction=WIDTH_FRACTION,
            x_values=list(X_VALUES),
            best_overall=best(scaled),
            best_by_h_c=best_by_hc,
            rows=scaled,
        ),
    )
    print("TRANSITION_LOCATION_JSON=" + json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
