"""Systematic screen of localized sound-speed softening.

This is a screening calculation.  It scans transition location, width, depth,
and stellar central enthalpy with 33 EOS nodes.  Candidate extrema are later
refined with 65 nodes.

For every finite stable star it records:
- soft/hard full normal RMS ratio;
- alignment of the soft covector with the embedded I--Love normal;
- I--Love/C--Love RMS ratio;
- change in covariance-weighted I--Love variance relative to the zero-profile
  star at the same h_c;
- fraction of the positive excess direct I--Love sensitivity localized
  inside the imposed softening window.

The transition is a smooth latent-field top-hat and remains causal by
construction.
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

N_NODES = 33
N_STEPS = 1024
N_HIGH = 1024
EDGE = 0.008

CENTERS = (0.10, 0.14, 0.18, 0.22)
WIDTHS = (0.04, 0.08)
DEPTHS = (0.5, 1.0, 1.5, 2.0)
H_C_VALUES = (0.20, 0.30, 0.40)

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


def transition_profile(center, width, depth):
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


def finite_scalar(x):
    return bool(np.isfinite(float(x)))


def summarize_case(values, h_c, baseline, center, width, depth):
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
    ) = response_state(values, jnp.asarray(h_c))

    q = np.asarray(y)
    compactness = float(np.exp(q[0]))

    finite = (
        np.all(np.isfinite(q))
        and np.all(np.isfinite(np.asarray(eigenvalues)))
        and finite_scalar(ilove_var)
        and finite_scalar(clove_var)
        and finite_scalar(dM_dh)
    )
    stable = finite and float(dM_dh) > 0.0
    physical_screen = stable and 0.0 < compactness < 0.35

    ilove_response = np.asarray(ilove @ J)
    base_response = baseline["ilove_response"]

    direct_density = ilove_response**2
    base_density = base_response**2
    excess = np.maximum(direct_density - base_density, 0.0)

    left = center - width / 2.0
    right = center + width / 2.0
    h = np.asarray(nodes)
    window = (h >= left) & (h <= right)

    C = np.asarray(response_covariance)
    transition_variance = float(ilove_response @ C @ ilove_response)
    base_variance = baseline["variance"]

    excess_sum = float(excess.sum())
    localized_excess = (
        float(excess[window].sum() / excess_sum)
        if excess_sum > 0.0
        else 0.0
    )

    return dict(
        center=float(center),
        width=float(width),
        depth=float(depth),
        h_c=float(h_c),
        compactness=compactness,
        stable=bool(stable),
        physical_screen=bool(physical_screen),
        dM_dh=float(dM_dh),
        min_cs2=float(min_cs2),
        normal_rms_ratio=float(
            jnp.sqrt(jnp.maximum(eigenvalues[1], 0.0) / eigenvalues[2])
        ) if finite else None,
        ilove_alignment=float(jnp.abs(jnp.dot(soft, ilove)))
        if finite else None,
        ilove_clove_rms_ratio=float(jnp.sqrt(ilove_var / clove_var))
        if finite else None,
        ilove_variance_factor=float(transition_variance / base_variance)
        if finite else None,
        localized_positive_excess_fraction=localized_excess,
        excess_peak_enthalpy=float(h[np.argmax(excess)])
        if excess_sum > 0.0 else None,
    )


def baseline_at_h(h_c):
    zero = jnp.zeros(N_NODES)
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
    ) = response_state(zero, jnp.asarray(h_c))

    response = np.asarray(ilove @ J)
    C = np.asarray(response_covariance)
    return dict(
        compactness=float(jnp.exp(y[0])),
        ilove_response=response,
        variance=float(response @ C @ response),
        normal_rms_ratio=float(jnp.sqrt(eigenvalues[1] / eigenvalues[2])),
        ilove_alignment=float(jnp.abs(jnp.dot(soft, ilove))),
        ilove_clove_rms_ratio=float(jnp.sqrt(ilove_var / clove_var)),
        dM_dh=float(dM_dh),
        min_cs2=float(min_cs2),
    )


def valid_rows(rows):
    return [r for r in rows if r["physical_screen"]]


def extrema(rows):
    good = valid_rows(rows)
    return dict(
        n_total=len(rows),
        n_physical=len(good),
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
        strongest_localization=max(
            good, key=lambda r: r["localized_positive_excess_fraction"]
        ),
    )


def main():
    response_state(
        jnp.zeros(N_NODES), jnp.asarray(0.30)
    )[0].block_until_ready()

    baselines = {str(h): baseline_at_h(h) for h in H_C_VALUES}
    rows = []

    for center in CENTERS:
        for width in WIDTHS:
            for depth in DEPTHS:
                values = transition_profile(center, width, depth)
                for h_c in H_C_VALUES:
                    rows.append(
                        summarize_case(
                            values,
                            h_c,
                            baselines[str(h_c)],
                            center,
                            width,
                            depth,
                        )
                    )

    payload = dict(
        screening=True,
        n_nodes=N_NODES,
        n_steps=N_STEPS,
        n_high=N_HIGH,
        edge=EDGE,
        grid=dict(
            centers=list(CENTERS),
            widths=list(WIDTHS),
            depths=list(DEPTHS),
            h_c_values=list(H_C_VALUES),
        ),
        baselines=baselines,
        extrema=extrema(rows),
        rows=rows,
    )
    print("TRANSITION_SCAN_JSON=" + json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
