"""Controlled breakdown with localized sound-speed softening.

A phase-transition-like EOS feature is modeled as a smooth latent-field
window that strongly lowers c_s^2 over a finite enthalpy interval and then
returns to the baseline high-density stiffness.  The construction remains
causal and thermodynamically consistent.

The script evaluates two refined transition cases with 65 EOS nodes and
compares them with the baseline response.  It also localizes the increase in
I--Love sensitivity in the nodal functional derivative.
"""

import json
import numpy as np

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from ns_response_geometry.eos import RelativisticPolytrope
from ns_response_geometry.geometry import (
    induced_response,
    plane_normal_variance,
)
from ns_response_geometry.observables import solve_observables
from ns_response_geometry.response_eos import (
    build_nodal_sound_speed_eos,
    squared_exponential_covariance,
)

N_NODES = 65
N_STEPS = 2048
N_HIGH = 2048

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


def transition_profile(center, width, depth, edge=0.008):
    left = center - width / 2.0
    right = center + width / 2.0
    return -0.5 * depth * (
        jnp.tanh((nodes - left) / edge)
        - jnp.tanh((nodes - right) / edge)
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
        tangent,
        soft,
        ilove,
        eigenvalues,
        jnp.array(
            [
                dM_dh,
                jnp.sqrt(ilove_var / clove_var),
                jnp.min(eos.cs2_grid[128:]),
            ]
        ),
    )


def summarize(values, h_c):
    (
        y,
        J,
        tangent,
        soft,
        ilove,
        eigenvalues,
        extra,
    ) = response_state(values, jnp.asarray(h_c))

    return dict(
        h_c=float(h_c),
        compactness=float(jnp.exp(y[0])),
        ibar=float(jnp.exp(y[1])),
        lambda2=float(jnp.exp(y[2])),
        dM_dh=float(extra[0]),
        normal_rms_ratio=float(
            jnp.sqrt(eigenvalues[1] / eigenvalues[2])
        ),
        ilove_alignment=float(jnp.abs(jnp.dot(soft, ilove))),
        ilove_clove_rms_ratio=float(extra[1]),
        min_cs2=float(extra[2]),
        soft_covector=np.asarray(soft).tolist(),
    )


def localization(values_transition, h_c=0.30):
    zero = jnp.zeros(N_NODES)

    y0, J0, _, _, ilove0, _, _ = response_state(
        zero, jnp.asarray(h_c)
    )
    yt, Jt, _, _, ilovet, _, _ = response_state(
        values_transition, jnp.asarray(h_c)
    )

    response0 = np.asarray(ilove0 @ J0)
    responset = np.asarray(ilovet @ Jt)
    h = np.asarray(nodes)

    density0 = response0**2
    densityt = responset**2

    window = (h >= 0.14) & (h <= 0.18)
    excess = np.maximum(densityt - density0, 0.0)

    C = np.asarray(response_covariance)
    variance0 = float(response0 @ C @ response0)
    variancet = float(responset @ C @ responset)

    return dict(
        baseline_compactness=float(jnp.exp(y0[0])),
        transition_compactness=float(jnp.exp(yt[0])),
        covariance_weighted_ilove_variance_baseline=variance0,
        covariance_weighted_ilove_variance_transition=variancet,
        covariance_weighted_variance_factor=variancet / variance0,
        transition_window=[0.14, 0.18],
        transition_window_fraction_of_direct_sensitivity=float(
            densityt[window].sum() / densityt.sum()
        ),
        transition_window_fraction_of_positive_excess=float(
            excess[window].sum() / excess.sum()
        ),
        excess_peak_enthalpy=float(h[np.argmax(excess)]),
        excess_centroid_enthalpy=float(
            (h * excess).sum() / excess.sum()
        ),
    )


def main():
    response_state(
        jnp.zeros(N_NODES), jnp.asarray(0.30)
    )[0].block_until_ready()

    case_a = transition_profile(
        center=0.16, width=0.04, depth=2.0
    )
    case_b = transition_profile(
        center=0.16, width=0.08, depth=2.0
    )

    payload = dict(
        n_nodes=N_NODES,
        n_steps=N_STEPS,
        n_high=N_HIGH,
        baseline_h030=summarize(jnp.zeros(N_NODES), 0.30),
        baseline_h036=summarize(jnp.zeros(N_NODES), 0.36),
        transition_A=dict(
            profile=dict(center=0.16, width=0.04, depth=2.0, edge=0.008),
            response=summarize(case_a, 0.30),
        ),
        transition_B=dict(
            profile=dict(center=0.16, width=0.08, depth=2.0, edge=0.008),
            response=summarize(case_b, 0.36),
        ),
        localization_A=localization(case_a, 0.30),
    )

    print("BREAKDOWN_JSON=" + json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
