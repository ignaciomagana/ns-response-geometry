"""Robustness audit for the local I--Love response hierarchy.

This is the first production-oriented stress test of H1/H2. It uses nodal
latent sound-speed coordinates so basis refinement approximates a fixed GP
metric rather than changing the coefficient meaning.
"""

import json

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

from ns_response_geometry.eos import RelativisticPolytrope
from ns_response_geometry.geometry import (
    induced_response,
    plane_normal_variance,
)
from ns_response_geometry.observables import solve_observables
from ns_response_geometry.response import (
    log_observable_vector,
    response_jacobian,
    sequence_tangent,
)
from ns_response_geometry.response_eos import (
    build_nodal_sound_speed_eos,
    exponential_covariance,
    matern32_covariance,
    squared_exponential_covariance,
)


KERNELS = {
    "se": squared_exponential_covariance,
    "matern32": matern32_covariance,
    "exponential": exponential_covariance,
}


def response_state(gamma, h_c, n_nodes, *, n_steps=1024, n_high=512):
    reference = RelativisticPolytrope(K=100.0, gamma=gamma)
    nodes = jnp.linspace(0.02, 0.50, n_nodes)
    values = jnp.zeros(n_nodes)
    kwargs = dict(
        h_match=0.02,
        h_max=0.50,
        h_nodes=nodes,
        n_low=128,
        n_high=n_high,
    )
    J = response_jacobian(
        reference, values, h_c, n_steps=n_steps, eos_kwargs=kwargs
    )
    tangent = sequence_tangent(
        reference, values, h_c, n_steps=n_steps, eos_kwargs=kwargs
    )
    eos = build_nodal_sound_speed_eos(reference, values, **kwargs)
    y = log_observable_vector(eos, h_c, n_steps=n_steps)
    return reference, nodes, values, kwargs, J, tangent, y


def plane_summary(G, tangent, metric):
    ilove_var, _ = plane_normal_variance(G, tangent, [1, 2], metric)
    clove_var, _ = plane_normal_variance(G, tangent, [0, 2], metric)
    return dict(
        ilove_variance=float(ilove_var),
        clove_variance=float(clove_var),
        ilove_rms=float(jnp.sqrt(ilove_var)),
        clove_rms=float(jnp.sqrt(clove_var)),
        variance_ratio=float(ilove_var / clove_var),
        rms_ratio=float(jnp.sqrt(ilove_var / clove_var)),
    )


def finite_difference_check():
    gamma = 2.0
    h_c = 0.16
    n_nodes = 9
    reference, nodes, values, kwargs, J, _, _ = response_state(
        gamma, h_c, n_nodes, n_steps=1024, n_high=512
    )
    eps = 2.0e-4
    columns = []
    for i in range(n_nodes):
        delta = jnp.zeros(n_nodes).at[i].set(eps)
        eos_p = build_nodal_sound_speed_eos(reference, values + delta, **kwargs)
        eos_m = build_nodal_sound_speed_eos(reference, values - delta, **kwargs)
        yp = log_observable_vector(eos_p, h_c, n_steps=1024)
        ym = log_observable_vector(eos_m, h_c, n_steps=1024)
        columns.append((yp - ym) / (2.0 * eps))
    Jfd = jnp.stack(columns, axis=1)
    absolute = jnp.abs(J - Jfd)
    scale = jnp.maximum(jnp.abs(Jfd), 1.0e-10)
    relative = absolute / scale
    return dict(
        n_nodes=n_nodes,
        step=eps,
        max_absolute_error=float(jnp.max(absolute)),
        max_relative_error=float(jnp.max(relative)),
        rms_relative_error=float(jnp.sqrt(jnp.mean(relative**2))),
    )


def basis_convergence():
    rows = []
    for n_nodes in (5, 9, 17, 33):
        _, nodes, _, _, J, tangent, y = response_state(
            2.0, 0.16, n_nodes
        )
        covariance = squared_exponential_covariance(
            nodes, amplitude=0.10, length_scale=0.08
        )
        G = induced_response(J, covariance)
        summary = plane_summary(G, tangent, jnp.eye(3))
        rows.append(
            dict(
                n_nodes=n_nodes,
                compactness=float(jnp.exp(y[0])),
                **summary,
            )
        )
    limit = rows[-1]["rms_ratio"]
    for row in rows:
        row["fractional_difference_from_33"] = (row["rms_ratio"] - limit) / limit
    return rows


def covariance_robustness():
    _, nodes, _, _, J, tangent, y = response_state(2.0, 0.16, 17)
    metrics = {
        "identity": jnp.eye(3),
        "I_weighted": jnp.diag(jnp.array([1.0, 4.0, 1.0])),
        "Love_weighted": jnp.diag(jnp.array([1.0, 1.0, 4.0])),
    }
    rows = []
    for kernel_name, kernel in KERNELS.items():
        for length_scale in (0.04, 0.08, 0.16):
            covariance = kernel(nodes, amplitude=0.10, length_scale=length_scale)
            G = induced_response(J, covariance)
            for metric_name, metric in metrics.items():
                rows.append(
                    dict(
                        kernel=kernel_name,
                        length_scale=length_scale,
                        observable_metric=metric_name,
                        compactness=float(jnp.exp(y[0])),
                        **plane_summary(G, tangent, metric),
                    )
                )
    return rows


def reference_eos_robustness():
    rows = []
    for gamma in (1.70, 1.85, 2.00):
        for h_c in (0.12, 0.20, 0.30):
            _, nodes, _, _, J, tangent, y = response_state(
                gamma, h_c, 17
            )
            covariance = squared_exponential_covariance(
                nodes, amplitude=0.10, length_scale=0.08
            )
            G = induced_response(J, covariance)
            rows.append(
                dict(
                    gamma=gamma,
                    h_c=h_c,
                    compactness=float(jnp.exp(y[0])),
                    ibar=float(jnp.exp(y[1])),
                    lambda2=float(jnp.exp(y[2])),
                    **plane_summary(G, tangent, jnp.eye(3)),
                )
            )
    return rows


def coordinate_invariance():
    _, nodes, _, _, J, tangent, y = response_state(2.0, 0.16, 17)
    covariance = squared_exponential_covariance(
        nodes, amplitude=0.10, length_scale=0.08
    )
    G = induced_response(J, covariance)
    g = jnp.eye(3)
    base = plane_summary(G, tangent, g)

    # Change from log observables z=(ln C,ln Ibar,ln Lambda) to raw
    # q=(C,Ibar,Lambda). Local tensor transformation has Jacobian diag(q).
    raw = jnp.exp(y)
    A = jnp.diag(raw)
    Ainv = jnp.diag(1.0 / raw)
    G_raw = A @ G @ A.T
    t_raw = A @ tangent
    g_raw = Ainv.T @ g @ Ainv
    transformed = plane_summary(G_raw, t_raw, g_raw)

    return dict(
        log_coordinates=base,
        raw_coordinates_with_transformed_metric=transformed,
        ilove_relative_difference=(
            transformed["ilove_variance"] - base["ilove_variance"]
        ) / base["ilove_variance"],
        clove_relative_difference=(
            transformed["clove_variance"] - base["clove_variance"]
        ) / base["clove_variance"],
    )


def sequence_resolution_check():
    rows = []
    for n_steps, n_high in ((512, 256), (1024, 512), (2048, 1024)):
        _, nodes, _, _, J, tangent, y = response_state(
            2.0, 0.16, 17, n_steps=n_steps, n_high=n_high
        )
        covariance = squared_exponential_covariance(
            nodes, amplitude=0.10, length_scale=0.08
        )
        G = induced_response(J, covariance)
        rows.append(
            dict(
                n_steps=n_steps,
                n_high=n_high,
                compactness=float(jnp.exp(y[0])),
                **plane_summary(G, tangent, jnp.eye(3)),
            )
        )
    limit = rows[-1]["rms_ratio"]
    for row in rows:
        row["fractional_difference_from_finest"] = (
            row["rms_ratio"] - limit
        ) / limit
    return rows


def main():
    payload = dict(
        production_representation="nodal latent sound-speed perturbation",
        derivative_check=finite_difference_check(),
        basis_convergence=basis_convergence(),
        covariance_and_observable_metric=covariance_robustness(),
        reference_eos=reference_eos_robustness(),
        coordinate_invariance=coordinate_invariance(),
        numerical_resolution=sequence_resolution_check(),
    )
    print("ROBUSTNESS_JSON=" + json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
