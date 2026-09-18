"""Preliminary response-geometry calculation.

This is a diagnostic, not a production science result. It uses one reference
Gamma=2 relativistic polytrope, five latent sound-speed modes, a smooth
coefficient covariance, and a Euclidean metric in log-observable coordinates.
"""

import json

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

from ns_response_geometry.eos import RelativisticPolytrope
from ns_response_geometry.geometry import induced_response, plane_normal_variance
from ns_response_geometry.response import (
    log_observable_vector,
    response_jacobian,
    sequence_tangent,
)
from ns_response_geometry.response_eos import (
    build_sound_speed_eos,
    squared_exponential_covariance,
)


def main():
    reference = RelativisticPolytrope(K=100.0, gamma=2.0)
    n_modes = 5
    h_match = 0.02
    h_max = 0.55
    centers = jnp.linspace(h_match, h_max, n_modes)
    coeffs = jnp.zeros(n_modes)
    eos_kwargs = dict(
        h_match=h_match,
        h_max=h_max,
        n_low=128,
        n_high=1024,
        centers=centers,
        width=(h_max - h_match) / (n_modes - 1),
    )
    covariance = squared_exponential_covariance(
        centers, amplitude=0.10, length_scale=0.08
    )

    rows = []
    for h_c in (0.08, 0.12, 0.16, 0.20, 0.25, 0.30):
        J = response_jacobian(
            reference,
            coeffs,
            h_c,
            n_steps=2048,
            eos_kwargs=eos_kwargs,
        )
        tangent = sequence_tangent(
            reference,
            coeffs,
            h_c,
            n_steps=2048,
            eos_kwargs=eos_kwargs,
        )
        G = induced_response(J, covariance)

        ilove_var, ilove_n = plane_normal_variance(G, tangent, [1, 2])
        clove_var, clove_n = plane_normal_variance(G, tangent, [0, 2])

        base_eos = build_sound_speed_eos(reference, coeffs, **eos_kwargs)
        y = log_observable_vector(base_eos, h_c, n_steps=2048)
        rows.append(
            dict(
                h_c=float(h_c),
                compactness=float(jnp.exp(y[0])),
                ibar=float(jnp.exp(y[1])),
                lambda2=float(jnp.exp(y[2])),
                ilove_variance=float(ilove_var),
                clove_variance=float(clove_var),
                ilove_rms=float(jnp.sqrt(ilove_var)),
                clove_rms=float(jnp.sqrt(clove_var)),
                variance_ratio=float(ilove_var / clove_var),
                rms_ratio=float(jnp.sqrt(ilove_var / clove_var)),
                ilove_normal=np.asarray(ilove_n).tolist(),
                clove_normal=np.asarray(clove_n).tolist(),
            )
        )

    # One explicit autodiff vs symmetric finite-difference check at h_c=0.16.
    h_c = 0.16
    J = response_jacobian(
        reference, coeffs, h_c, n_steps=2048, eos_kwargs=eos_kwargs
    )
    eps = 2.0e-4
    fd_cols = []
    for i in range(n_modes):
        delta = jnp.zeros(n_modes).at[i].set(eps)
        eos_p = build_sound_speed_eos(reference, coeffs + delta, **eos_kwargs)
        eos_m = build_sound_speed_eos(reference, coeffs - delta, **eos_kwargs)
        yp = log_observable_vector(eos_p, h_c, n_steps=2048)
        ym = log_observable_vector(eos_m, h_c, n_steps=2048)
        fd_cols.append((yp - ym) / (2.0 * eps))
    J_fd = jnp.stack(fd_cols, axis=1)

    scale = jnp.maximum(jnp.abs(J_fd), 1.0e-10)
    rel = jnp.abs(J - J_fd) / scale
    derivative_check = dict(
        max_absolute_error=float(jnp.max(jnp.abs(J - J_fd))),
        max_relative_error=float(jnp.max(rel)),
        rms_relative_error=float(jnp.sqrt(jnp.mean(rel**2))),
    )

    payload = dict(
        diagnostic_only=True,
        reference_eos="Gamma=2 relativistic polytrope, K=100",
        n_modes=n_modes,
        covariance=dict(amplitude=0.10, length_scale=0.08),
        observable_metric="Euclidean in (ln C, ln Ibar, ln Lambda2)",
        derivative_check=derivative_check,
        rows=rows,
    )
    print("RESULT_JSON=" + json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
