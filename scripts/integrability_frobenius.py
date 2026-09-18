"""Frobenius integrability audit for the soft EOS-response covector.

The calculation works in observable coordinates
    y = (ln C, ln Ibar, ln Lambda2)
and evaluates the soft normal covector n_A after quotienting the stellar
sequence tangent.  A local three-dimensional domain patch is built from
(h_c, alpha_hard, alpha_soft), where the EOS directions are the two
covariance-normalized right singular vectors of the projected response.

For each patch we estimate dn_A/dy^B with centered finite differences and
evaluate the Frobenius obstruction

    n . curl_y n.

The normalized helicity |n.curl n|/|curl n| is zero for an exactly
integrable codimension-one distribution.  The calculation is deliberately
run at high resolution because derivatives of the response eigenvector are
more sensitive than the stellar observables themselves.
"""

import json
import numpy as np

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from ns_response_geometry.eos import RelativisticPolytrope
from ns_response_geometry.geometry import induced_response
from ns_response_geometry.observables import solve_observables
from ns_response_geometry.response_eos import (
    build_nodal_sound_speed_eos,
    matern32_covariance,
    squared_exponential_covariance,
)

SEED = 260918
N_NODES = 17
N_STEPS = 4096
N_HIGH = 2048
DELTA_ALPHA = 0.03
DELTA_H = 0.001

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
domain_covariance = squared_exponential_covariance(
    nodes, amplitude=1.0, length_scale=0.08
)
domain_cholesky = jnp.linalg.cholesky(
    domain_covariance + 1.0e-10 * jnp.eye(N_NODES)
)


def log_observables(values, h_c):
    eos = build_nodal_sound_speed_eos(reference, values, **eos_kwargs)
    r = solve_observables(eos, h_c, n_steps=N_STEPS)
    return jnp.log(jnp.stack((r.compactness, r.ibar, r.lambda2)))


@jax.jit
def local_state(values, h_c):
    J = jax.jacfwd(lambda a: log_observables(a, h_c))(values)
    tangent = jax.jacfwd(lambda h: log_observables(values, h))(h_c)
    G = induced_response(J, response_covariance)

    that = tangent / jnp.linalg.norm(tangent)
    projector = jnp.eye(3) - jnp.outer(that, that)
    eigenvalues, eigenvectors = jnp.linalg.eigh(projector @ G @ projector)
    soft = eigenvectors[:, 1]

    # Fix the sign gauge by alignment with the embedded I--Love normal.
    ilove = jnp.array([0.0, -tangent[2], tangent[1]])
    ilove = ilove / jnp.linalg.norm(ilove)
    soft = jnp.where(jnp.dot(soft, ilove) < 0.0, -soft, soft)

    # Build a well-conditioned domain patch from projected-response SVD.
    W = projector @ J @ domain_cholesky
    _, singular_values, Vh = jnp.linalg.svd(W, full_matrices=False)
    hard_direction = domain_cholesky @ Vh[0]
    soft_direction = domain_cholesky @ Vh[1]

    # Normalize directions in the domain covariance metric.
    invC_hard = jnp.linalg.solve(domain_covariance, hard_direction)
    invC_soft = jnp.linalg.solve(domain_covariance, soft_direction)
    hard_direction = hard_direction / jnp.sqrt(hard_direction @ invC_hard)
    soft_direction = soft_direction / jnp.sqrt(soft_direction @ invC_soft)

    return (
        log_observables(values, h_c),
        soft,
        eigenvalues,
        hard_direction,
        soft_direction,
        singular_values,
    )


def patch_diagnostic(values0, h0):
    y0, n0, evals, d_hard, d_soft, singular_values = local_state(
        values0, jnp.asarray(h0)
    )

    def evaluate(h, alpha_hard, alpha_soft):
        values = values0 + alpha_hard * d_hard + alpha_soft * d_soft
        y, n = local_state(values, jnp.asarray(h))[:2]
        n = jnp.where(jnp.dot(n, n0) < 0.0, -n, n)
        return y, n

    y_columns = []
    n_columns = []

    for label, step in (
        ("h", DELTA_H),
        ("hard", DELTA_ALPHA),
        ("soft", DELTA_ALPHA),
    ):
        if label == "h":
            yp, np_ = evaluate(h0 + step, 0.0, 0.0)
            ym, nm = evaluate(h0 - step, 0.0, 0.0)
        elif label == "hard":
            yp, np_ = evaluate(h0, step, 0.0)
            ym, nm = evaluate(h0, -step, 0.0)
        else:
            yp, np_ = evaluate(h0, 0.0, step)
            ym, nm = evaluate(h0, 0.0, -step)

        y_columns.append((yp - ym) / (2.0 * step))
        n_columns.append((np_ - nm) / (2.0 * step))

    dy_dx = jnp.stack(y_columns, axis=1)
    dn_dx = jnp.stack(n_columns, axis=1)
    dn_dy = dn_dx @ jnp.linalg.inv(dy_dx)

    curl = jnp.array(
        [
            dn_dy[2, 1] - dn_dy[1, 2],
            dn_dy[0, 2] - dn_dy[2, 0],
            dn_dy[1, 0] - dn_dy[0, 1],
        ]
    )
    frobenius = jnp.dot(n0, curl)
    normalized_helicity = jnp.abs(frobenius) / (
        jnp.linalg.norm(curl) + 1.0e-30
    )
    frobenius_over_gradient = jnp.abs(frobenius) / (
        jnp.linalg.norm(dn_dy) + 1.0e-30
    )

    return dict(
        h_c=float(h0),
        compactness=float(jnp.exp(y0[0])),
        soft_covector=np.asarray(n0).tolist(),
        normal_eigenvalues=np.asarray(evals).tolist(),
        input_singular_values=np.asarray(singular_values).tolist(),
        condition_dy_dx=float(jnp.linalg.cond(dy_dx)),
        frobenius=float(frobenius),
        curl_norm=float(jnp.linalg.norm(curl)),
        normalized_helicity=float(normalized_helicity),
        frobenius_over_gradient=float(frobenius_over_gradient),
    )


def background_draws():
    rng = np.random.default_rng(SEED)
    smooth_cov = np.asarray(
        squared_exponential_covariance(
            nodes, amplitude=1.0, length_scale=0.08
        )
    )
    rough_cov = np.asarray(
        matern32_covariance(
            nodes, amplitude=1.0, length_scale=0.04
        )
    )
    Ls = np.linalg.cholesky(smooth_cov + 1.0e-10 * np.eye(N_NODES))
    Lr = np.linalg.cholesky(rough_cov + 1.0e-10 * np.eye(N_NODES))

    smooth = [Ls @ rng.normal(size=N_NODES) for _ in range(8)]
    rough = [Lr @ rng.normal(size=N_NODES) for _ in range(8)]
    return [
        ("zero", np.zeros(N_NODES)),
        ("smooth_1", smooth[0]),
        ("rough_1", rough[0]),
    ]


def main():
    # Compile once.
    local_state(jnp.zeros(N_NODES), jnp.asarray(0.16))[0].block_until_ready()

    rows = []
    for background_name, values in background_draws():
        values = jnp.asarray(values)
        for h_c in (0.12, 0.20, 0.28):
            row = patch_diagnostic(values, h_c)
            row["background"] = background_name
            rows.append(row)

    helicity = np.asarray([r["normalized_helicity"] for r in rows])
    gradnorm = np.asarray([r["frobenius_over_gradient"] for r in rows])

    payload = dict(
        seed=SEED,
        n_nodes=N_NODES,
        n_steps=N_STEPS,
        n_high=N_HIGH,
        delta_alpha=DELTA_ALPHA,
        delta_h=DELTA_H,
        summary=dict(
            n=len(rows),
            normalized_helicity_min=float(np.min(helicity)),
            normalized_helicity_median=float(np.median(helicity)),
            normalized_helicity_p90=float(np.quantile(helicity, 0.90)),
            normalized_helicity_max=float(np.max(helicity)),
            frobenius_over_gradient_median=float(np.median(gradnorm)),
            frobenius_over_gradient_max=float(np.max(gradnorm)),
        ),
        rows=rows,
    )
    print("INTEGRABILITY_JSON=" + json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
