"""Second-order curvature test for finite EOS displacements.

At each background star, freeze the local soft normal covector n0 and study

    R(s) = n0 . [ y(a0 + s d) - y(a0) ]

for random GP EOS directions d.  The exact finite displacement is compared
with

    R_1(s) = s n0 . J d

and

    R_2(s) = R_1(s) + 1/2 s^2 n0 . H[d,d].

The directional curvature is evaluated by a centered second difference at
small EOS amplitude.  The purpose is to determine the finite radius over
which second-order geometry predicts departures from the local soft tangent
plane.
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

SEED_BACKGROUNDS = 260918
SEED_DIRECTIONS = 2718
N_NODES = 17
N_STEPS = 2048
N_HIGH = 1024
N_DIRECTIONS = 48
CURVATURE_STEP = 0.03

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
draw_covariance = squared_exponential_covariance(
    nodes, amplitude=1.0, length_scale=0.08
)
draw_cholesky = np.linalg.cholesky(
    np.asarray(draw_covariance) + 1.0e-10 * np.eye(N_NODES)
)


def log_observables(values, h_c):
    eos = build_nodal_sound_speed_eos(reference, values, **eos_kwargs)
    r = solve_observables(eos, h_c, n_steps=N_STEPS)
    return jnp.log(jnp.stack((r.compactness, r.ibar, r.lambda2)))


@jax.jit
def observable_map(values, h_c):
    return log_observables(values, h_c)


@jax.jit
def center_geometry(values, h_c):
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

    return y, J, soft, eigenvalues


def random_directions():
    rng = np.random.default_rng(SEED_DIRECTIONS)
    directions = []
    for _ in range(N_DIRECTIONS):
        d = draw_cholesky @ rng.normal(size=N_NODES)
        d = d / np.sqrt(np.mean(d * d))
        directions.append(d)
    return directions


def evaluate_patch(values0, h_c):
    values0 = jnp.asarray(values0)
    h_c = jnp.asarray(h_c)
    y0, J, soft, eigenvalues = center_geometry(values0, h_c)

    rows = []
    for direction in random_directions():
        d = jnp.asarray(direction)

        linear_coefficient = float(soft @ (J @ d))

        yp = observable_map(
            values0 + CURVATURE_STEP * d, h_c
        )
        ym = observable_map(
            values0 - CURVATURE_STEP * d, h_c
        )
        quadratic_coefficient = float(
            soft
            @ (yp - 2.0 * y0 + ym)
            / CURVATURE_STEP**2
        )

        actual = {}
        for amplitude in (0.05, 0.10, 0.25, 0.50):
            y = observable_map(values0 + amplitude * d, h_c)
            actual[str(amplitude)] = float(soft @ (y - y0))

        rows.append(
            dict(
                linear=linear_coefficient,
                quadratic=quadratic_coefficient,
                actual=actual,
            )
        )

    output = dict(
        compactness=float(jnp.exp(y0[0])),
        normal_ratio=float(jnp.sqrt(eigenvalues[1] / eigenvalues[2])),
        amplitudes={},
    )

    for amplitude in (0.05, 0.10, 0.25, 0.50):
        key = str(amplitude)
        exact = np.asarray([r["actual"][key] for r in rows])
        linear = np.asarray(
            [r["linear"] * amplitude for r in rows]
        )
        quadratic = np.asarray(
            [
                r["linear"] * amplitude
                + 0.5 * r["quadratic"] * amplitude**2
                for r in rows
            ]
        )

        actual_rms = np.sqrt(np.mean(exact**2))

        def score(prediction):
            return dict(
                relative_rmse=float(
                    np.sqrt(np.mean((prediction - exact) ** 2))
                    / (actual_rms + 1.0e-30)
                ),
                correlation=float(np.corrcoef(prediction, exact)[0, 1]),
                predicted_rms_over_actual=float(
                    np.sqrt(np.mean(prediction**2))
                    / (actual_rms + 1.0e-30)
                ),
            )

        curvature_only = np.asarray(
            [
                0.5 * r["quadratic"] * amplitude**2
                for r in rows
            ]
        )

        output["amplitudes"][key] = dict(
            actual_rms=float(actual_rms),
            linear=score(linear),
            quadratic=score(quadratic),
            quadratic_fraction=float(
                np.sqrt(np.mean(curvature_only**2))
                / (actual_rms + 1.0e-30)
            ),
        )

    return output


def backgrounds():
    rng = np.random.default_rng(SEED_BACKGROUNDS)
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
    Ls = np.linalg.cholesky(
        smooth_cov + 1.0e-10 * np.eye(N_NODES)
    )
    Lr = np.linalg.cholesky(
        rough_cov + 1.0e-10 * np.eye(N_NODES)
    )

    smooth = [Ls @ rng.normal(size=N_NODES) for _ in range(8)]
    rough = [Lr @ rng.normal(size=N_NODES) for _ in range(8)]
    return (
        ("zero", np.zeros(N_NODES)),
        ("smooth_1", smooth[0]),
        ("rough_1", rough[0]),
    )


def main():
    center_geometry(
        jnp.zeros(N_NODES), jnp.asarray(0.16)
    )[0].block_until_ready()

    rows = []
    for name, values in backgrounds():
        for h_c in (0.12, 0.20, 0.28):
            row = evaluate_patch(values, h_c)
            row["background"] = name
            row["h_c"] = h_c
            rows.append(row)

    summary = {}
    for amplitude in ("0.05", "0.1", "0.25", "0.5"):
        summary[amplitude] = {}
        for model in ("linear", "quadratic"):
            values = np.asarray(
                [
                    row["amplitudes"][amplitude][model][
                        "relative_rmse"
                    ]
                    for row in rows
                ]
            )
            summary[amplitude][model] = dict(
                median_relative_rmse=float(np.median(values)),
                p90_relative_rmse=float(np.quantile(values, 0.90)),
                max_relative_rmse=float(np.max(values)),
            )

        curvature_fraction = np.asarray(
            [
                row["amplitudes"][amplitude][
                    "quadratic_fraction"
                ]
                for row in rows
            ]
        )
        summary[amplitude]["quadratic_fraction_median"] = float(
            np.median(curvature_fraction)
        )

    payload = dict(
        n_nodes=N_NODES,
        n_steps=N_STEPS,
        n_high=N_HIGH,
        n_directions=N_DIRECTIONS,
        curvature_step=CURVATURE_STEP,
        summary=summary,
        rows=rows,
    )
    print("CURVATURE_JSON=" + json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
