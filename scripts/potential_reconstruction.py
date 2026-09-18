"""Reconstruct a global scalar quasi-invariant from soft response covectors.

The soft covector is normalized by its ln(Ibar) component,

    m = n / n_I,

so that locally

    d ln(Ibar) + m_C d ln C + m_L d ln Lambda ~= 0.

If the field is integrable, there exists a scalar
    F = ln(Ibar) + g(ln C, ln Lambda)
whose gradient matches m.  We fit only the gradient samples, never the
observable values themselves.  Cross-validation by EOS background then tests
whether the integrated potential is common across unseen backgrounds.

For comparison, a one-dimensional I--Love reconstruction uses
    F_1D = ln(Ibar) + g(ln Lambda).
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
N_STEPS = 2048
N_HIGH = 1024

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


def log_observables(values, h_c):
    eos = build_nodal_sound_speed_eos(reference, values, **eos_kwargs)
    r = solve_observables(eos, h_c, n_steps=N_STEPS)
    return jnp.log(jnp.stack((r.compactness, r.ibar, r.lambda2)))


def mass(values, h_c):
    eos = build_nodal_sound_speed_eos(reference, values, **eos_kwargs)
    return solve_observables(eos, h_c, n_steps=N_STEPS).mass


@jax.jit
def soft_state(values, h_c):
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

    dh = 1.0e-3
    dM_dh = (
        mass(values, h_c + dh) - mass(values, h_c - dh)
    ) / (2.0 * dh)

    return jnp.concatenate(
        (
            y,
            soft,
            jnp.array([dM_dh, jnp.sqrt(eigenvalues[1] / eigenvalues[2])]),
        )
    )


def draw_backgrounds():
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

    backgrounds = []
    for family, L in (("smooth", Ls), ("rough", Lr)):
        for index in range(1, 9):
            backgrounds.append(
                (family, index, L @ rng.normal(size=N_NODES))
            )
    return backgrounds


def polynomial_powers(degree):
    powers = []
    for total in range(1, degree + 1):
        for p in range(total + 1):
            powers.append((p, total - p))
    return powers


def gradient_design(u, v, degree, sx, sz):
    row_c = []
    row_l = []
    for p, q in polynomial_powers(degree):
        row_c.append(
            0.0 if p == 0 else p * u ** (p - 1) * v**q / sx
        )
        row_l.append(
            0.0 if q == 0 else q * u**p * v ** (q - 1) / sz
        )
    return np.asarray(row_c), np.asarray(row_l)


def evaluate_polynomial(u, v, coefficients, degree):
    return sum(
        coefficient * u**p * v**q
        for coefficient, (p, q) in zip(
            coefficients, polynomial_powers(degree)
        )
    )


def fit_potential(rows, degree, one_dimensional=False):
    logC = np.asarray([r["logC"] for r in rows])
    logL = np.asarray([r["logLambda"] for r in rows])
    x0, z0 = logC.mean(), logL.mean()
    sx, sz = logC.std(), logL.std()

    A = []
    b = []

    if one_dimensional:
        for r in rows:
            v = (r["logLambda"] - z0) / sz
            A.append(
                [
                    q * v ** (q - 1) / sz
                    for q in range(1, degree + 1)
                ]
            )
            b.append(r["mL"])
    else:
        for r in rows:
            u = (r["logC"] - x0) / sx
            v = (r["logLambda"] - z0) / sz
            row_c, row_l = gradient_design(
                u, v, degree, sx, sz
            )
            A.extend((row_c, row_l))
            b.extend((r["mC"], r["mL"]))

    coefficients = np.linalg.lstsq(
        np.asarray(A), np.asarray(b), rcond=None
    )[0]

    return dict(
        degree=degree,
        one_dimensional=one_dimensional,
        x0=x0,
        z0=z0,
        sx=sx,
        sz=sz,
        coefficients=coefficients,
    )


def predict(model, row):
    u = (row["logC"] - model["x0"]) / model["sx"]
    v = (row["logLambda"] - model["z0"]) / model["sz"]
    degree = model["degree"]
    coefficients = model["coefficients"]

    if model["one_dimensional"]:
        g_c = 0.0
        g_l = sum(
            c * q * v ** (q - 1) / model["sz"]
            for c, q in zip(coefficients, range(1, degree + 1))
        )
        g = sum(
            c * v**q
            for c, q in zip(coefficients, range(1, degree + 1))
        )
    else:
        row_c, row_l = gradient_design(
            u, v, degree, model["sx"], model["sz"]
        )
        g_c = float(row_c @ coefficients)
        g_l = float(row_l @ coefficients)
        g = evaluate_polynomial(u, v, coefficients, degree)

    reconstructed_gradient = np.asarray([g_c, 1.0, g_l])
    target_gradient = np.asarray(row["mvec"])
    alignment = abs(reconstructed_gradient @ target_gradient) / (
        np.linalg.norm(reconstructed_gradient)
        * np.linalg.norm(target_gradient)
    )

    invariant = row["logI"] + g
    return alignment, invariant


def assess(model, rows):
    predictions = [predict(model, row) for row in rows]
    alignment = np.asarray([p[0] for p in predictions])
    invariant = np.asarray([p[1] for p in predictions])
    centered = invariant - np.median(invariant)

    return dict(
        n=len(rows),
        alignment_median=float(np.median(alignment)),
        alignment_p10=float(np.quantile(alignment, 0.10)),
        alignment_min=float(np.min(alignment)),
        invariant_std=float(np.std(invariant)),
        invariant_p90_abs=float(np.quantile(np.abs(centered), 0.90)),
        invariant_range=float(np.ptp(invariant)),
    )


def build_dataset():
    soft_state(jnp.zeros(N_NODES), jnp.asarray(0.16)).block_until_ready()

    rows = []
    h_values = (0.10, 0.14, 0.18, 0.22, 0.26, 0.30)

    for background_id, (family, index, values) in enumerate(
        draw_backgrounds(), 1
    ):
        values = jnp.asarray(values)
        for h_c in h_values:
            q = np.asarray(soft_state(values, jnp.asarray(h_c)))
            y = q[:3]
            soft = q[3:6]
            dM_dh = q[6]

            if dM_dh <= 0.0 or np.exp(y[0]) < 0.05:
                continue

            normalized = soft / soft[1]
            rows.append(
                dict(
                    background=background_id,
                    family=family,
                    index=index,
                    h_c=h_c,
                    logC=float(y[0]),
                    logI=float(y[1]),
                    logLambda=float(y[2]),
                    compactness=float(np.exp(y[0])),
                    mC=float(normalized[0]),
                    mL=float(normalized[2]),
                    mvec=np.asarray(normalized).tolist(),
                    normal_ratio=float(q[7]),
                )
            )
    return rows


def four_fold_cv(rows, degree, one_dimensional):
    folds = (
        (1, 5, 9, 13),
        (2, 6, 10, 14),
        (3, 7, 11, 15),
        (4, 8, 12, 16),
    )
    outputs = []

    for fold in folds:
        train = [r for r in rows if r["background"] not in fold]
        test = [r for r in rows if r["background"] in fold]
        model = fit_potential(train, degree, one_dimensional)
        result = assess(model, test)
        result["held_out_backgrounds"] = list(fold)
        outputs.append(result)

    return outputs


def family_holdout(rows, degree, one_dimensional):
    result = {}
    for train_family, test_family in (
        ("smooth", "rough"),
        ("rough", "smooth"),
    ):
        train = [r for r in rows if r["family"] == train_family]
        test = [r for r in rows if r["family"] == test_family]
        model = fit_potential(train, degree, one_dimensional)
        result[f"{train_family}_to_{test_family}"] = assess(model, test)
    return result


def summarize_folds(folds):
    keys = (
        "alignment_median",
        "alignment_p10",
        "alignment_min",
        "invariant_std",
        "invariant_p90_abs",
        "invariant_range",
    )
    return {
        key: dict(
            median=float(np.median([f[key] for f in folds])),
            min=float(np.min([f[key] for f in folds])),
            max=float(np.max([f[key] for f in folds])),
        )
        for key in keys
    }


def main():
    rows = build_dataset()
    output = dict(
        seed=SEED,
        n_nodes=N_NODES,
        n_steps=N_STEPS,
        n_high=N_HIGH,
        n_points=len(rows),
        degrees={},
    )

    for degree in (2, 3, 4):
        output["degrees"][str(degree)] = {}
        for one_dimensional, name in (
            (True, "ilove_1d"),
            (False, "soft_surface_2d"),
        ):
            folds = four_fold_cv(rows, degree, one_dimensional)
            output["degrees"][str(degree)][name] = dict(
                cv_summary=summarize_folds(folds),
                cv_folds=folds,
                family_holdout=family_holdout(
                    rows, degree, one_dimensional
                ),
            )

    print("POTENTIAL_JSON=" + json.dumps(output, sort_keys=True))


if __name__ == "__main__":
    main()
