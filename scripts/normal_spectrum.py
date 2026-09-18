"""Full normal-spectrum audit of the EOS response operator.

The observable space is (ln C, ln Ibar, ln Lambda2). After projecting out the
stellar-sequence tangent, the remaining two normal response eigenvalues are
compared. The low-response covector is also compared with the I--Love plane
normal.

The script evaluates both the Gamma=2 reference sequence and the same
deterministic broad latent-field backgrounds used by background_ensemble.py.
"""

import json
import numpy as np

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from ns_response_geometry.eos import RelativisticPolytrope
from ns_response_geometry.geometry import (
    induced_response,
    normal_response_spectrum,
    plane_normal_variance,
)
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


def draw_gp(rng, covariance):
    C = np.asarray(covariance, dtype=float)
    L = np.linalg.cholesky(C + 1.0e-10 * np.eye(C.shape[0]))
    return L @ rng.normal(size=C.shape[0])


def summary(rows, compactness_min=None):
    stable = [r for r in rows if r["stable"]]
    if compactness_min is not None:
        stable = [r for r in stable if r["compactness"] >= compactness_min]
    ratio = np.asarray([r["normal_rms_ratio"] for r in stable])
    align = np.asarray([r["ilove_alignment"] for r in stable])
    ccomp = np.asarray([r["abs_compactness_component"] for r in stable])
    return dict(
        n=len(stable),
        normal_rms_ratio_min=float(np.min(ratio)),
        normal_rms_ratio_median=float(np.median(ratio)),
        normal_rms_ratio_p90=float(np.quantile(ratio, 0.90)),
        normal_rms_ratio_max=float(np.max(ratio)),
        ilove_alignment_min=float(np.min(align)),
        ilove_alignment_median=float(np.median(align)),
        abs_compactness_component_median=float(np.median(ccomp)),
        abs_compactness_component_max=float(np.max(ccomp)),
    )


def main():
    reference = RelativisticPolytrope(K=100.0, gamma=2.0)
    nodes = jnp.linspace(0.02, 0.50, N_NODES)
    eos_kwargs = dict(
        h_match=0.02,
        h_max=0.50,
        h_nodes=nodes,
        n_low=128,
        n_high=N_HIGH,
    )
    covariance = squared_exponential_covariance(
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
    def calculate(values, h_c):
        J = jax.jacfwd(lambda a: log_observables(a, h_c))(values)
        tangent = jax.jacfwd(lambda h: log_observables(values, h))(h_c)
        G = induced_response(J, covariance)

        evals, covectors = normal_response_spectrum(
            G, tangent, jnp.eye(3)
        )
        low = covectors[:, 1]

        _, ilove_normal_2d = plane_normal_variance(
            G, tangent, [1, 2], jnp.eye(3)
        )
        ilove_normal = jnp.array(
            [0.0, ilove_normal_2d[0], ilove_normal_2d[1]]
        )
        alignment = jnp.abs(jnp.dot(low, ilove_normal)) / (
            jnp.linalg.norm(low) * jnp.linalg.norm(ilove_normal)
        )

        y = log_observables(values, h_c)
        dh = 1.0e-3
        dM_dh = (
            mass(values, h_c + dh) - mass(values, h_c - dh)
        ) / (2.0 * dh)

        return jnp.concatenate(
            (
                jnp.exp(y),
                jnp.array(
                    [
                        dM_dh,
                        evals[1],
                        evals[2],
                        jnp.sqrt(evals[1] / evals[2]),
                        alignment,
                        jnp.abs(low[0]),
                    ]
                ),
                low,
            )
        )

    zero = jnp.zeros(N_NODES)
    calculate(zero, jnp.asarray(0.12)).block_until_ready()

    reference_rows = []
    for h_c in (0.08, 0.12, 0.16, 0.20, 0.25, 0.30):
        q = np.asarray(calculate(zero, jnp.asarray(h_c)))
        reference_rows.append(
            dict(
                h_c=float(h_c),
                compactness=float(q[0]),
                normal_rms_ratio=float(q[6]),
                ilove_alignment=float(q[7]),
                abs_compactness_component=float(q[8]),
                low_covector=[float(x) for x in q[9:12]],
            )
        )

    rng = np.random.default_rng(SEED)
    families = {
        "smooth_se": squared_exponential_covariance(
            nodes, amplitude=1.0, length_scale=0.08
        ),
        "rough_matern32": matern32_covariance(
            nodes, amplitude=1.0, length_scale=0.04
        ),
    }

    rows = []
    draw_id = 0
    for family_name, family_covariance in families.items():
        for family_index in range(8):
            values = jnp.asarray(draw_gp(rng, family_covariance))
            draw_id += 1
            for h_c in (0.12, 0.20, 0.28):
                q = np.asarray(calculate(values, jnp.asarray(h_c)))
                rows.append(
                    dict(
                        family=family_name,
                        family_index=family_index,
                        draw_id=draw_id,
                        h_c=float(h_c),
                        compactness=float(q[0]),
                        ibar=float(q[1]),
                        lambda2=float(q[2]),
                        stable=bool(q[3] > 0.0),
                        dM_dh=float(q[3]),
                        low_variance=float(q[4]),
                        high_variance=float(q[5]),
                        normal_rms_ratio=float(q[6]),
                        ilove_alignment=float(q[7]),
                        abs_compactness_component=float(q[8]),
                        low_covector=[float(x) for x in q[9:12]],
                    )
                )

    payload = dict(
        seed=SEED,
        n_nodes=N_NODES,
        n_steps=N_STEPS,
        n_high=N_HIGH,
        reference_sequence=reference_rows,
        ensemble_summary=summary(rows),
        compactness_cuts={
            "0.05": summary(rows, 0.05),
            "0.08": summary(rows, 0.08),
            "0.10": summary(rows, 0.10),
            "0.12": summary(rows, 0.12),
        },
        rows=rows,
    )
    print("NORMAL_SPECTRUM_JSON=" + json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
