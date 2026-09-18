"""Fine-resolution broad nonparametric background audit.

Sixteen deterministic causal latent sound-speed backgrounds are drawn from
smooth and rough GP families. The local EOS-response Jacobian is recomputed
around every background at three stellar configurations. The expensive map is
JIT compiled once and reused for all draws.

This is a functional-space stress test, not a nuclear-physics posterior over
realistic EOSs.
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


def summarize(rows):
    stable = [r for r in rows if r["stable"]]
    ratios = np.asarray([r["rms_ratio"] for r in stable], dtype=float)
    return dict(
        total_points=len(rows),
        stable_points=len(stable),
        stable_fraction=len(stable) / len(rows),
        rms_ratio_min=float(np.min(ratios)),
        rms_ratio_median=float(np.median(ratios)),
        rms_ratio_p90=float(np.quantile(ratios, 0.90)),
        rms_ratio_max=float(np.max(ratios)),
        fraction_ratio_below_0p25=float(np.mean(ratios < 0.25)),
        fraction_ratio_below_0p5=float(np.mean(ratios < 0.50)),
    )


def main():
    rng = np.random.default_rng(SEED)
    reference = RelativisticPolytrope(K=100.0, gamma=2.0)
    nodes = jnp.linspace(0.02, 0.50, N_NODES)
    eos_kwargs = dict(
        h_match=0.02,
        h_max=0.50,
        h_nodes=nodes,
        n_low=128,
        n_high=N_HIGH,
    )
    local_covariance = squared_exponential_covariance(
        nodes, amplitude=0.10, length_scale=0.08
    )

    def log_observables(values, h_c):
        eos = build_nodal_sound_speed_eos(reference, values, **eos_kwargs)
        result = solve_observables(eos, h_c, n_steps=N_STEPS)
        return jnp.log(
            jnp.stack((result.compactness, result.ibar, result.lambda2))
        )

    def mass(values, h_c):
        eos = build_nodal_sound_speed_eos(reference, values, **eos_kwargs)
        return solve_observables(eos, h_c, n_steps=N_STEPS).mass

    @jax.jit
    def calculate(values, h_c):
        J = jax.jacfwd(lambda a: log_observables(a, h_c))(values)
        tangent = jax.jacfwd(lambda h: log_observables(values, h))(h_c)
        G = induced_response(J, local_covariance)

        ilove_var, _ = plane_normal_variance(
            G, tangent, [1, 2], jnp.eye(3)
        )
        clove_var, _ = plane_normal_variance(
            G, tangent, [0, 2], jnp.eye(3)
        )

        y = log_observables(values, h_c)
        dh = 1.0e-3
        dM_dh = (
            mass(values, h_c + dh) - mass(values, h_c - dh)
        ) / (2.0 * dh)

        eos = build_nodal_sound_speed_eos(reference, values, **eos_kwargs)
        return jnp.array(
            [
                jnp.exp(y[0]),
                jnp.exp(y[1]),
                jnp.exp(y[2]),
                jnp.sqrt(ilove_var),
                jnp.sqrt(clove_var),
                jnp.sqrt(ilove_var / clove_var),
                dM_dh,
                jnp.max(eos.cs2_grid),
                jnp.min(eos.cs2_grid[128:]),
            ]
        )

    # Compile before entering the deterministic draw loop.
    calculate(jnp.zeros(N_NODES), jnp.asarray(0.12)).block_until_ready()

    background_families = {
        "smooth_se": squared_exponential_covariance(
            nodes, amplitude=1.0, length_scale=0.08
        ),
        "rough_matern32": matern32_covariance(
            nodes, amplitude=1.0, length_scale=0.04
        ),
    }

    rows = []
    draw_id = 0
    for family_name, covariance in background_families.items():
        for family_index in range(8):
            values_np = draw_gp(rng, covariance)
            values = jnp.asarray(values_np)
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
                        ilove_rms=float(q[3]),
                        clove_rms=float(q[4]),
                        rms_ratio=float(q[5]),
                        stable=bool(q[6] > 0.0),
                        dM_dh=float(q[6]),
                        max_cs2=float(q[7]),
                        min_cs2_above_match=float(q[8]),
                        latent_rms=float(np.sqrt(np.mean(values_np**2))),
                        latent_max_abs=float(np.max(np.abs(values_np))),
                    )
                )

    stable = [r for r in rows if r["stable"]]
    compactness_cuts = {}
    for cut in (0.05, 0.08, 0.10, 0.12):
        subset = [r for r in stable if r["compactness"] >= cut]
        ratios = np.asarray([r["rms_ratio"] for r in subset])
        compactness_cuts[str(cut)] = dict(
            n=len(subset),
            rms_ratio_median=float(np.median(ratios)),
            rms_ratio_p90=float(np.quantile(ratios, 0.90)),
            rms_ratio_max=float(np.max(ratios)),
            fraction_ratio_below_0p25=float(np.mean(ratios < 0.25)),
        )

    payload = dict(
        seed=SEED,
        n_background_draws=16,
        n_nodes=N_NODES,
        n_steps=N_STEPS,
        n_high=N_HIGH,
        background_amplitude=1.0,
        background_families={
            "smooth_se": dict(kernel="squared_exponential", length_scale=0.08),
            "rough_matern32": dict(kernel="matern32", length_scale=0.04),
        },
        local_response_metric=dict(
            kernel="squared_exponential",
            amplitude=0.10,
            length_scale=0.08,
            observable_metric="identity in log observables",
        ),
        summary=summarize(rows),
        family_summary={
            name: summarize([r for r in rows if r["family"] == name])
            for name in background_families
        },
        compactness_cuts=compactness_cuts,
        rows=rows,
    )
    print("BACKGROUND_ENSEMBLE_JSON=" + json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
