"""Broad nonparametric background audit of the local response hierarchy.

Background EOSs are finite draws of a causal latent sound-speed field anchored
to the same low-density reference. The response Jacobian is then recomputed
around each nonzero background field. This is a functional-space stress test,
not yet a nuclear-physics posterior over realistic EOSs.
"""

import json

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

from ns_response_geometry.eos import RelativisticPolytrope
from ns_response_geometry.geometry import induced_response, plane_normal_variance
from ns_response_geometry.observables import solve_observables
from ns_response_geometry.response import response_jacobian, sequence_tangent
from ns_response_geometry.response_eos import (
    build_nodal_sound_speed_eos,
    matern32_covariance,
    squared_exponential_covariance,
)


def draw_gp(rng, covariance):
    C = np.asarray(covariance, dtype=float)
    L = np.linalg.cholesky(C + 1.0e-10 * np.eye(C.shape[0]))
    return L @ rng.normal(size=C.shape[0])


def stable_mass_slope(eos, h_c, n_steps=768):
    dh = 2.0e-3
    hm = max(h_c - dh, 1.0e-3)
    hp = h_c + dh
    mm = float(solve_observables(eos, hm, n_steps=n_steps).mass)
    mp = float(solve_observables(eos, hp, n_steps=n_steps).mass)
    return (mp - mm) / (hp - hm)


def response_ratio(reference, values, nodes, h_c, *, n_steps=768, n_high=512):
    kwargs = dict(
        h_match=0.02,
        h_max=0.50,
        h_nodes=nodes,
        n_low=128,
        n_high=n_high,
    )
    eos = build_nodal_sound_speed_eos(reference, values, **kwargs)
    obs = solve_observables(eos, h_c, n_steps=n_steps)
    slope = stable_mass_slope(eos, h_c, n_steps=n_steps)

    J = response_jacobian(
        reference, values, h_c, n_steps=n_steps, eos_kwargs=kwargs
    )
    tangent = sequence_tangent(
        reference, values, h_c, n_steps=n_steps, eos_kwargs=kwargs
    )
    local_cov = squared_exponential_covariance(
        nodes, amplitude=0.10, length_scale=0.08
    )
    G = induced_response(J, local_cov)
    ilove_var, _ = plane_normal_variance(G, tangent, [1, 2], jnp.eye(3))
    clove_var, _ = plane_normal_variance(G, tangent, [0, 2], jnp.eye(3))

    return dict(
        h_c=float(h_c),
        stable=bool(slope > 0.0),
        dM_dh=float(slope),
        compactness=float(obs.compactness),
        ibar=float(obs.ibar),
        lambda2=float(obs.lambda2),
        max_cs2=float(jnp.max(eos.cs2_grid)),
        min_cs2_above_match=float(jnp.min(eos.cs2_grid[128:])),
        ilove_rms=float(jnp.sqrt(ilove_var)),
        clove_rms=float(jnp.sqrt(clove_var)),
        rms_ratio=float(jnp.sqrt(ilove_var / clove_var)),
        variance_ratio=float(ilove_var / clove_var),
    )


def summarize(rows):
    stable = [r for r in rows if r["stable"]]
    ratios = np.array([r["rms_ratio"] for r in stable], dtype=float)
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
    rng = np.random.default_rng(260918)
    reference = RelativisticPolytrope(K=100.0, gamma=2.0)
    n_nodes = 17
    nodes = jnp.linspace(0.02, 0.50, n_nodes)

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
    for family_name, covariance in families.items():
        for family_index in range(8):
            values = draw_gp(rng, covariance)
            draw_id += 1
            for h_c in (0.12, 0.20, 0.28):
                row = response_ratio(reference, jnp.asarray(values), nodes, h_c)
                row.update(
                    family=family_name,
                    family_index=family_index,
                    draw_id=draw_id,
                    latent_rms=float(np.sqrt(np.mean(values**2))),
                    latent_max_abs=float(np.max(np.abs(values))),
                )
                rows.append(row)

    family_summary = {}
    for family_name in families:
        family_summary[family_name] = summarize(
            [r for r in rows if r["family"] == family_name]
        )

    payload = dict(
        seed=260918,
        n_background_draws=16,
        n_nodes=n_nodes,
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
        family_summary=family_summary,
        rows=rows,
    )
    print("BACKGROUND_ENSEMBLE_JSON=" + json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
