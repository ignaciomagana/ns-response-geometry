"""Matched-compactness reference-EOS transition-location audit.

The purpose is to test whether the vulnerable localized-softening layer found
for the Gamma=2 reference is tied to h_tr/h_c, to a physical stellar shell,
or is strongly reference-EOS dependent.

For Gamma = 1.70, 1.85, 2.00:
1. solve for h_c at fixed baseline compactness C = 0.12 and 0.16;
2. impose a smooth causal sound-speed softening of depth 1.5 and width
   0.25 h_c;
3. scan x = h_tr/h_c from 0.25 to 0.70;
4. record the location of strongest I--Love degradation and the corresponding
   r/R and m/M in the deformed and baseline stars.

All response calculations use 65 EOS nodes and the same response metric as
the production H5 analysis.
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
from ns_response_geometry.tov import solve_star_profile


N_NODES = 65
N_STEPS = 2048
N_HIGH = 2048
EDGE = 0.008
DEPTH = 1.5
WIDTH_FRACTION = 0.25

GAMMAS = (1.70, 1.85, 2.00)
TARGET_COMPACTNESS = (0.12, 0.16)
X_VALUES = tuple(np.arange(0.25, 0.701, 0.05))

nodes = jnp.linspace(0.02, 0.50, N_NODES)
response_covariance = squared_exponential_covariance(
    nodes, amplitude=0.10, length_scale=0.08
)


def transition_profile(center, width, depth=DEPTH):
    left = center - width / 2.0
    right = center + width / 2.0
    return -0.5 * depth * (
        jnp.tanh((nodes - left) / EDGE)
        - jnp.tanh((nodes - right) / EDGE)
    )


def find_hc_for_compactness(reference, target):
    """Bisection on the stable branch using a jitted compactness map."""
    lo, hi = 0.03, 0.55

    compactness_jit = jax.jit(
        lambda h: solve_observables(
            reference, h, n_steps=1024
        ).compactness
    )
    compactness_jit(jnp.asarray(0.20)).block_until_ready()

    def compactness(h):
        return float(compactness_jit(jnp.asarray(h)))

    clo = compactness(lo)
    chi = compactness(hi)
    if not (clo < target < chi):
        raise RuntimeError(
            f"Target C={target} not bracketed for gamma={reference.gamma}: "
            f"C(lo)={clo}, C(hi)={chi}"
        )

    for _ in range(40):
        mid = 0.5 * (lo + hi)
        cmid = compactness(mid)
        if cmid < target:
            lo = mid
        else:
            hi = mid

    return 0.5 * (lo + hi)


def physical_shell(eos, h_c, center, width):
    profile = solve_star_profile(eos, h_c, n_steps=N_STEPS)

    h_desc = np.asarray(profile.enthalpy)
    r_desc = np.asarray(profile.radius)
    m_desc = np.asarray(profile.mass)
    R = float(r_desc[-1])
    M = float(m_desc[-1])

    h = h_desc[::-1]
    radius = r_desc[::-1]
    mass = m_desc[::-1]

    def at(target):
        rr = float(np.interp(target, h, radius))
        mm = float(np.interp(target, h, mass))
        return dict(r_over_R=rr / R, m_over_M=mm / M)

    left = max(center - width / 2.0, 0.0)
    right = min(center + width / 2.0, h_c)

    return dict(
        center=at(min(center, h_c)),
        inner_edge=at(min(right, h_c)),
        outer_edge=at(min(left, h_c)),
    )


def make_response_functions(reference):
    eos_kwargs = dict(
        h_match=0.02,
        h_max=0.50,
        h_nodes=nodes,
        n_low=128,
        n_high=N_HIGH,
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
            soft,
            ilove,
            eigenvalues,
            ilove_var,
            clove_var,
            dM_dh,
            jnp.min(eos.cs2_grid[128:]),
        )

    return eos_kwargs, response_state


def evaluate(reference, eos_kwargs, response_state, h_c, x):
    center = x * h_c
    width = WIDTH_FRACTION * h_c
    values = transition_profile(center, width)
    zero = jnp.zeros(N_NODES)

    (
        y,
        soft,
        ilove,
        eigenvalues,
        ilove_var,
        clove_var,
        dM_dh,
        min_cs2,
    ) = response_state(values, jnp.asarray(h_c))

    (
        y0,
        soft0,
        ilove0,
        evals0,
        ilove_var0,
        clove_var0,
        dM_dh0,
        min_cs20,
    ) = response_state(zero, jnp.asarray(h_c))

    compactness = float(jnp.exp(y[0]))
    baseline_compactness = float(jnp.exp(y0[0]))
    finite = (
        np.all(np.isfinite(np.asarray(y)))
        and np.all(np.isfinite(np.asarray(eigenvalues)))
        and np.isfinite(float(dM_dh))
    )
    stable = finite and float(dM_dh) > 0.0
    physical = stable and 0.0 < compactness < 0.35

    return dict(
        gamma=float(reference.gamma),
        h_c=float(h_c),
        target_baseline_compactness=float(baseline_compactness),
        x=float(x),
        center=float(center),
        width=float(width),
        compactness=compactness,
        stable=bool(stable),
        physical_screen=bool(physical),
        dM_dh=float(dM_dh),
        min_cs2=float(min_cs2),
        ilove_alignment=float(jnp.abs(jnp.dot(soft, ilove)))
        if finite else None,
        ilove_clove_rms_ratio=float(jnp.sqrt(ilove_var / clove_var))
        if finite else None,
        normal_rms_ratio=float(jnp.sqrt(eigenvalues[1] / eigenvalues[2]))
        if finite else None,
        baseline_ilove_alignment=float(jnp.abs(jnp.dot(soft0, ilove0))),
        baseline_ilove_clove_rms_ratio=float(
            jnp.sqrt(ilove_var0 / clove_var0)
        ),
        baseline_normal_rms_ratio=float(jnp.sqrt(evals0[1] / evals0[2])),
    )


def strongest(rows):
    good = [r for r in rows if r["physical_screen"]]
    if not good:
        return None

    # Primary definition: largest I--Love/C--Love response ratio.
    best = max(good, key=lambda r: r["ilove_clove_rms_ratio"])
    return best


def main():
    rows = []
    matched = []

    for gamma in GAMMAS:
        reference = RelativisticPolytrope(K=100.0, gamma=gamma)
        eos_kwargs, response_state = make_response_functions(reference)

        # Compile once per reference EOS.
        response_state(
            jnp.zeros(N_NODES), jnp.asarray(0.20)
        )[0].block_until_ready()

        for target in TARGET_COMPACTNESS:
            h_c = find_hc_for_compactness(reference, target)

            baseline = solve_observables(
                reference, h_c, n_steps=N_STEPS
            )
            matched.append(
                dict(
                    gamma=float(gamma),
                    target_compactness=float(target),
                    h_c=float(h_c),
                    achieved_compactness=float(baseline.compactness),
                )
            )

            subset = []
            for x in X_VALUES:
                row = evaluate(
                    reference, eos_kwargs, response_state, h_c, x
                )
                row["requested_target_compactness"] = float(target)
                rows.append(row)
                subset.append(row)

    best_by_case = []
    for gamma in GAMMAS:
        for target in TARGET_COMPACTNESS:
            subset = [
                r for r in rows
                if abs(r["gamma"] - gamma) < 1e-12
                and abs(r["requested_target_compactness"] - target) < 1e-12
            ]
            best = strongest(subset)
            if best is not None:
                # Compute physical-shell mappings only for the six winners.
                reference = RelativisticPolytrope(K=100.0, gamma=gamma)
                eos_kwargs = dict(
                    h_match=0.02,
                    h_max=0.50,
                    h_nodes=nodes,
                    n_low=128,
                    n_high=N_HIGH,
                )
                values = transition_profile(best["center"], best["width"])
                eos = build_nodal_sound_speed_eos(
                    reference, values, **eos_kwargs
                )
                eos0 = build_nodal_sound_speed_eos(
                    reference, jnp.zeros(N_NODES), **eos_kwargs
                )
                best = dict(best)
                best["transition_shell"] = physical_shell(
                    eos, best["h_c"], best["center"], best["width"]
                )
                best["baseline_shell"] = physical_shell(
                    eos0, best["h_c"], best["center"], best["width"]
                )
                best_by_case.append(best)

    x_values = np.asarray([r["x"] for r in best_by_case])
    r_values = np.asarray([
        r["transition_shell"]["center"]["r_over_R"]
        for r in best_by_case
    ])
    m_values = np.asarray([
        r["transition_shell"]["center"]["m_over_M"]
        for r in best_by_case
    ])

    payload = dict(
        n_nodes=N_NODES,
        n_steps=N_STEPS,
        n_high=N_HIGH,
        depth=DEPTH,
        edge=EDGE,
        width_fraction=WIDTH_FRACTION,
        gammas=list(GAMMAS),
        target_compactness=list(TARGET_COMPACTNESS),
        x_grid=list(X_VALUES),
        matched_baselines=matched,
        best_by_case=best_by_case,
        summary=dict(
            n_cases=len(best_by_case),
            x_min=float(np.min(x_values)),
            x_median=float(np.median(x_values)),
            x_max=float(np.max(x_values)),
            r_over_R_min=float(np.min(r_values)),
            r_over_R_median=float(np.median(r_values)),
            r_over_R_max=float(np.max(r_values)),
            m_over_M_min=float(np.min(m_values)),
            m_over_M_median=float(np.median(m_values)),
            m_over_M_max=float(np.max(m_values)),
        ),
        rows=rows,
    )

    print("MATCHED_COMPACTNESS_JSON=" + json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
