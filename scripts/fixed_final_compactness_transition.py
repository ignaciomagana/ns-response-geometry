"""Transition-location audit at fixed post-softening compactness.

This removes the principal confound found in the fixed-depth transition scan.
For each reference EOS (Gamma = 1.70, 1.85, 2.00), we first match an
unperturbed star at baseline compactness C=0.16.  At each transition location
x=h_tr/h_c, we then solve for the latent softening depth D such that the
deformed configuration has the same target compactness C=0.12.

Only after matching the final compactness do we compare:
- I--Love/C--Love transverse RMS response;
- full soft/hard normal RMS ratio;
- soft/I--Love alignment;
- physical shell location r/R and m/M.

Thus any residual location dependence is not caused merely by different
global expansion of the star.

The window width is fixed at 0.25 h_c and the edge scale at 0.008.
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
WIDTH_FRACTION = 0.25
BASELINE_COMPACTNESS = 0.16
TARGET_COMPACTNESS = 0.12
GAMMAS = (1.70, 1.85, 2.00)
X_VALUES = tuple(np.arange(0.25, 0.651, 0.05))
MAX_DEPTH = 3.0

nodes = jnp.linspace(0.02, 0.50, N_NODES)
response_covariance = squared_exponential_covariance(
    nodes, amplitude=0.10, length_scale=0.08
)


def transition_profile(center, width, depth):
    left = center - width / 2.0
    right = center + width / 2.0
    return -0.5 * depth * (
        jnp.tanh((nodes - left) / EDGE)
        - jnp.tanh((nodes - right) / EDGE)
    )


def find_hc(reference, target):
    compactness_jit = jax.jit(
        lambda h: solve_observables(reference, h, n_steps=1024).compactness
    )
    compactness_jit(jnp.asarray(0.20)).block_until_ready()

    lo, hi = 0.03, 0.55
    clo = float(compactness_jit(jnp.asarray(lo)))
    chi = float(compactness_jit(jnp.asarray(hi)))
    if not (clo < target < chi):
        raise RuntimeError(
            f"C={target} not bracketed for gamma={reference.gamma}: "
            f"{clo}, {chi}"
        )

    for _ in range(40):
        mid = 0.5 * (lo + hi)
        cmid = float(compactness_jit(jnp.asarray(mid)))
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


def make_functions(reference, h_c):
    eos_kwargs = dict(
        h_match=0.02,
        h_max=0.50,
        h_nodes=nodes,
        n_low=128,
        n_high=N_HIGH,
    )

    def log_observables(values):
        eos = build_nodal_sound_speed_eos(reference, values, **eos_kwargs)
        r = solve_observables(eos, h_c, n_steps=N_STEPS)
        return jnp.log(jnp.stack((r.compactness, r.ibar, r.lambda2)))

    def log_observables_h(values, h):
        eos = build_nodal_sound_speed_eos(reference, values, **eos_kwargs)
        r = solve_observables(eos, h, n_steps=N_STEPS)
        return jnp.log(jnp.stack((r.compactness, r.ibar, r.lambda2)))

    def mass_h(values, h):
        eos = build_nodal_sound_speed_eos(reference, values, **eos_kwargs)
        return solve_observables(eos, h, n_steps=N_STEPS).mass

    @jax.jit
    def compactness_for(values):
        eos = build_nodal_sound_speed_eos(reference, values, **eos_kwargs)
        return solve_observables(eos, h_c, n_steps=N_STEPS).compactness

    @jax.jit
    def response_state(values):
        y = log_observables(values)
        J = jax.jacfwd(log_observables)(values)
        tangent = jax.jacfwd(lambda h: log_observables_h(values, h))(
            jnp.asarray(h_c)
        )
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
            mass_h(values, h_c + dh) - mass_h(values, h_c - dh)
        ) / (2.0 * dh)

        return (
            y,
            soft,
            ilove,
            eigenvalues,
            ilove_var,
            clove_var,
            dM_dh,
        )

    compactness_for(jnp.zeros(N_NODES)).block_until_ready()
    response_state(jnp.zeros(N_NODES))[0].block_until_ready()

    return eos_kwargs, compactness_for, response_state


def solve_depth(compactness_for, center, width):
    """Find depth giving TARGET_COMPACTNESS. Return None if unreachable/stable
    target cannot be bracketed by compactness alone.
    """
    def C(depth):
        values = transition_profile(center, width, depth)
        return float(compactness_for(values))

    c0 = C(0.0)
    cmax = C(MAX_DEPTH)

    if not (np.isfinite(c0) and np.isfinite(cmax)):
        return None
    if c0 < TARGET_COMPACTNESS:
        return None
    if cmax > TARGET_COMPACTNESS:
        return None

    lo, hi = 0.0, MAX_DEPTH
    for _ in range(32):
        mid = 0.5 * (lo + hi)
        cmid = C(mid)
        if not np.isfinite(cmid):
            hi = mid
            continue
        if cmid > TARGET_COMPACTNESS:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def evaluate(reference, h_c, x, eos_kwargs, compactness_for, response_state):
    center = x * h_c
    width = WIDTH_FRACTION * h_c

    depth = solve_depth(compactness_for, center, width)
    if depth is None:
        return dict(
            gamma=float(reference.gamma),
            h_c=float(h_c),
            x=float(x),
            center=float(center),
            width=float(width),
            reachable=False,
        )

    values = transition_profile(center, width, depth)
    y, soft, ilove, eigenvalues, ilove_var, clove_var, dM_dh = (
        response_state(values)
    )

    compactness = float(jnp.exp(y[0]))
    finite = (
        np.all(np.isfinite(np.asarray(y)))
        and np.all(np.isfinite(np.asarray(eigenvalues)))
        and np.isfinite(float(dM_dh))
    )
    stable = finite and float(dM_dh) > 0.0

    eos = build_nodal_sound_speed_eos(reference, values, **eos_kwargs)

    return dict(
        gamma=float(reference.gamma),
        h_c=float(h_c),
        x=float(x),
        center=float(center),
        width=float(width),
        depth=float(depth),
        reachable=True,
        compactness=compactness,
        stable=bool(stable),
        dM_dh=float(dM_dh),
        ilove_alignment=float(jnp.abs(jnp.dot(soft, ilove)))
        if finite else None,
        ilove_clove_rms_ratio=float(jnp.sqrt(ilove_var / clove_var))
        if finite else None,
        normal_rms_ratio=float(jnp.sqrt(eigenvalues[1] / eigenvalues[2]))
        if finite else None,
        transition_shell=physical_shell(eos, h_c, center, width)
        if stable else None,
    )


def main():
    rows = []
    baselines = []

    for gamma in GAMMAS:
        reference = RelativisticPolytrope(K=100.0, gamma=gamma)
        h_c = find_hc(reference, BASELINE_COMPACTNESS)
        baseline = solve_observables(reference, h_c, n_steps=N_STEPS)
        baselines.append(
            dict(
                gamma=float(gamma),
                h_c=float(h_c),
                compactness=float(baseline.compactness),
            )
        )

        eos_kwargs, compactness_for, response_state = make_functions(
            reference, h_c
        )

        for x in X_VALUES:
            row = evaluate(
                reference,
                h_c,
                x,
                eos_kwargs,
                compactness_for,
                response_state,
            )
            rows.append(row)

    stable = [r for r in rows if r.get("stable", False)]

    best_by_gamma = []
    for gamma in GAMMAS:
        subset = [
            r for r in stable if abs(r["gamma"] - gamma) < 1e-12
        ]
        if subset:
            best_by_gamma.append(
                max(subset, key=lambda r: r["ilove_clove_rms_ratio"])
            )

    x_best = np.asarray([r["x"] for r in best_by_gamma])
    r_best = np.asarray([
        r["transition_shell"]["center"]["r_over_R"]
        for r in best_by_gamma
    ])
    m_best = np.asarray([
        r["transition_shell"]["center"]["m_over_M"]
        for r in best_by_gamma
    ])

    payload = dict(
        n_nodes=N_NODES,
        n_steps=N_STEPS,
        n_high=N_HIGH,
        baseline_compactness=BASELINE_COMPACTNESS,
        target_compactness=TARGET_COMPACTNESS,
        width_fraction=WIDTH_FRACTION,
        edge=EDGE,
        max_depth=MAX_DEPTH,
        x_grid=list(X_VALUES),
        baselines=baselines,
        n_reachable=sum(r.get("reachable", False) for r in rows),
        n_stable=len(stable),
        best_by_gamma=best_by_gamma,
        summary=dict(
            x_min=float(np.min(x_best)),
            x_median=float(np.median(x_best)),
            x_max=float(np.max(x_best)),
            r_over_R_min=float(np.min(r_best)),
            r_over_R_median=float(np.median(r_best)),
            r_over_R_max=float(np.max(r_best)),
            m_over_M_min=float(np.min(m_best)),
            m_over_M_median=float(np.median(m_best)),
            m_over_M_max=float(np.max(m_best)),
        ) if best_by_gamma else None,
        rows=rows,
    )

    print("FIXED_FINAL_COMPACTNESS_JSON=" + json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
