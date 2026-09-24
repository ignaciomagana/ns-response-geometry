"""Branch-safe fixed-final-compactness transition-location audit.

This supersedes the first fixed-final-compactness refinement.  The earlier
control used a baseline C=0.16 and a globally monotone bisection in softening
depth.  A later audit showed that the Gamma=1.70 C=0.16 baseline had
dM/dh_c < 0 and that C(D) is non-monotone at some transition locations.

Here all reference EOSs are instead matched at baseline C=0.14 and are
required to have positive dM/dh_c.  For each transition location and target
final compactness, the full sampled C(D) curve is searched for roots.  We
select the smallest-depth root, i.e. the root continuously connected to the
undeformed D=0 configuration, and then apply the same positive-mass-slope
screen to the deformed star.

Window width: 0.25 h_c.
Transition edge scale: 0.008.
EOS nodes: 65.
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
BASELINE_COMPACTNESS = 0.14
TARGET_COMPACTNESS = (0.10, 0.12)
GAMMAS = (1.70, 1.85, 2.00)
X_VALUES = tuple(np.arange(0.25, 0.651, 0.025))
MAX_DEPTH = 3.0
DEPTH_GRID = tuple(np.linspace(0.0, MAX_DEPTH, 121))
ROOT_TOL = 1.0e-10

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
    cfun = jax.jit(
        lambda h: solve_observables(
            reference, h, n_steps=1024
        ).compactness
    )
    cfun(jnp.asarray(0.20)).block_until_ready()

    lo, hi = 0.03, 0.55
    for _ in range(42):
        mid = 0.5 * (lo + hi)
        if float(cfun(jnp.asarray(mid))) < target:
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

    def observables(values, h):
        eos = build_nodal_sound_speed_eos(reference, values, **eos_kwargs)
        r = solve_observables(eos, h, n_steps=N_STEPS)
        return jnp.log(jnp.stack((r.compactness, r.ibar, r.lambda2)))

    def mass(values, h):
        eos = build_nodal_sound_speed_eos(reference, values, **eos_kwargs)
        return solve_observables(eos, h, n_steps=N_STEPS).mass

    @jax.jit
    def compactness_for(values):
        eos = build_nodal_sound_speed_eos(reference, values, **eos_kwargs)
        return solve_observables(eos, h_c, n_steps=N_STEPS).compactness

    @jax.jit
    def response_state(values):
        y = observables(values, h_c)
        J = jax.jacfwd(lambda a: observables(a, h_c))(values)
        tangent = jax.jacfwd(lambda h: observables(values, h))(
            jnp.asarray(h_c)
        )
        G = induced_response(J, response_covariance)

        that = tangent / jnp.linalg.norm(tangent)
        P = jnp.eye(3) - jnp.outer(that, that)
        eigenvalues, eigenvectors = jnp.linalg.eigh(P @ G @ P)
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
        dMdh = (
            mass(values, h_c + dh) - mass(values, h_c - dh)
        ) / (2.0 * dh)

        return (
            y,
            soft,
            ilove,
            eigenvalues,
            ilove_var,
            clove_var,
            dMdh,
        )

    compactness_for(jnp.zeros(N_NODES)).block_until_ready()
    response_state(jnp.zeros(N_NODES))[0].block_until_ready()
    return eos_kwargs, compactness_for, response_state


def depth_curve(compactness_for, center, width):
    compactness = []
    for depth in DEPTH_GRID:
        value = float(
            compactness_for(transition_profile(center, width, depth))
        )
        compactness.append(value)
    return np.asarray(DEPTH_GRID), np.asarray(compactness)


def root_brackets(depths, compactness, target):
    f = compactness - target
    brackets = []
    for i in range(len(depths) - 1):
        if not (np.isfinite(f[i]) and np.isfinite(f[i + 1])):
            continue
        if abs(f[i]) <= ROOT_TOL:
            brackets.append((depths[i], depths[i]))
            continue
        if f[i] * f[i + 1] < 0.0:
            brackets.append((depths[i], depths[i + 1]))
    if np.isfinite(f[-1]) and abs(f[-1]) <= ROOT_TOL:
        brackets.append((depths[-1], depths[-1]))

    unique = []
    for lo, hi in brackets:
        if not unique or abs(lo - unique[-1][0]) > 1.0e-8:
            unique.append((float(lo), float(hi)))
    return unique


def refine_root(compactness_for, center, width, target, bracket):
    lo, hi = bracket
    if lo == hi:
        return float(lo)

    def f(depth):
        return float(
            compactness_for(transition_profile(center, width, depth))
        ) - target

    flo = f(lo)
    fhi = f(hi)
    if flo * fhi > 0.0:
        return None

    for _ in range(40):
        mid = 0.5 * (lo + hi)
        fm = f(mid)
        if not np.isfinite(fm):
            return None
        if flo * fm <= 0.0:
            hi = mid
            fhi = fm
        else:
            lo = mid
            flo = fm
    return 0.5 * (lo + hi)


def evaluate(
    reference,
    h_c,
    x,
    target,
    eos_kwargs,
    compactness_for,
    response_state,
):
    center = x * h_c
    width = WIDTH_FRACTION * h_c
    depths, compactness = depth_curve(
        compactness_for, center, width
    )
    brackets = root_brackets(depths, compactness, target)
    roots = []
    for bracket in brackets:
        root = refine_root(
            compactness_for, center, width, target, bracket
        )
        if root is not None:
            roots.append(root)
    roots = sorted(roots)

    base = dict(
        gamma=float(reference.gamma),
        h_c=float(h_c),
        x=float(x),
        center=float(center),
        width=float(width),
        target_compactness=float(target),
        n_roots=len(roots),
        first_root_depth=float(roots[0]) if roots else None,
        depth_curve_min=float(np.nanmin(compactness)),
        depth_curve_max=float(np.nanmax(compactness)),
        depth_curve_max_increase=float(
            np.nanmax(np.diff(compactness))
        ),
    )

    if not roots:
        return dict(base, reachable=False)

    depth = roots[0]
    values = transition_profile(center, width, depth)
    y, soft, ilove, eigenvalues, ilove_var, clove_var, dMdh = (
        response_state(values)
    )

    finite = (
        np.all(np.isfinite(np.asarray(y)))
        and np.all(np.isfinite(np.asarray(eigenvalues)))
        and np.isfinite(float(dMdh))
    )
    positive_mass_slope = finite and float(dMdh) > 0.0
    compactness_value = float(jnp.exp(y[0]))
    root_residual = abs(compactness_value - target)

    return dict(
        base,
        reachable=True,
        retained=bool(positive_mass_slope),
        compactness=compactness_value,
        root_residual=float(root_residual),
        depth=float(depth),
        dM_dh=float(dMdh),
        ilove_alignment=float(jnp.abs(jnp.dot(soft, ilove)))
        if finite else None,
        ilove_clove_rms_ratio=float(jnp.sqrt(ilove_var / clove_var))
        if finite else None,
        normal_rms_ratio=float(jnp.sqrt(eigenvalues[1] / eigenvalues[2]))
        if finite else None,
    )


def attach_shell(reference, eos_kwargs, row):
    values = transition_profile(row["center"], row["width"], row["depth"])
    eos = build_nodal_sound_speed_eos(reference, values, **eos_kwargs)
    out = dict(row)
    out["transition_shell"] = physical_shell(
        eos, row["h_c"], row["center"], row["width"]
    )
    return out


def main():
    rows = []
    baselines = []
    reference_context = {}

    for gamma in GAMMAS:
        reference = RelativisticPolytrope(K=100.0, gamma=gamma)
        h_c = find_hc(reference, BASELINE_COMPACTNESS)
        baseline = solve_observables(reference, h_c, n_steps=N_STEPS)
        eos_kwargs, compactness_for, response_state = make_functions(
            reference, h_c
        )
        reference_context[gamma] = (reference, eos_kwargs)

        baseline_state = response_state(jnp.zeros(N_NODES))
        baseline_dMdh = float(baseline_state[-1])
        baseline_retained = (
            np.isfinite(baseline_dMdh) and baseline_dMdh > 0.0
        )
        baselines.append(
            dict(
                gamma=float(gamma),
                h_c=float(h_c),
                compactness=float(baseline.compactness),
                dM_dh=baseline_dMdh,
                retained=bool(baseline_retained),
            )
        )
        if not baseline_retained:
            raise RuntimeError(
                f"Baseline Gamma={gamma} at C={BASELINE_COMPACTNESS} "
                f"has dM/dh_c={baseline_dMdh}"
            )

        for target in TARGET_COMPACTNESS:
            for x in X_VALUES:
                rows.append(
                    evaluate(
                        reference,
                        h_c,
                        x,
                        target,
                        eos_kwargs,
                        compactness_for,
                        response_state,
                    )
                )

    best = []
    for gamma in GAMMAS:
        reference, eos_kwargs = reference_context[gamma]
        for target in TARGET_COMPACTNESS:
            subset = [
                r for r in rows
                if abs(r["gamma"] - gamma) < 1e-12
                and abs(r["target_compactness"] - target) < 1e-12
                and r.get("retained", False)
            ]
            if subset:
                winner = max(
                    subset, key=lambda r: r["ilove_clove_rms_ratio"]
                )
                best.append(attach_shell(reference, eos_kwargs, winner))

    summary_by_target = {}
    for target in TARGET_COMPACTNESS:
        subset = [
            r for r in best
            if abs(r["target_compactness"] - target) < 1e-12
        ]
        if not subset:
            continue
        xs = np.asarray([r["x"] for r in subset])
        rs = np.asarray([
            r["transition_shell"]["center"]["r_over_R"]
            for r in subset
        ])
        ms = np.asarray([
            r["transition_shell"]["center"]["m_over_M"]
            for r in subset
        ])
        summary_by_target[str(target)] = dict(
            n=len(subset),
            x_min=float(np.min(xs)),
            x_median=float(np.median(xs)),
            x_max=float(np.max(xs)),
            r_over_R_min=float(np.min(rs)),
            r_over_R_median=float(np.median(rs)),
            r_over_R_max=float(np.max(rs)),
            m_over_M_min=float(np.min(ms)),
            m_over_M_median=float(np.median(ms)),
            m_over_M_max=float(np.max(ms)),
        )

    root_counts = [r["n_roots"] for r in rows]
    residuals = [
        r["root_residual"] for r in rows if r.get("reachable", False)
    ]
    payload = dict(
        analysis="branch_safe_fixed_final_compactness",
        supersedes="results/fixed_final_compactness_refinement_20260918.json",
        n_nodes=N_NODES,
        n_steps=N_STEPS,
        n_high=N_HIGH,
        baseline_compactness=BASELINE_COMPACTNESS,
        target_compactness=list(TARGET_COMPACTNESS),
        width_fraction=WIDTH_FRACTION,
        edge=EDGE,
        max_depth=MAX_DEPTH,
        depth_grid_size=len(DEPTH_GRID),
        x_grid=list(X_VALUES),
        baselines=baselines,
        all_baselines_retained=all(b["retained"] for b in baselines),
        n_reachable=sum(r.get("reachable", False) for r in rows),
        n_retained=sum(r.get("retained", False) for r in rows),
        max_root_count=int(max(root_counts)),
        n_multiple_root_cases=int(sum(n > 1 for n in root_counts)),
        max_root_residual=float(max(residuals)) if residuals else None,
        best_by_gamma_and_target=best,
        summary_by_target=summary_by_target,
        rows=rows,
    )

    print(
        "BRANCHSAFE_FIXED_FINAL_JSON="
        + json.dumps(payload, sort_keys=True)
    )


if __name__ == "__main__":
    main()
