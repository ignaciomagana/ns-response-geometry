"""Named realistic-EOS validation of the response geometry.

The seven reference EOSs are the Read et al. piecewise-polytropic fits
tabulated by Lackey & Wade (2015): SLy, ENG, MPA1, MS1, MS1b, H4, ALF2.

The script first validates the stellar sequences against the published fit
predictions for M_max and R_1.4, then evaluates the full response spectrum at
matched compactness C=0.12, 0.16, 0.20.  A Frobenius screen is evaluated at
C=0.16 for every EOS.

This is a named realistic-EOS validation of the existing paper-1 geometry,
not a new EOS posterior.
"""

import json
import math

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
from scipy.optimize import brentq, minimize_scalar

from ns_response_geometry.geometry import (
    induced_response,
    normal_response_spectrum,
    plane_normal_variance,
)
from ns_response_geometry.observables import solve_observables
from ns_response_geometry.realistic_eos import (
    M_SUN_KM,
    NAMED_EOS_FITS,
    NamedPiecewisePolytrope,
)
from ns_response_geometry.response import response_jacobian, sequence_tangent
from ns_response_geometry.response_eos import (
    build_nodal_sound_speed_eos,
    squared_exponential_covariance,
)
from ns_response_geometry.tov import solve_star


EOS_NAMES = ("SLy", "ENG", "MPA1", "MS1", "MS1b", "H4", "ALF2")
TARGET_COMPACTNESS = (0.12, 0.16, 0.20)
N_NODES = 17
N_STEPS = 2048
N_HIGH = 1024

# Table I of Lackey & Wade (2015): tabulated value and percentage error of
# the best-fit piecewise-polytrope relative to the table.
PUBLISHED = {
    "SLy":  dict(mmax_tab=2.049, mmax_err_pct=0.02,  r14_tab=11.736, r14_err_pct=-0.21),
    "ENG":  dict(mmax_tab=2.240, mmax_err_pct=-0.05, r14_tab=12.059, r14_err_pct=-0.69),
    "MPA1": dict(mmax_tab=2.461, mmax_err_pct=-0.16, r14_tab=12.473, r14_err_pct=-0.26),
    "MS1":  dict(mmax_tab=2.767, mmax_err_pct=-0.54, r14_tab=14.918, r14_err_pct=0.06),
    "MS1b": dict(mmax_tab=2.776, mmax_err_pct=-1.03, r14_tab=14.583, r14_err_pct=-0.32),
    "H4":   dict(mmax_tab=2.032, mmax_err_pct=-0.85, r14_tab=13.774, r14_err_pct=1.34),
    "ALF2": dict(mmax_tab=2.086, mmax_err_pct=-5.26, r14_tab=13.188, r14_err_pct=-3.66),
}


def published_fit_values(name):
    p = PUBLISHED[name]
    return (
        p["mmax_tab"] * (1.0 + p["mmax_err_pct"] / 100.0),
        p["r14_tab"] * (1.0 + p["r14_err_pct"] / 100.0),
    )


def star_scalar(eos, h):
    s = solve_star(eos, h, n_steps=2048)
    return float(s.mass / M_SUN_KM), float(s.radius), float(s.compactness)


def stable_sequence(eos):
    hs = np.linspace(0.025, 0.95, 96)
    rows = []
    for h in hs:
        m, r, c = star_scalar(eos, float(h))
        if np.isfinite(m) and np.isfinite(r) and 0.0 < c < 0.4:
            rows.append((h, m, r, c))
    arr = np.asarray(rows)
    i = int(np.argmax(arr[:, 1]))
    if i < 2 or i >= len(arr) - 2:
        raise RuntimeError(f"Could not bracket maximum mass for {eos.name}")
    return arr, i


def sequence_benchmarks(eos):
    arr, imax = stable_sequence(eos)
    left = arr[imax - 2, 0]
    right = arr[imax + 2, 0]

    opt = minimize_scalar(
        lambda h: -star_scalar(eos, float(h))[0],
        bounds=(left, right),
        method="bounded",
        options={"xatol": 2.0e-6},
    )
    hmax = float(opt.x)
    mmax, rmax, cmax = star_scalar(eos, hmax)

    stable = arr[: imax + 1]
    masses = stable[:, 1]
    crossing = np.where((masses[:-1] - 1.4) * (masses[1:] - 1.4) <= 0.0)[0]
    if len(crossing) == 0:
        raise RuntimeError(f"No 1.4 Msun crossing for {eos.name}")
    j = int(crossing[-1])
    h14 = brentq(
        lambda h: star_scalar(eos, float(h))[0] - 1.4,
        stable[j, 0],
        stable[j + 1, 0],
        xtol=1.0e-9,
    )
    m14, r14, c14 = star_scalar(eos, h14)

    mfit, rfit = published_fit_values(eos.name)
    return dict(
        h_max=hmax,
        mmax_msun=mmax,
        r_at_mmax_km=rmax,
        c_at_mmax=cmax,
        h_1p4=float(h14),
        r_1p4_km=r14,
        c_1p4=c14,
        published_fit_mmax_msun=mfit,
        published_fit_r_1p4_km=rfit,
        mmax_fractional_error=(mmax - mfit) / mfit,
        r14_fractional_error=(r14 - rfit) / rfit,
    )


def h_for_compactness(eos, target, h_turn):
    grid = np.linspace(0.025, h_turn * 0.999, 80)
    comp = np.asarray([star_scalar(eos, float(h))[2] for h in grid])
    crossing = np.where((comp[:-1] - target) * (comp[1:] - target) <= 0.0)[0]
    if len(crossing) == 0:
        raise RuntimeError(f"{eos.name} does not reach C={target} on stable branch")
    j = int(crossing[0])
    return brentq(
        lambda h: star_scalar(eos, float(h))[2] - target,
        grid[j],
        grid[j + 1],
        xtol=1.0e-9,
    )


def response_at(eos, h_c, h_max):
    nodes = jnp.linspace(0.02, h_max, N_NODES)
    values = jnp.zeros(N_NODES)
    kwargs = dict(
        h_match=0.02,
        h_max=h_max,
        h_nodes=nodes,
        n_low=128,
        n_high=N_HIGH,
    )
    covariance = squared_exponential_covariance(
        nodes, amplitude=0.10, length_scale=0.08
    )

    J = response_jacobian(
        eos, values, h_c, n_steps=N_STEPS, eos_kwargs=kwargs
    )
    tangent = sequence_tangent(
        eos, values, h_c, n_steps=N_STEPS, eos_kwargs=kwargs
    )
    G = induced_response(J, covariance)
    evals, covectors = normal_response_spectrum(G, tangent, jnp.eye(3))

    soft = covectors[:, 1]
    ilove = jnp.array([0.0, -tangent[2], tangent[1]])
    ilove = ilove / jnp.linalg.norm(ilove)
    soft = jnp.where(jnp.dot(soft, ilove) < 0.0, -soft, soft)
    alignment = jnp.abs(jnp.dot(soft, ilove))

    ilove_var, _ = plane_normal_variance(G, tangent, [1, 2], jnp.eye(3))
    clove_var, _ = plane_normal_variance(G, tangent, [0, 2], jnp.eye(3))

    base = build_nodal_sound_speed_eos(eos, values, **kwargs)
    obs = solve_observables(base, h_c, n_steps=N_STEPS)
    cs2_center = float(base.sound_speed_squared(h_c))

    return dict(
        h_c=float(h_c),
        compactness=float(obs.compactness),
        mass_msun=float(obs.mass / M_SUN_KM),
        radius_km=float(obs.radius),
        lambda2=float(obs.lambda2),
        ibar=float(obs.ibar),
        center_cs2=cs2_center,
        normal_rms_ratio=float(jnp.sqrt(evals[1] / evals[2])),
        ilove_alignment=float(alignment),
        ilove_clove_rms_ratio=float(jnp.sqrt(ilove_var / clove_var)),
        soft_covector=np.asarray(soft).tolist(),
    ), (nodes, values, kwargs, covariance)


def frobenius_at(eos, h0, h_max):
    nodes = jnp.linspace(0.02, h_max, N_NODES)
    values0 = jnp.zeros(N_NODES)
    kwargs = dict(
        h_match=0.02,
        h_max=h_max,
        h_nodes=nodes,
        n_low=128,
        n_high=N_HIGH,
    )
    covariance = squared_exponential_covariance(
        nodes, amplitude=0.10, length_scale=0.08
    )
    domain_covariance = squared_exponential_covariance(
        nodes, amplitude=1.0, length_scale=0.08
    )
    domain_cholesky = jnp.linalg.cholesky(
        domain_covariance + 1.0e-10 * jnp.eye(N_NODES)
    )

    def log_observables(a, h):
        ref = build_nodal_sound_speed_eos(eos, a, **kwargs)
        r = solve_observables(ref, h, n_steps=N_STEPS)
        return jnp.log(jnp.stack((r.compactness, r.ibar, r.lambda2)))

    def local_state(a, h):
        y = log_observables(a, h)
        J = jax.jacfwd(lambda x: log_observables(x, h))(a)
        tangent = jax.jacfwd(lambda x: log_observables(a, x))(h)
        G = induced_response(J, covariance)

        that = tangent / jnp.linalg.norm(tangent)
        projector = jnp.eye(3) - jnp.outer(that, that)
        evals, evecs = jnp.linalg.eigh(projector @ G @ projector)
        soft = evecs[:, 1]
        ilove = jnp.array([0.0, -tangent[2], tangent[1]])
        ilove = ilove / jnp.linalg.norm(ilove)
        soft = jnp.where(jnp.dot(soft, ilove) < 0.0, -soft, soft)

        W = projector @ J @ domain_cholesky
        _, singular, Vh = jnp.linalg.svd(W, full_matrices=False)
        hard = domain_cholesky @ Vh[0]
        soft_input = domain_cholesky @ Vh[1]

        inv_hard = jnp.linalg.solve(domain_covariance, hard)
        inv_soft = jnp.linalg.solve(domain_covariance, soft_input)
        hard = hard / jnp.sqrt(hard @ inv_hard)
        soft_input = soft_input / jnp.sqrt(soft_input @ inv_soft)
        return y, soft, hard, soft_input, singular

    y0, n0, hard, soft_input, singular = local_state(
        values0, jnp.asarray(h0)
    )

    da = 0.03
    dh = 0.001

    def evaluate(h, ah, aso):
        y, n, _, _, _ = local_state(
            values0 + ah * hard + aso * soft_input, jnp.asarray(h)
        )
        n = jnp.where(jnp.dot(n, n0) < 0.0, -n, n)
        return y, n

    ycols = []
    ncols = []
    for label, step in (("h", dh), ("hard", da), ("soft", da)):
        if label == "h":
            yp, np_ = evaluate(h0 + step, 0.0, 0.0)
            ym, nm = evaluate(h0 - step, 0.0, 0.0)
        elif label == "hard":
            yp, np_ = evaluate(h0, step, 0.0)
            ym, nm = evaluate(h0, -step, 0.0)
        else:
            yp, np_ = evaluate(h0, 0.0, step)
            ym, nm = evaluate(h0, 0.0, -step)
        ycols.append((yp - ym) / (2.0 * step))
        ncols.append((np_ - nm) / (2.0 * step))

    dy_dx = jnp.stack(ycols, axis=1)
    dn_dx = jnp.stack(ncols, axis=1)
    dn_dy = dn_dx @ jnp.linalg.inv(dy_dx)
    curl = jnp.array(
        [
            dn_dy[2, 1] - dn_dy[1, 2],
            dn_dy[0, 2] - dn_dy[2, 0],
            dn_dy[1, 0] - dn_dy[0, 1],
        ]
    )
    obstruction = jnp.dot(n0, curl)
    eta = jnp.abs(obstruction) / (jnp.linalg.norm(curl) + 1.0e-30)

    return dict(
        compactness=float(jnp.exp(y0[0])),
        normalized_helicity=float(eta),
        dy_dx_condition=float(jnp.linalg.cond(dy_dx)),
        input_singular_values=np.asarray(singular).tolist(),
    )


def main():
    output = dict(
        eos_names=list(EOS_NAMES),
        target_compactness=list(TARGET_COMPACTNESS),
        published_source="Lackey & Wade 2015 Table I / Read et al. piecewise fits",
        results={},
    )

    all_normal = []
    all_alignment = []
    all_ilove_control = []
    all_frobenius = []

    for name in EOS_NAMES:
        eos = NamedPiecewisePolytrope.from_name(name)
        benchmark = sequence_benchmarks(eos)
        h_domain_max = max(0.65, 1.10 * benchmark["h_max"])

        points = []
        h_by_c = {}
        for target in TARGET_COMPACTNESS:
            h = h_for_compactness(eos, target, benchmark["h_max"])
            h_by_c[str(target)] = float(h)
            response, _ = response_at(eos, h, h_domain_max)
            points.append(response)
            all_normal.append(response["normal_rms_ratio"])
            all_alignment.append(response["ilove_alignment"])
            all_ilove_control.append(response["ilove_clove_rms_ratio"])

        frob = frobenius_at(eos, h_by_c["0.16"], h_domain_max)
        all_frobenius.append(frob["normalized_helicity"])

        output["results"][name] = dict(
            fit_parameters=dict(
                log10_p1=NAMED_EOS_FITS[name][0],
                gamma1=NAMED_EOS_FITS[name][1],
                gamma2=NAMED_EOS_FITS[name][2],
                gamma3=NAMED_EOS_FITS[name][3],
                match_density_cgs=eos.match_density_cgs,
            ),
            benchmark=benchmark,
            response_points=points,
            frobenius_C0p16=frob,
        )

    output["summary"] = dict(
        n_eos=len(EOS_NAMES),
        n_response_points=len(all_normal),
        normal_rms_ratio_min=float(np.min(all_normal)),
        normal_rms_ratio_median=float(np.median(all_normal)),
        normal_rms_ratio_max=float(np.max(all_normal)),
        ilove_alignment_min=float(np.min(all_alignment)),
        ilove_alignment_median=float(np.median(all_alignment)),
        ilove_alignment_max=float(np.max(all_alignment)),
        ilove_clove_rms_ratio_min=float(np.min(all_ilove_control)),
        ilove_clove_rms_ratio_median=float(np.median(all_ilove_control)),
        ilove_clove_rms_ratio_max=float(np.max(all_ilove_control)),
        frobenius_min=float(np.min(all_frobenius)),
        frobenius_median=float(np.median(all_frobenius)),
        frobenius_max=float(np.max(all_frobenius)),
        benchmark_mmax_abs_fractional_error_max=float(
            max(abs(output["results"][n]["benchmark"]["mmax_fractional_error"]) for n in EOS_NAMES)
        ),
        benchmark_r14_abs_fractional_error_max=float(
            max(abs(output["results"][n]["benchmark"]["r14_fractional_error"]) for n in EOS_NAMES)
        ),
    )

    print("REALISTIC_EOS_JSON=" + json.dumps(output, sort_keys=True))


if __name__ == "__main__":
    main()
