"""Linear response geometry utilities.

Observable displacements are contravariant vectors and relation normals are
covectors. Keep those roles explicit: the covariance G has upper indices,
while a relation n_A dy^A=0 lives in the cotangent space.
"""

from __future__ import annotations

import jax.numpy as jnp


def induced_response(jacobian, eos_covariance):
    """Return the observable displacement covariance G = J C_EOS J^T."""
    J = jnp.asarray(jacobian)
    C = jnp.asarray(eos_covariance)
    return J @ C @ J.T


def metric_projector(tangent, metric=None):
    """Project contravariant vectors orthogonal to tangent under metric g."""
    t = jnp.asarray(tangent)
    g = jnp.eye(t.shape[0], dtype=t.dtype) if metric is None else jnp.asarray(metric)
    denom = t @ g @ t
    return jnp.eye(t.shape[0], dtype=t.dtype) - jnp.outer(t, t @ g) / denom


def transverse_response(response, tangent, metric=None):
    """Projected displacement covariance, mainly useful for visualization."""
    G = jnp.asarray(response)
    P = metric_projector(tangent, metric)
    return P @ G @ P.T


def normal_response_spectrum(response, tangent, metric=None):
    """Metric-normalized response spectrum normal to a stellar sequence.

    We minimize n_A G^{AB} n_B subject to n_A t^A = 0 and
    n_A g^{AB} n_B = 1. If g = L L^T and n = L m, this becomes an
    ordinary Euclidean eigenproblem in m.
    """
    G = jnp.asarray(response)
    t = jnp.asarray(tangent)
    g = jnp.eye(t.shape[0], dtype=t.dtype) if metric is None else jnp.asarray(metric)

    L = jnp.linalg.cholesky(g)
    q = L.T @ t
    qhat = q / jnp.linalg.norm(q)
    P = jnp.eye(t.shape[0], dtype=t.dtype) - jnp.outer(qhat, qhat)

    whitened = L.T @ G @ L
    normal_operator = P @ whitened @ P
    evals, evecs = jnp.linalg.eigh(normal_operator)

    # Columns are covectors n_A in the original observable coordinates.
    covectors = L @ evecs
    return evals, covectors


def plane_normal_variance(response, tangent, indices, metric=None):
    """Unique metric-normalized normal variance in a two-observable plane."""
    G = jnp.asarray(response)
    t = jnp.asarray(tangent)
    idx = jnp.asarray(indices)

    G2 = G[jnp.ix_(idx, idx)]
    t2 = t[idx]
    if metric is None:
        g2 = jnp.eye(2, dtype=G.dtype)
    else:
        g = jnp.asarray(metric)
        g2 = g[jnp.ix_(idx, idx)]

    L = jnp.linalg.cholesky(g2)
    q = L.T @ t2
    m = jnp.stack((-q[1], q[0]))
    m = m / jnp.linalg.norm(m)
    n = L @ m
    return n @ G2 @ n, n
