"""Linear response geometry utilities."""

from __future__ import annotations

import jax.numpy as jnp


def induced_response(jacobian, eos_covariance):
    """Return G = J C_EOS J^T."""
    J = jnp.asarray(jacobian)
    C = jnp.asarray(eos_covariance)
    return J @ C @ J.T


def metric_projector(tangent, metric=None):
    """Project vectors onto the metric-orthogonal complement of tangent.

    P acts on contravariant vectors.  For a positive metric g,

        P = I - t (t^T g) / (t^T g t).
    """
    t = jnp.asarray(tangent)
    g = jnp.eye(t.shape[0], dtype=t.dtype) if metric is None else jnp.asarray(metric)
    denom = t @ g @ t
    return jnp.eye(t.shape[0], dtype=t.dtype) - jnp.outer(t, t @ g) / denom


def transverse_response(response, tangent, metric=None):
    """Project EOS-induced observable response away from sequence motion."""
    G = jnp.asarray(response)
    P = metric_projector(tangent, metric)
    return P @ G @ P.T
