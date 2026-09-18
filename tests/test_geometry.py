import numpy as np

from ns_response_geometry.geometry import (
    induced_response,
    metric_projector,
    transverse_response,
)


def test_induced_response_is_invariant_under_eos_basis_change():
    rng = np.random.default_rng(7)
    J = rng.normal(size=(3, 4))
    A = rng.normal(size=(4, 4))
    while abs(np.linalg.det(A)) < 0.1:
        A = rng.normal(size=(4, 4))

    X = rng.normal(size=(4, 4))
    C = X @ X.T

    G = np.asarray(induced_response(J, C))

    Ainv = np.linalg.inv(A)
    J_new = J @ Ainv
    C_new = A @ C @ A.T
    G_new = np.asarray(induced_response(J_new, C_new))

    np.testing.assert_allclose(G_new, G, rtol=2e-6, atol=2e-6)


def test_metric_projector_removes_tangent():
    t = np.array([1.0, -2.0, 0.5])
    g = np.diag([1.0, 3.0, 2.0])
    P = np.asarray(metric_projector(t, g))
    np.testing.assert_allclose(P @ t, 0.0, atol=2e-7)


def test_transverse_response_has_no_tangent_component_in_euclidean_metric():
    G = np.array([[2.0, 0.4], [0.4, 1.0]])
    t = np.array([1.0, 2.0])
    Gp = np.asarray(transverse_response(G, t))
    np.testing.assert_allclose(Gp @ t, 0.0, atol=2e-7)
