import numpy as np

from ns_response_geometry.geometry import (
    induced_response,
    metric_projector,
    normal_response_spectrum,
    plane_normal_variance,
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


def test_plane_normal_is_unit_covector_and_annihilates_tangent():
    G = np.array([[2.0, 0.4], [0.4, 1.0]])
    t = np.array([1.0, 2.0])
    g = np.array([[2.0, 0.2], [0.2, 1.0]])

    variance, n = plane_normal_variance(G, t, [0, 1], g)
    n = np.asarray(n)

    np.testing.assert_allclose(n @ t, 0.0, atol=2e-6)
    np.testing.assert_allclose(n @ np.linalg.inv(g) @ n, 1.0, atol=2e-6)
    assert float(variance) > 0.0


def test_plane_normal_variance_invariant_under_linear_observable_change():
    G = np.array([[2.0, 0.4], [0.4, 1.0]])
    t = np.array([1.0, 2.0])
    g = np.array([[2.0, 0.2], [0.2, 1.0]])
    variance, _ = plane_normal_variance(G, t, [0, 1], g)

    A = np.array([[1.3, 0.4], [-0.2, 0.8]])
    Ainv = np.linalg.inv(A)

    G_new = A @ G @ A.T
    t_new = A @ t
    g_new = Ainv.T @ g @ Ainv

    variance_new, _ = plane_normal_variance(G_new, t_new, [0, 1], g_new)
    np.testing.assert_allclose(variance_new, variance, rtol=2e-6, atol=2e-6)


def test_full_normal_spectrum_has_projected_tangent_null():
    rng = np.random.default_rng(11)
    X = rng.normal(size=(3, 3))
    G = X @ X.T
    t = rng.normal(size=3)
    g = np.diag([1.0, 2.0, 0.5])

    evals, covectors = normal_response_spectrum(G, t, g)
    evals = np.asarray(evals)
    covectors = np.asarray(covectors)

    assert abs(evals[0]) < 2e-6
    for k in (1, 2):
        np.testing.assert_allclose(covectors[:, k] @ t, 0.0, atol=3e-6)
