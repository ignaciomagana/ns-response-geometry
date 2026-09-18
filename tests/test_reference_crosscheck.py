import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)

from ns_response_geometry.eos import IncompressibleEOS, RelativisticPolytrope
from ns_response_geometry.observables import solve_observables

from tests.reference_scipy import solve_reference


@pytest.mark.parametrize(
    "eos,h_c",
    [
        (IncompressibleEOS(epsilon0=1.0), 0.15),
        (RelativisticPolytrope(K=100.0, gamma=2.0), 0.10),
    ],
)
def test_jax_enthalpy_solver_matches_adaptive_radius_reference(eos, h_c):
    reference = solve_reference(eos, h_c)
    result = solve_observables(eos, h_c, n_steps=16384)
    production = np.asarray(result[:-1], dtype=float)

    np.testing.assert_allclose(production, reference, rtol=3.0e-5, atol=1.0e-9)
