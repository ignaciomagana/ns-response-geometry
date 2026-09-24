import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from ns_response_geometry.realistic_eos import (
    NAMED_EOS_FITS,
    NamedPiecewisePolytrope,
)


@pytest.mark.parametrize("name", sorted(NAMED_EOS_FITS))
def test_named_eos_is_finite_and_causal_over_moderate_enthalpy(name):
    eos = NamedPiecewisePolytrope.from_name(name)
    h = jnp.linspace(1.0e-6, 0.45, 128)

    p = np.asarray(eos.pressure(h))
    eps = np.asarray(eos.energy_density(h))
    cs2 = np.asarray(eos.sound_speed_squared(h))

    assert np.all(np.isfinite(p))
    assert np.all(np.isfinite(eps))
    assert np.all(np.isfinite(cs2))
    assert np.all(np.diff(p) > 0.0)
    assert np.all(np.diff(eps) > 0.0)
    assert np.all(cs2 >= 0.0)


@pytest.mark.parametrize("name", sorted(NAMED_EOS_FITS))
def test_named_eos_pressure_and_energy_are_continuous(name):
    eos = NamedPiecewisePolytrope.from_name(name)
    _, _, _, upper_h, _ = eos._segment_data()

    for hb in np.asarray(upper_h):
        dh = 1.0e-8
        pm = float(eos.pressure(hb - dh))
        pp = float(eos.pressure(hb + dh))
        em = float(eos.energy_density(hb - dh))
        ep = float(eos.energy_density(hb + dh))

        assert abs(pp - pm) / max(abs(pp), abs(pm), 1.0e-30) < 2.0e-5
        assert abs(ep - em) / max(abs(ep), abs(em), 1.0e-30) < 2.0e-5


def test_named_eos_match_density_is_above_last_fixed_crust_boundary():
    for name in NAMED_EOS_FITS:
        eos = NamedPiecewisePolytrope.from_name(name)
        assert eos.match_density_cgs > 2.62780e12
        assert eos.match_density_cgs < 10.0**14.7
