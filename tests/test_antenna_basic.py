"""Basic sanity checks derived from examples/example_1_antenna.py.

Mirrors the single-antenna workflow (define an Antenna, expose azimuth/
elevation cuts and a far-field sphere) but asserts on shapes/finiteness
instead of plotting, so it can run headless in CI.
"""
import numpy as np

import optycal as opt

FREQ = 2e9


def test_single_dipole_farfield_cuts():
    ant = opt.Antenna(0, 0, 0, FREQ, opt.GCS, opt.dipole_pattern_nf, opt.dipole_pattern_ff)

    azi, ele = opt.FF1D.aziele(dangle=10)
    ant.expose_ff(azi)
    ant.expose_ff(ele)

    assert azi.field is not None and ele.field is not None
    assert azi.field.normE.shape == azi.phi.shape
    assert ele.field.normE.shape == ele.theta.shape
    assert np.all(np.isfinite(azi.field.normE))
    assert np.all(np.isfinite(ele.field.normE))
    # A dipole is omnidirectional in the azimuth (phi) cut at theta=90deg.
    assert np.allclose(azi.field.normE, azi.field.normE[0], rtol=1e-5)


def test_single_dipole_farfield_sphere():
    ant = opt.Antenna(0, 0, 0, FREQ, opt.GCS, opt.dipole_pattern_nf, opt.dipole_pattern_ff)

    sphere = opt.FF2D.sphere(10)
    ant.expose_ff(sphere)

    assert sphere.field is not None
    assert sphere.field.normE.shape == sphere.theta.shape
    assert np.all(np.isfinite(sphere.field.normE))
    # A z-oriented dipole has a null on its own axis (theta=0/pi).
    on_axis = np.isclose(sphere.theta, 0) | np.isclose(sphere.theta, np.pi)
    assert np.all(sphere.field.normE[on_axis] < 1e-6 * np.max(sphere.field.normE))


def test_expose_xyz_matches_expose_thetaphi_far_away():
    """expose_xyz at a large radius should agree with the far-field pattern
    (expose_thetaphi), up to the 1/R free-space phase/amplitude factor that
    expose_xyz includes and expose_thetaphi doesn't."""
    ant = opt.Antenna(0, 0, 0, FREQ, opt.GCS, opt.dipole_pattern_nf, opt.dipole_pattern_ff)

    theta = np.array([np.pi / 4, np.pi / 2, 3 * np.pi / 4])
    phi = np.array([0.0, 1.0, 2.5])
    r = 1000.0  # electrically huge compared to a point dipole

    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)

    near = ant.expose_xyz(x, y, z)
    far = ant.expose_thetaphi(theta, phi)

    k0 = 2 * np.pi * FREQ / 299792458.0
    predicted_far_from_near = near.E * r * np.exp(1j * k0 * r)

    assert np.allclose(predicted_far_from_near, far.E.F, rtol=1e-3, atol=1e-6)
