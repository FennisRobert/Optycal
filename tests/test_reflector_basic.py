"""Basic sanity checks derived from examples/example_4_reflector.py.

Uses a smaller/coarser reflector than the example purely to keep the PO
surface exposure (`Antenna.expose_surface` over every mesh edge) fast in a
test run.
"""
import numpy as np

import optycal as opt

FREQ = 9e9
C0 = 299792458.0
WAVELENGTH = C0 / FREQ
DS = WAVELENGTH * 0.4  # coarse mesh, just for a smoke test
APERTURE_RADIUS = 0.2
FOCAL_DISTANCE = 0.4 / 0.6 * APERTURE_RADIUS  # keep the f/D ratio of the example


def _build_reflector_and_feed():
    feed_cs = opt.GCS.displace().rotate_basis((0, 0, 1), 180)

    parametric_line = opt.ParametricLine(fz=lambda t: APERTURE_RADIUS * t)
    mapping = opt.Mapping.parabolic_reflector((0, 0, 0), FOCAL_DISTANCE, (1, 0, 0))
    sweep_function = mapping.map(opt.SweepFunction.revolve((1, 0, 0)))
    mesh = sweep_function.mesh(parametric_line, DS, opt.GCS, alignment_function=opt.AlignX)

    surface = opt.Surface(mesh, opt.FRES_PEC)
    antenna = opt.Antenna(
        0, 0, 0, FREQ, feed_cs,
        *opt.generate_patch_pattern(0.3 * WAVELENGTH, 0.3 * WAVELENGTH, 2 * np.pi * FREQ / C0),
    )
    antenna.expose_surface(surface)
    return antenna, surface


def test_reflector_mesh_and_feed_are_built():
    antenna, surface = _build_reflector_and_feed()
    assert surface.mesh.nedges > 0
    # PEC reflector should be fully reflective: transmission is zero on both
    # sides (Tte=Ttm=0 for FRES_PEC), so all of the induced field ends up on
    # whichever side faces the feed -- here that's side 2.
    assert np.max(np.abs(surface.E2)) > 0
    assert np.max(np.abs(surface.E1)) == 0


def test_reflector_farfield_is_finite_and_peaks_near_boresight():
    antenna, surface = _build_reflector_and_feed()

    azi, ele = opt.FF1D.aziele(dangle=2)
    surface.expose_ff(azi)
    surface.expose_ff(ele)

    assert np.all(np.isfinite(azi.field.normE))
    assert np.all(np.isfinite(ele.field.normE))
    assert np.max(azi.field.normE) > 0
    assert np.max(ele.field.normE) > 0

    # A front-fed paraboloid pointed along +X should have its peak gain
    # close to boresight (theta=90deg, phi=0deg) in these cuts.
    boresight_idx = np.argmin(np.abs(ele.theta - np.pi / 2))
    peak_idx = np.argmax(ele.field.normE)
    assert abs(peak_idx - boresight_idx) <= 2
