"""Basic sanity checks derived from examples/example_2_antenna_array.py and
examples/example_3_rotated_array.py.

Uses a much smaller array (4x3 instead of 20x10) purely to keep the
power-normalization sphere integral (`AntennaArray._normalize_power`, which
generates and exposes its own mesh) fast in a test run.
"""
import numpy as np

import optycal as opt

FREQ = 3.1e9
C0 = 299792458.0
WAVELENGTH = C0 / FREQ
ELEMENT_SPACING = 0.55 * WAVELENGTH


def _small_array(cs=None) -> "opt.AntennaArray":
    array = opt.AntennaArray(FREQ, cs if cs is not None else opt.GCS)
    array.add_2d_array(
        opt.taper.taylor(4, 3, 20),
        opt.taper.uniform(3),
        (0, ELEMENT_SPACING, 0),
        (0, 0, ELEMENT_SPACING),
        0,
        opt.dipole_pattern_nf,
        opt.dipole_pattern_ff,
    )
    return array


def test_array_broadside_farfield_is_finite_and_peaks_on_boresight():
    array = _small_array()

    azi, ele = opt.FF1D.aziele(dangle=2)
    array.expose_ff(azi)
    array.expose_ff(ele)

    assert np.all(np.isfinite(azi.field.normE))
    assert np.all(np.isfinite(ele.field.normE))

    # Broadside (no scan direction set): peak gain should be at theta=90deg
    # (i.e. phi=0 index) in the elevation cut, since that's boresight for
    # an array in the y/z plane radiating along x.
    boresight_idx = np.argmin(np.abs(ele.theta - np.pi / 2))
    assert np.argmax(ele.field.normE) == boresight_idx


def test_array_normalizes_to_requested_power():
    """`add_2d_array` calls `AntennaArray._normalize_power` internally to
    scale the array to radiate `array.power` (1 W by default). Verify that
    by independently measuring the radiated power through a surrounding
    sphere surface (`Surface.powerflux`), the same mechanism the library
    itself already relies on elsewhere (e.g. `Antenna.normalize_power`).

    Note: `EHFieldFF.total_radiated_power_integral` (new in emsutil>=1.0)
    is not used here because it requires a (3, N, M) grid-shaped far field,
    while `AntennaArray.expose_thetaphi`/`FF2D` always produce flattened
    (3, N*M) data -- a real data-layout mismatch between Optycal and the
    newer emsutil, not something this test should paper over.

    This is a power-conservation check only, not an equivalence-theorem
    validation: array elements here use isolated (non-embedded) dipole
    patterns with no mutual coupling between them, so the combined
    near/far field is a plain superposition of independent point sources,
    not what a physically coupled array would produce. That's fine for
    checking "does the total radiated power come out ~1 W", but it's not
    a meaningful accuracy benchmark the way the single-dipole Stratton-Chu
    equivalence test in test_stratton_chu_dipole_sphere.py is.
    """
    array = _small_array()

    _lambda = 2 * np.pi / array.k0
    radius = 5 * _lambda
    mesh = opt.generate_sphere(np.array([0.0, 0.0, 0.0]), radius, _lambda / 3, array.cs.get_global())
    surface = opt.Surface(mesh, opt.FRES_AIR)
    array.expose_surface(surface)

    Prad = sum(surface.powerflux())
    assert np.isfinite(Prad)
    assert 0.5 < Prad < 2.0


def test_rotated_and_steered_array_farfield_is_finite():
    """From example_3: array mounted in a rotated/tilted coordinate system,
    with electronic beam steering applied."""
    cs = opt.GCS.copy().rotate_basis((0, -1, 0), 15, True).rotate_basis((0, 0, 1), 10)
    array = _small_array(cs)
    array.set_scan_direction(90, 0)

    azi, ele = opt.FF1D.aziele(dangle=5)
    array.expose_ff(azi)
    array.expose_ff(ele)

    assert np.all(np.isfinite(azi.field.normE))
    assert np.all(np.isfinite(ele.field.normE))
    assert np.max(azi.field.normE) > 0
    assert np.max(ele.field.normE) > 0
