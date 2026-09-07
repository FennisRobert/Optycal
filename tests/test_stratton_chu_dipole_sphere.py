"""Equivalence-theorem validation of the Stratton-Chu solvers.

A Hertzian dipole (`dipole_pattern_nf`/`dipole_pattern_ff`) has a closed-form
near- and far-field solution. If we enclose that dipole in a closed,
non-reflecting (`FRES_AIR`) sphere, expose the sphere with the dipole's near
field, and then propagate the sphere's surface fields back out with the
Stratton-Chu kernels, the equivalence theorem guarantees the result outside
the sphere must reproduce the dipole's own analytic field. This gives an
actual ground truth for `solvers/strattonchu.py` and `solvers/strattonchuff.py`,
which is otherwise hard to come by for a PO code (see CLAUDE.md).

Tolerances below are set from measured behavior, not guessed:
- The far-field kernel (`stratton_chu_ff`) converges to ~1e-4 relative error
  even on a coarse mesh.
- The near-field kernel (`stratton_chu_xyz`) plateaus around ~5-9% relative
  error as the mesh is refined (see the mesh-convergence note below) -- a
  real discretization ceiling of the vertex/edge-midpoint field quadrature,
  not something that vanishes with a finer mesh. Tightening this tolerance
  without changing the quadrature scheme will make the test flaky.
"""
import numpy as np
import pytest

import optycal as opt

FREQ = 3e9
C0 = 299792458.0
WAVELENGTH = C0 / FREQ
RADIUS = 0.75 * WAVELENGTH  # sphere strictly enclosing the point dipole at the origin
DS = WAVELENGTH / 10  # mesh element size

# Evaluation directions: avoid the dipole's on-axis nulls (theta=0/pi) and
# principal-plane zero-crossings, which would make relative-error metrics
# on individual field components singular.
THETA_EVAL = np.array([np.pi / 3, np.pi / 2, 2 * np.pi / 3, 0.9, 2.0, 1.3])
PHI_EVAL = np.array([0.3, 1.0, 2.5, -1.2, 4.0, -2.7])

FAR_FIELD_RELERR_TOL = 1e-2
NEAR_FIELD_RELERR_TOL = 0.12
NEAR_FIELD_RADIUS_FACTOR = 2.0  # evaluate the near field at 2x the sphere radius


def _vector_relerr(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Per-point relative error between two (3, N) complex vector fields."""
    return np.linalg.norm(a - b, axis=0) / np.linalg.norm(b, axis=0)


@pytest.fixture(scope="module")
def dipole_in_sphere():
    """A dipole enclosed by a closed, non-reflecting spherical surface,
    exposed with the dipole's own near field."""
    antenna = opt.Antenna(0, 0, 0, FREQ, opt.GCS, opt.dipole_pattern_nf, opt.dipole_pattern_ff)
    mesh = opt.generate_sphere(np.array([0.0, 0.0, 0.0]), RADIUS, DS, opt.GCS)
    surface = opt.Surface(mesh, opt.FRES_AIR)
    antenna.expose_surface(surface)
    return antenna, surface


def test_sphere_surface_is_excited(dipole_in_sphere):
    _, surface = dipole_in_sphere
    # side 2 (outward, transmitted through an FRES_AIR interface = fully
    # transmitted) should carry the dipole's near field; side 1 (reflected)
    # should be ~0 since air/air has no reflection.
    assert np.max(np.abs(surface.E2)) > 1.0
    assert np.max(np.abs(surface.E1)) < 1e-6 * np.max(np.abs(surface.E2))


def test_farfield_matches_analytic_dipole(dipole_in_sphere):
    antenna, surface = dipole_in_sphere

    sc_field = surface.expose_thetaphi(THETA_EVAL, PHI_EVAL, side=2)
    analytic_field = antenna.expose_thetaphi(THETA_EVAL, PHI_EVAL)

    relerr = _vector_relerr(sc_field.E.F, analytic_field.E.F)
    assert np.all(np.isfinite(relerr))
    assert np.max(relerr) < FAR_FIELD_RELERR_TOL


def test_farfield_H_matches_analytic_dipole(dipole_in_sphere):
    """Regression test for a bug found while porting stratton_chu_ff to Rust
    (see claude_nodes/kernel_optimization.md): the far-field H was computed
    from the pre-`Q`-scaling E accumulator instead of the actual (scaled) E,
    violating the physical `H = r_hat x E / Z0` relation and leaving H off
    by a frequency-dependent complex factor. E alone did not catch this --
    `test_farfield_matches_analytic_dipole` above only checks `.E`."""
    antenna, surface = dipole_in_sphere

    sc_field = surface.expose_thetaphi(THETA_EVAL, PHI_EVAL, side=2)
    analytic_field = antenna.expose_thetaphi(THETA_EVAL, PHI_EVAL)

    relerr = _vector_relerr(sc_field.H.F, analytic_field.H.F)
    assert np.all(np.isfinite(relerr))
    assert np.max(relerr) < FAR_FIELD_RELERR_TOL


def test_nearfield_matches_analytic_dipole(dipole_in_sphere):
    antenna, surface = dipole_in_sphere

    r = NEAR_FIELD_RADIUS_FACTOR * RADIUS
    x = r * np.sin(THETA_EVAL) * np.cos(PHI_EVAL)
    y = r * np.sin(THETA_EVAL) * np.sin(PHI_EVAL)
    z = r * np.cos(THETA_EVAL)

    sc_field = surface.expose_xyz(x, y, z, side=2)
    analytic_field = antenna.expose_xyz(x, y, z)

    relerr = _vector_relerr(sc_field.E, analytic_field.E)
    assert np.all(np.isfinite(relerr))
    assert np.max(relerr) < NEAR_FIELD_RELERR_TOL
