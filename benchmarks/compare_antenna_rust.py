"""Numeric correctness check: Rust antenna-pattern evaluation
(`antenna_expose_xyz`/`antenna_expose_thetaphi` + `AntennaPattern`) vs. the
existing pure-Python `Antenna.expose_xyz`/`expose_thetaphi` on identical
inputs, for: the native Dipole pattern, the native HalfDipole pattern, and
an arbitrary custom pattern (patch) routed through the Interpolated/gridded
path.

Usage: python benchmarks/compare_antenna_rust.py
"""
import _cap_cores  # noqa: F401

import numpy as np

import optycal as opt
import optycal_kernels
from optycal.antennas.interpolation_pattern import AntennaPattern as PyAntennaPattern
from optycal.settings import Precision

FREQ = 3e9
C0 = 299792458.0
K0 = 2 * np.pi * FREQ / C0


def relerr(a, b):
    a = np.asarray(a)
    b = np.asarray(b)
    return float(np.max(np.abs(a - b)) / np.max(np.abs(b)))


def check_pattern(name, nf_pattern, ff_pattern, cs, rust_pattern, nf_tol=1e-8, ff_tol=1e-4):
    ant = opt.Antenna(0.3, -0.2, 0.5, FREQ, cs, nf_pattern, ff_pattern)

    rng = np.random.default_rng(0)
    n = 200
    # near-field points, well outside a small radius so dipole reactive
    # terms are well-conditioned
    theta = np.arccos(rng.uniform(-1, 1, n))
    phi = rng.uniform(-np.pi, np.pi, n)
    r = rng.uniform(2.0, 20.0, n) * (C0 / FREQ)
    x = ant.gx + r * np.sin(theta) * np.cos(phi)
    y = ant.gy + r * np.sin(theta) * np.sin(phi)
    z = ant.gz + r * np.cos(theta)

    py_nf = ant.expose_xyz(x, y, z)

    basis = np.ascontiguousarray(ant.cs.global_basis.astype(np.float64))
    basis_inv = np.ascontiguousarray(np.linalg.pinv(ant.cs.global_basis).astype(np.float64))
    E_rs, H_rs = optycal_kernels.antenna_expose_xyz(
        x, y, z, list(ant.gxyz), basis, basis_inv, rust_pattern, complex(ant.amplitude), ant.k0,
    )
    e_err = relerr(np.asarray(E_rs), py_nf.E)
    h_err = relerr(np.asarray(H_rs), py_nf.H)
    print(f"[{name}] near field: E relerr={e_err:.3e}  H relerr={h_err:.3e}")
    assert e_err < nf_tol, f"{name} near-field E mismatch: {e_err}"
    assert h_err < nf_tol, f"{name} near-field H mismatch: {h_err}"

    # far field
    gtheta = np.arccos(rng.uniform(-1, 1, n)).astype(np.float32)
    gphi = rng.uniform(-np.pi, np.pi, n).astype(np.float32)
    py_ff = ant.expose_thetaphi(gtheta, gphi)

    E_rs, H_rs = optycal_kernels.antenna_expose_thetaphi(
        gtheta, gphi, list(ant.local_xyz), basis, basis_inv, rust_pattern, complex(ant.amplitude), ant.k0,
    )
    e_err = relerr(np.asarray(E_rs), py_ff.E.F)
    h_err = relerr(np.asarray(H_rs), py_ff.H.F)
    print(f"[{name}] far field:  E relerr={e_err:.3e}  H relerr={h_err:.3e}")
    # Antenna.expose_thetaphi downcasts theta/phi to float32 partway
    # through (`gtheta = gtheta.astype(np.float32)`); this Rust port stays
    # in f64 throughout, so it's *more* accurate than the Python reference
    # here, not less -- the expected residual is float32-epsilon-scale
    # (~1e-7 per op, compounding across a few trig calls to ~1e-6), not
    # machine precision. Tolerance reflects that; tightening it would just
    # be asserting Python's float32 rounding as ground truth.
    assert e_err < ff_tol, f"{name} far-field E mismatch: {e_err}"
    assert h_err < ff_tol, f"{name} far-field H mismatch: {h_err}"


if __name__ == "__main__":
    cs = opt.GCS.copy().rotate_basis((0, -1, 0), 25).rotate_basis((0, 0, 1), 40)

    check_pattern("dipole", opt.dipole_pattern_nf, opt.dipole_pattern_ff, cs,
                  optycal_kernels.AntennaPattern.dipole())
    check_pattern("half_dipole", opt.half_dipole_pattern_nf, opt.half_dipole_pattern_ff, cs,
                  optycal_kernels.AntennaPattern.half_dipole())

    # Custom pattern (patch) via the Interpolated/gridded path.
    theta_grid = np.linspace(0, np.pi, 181, dtype=np.float32)
    phi_grid = np.linspace(-np.pi, np.pi, 361, dtype=np.float32)
    pat = PyAntennaPattern.from_function(opt.patch_pattern_ff, theta_grid, phi_grid, K0)
    full = pat.full_matrix(Precision.DOUBLE)
    rust_patch = optycal_kernels.AntennaPattern.interpolated(theta_grid, phi_grid, full)

    # Note: patch_pattern_nf ignores r (same as ff), so the "far-field
    # shape only" near-field approximation used by Interpolated isn't an
    # approximation here -- the only expected residual is genuine
    # bicubic-spline interpolation error against the exact analytic
    # pattern (grid is 181x361, so ~1e-5 is the expected ballpark, not a
    # bug -- this is inherent to gridded interpolation, not something a
    # tighter tolerance should chase).
    check_pattern("patch (interpolated)", opt.patch_pattern_nf, opt.patch_pattern_ff, cs, rust_patch,
                  nf_tol=5e-4, ff_tol=2e-3)

    print("OK: Rust antenna evaluation matches pure-Python Antenna.")
