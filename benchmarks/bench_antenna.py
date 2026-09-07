"""Throughput comparison: Rust-backed Antenna.expose_xyz/expose_thetaphi
vs. a reconstruction of the pre-migration pure-Python implementation, for
the native Dipole pattern and a custom (patch, interpolated) pattern.

Usage: python benchmarks/bench_antenna.py
"""
import _cap_cores  # noqa: F401

import time

import numpy as np

import optycal as opt
import optycal_kernels

FREQ = 3e9
C0 = 299792458.0
K0 = 2 * np.pi * FREQ / C0
REPEATS = 5


def _time_calls(fn, repeats: int) -> float:
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return min(times)


def old_expose_thetaphi(ant, gtheta, gphi):
    """Reconstruction of Antenna.expose_thetaphi before the Rust migration."""
    gtheta = gtheta.astype(np.float32)
    gphi = gphi.astype(np.float32)
    theta_local, phi_local = ant.cs.ae_from_global_cs(gtheta, gphi)
    cst = np.cos(theta_local)
    csp = np.cos(phi_local)
    snt = np.sin(theta_local)
    snp = np.sin(phi_local)
    kxh = snt * csp
    kyh = snt * snp
    kzh = cst
    theta_local = np.arccos(kzh)
    phi_local = np.arctan2(kyh, kxh)
    kx, ky, kz = ant.k0 * kxh, ant.k0 * kyh, ant.k0 * kzh
    gx, gy, gz = ant.local_xyz
    B = ant.amplitude * np.exp(1j * (kx * gx + ky * gy + kz * gz))
    [ex, ey, ez, hx, hy, hz] = ant.ff_pattern(theta_local, phi_local, ant.k0)
    E1 = np.array(ant.cs.in_global_basis(ex, ey, ez))
    H1 = np.array(ant.cs.in_global_basis(hx, hy, hz))
    return B * E1, B * H1


def old_expose_xyz(ant, gx, gy, gz):
    """Reconstruction of Antenna.expose_xyz before the Rust migration."""
    sx, sy, sz = ant.gxyz
    dx, dy, dz = gx - sx, gy - sy, gz - sz
    R = np.sqrt(dx**2 + dy**2 + dz**2)
    kx, ky, kz = dx / R, dy / R, dz / R
    lkx, lky, lkz = ant.cs.from_global_basis(kx, ky, kz)
    thetac = np.arccos(lkz)
    phic = np.arctan2(lky, lkx)
    B = ant.amplitude * np.exp(-1j * ant.k0 * R) / R
    [ex, ey, ez, hx, hy, hz] = ant.nf_pattern(thetac, phic, R, ant.k0)
    ex, ey, ez = ant.cs.in_global_basis(ex, ey, ez)
    hx, hy, hz = ant.cs.in_global_basis(hx, hy, hz)
    return np.array([ex, ey, ez]) * B, np.array([hx, hy, hz]) * B


def bench(name, ant, n):
    rng = np.random.default_rng(0)
    gtheta = np.arccos(rng.uniform(-1, 1, n))
    gphi = rng.uniform(-np.pi, np.pi, n)

    t_old = _time_calls(lambda: old_expose_thetaphi(ant, gtheta, gphi), REPEATS)
    t_new = _time_calls(lambda: ant.expose_thetaphi(gtheta, gphi), REPEATS)
    print(f"[{name}] expose_thetaphi  N={n:7d}  old={t_old:8.5f}s  "
          f"rust={t_new:8.5f}s  speedup={t_old/t_new:5.2f}x")

    r = rng.uniform(2, 20, n) * (C0 / FREQ)
    x = ant.gx + r * np.sin(gtheta) * np.cos(gphi)
    y = ant.gy + r * np.sin(gtheta) * np.sin(gphi)
    z = ant.gz + r * np.cos(gtheta)
    t_old = _time_calls(lambda: old_expose_xyz(ant, x, y, z), REPEATS)
    t_new = _time_calls(lambda: ant.expose_xyz(x, y, z), REPEATS)
    print(f"[{name}] expose_xyz       N={n:7d}  old={t_old:8.5f}s  "
          f"rust={t_new:8.5f}s  speedup={t_old/t_new:5.2f}x")


if __name__ == "__main__":
    dipole_ant = opt.Antenna(0, 0, 0, FREQ, opt.GCS, opt.dipole_pattern_nf, opt.dipole_pattern_ff)
    # generate_patch_pattern is native too now (Antenna._build_rust_pattern
    # recognizes its tagged closures) -- use generate_triang_pattern for a
    # genuinely non-native pattern that actually exercises the gridded
    # Interpolated path this benchmark is meant to characterize.
    triang_ant = opt.Antenna(0, 0, 0, FREQ, opt.GCS,
                              *opt.generate_triang_pattern(10.0, -20.0, 60.0, 30.0, K0))

    for n in [1_000, 20_000, 64_620]:
        bench("dipole (native)", dipole_ant, n)
        bench("triang (interpolated)", triang_ant, n)
