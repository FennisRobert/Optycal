"""Shared synthetic-data generators for the Rust Stratton-Chu kernel
benchmarks (`bench_all.py`, `sweep_config.py`, `sweep_config_large.py`).

Generates synthetic but realistically-scaled random source-surface data
(fields + positions + area-weighted normals, a couple of wavelengths
across) and target grids (near field: points on a sphere well outside the
source region; far field: random theta/phi directions) -- no meshing,
`Antenna`/`Surface` exposure, or Fresnel-coefficient machinery involved,
so these isolate the raw kernel from everything else in the
Antenna/Surface call path.
"""
import _cap_cores  # noqa: F401  (side effect: caps numba/Rayon thread pools; import first)

import numpy as np

C0 = 299792458.0
FREQ = 3e9
K0 = 2 * np.pi * FREQ / C0
WAVELENGTH = C0 / FREQ

REPEATS = 3

# (n_source, n_target) problem sizes. Sized to be "dense enough" to reach
# steady-state throughput (not dominated by call overhead / dispatch)
# while keeping a full benchmark run in the tens-of-seconds range.
PROBLEM_SIZES = [
    (2_000, 2_000),
    (8_000, 8_000),
    (20_000, 20_000),
    (8_000, 64_620),   # source mesh vs. a full 1-degree far-field sphere (FF2D.sphere(1))
]


def _random_source(n_source: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Synthetic source-surface data: fields (Ein, Hin), point positions
    (vis) and area-weighted normals (wns), all shape (3, n_source)."""
    rng = np.random.default_rng(seed)
    extent = 2.0 * WAVELENGTH  # a source region a couple of wavelengths across

    vis = (rng.random((3, n_source)) - 0.5) * extent
    wns = rng.normal(size=(3, n_source)) * (WAVELENGTH ** 2 / n_source)

    Ein = (rng.normal(size=(3, n_source)) + 1j * rng.normal(size=(3, n_source))).astype(np.complex128)
    Hin = (rng.normal(size=(3, n_source)) + 1j * rng.normal(size=(3, n_source))).astype(np.complex128)
    return Ein, Hin, vis, wns


def _random_near_targets(n_target: int, seed: int) -> np.ndarray:
    """Target points (3, n_target) on a sphere well outside the source
    region, so the near-field kernel's 1/R, 1/R^2, 1/R^3 terms are all well
    conditioned (matches how Surface.expose_xyz is actually used)."""
    rng = np.random.default_rng(seed + 1)
    r = 20.0 * WAVELENGTH
    theta = np.arccos(rng.uniform(-1, 1, n_target))
    phi = rng.uniform(-np.pi, np.pi, n_target)
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    return np.array([x, y, z], dtype=np.float64)


def _random_far_targets(n_target: int, seed: int) -> np.ndarray:
    """Target directions (2, n_target): theta, phi."""
    rng = np.random.default_rng(seed + 2)
    theta = np.arccos(rng.uniform(-1, 1, n_target))
    phi = rng.uniform(-np.pi, np.pi, n_target)
    return np.array([theta, phi], dtype=np.float64)
