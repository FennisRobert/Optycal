"""Sweep `KernelConfig` (target_tile, source_tile) for the cache-blocked
`stratton_chu_xyz` / `stratton_chu_ff` kernels and report throughput
per configuration, to find good defaults empirically instead of guessing
(numba's far-field kernel hardcodes 128/128 -- this checks whether that's
actually good on this machine, and whether the same blocking helps the
near-field kernel).

Usage: python benchmarks/sweep_config.py
"""
import _cap_cores  # noqa: F401  (side effect: caps numba/Rayon thread pools; import first)

import time

import numpy as np

import optycal_kernels
from _synthetic_data import K0, _random_source, _random_near_targets, _random_far_targets

TILE_SIZES = [16, 32, 64, 128, 256, 512, 1024, 4096]
N_SOURCE = 12_000
N_TARGET = 12_000
REPEATS = 3


def _time_calls(fn, args, repeats: int) -> float:
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn(*args)
        times.append(time.perf_counter() - t0)
    return min(times)


def sweep_near():
    Ein, Hin, vis, wns = _random_source(N_SOURCE, seed=7)
    cout = _random_near_targets(N_TARGET, seed=7)
    n_pairs = N_SOURCE * N_TARGET

    print(f"\n=== stratton_chu_xyz sweep (N_src={N_SOURCE}, N_tgt={N_TARGET}) ===")
    print(f"{'target_tile':>12s} {'source_tile':>12s} {'best(s)':>10s} {'Mpairs/s':>10s}")
    best = None
    for tt in TILE_SIZES:
        for st in TILE_SIZES:
            cfg = optycal_kernels.KernelConfig(tt, st)
            args = (Ein, Hin, vis, wns, cout, K0, cfg)
            optycal_kernels.stratton_chu_xyz(*args)  # warm up (page faults, etc.)
            best_s = _time_calls(optycal_kernels.stratton_chu_xyz, args, REPEATS)
            mpairs = n_pairs / best_s / 1e6
            print(f"{tt:12d} {st:12d} {best_s:10.4f} {mpairs:10.2f}")
            if best is None or best_s < best[0]:
                best = (best_s, tt, st)
    print(f"BEST: target_tile={best[1]} source_tile={best[2]} -> {best[0]:.4f}s "
          f"({n_pairs/best[0]/1e6:.2f} Mpairs/s)")
    return best


def sweep_far():
    Ein, Hin, vis, wns = _random_source(N_SOURCE, seed=8)
    tpout = _random_far_targets(N_TARGET, seed=8)
    n_pairs = N_SOURCE * N_TARGET

    print(f"\n=== stratton_chu_ff sweep (N_src={N_SOURCE}, N_tgt={N_TARGET}) ===")
    print(f"{'target_tile':>12s} {'source_tile':>12s} {'best(s)':>10s} {'Mpairs/s':>10s}")
    best = None
    for tt in TILE_SIZES:
        for st in TILE_SIZES:
            cfg = optycal_kernels.KernelConfig(tt, st)
            args = (Ein, Hin, vis, wns, tpout, K0, cfg)
            optycal_kernels.stratton_chu_ff(*args)
            best_s = _time_calls(optycal_kernels.stratton_chu_ff, args, REPEATS)
            mpairs = n_pairs / best_s / 1e6
            print(f"{tt:12d} {st:12d} {best_s:10.4f} {mpairs:10.2f}")
            if best is None or best_s < best[0]:
                best = (best_s, tt, st)
    print(f"BEST: target_tile={best[1]} source_tile={best[2]} -> {best[0]:.4f}s "
          f"({n_pairs/best[0]/1e6:.2f} Mpairs/s)")
    return best


if __name__ == "__main__":
    sweep_near()
    sweep_far()
