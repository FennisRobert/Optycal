"""Compare Rust Stratton-Chu kernel `KernelConfig` settings against each
other on the same problem sizes.

Historical note: this used to also compare against the numba
implementation these kernels replaced (`solvers/strattonchu.py::
stratton_chu_xyz`, `solvers/strattonchuff.py::stratton_chu_ff`) -- that
numba code has since been deleted (the Rust port is the only
implementation now; see claude_nodes/kernel_optimization.md for the full
numba-vs-Rust benchmark history and numbers, which remain valid as a
historical record even though the numba code no longer exists in the
tree). The ongoing, numba-independent correctness gate is
`tests/test_stratton_chu_dipole_sphere.py` (the closed-sphere
analytic-dipole equivalence test).

Usage:
    python benchmarks/bench_all.py
"""
from __future__ import annotations

import time

import optycal_kernels

from _synthetic_data import (
    K0,
    PROBLEM_SIZES,
    REPEATS,
    _random_source,
    _random_near_targets,
    _random_far_targets,
)

_DEFAULT_CFG = optycal_kernels.KernelConfig()  # tuned defaults, see KernelConfig docstring
_BEST_NEAR_CFG = optycal_kernels.KernelConfig(16, 64)
_BEST_FAR_CFG = optycal_kernels.KernelConfig(16, 4096)
_LUT128_CFG = optycal_kernels.KernelConfig(32, 512, 128)

NEAR_FIELD_IMPLS = {
    "rust(default_cfg)": lambda *a: optycal_kernels.stratton_chu_xyz(*a, _DEFAULT_CFG),
    "rust(swept_cfg)": lambda *a: optycal_kernels.stratton_chu_xyz(*a, _BEST_NEAR_CFG),
    "rust(lut=128)": lambda *a: optycal_kernels.stratton_chu_xyz(*a, _LUT128_CFG),
}
FAR_FIELD_IMPLS = {
    "rust(default_cfg)": lambda *a: optycal_kernels.stratton_chu_ff(*a, _DEFAULT_CFG),
    "rust(swept_cfg)": lambda *a: optycal_kernels.stratton_chu_ff(*a, _BEST_FAR_CFG),
}


def _time_calls(fn, args, repeats: int) -> tuple[float, float]:
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn(*args)
        times.append(time.perf_counter() - t0)
    times.sort()
    return times[len(times) // 2], times[0]


def run():
    print(f"{'kernel':10s} {'impl':20s} {'N_src':>8s} {'N_tgt':>8s} {'pairs':>14s} "
          f"{'best(s)':>10s} {'median(s)':>10s} {'Mpairs/s':>10s}")

    for n_source, n_target in PROBLEM_SIZES:
        Ein, Hin, vis, wns = _random_source(n_source, seed=1)
        cout = _random_near_targets(n_target, seed=1)
        for name, fn in NEAR_FIELD_IMPLS.items():
            args = (Ein, Hin, vis, wns, cout, K0)
            fn(*args)  # warm up
            median_s, best_s = _time_calls(fn, args, REPEATS)
            n_pairs = n_source * n_target
            print(f"{'near':10s} {name:20s} {n_source:8d} {n_target:8d} {n_pairs:14,d} "
                  f"{best_s:10.4f} {median_s:10.4f} {n_pairs/best_s/1e6:10.2f}")

    for n_source, n_target in PROBLEM_SIZES:
        Ein, Hin, vis, wns = _random_source(n_source, seed=2)
        tpout = _random_far_targets(n_target, seed=2)
        for name, fn in FAR_FIELD_IMPLS.items():
            args = (Ein, Hin, vis, wns, tpout, K0)
            fn(*args)  # warm up
            median_s, best_s = _time_calls(fn, args, REPEATS)
            n_pairs = n_source * n_target
            print(f"{'far':10s} {name:20s} {n_source:8d} {n_target:8d} {n_pairs:14,d} "
                  f"{best_s:10.4f} {median_s:10.4f} {n_pairs/best_s/1e6:10.2f}")


if __name__ == "__main__":
    run()
