"""Follow-up to sweep_config.py: at N_source=12,000 the masked-source working
set (~11 arrays x 12,000 x 16 bytes =~ 2 MB) comfortably fits in cache, which
is likely why source_tile barely mattered there. Fix target_tile at the
winning small value and push N_source much higher (~150,000, ~26 MB of
source data -- should exceed typical per-core L2) to see whether source-side
cache blocking actually starts to pay off at a scale where it should.
"""
import _cap_cores  # noqa: F401  (side effect: caps numba/Rayon thread pools; import first)

import time

import optycal_kernels
from _synthetic_data import K0, _random_source, _random_near_targets

TARGET_TILE = 32
SOURCE_TILES = [64, 256, 1024, 4096, 8192, 16384, 65536, 150_000]
N_SOURCE = 150_000
N_TARGET = 3_000
REPEATS = 3


def _time_calls(fn, args, repeats: int) -> float:
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn(*args)
        times.append(time.perf_counter() - t0)
    return min(times)


def main():
    Ein, Hin, vis, wns = _random_source(N_SOURCE, seed=9)
    cout = _random_near_targets(N_TARGET, seed=9)
    n_pairs = N_SOURCE * N_TARGET

    print(f"N_src={N_SOURCE} (~{N_SOURCE*11*16/1e6:.1f} MB masked-source working set), "
          f"N_tgt={N_TARGET}, target_tile fixed at {TARGET_TILE}")
    print(f"{'source_tile':>12s} {'best(s)':>10s} {'Mpairs/s':>10s}")
    for st in SOURCE_TILES:
        cfg = optycal_kernels.KernelConfig(TARGET_TILE, st)
        args = (Ein, Hin, vis, wns, cout, K0, cfg)
        optycal_kernels.stratton_chu_xyz(*args)
        best_s = _time_calls(optycal_kernels.stratton_chu_xyz, args, REPEATS)
        print(f"{st:12d} {best_s:10.4f} {n_pairs/best_s/1e6:10.2f}")


if __name__ == "__main__":
    main()
