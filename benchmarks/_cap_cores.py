"""Caps numba/Rayon thread pools to a fixed core count for benchmark runs on
this machine (an otherwise-in-use 28-core workstation), so benchmarking
doesn't hog every core. Import this before anything that touches numba or
`optycal_kernels` (both read their thread-count env var lazily, on first
parallel dispatch/first global-pool use -- so this only works if it runs
before those first uses, not merely before the benchmark timing starts).

An already-set `NUMBA_NUM_THREADS`/`RAYON_NUM_THREADS` in the environment is
left alone (`setdefault`), so `NUMBA_NUM_THREADS=4 python bench_all.py` still
overrides this default.
"""
import os

CAP = "10"
os.environ.setdefault("NUMBA_NUM_THREADS", CAP)
os.environ.setdefault("RAYON_NUM_THREADS", CAP)
