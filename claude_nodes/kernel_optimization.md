# Stratton-Chu kernel: Numba → Rust port log

Goal: port the two hot Stratton-Chu integral kernels (`solvers/strattonchu.py::stratton_chu_xyz`,
`solvers/strattonchuff.py::stratton_chu_ff`) from numba to a Rust extension module
(PyO3 + Rayon), eventually distributed via `cibuildwheel`. This file logs every
benchmarked iteration so speedups are traceable to a specific, justified change
rather than vibes.

**Benchmark harness:** `benchmarks/bench_stratton_chu.py`. Generates synthetic
but realistically-scaled random source-surface data (fields + positions +
area-weighted normals, a couple of wavelengths across) and target grids (near
field: points on a sphere well outside the source region; far field: random
theta/phi directions), then times the raw kernel call directly -- no meshing,
`Antenna`/`Surface` exposure, or Fresnel-coefficient machinery involved. Both
kernels are O(N_source * N_target); results are reported as `Mpairs/s`
(millions of source-point x target-point evaluations per second) so runs at
different problem sizes are comparable. Each configuration is JIT-warmed once
(excluded from timing) then timed 3x, reporting the best (least noisy) time.

**Correctness gate:** any Rust port must keep passing
`tests/test_stratton_chu_dipole_sphere.py` (the closed-sphere dipole
equivalence-theorem test — see `CLAUDE.md`) and must numerically match the
numba kernel on identical synthetic inputs to floating-point precision (this
is checked separately from the physics test, since it catches porting bugs
that happen to still pass a generous physics tolerance).

**Machine:** Apple M3 Ultra, 28 cores. Numba: `num_threads=28`,
`threading_layer=workqueue` (the default fallback layer — `tbb`/`omp` are not
installed in this venv, so this is not necessarily numba's best case).

---

## Iteration 0 — Numba baseline

No code changes; this is the existing implementation, run through the new
benchmark harness to establish the numbers everything else is measured
against.

| kernel | N_source | N_target | pairs | best time | throughput |
|---|---:|---:|---:|---:|---:|
| `stratton_chu_xyz` (near field) | 2,000 | 2,000 | 4.0M | 0.0119 s | 336 Mpairs/s |
| `stratton_chu_xyz` (near field) | 8,000 | 8,000 | 64.0M | 0.0927 s | 691 Mpairs/s |
| `stratton_chu_xyz` (near field) | 20,000 | 20,000 | 400.0M | 0.5468 s | 732 Mpairs/s |
| `stratton_chu_xyz` (near field) | 8,000 | 64,620 | 517.0M | 1.0663 s | 485 Mpairs/s |
| `stratton_chu_ff` (far field) | 2,000 | 2,000 | 4.0M | 0.0070 s | 573 Mpairs/s |
| `stratton_chu_ff` (far field) | 8,000 | 8,000 | 64.0M | 0.0574 s | 1116 Mpairs/s |
| `stratton_chu_ff` (far field) | 20,000 | 20,000 | 400.0M | 0.2682 s | 1491 Mpairs/s |
| `stratton_chu_ff` (far field) | 8,000 | 64,620 | 517.0M | 0.3344 s | 1546 Mpairs/s |

**Observations:**
- Both kernels' throughput *increases* with problem size — expected: numba's
  `prange` parallelization overhead and JIT dispatch cost are amortized over
  more work, and at N=2,000 there simply isn't enough work per thread to
  reach steady state on 28 cores.
- `stratton_chu_ff` is consistently ~2x the throughput of `stratton_chu_xyz`
  at matched sizes. Two structural reasons, both relevant to how the Rust
  port should be shaped (see Iteration 2):
  1. `stratton_chu_ff` parallelizes over **target-point tiles**
     (`prange(N_OUT)`) with an explicit source-tile inner loop sized for
     cache reuse (`TILE_SRC = TILE_OUT = 128`) — an embarrassingly parallel
     reduction (each output tile is written by exactly one thread, once).
  2. `stratton_chu_xyz` parallelizes over **source points**
     (`prange(Nids)`), with every iteration doing a full length-`N_target`
     vectorized numpy update (`Eoutx += ...`) into the *same* shared output
     arrays. This is a cross-thread array reduction, not an embarrassingly
     parallel loop, and it also computes a fresh `1/R`, `exp(-i k0 R)`,
     `1/R²` etc. per (source, target) pair with no tiling/cache-blocking at
     all.
- This asymmetry is the single biggest known optimization target for the
  Rust near-field kernel: restructuring it to parallelize over target
  points the way the far-field kernel already does should close most of
  this 2x gap by itself, before any lower-level micro-optimization.

Next: scaffold the Rust crate, port both kernels as a faithful v1 (mirroring
numba's current parallelization strategy per kernel), confirm bit-for-bit
numeric agreement + the physics test, *then* benchmark v1 before changing
anything else. Restructuring the near-field kernel's parallelization
strategy is planned as its own, separately-benchmarked iteration so its
effect is isolated from "porting to Rust" itself.

---

## Bug found in the numba reference during correctness checking (not fixed here)

While validating the Rust port numerically against numba
(`benchmarks/compare_rust_numba.py`, elementwise comparison on identical
synthetic input, not the physics test), the far-field H output disagreed by
a factor matching `|Q| = k0/(4π)` (measured ratio ≈5.10 at the benchmark's
3 GHz, and `k0/(4π)` at 3 GHz is exactly 5.0 — same thing up to the
relative-error formula used).

Root cause, in `solvers/strattonchuff.py::stratton_chu_ff` (existing numba
code, unchanged by this port):

```python
Eout[0, :] = Q * Eoutx          # E uses the Q-scaled accumulator
...
Hout[0, :] = (ry * Eoutz - rz * Eouty) / Z0   # H uses the RAW, un-scaled accumulator
```

Physically, far-zone radiated fields satisfy `H = r_hat × E / Z0` using the
*same* E that's actually returned (i.e. the `Q`-scaled one) — that's a
general asymptotic relation, not specific to this derivation's constant.
Computing `Hout` from the pre-`Q` accumulator instead means the far-field H
(and therefore `Ez.Hx/Hy/Hz`, `normH`, and anything computing Poynting flux
from a far-field `Surface`/`Antenna` exposure, e.g. a correct
`total_radiated_power_integral`-style calculation) comes out wrong by a
factor of `1/Q` — a complex number with both a magnitude error
(`4π/k0`, i.e. frequency-dependent!) and a 90° phase error, relative to what
the physical relation predicts.

**This was not introduced by the Rust port.** `stratton_chu_xyz_v1` and
`stratton_chu_ff_v1` initially reproduced numba's exact current output
bug-for-bug for v1 numeric parity while this was still just a flagged
observation. **Fixed** (with the maintainer's explicit go-ahead) in both
`solvers/strattonchuff.py` (numba) and `stratton_chu_ff_v1`
(`rust/optycal_kernels/src/lib.rs`) by computing `Hout` from the actual
(`Q`-scaled) `Eout` instead of the raw pre-scaling accumulator, in both
places identically so they still agree with each other.

Also checked, on the maintainer's suggestion, whether the analogous bug
exists in the *near-field* kernel (`stratton_chu_xyz`), since it also has a
`Q`-like prefactor per term. It does not: `stratton_chu_xyz` computes E and
H symmetrically inside the same per-source-point accumulation (no separate
global rescale step afterwards), and its H already matched the analytic
dipole H even better than E does (~5.6e-5 max relative error vs. E's ~6.9%)
before any change here.

**Verification after the fix** (`benchmarks/compare_rust_numba.py` +
one-off scripts comparing directly against `dipole_pattern_ff`'s closed-form
H): Rust and numba far-field H still agree with each other to ~1e-14
relative error (both sides were fixed identically), and far-field H now
matches the analytic dipole H to ~4.3e-5 max relative error — the same
tight tolerance as far-field E, exactly as the physical `H = r_hat x E/Z0`
relation predicts. Before the fix this was off by a factor of `1/Q`
(frequency-dependent magnitude + 90° phase error). Added
`tests/test_stratton_chu_dipole_sphere.py::test_farfield_H_matches_analytic_dipole`
as a permanent regression test — the existing far-field test only ever
checked `.E`, which is why this shipped unnoticed. Full `pytest tests/`
suite re-run clean after the numba-side fix (11 → 12 tests, all passing).

---

## Iteration 1 — Rust v1, faithful structural port (PyO3 + Rayon)

`rust/optycal_kernels` (maturin, `cargo add numpy rayon num-complex`, `pyo3`
with the `extension-module` feature, `opt-level=3, lto=true,
codegen-units=1` release profile). `stratton_chu_xyz_v1` mirrors numba's
per-*source*-point `prange` + array-reduction strategy exactly: Rayon splits
the masked source-point range across worker threads via
`into_par_iter().fold(..).reduce(..)`, where `fold` gives each work-stealing
split its own private `(6, N_target)` accumulator (this is structurally the
same "private per-thread copy, summed at the end" that numba's automatic
array-reduction does under the hood for `arr += ...` inside `prange`).
`stratton_chu_ff_v1` mirrors numba's per-*target*-tile `prange(N_OUT)`
strategy: each output element is computed by exactly one Rayon-parallel
iterator step, no reduction needed (already numba's "good" pattern from
Iteration 0).

**Correctness:** `benchmarks/compare_rust_numba.py` — both kernels agree
with numba to ~1e-14/1e-15 relative error (floating point summation-order
noise, not a bug) on identical random inputs, once the far-field H quirk
above was reproduced rather than "fixed". Independently re-ran the
`tests/test_stratton_chu_dipole_sphere.py` physics scenario against
`stratton_chu_xyz_v1` directly — same ~6.9% max relative error against the
analytic dipole as the numba kernel, confirming this is a faithful,
physically-correct port, not just a bit-match against numba's specific
implementation.

| kernel | N_src | N_tgt | numba best | rust_v1 best | speedup |
|---|---:|---:|---:|---:|---:|
| near (`stratton_chu_xyz`) | 2,000 | 2,000 | 0.0112 s | 0.0083 s | 1.36x |
| near | 8,000 | 8,000 | 0.0845 s | 0.0929 s | 0.91x |
| near | 20,000 | 20,000 | 0.5618 s | 0.5566 s | 1.01x |
| near | 8,000 | 64,620 | 1.0946 s | 0.8250 s | 1.33x |
| far (`stratton_chu_ff`) | 2,000 | 2,000 | 0.0072 s | 0.0038 s | 1.90x |
| far | 8,000 | 8,000 | 0.0529 s | 0.0389 s | 1.36x |
| far | 20,000 | 20,000 | 0.2561 s | 0.2350 s | 1.09x |
| far | 8,000 | 64,620 | 0.3260 s | 0.3155 s | 1.03x |

(`benchmarks/bench_all.py`, best-of-3, 28-core M3 Ultra.)

**Reading these numbers:** a faithful structural port gets a modest,
inconsistent win (0.91x–1.90x) — basically "numba's JIT is already quite
good at this", not "Rust is magic". The far-field kernel wins more
consistently because it was already using the better (embarrassingly
parallel, cache-tiled) algorithm in numba; Rust mostly just shaves off
JIT-dispatch/interpreter overhead there. The near-field kernel's *loss* at
N=8,000 (0.91x) is the interesting result: it confirms the Iteration 0
hypothesis — a per-source-point array reduction (private-accumulator-per-
thread, sum at the end) has real overhead (allocating/summing a
`(6, N_target)` buffer per work-stealing split) that mostly cancels out
Rust's per-operation speed advantage. **This is exactly the case
Iteration 2 targets**: restructure the near-field kernel to parallelize over
*target* points instead, eliminating the reduction entirely.

---

## Iteration 2 — `stratton_chu_xyz_v2`: parallelize over target, not source

Restructured the near-field kernel: instead of `into_par_iter()` over
*source* points folding into private `(6, N_target)` accumulators, parallel
iterate directly over the *output* slots (`eoutx.par_iter_mut().zip(...)`)
so each target index is one independent Rayon work item that loops serially
over all (read-only, shared) masked source points and writes its own 6
scalars once. No reduction, no per-split buffer allocation — structurally
identical to what `stratton_chu_ff_v1` (and numba's far-field kernel) were
already doing. Everything else (masking, math, constants) is untouched
from v1.

**Correctness:** matches numba to ~1e-14 relative error, same as v1
(`benchmarks/compare_rust_numba.py`); independently re-verified against the
analytic dipole (E: ~6.9% max relerr, same discretization ceiling as
always; H: ~5.6e-5 max relerr).

| kernel | N_src | N_tgt | numba best | rust_v1 best | rust_v2 best | v2 vs numba | v2 vs v1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| near | 2,000 | 2,000 | 0.0110 s | 0.0073 s | 0.0061 s | 1.80x | 1.20x |
| near | 8,000 | 8,000 | 0.0857 s | 0.0924 s | 0.0834 s | 1.03x | 1.11x |
| near | 20,000 | 20,000 | 0.5234 s | 0.5514 s | 0.5149 s | 1.02x | 1.07x |
| near | 8,000 | 64,620 | 1.0654 s | 0.8280 s | 0.6606 s | 1.61x | 1.25x |

**Reading these numbers:** v2 beats v1 at every problem size (1.07x–1.25x),
confirming the reduction overhead was real and structurally worth removing.
But the win over *numba* is smaller and less consistent than hoped
(1.02x–1.80x, worse than `stratton_chu_ff_v1`'s already-decent numbers from
Iteration 1) — at N=8,000/20,000 it's barely ahead of numba at all. Two
likely reasons, both suggesting the *quadratic* O(N_source × N_target) inner
loop's cache behavior, not the parallelization axis, is now the bottleneck:

1. Every target's inner loop streams through the *entire* masked source
   array (all of `vx/vy/vz/nx_h[..]/nx_e[..]/e_dot_n/h_dot_n`, 11 arrays).
   At N_source=20,000 that's several MB per target — well past L2, so
   every target essentially re-reads all of main memory for the source
   data, 20,000 times over. Numba's JIT-vectorized inner ops likely get
   some prefetching/SIMD help that a naive scalar Rust loop doesn't
   automatically get for free.
2. There's no cache *blocking* here at all (unlike `stratton_chu_ff`'s
   numba implementation, which explicitly tiles both source and target in
   128-element blocks specifically to keep working set in L2).

This matches the maintainer's own suspicion that plain "numba vs. Rust, same
algorithm" wouldn't be a big win by itself (both ultimately compile via
LLVM), and points at cache-blocking/tiling as the next real lever — see
Iteration 3, which makes the tile sizes an explicit, externally-tunable
`KernelConfig` rather than a hardcoded guess, and benchmarks a sweep over
them instead of assuming a number.

---

## Iteration 3 — `KernelConfig`: externally-tunable cache blocking + a sweep

Added a `#[pyclass] KernelConfig { target_tile, source_tile }` (constructible
from Python: `optycal_kernels.KernelConfig(target_tile, source_tile)`,
`__repr__`-able, get/set fields) and two new kernels that use it:

- `stratton_chu_xyz_v3(..., config)`: same target-parallel structure as v2,
  but targets are processed in panels of `target_tile` (one panel = one
  Rayon work item, via `par_chunks_mut`), and for a fixed panel, sources are
  sequentially processed in blocks of `source_tile` so one source block's
  data is reused across the whole panel while (hopefully) still resident in
  cache — instead of every individual target re-streaming the *entire*
  source array once each, as v2 does.
- `stratton_chu_ff_v2(..., config)`: the same idea applied to the far-field
  kernel, replacing numba's hardcoded `TILE_SRC = TILE_OUT = 128` constants
  with the same tunable config, so numba's specific choice of 128 could be
  checked rather than assumed to be near-optimal.

The point of exposing this as a Python-constructible config object (rather
than Rust constants) is exactly what was asked for: tile sizes can be swept
and tuned from the outside — different machines, cache hierarchies, and
problem sizes can plausibly want different values — without recompiling
Rust for every trial.

**Correctness:** both new kernels match numba to ~1e-14 relative error at
default config (`benchmarks/compare_rust_numba.py`), same as v1/v2.

**Sweep methodology** (`benchmarks/sweep_config.py`,
`benchmarks/sweep_config_large.py`): grid-search `target_tile` and
`source_tile` independently over `{16, 32, 64, 128, 256, 512, 1024, 4096}`
at N_source=N_target=12,000 (masked-source working set ≈2 MB, fits
comfortably in cache), then a follow-up at N_source=150,000 (≈26 MB,
deliberately larger than a typical per-core L2/L3 slice) holding
`target_tile` fixed at a good value to isolate whether `source_tile`
matters more once the source data can no longer just sit in cache
regardless of blocking.

**Sweep results, condensed** (full grids logged by the scripts; every
`target_tile` row below is the best result over all 8 `source_tile` values
tested at that `target_tile`):

| target_tile | near-field best (Mpairs/s) | far-field best (Mpairs/s) |
|---:|---:|---:|
| 16 | **781.9** | **1723.0** |
| 32 | 770.1 | 1709.0 |
| 64 | 765.1 | 1678.2 |
| 128 | 715.2 | 1627.6 |
| 256 | 725.5 | 1605.1 |
| 512 | 716.5 | 1568.5 |
| 1024 | 407.1 | 903.5 |
| 4096 | 104.4 | 230.8 |

Large-N follow-up (N_source=150,000, `target_tile` fixed at 32): throughput
ranged only 675.6–701.7 Mpairs/s across `source_tile` from 64 to 150,000
(i.e. "no blocking at all") — a ~4% spread, smallest tile slightly *best*,
essentially flat.

**Conclusions (this is the actually useful result of Iteration 3, and it's
not the one originally expected):**

1. **`target_tile` (parallel granularity) dominates, `source_tile` (cache
   blocking) barely registers**, at every scale tested on this 28-core
   machine. Iteration 2's hypothesis — "v2 is slower than it should be
   because every target re-streams the whole source array past cache" —
   does not hold up: giving the near-field kernel explicit source-side
   blocking, even at 150,000 source points (12x more than any Iteration 2
   benchmark), gained at most ~4%, not the large win that hypothesis would
   predict.
2. What *does* matter enormously is not starving cores of parallel work:
   `target_tile=4096` is 5-8x **slower** than `target_tile=16` at matched
   problem sizes, because a 12,000-target problem only makes ~3 panels at
   that tile size — far fewer than 28 cores, so most cores sit idle. This
   is a load-balancing effect, not a cache effect.
3. The practical upshot: `stratton_chu_xyz_v3`/`stratton_chu_ff_v2` at their
   best swept config perform about the same as the untiled `v2`/`v1`
   (compare the "swept_cfg" rows below to `rust_v2`/`rust_v1` in the final
   table) — because the best `target_tile` found (16) is close to "one
   target per work item" i.e. close to what Rayon's own default
   `par_iter_mut` splitting heuristic in `v2`/`v1` was already doing. The
   `KernelConfig` API is still worth keeping (a wrong choice costs up to
   8x, so exposing it is legitimately useful for whoever tunes this on
   different hardware or vastly different problem sizes later), but on
   *this* machine at *these* problem sizes it's a safety net and a tuning
   hook, not a source of additional speedup beyond Iteration 2.
4. Default config was set to `KernelConfig(target_tile=32, source_tile=512)`
   — a safe, near-best value for both kernels (see the sweep table) rather
   than either kernel's individual optimum, since one config object is
   shared between both kernel families.

**Final comparison, all implementations, `benchmarks/bench_all.py`
(best-of-3, 28-core M3 Ultra):**

| kernel | N_src | N_tgt | numba | rust_v1 | rust_v2 | rust_v3/v2(default cfg) | rust_v3/v2(swept cfg) |
|---|---:|---:|---:|---:|---:|---:|---:|
| near | 2,000 | 2,000 | 1.00x | 1.48x | 1.88x | 1.81x | 1.84x |
| near | 8,000 | 8,000 | 1.00x | 1.01x | 1.12x | 1.08x | 1.11x |
| near | 20,000 | 20,000 | 1.00x | 0.93x | 1.01x | 1.00x | 1.02x |
| near | 8,000 | 64,620 | 1.00x | 1.29x | 1.60x | 1.60x | 1.62x |
| far | 2,000 | 2,000 | 1.00x | 2.51x | 2.53x* | — | 2.66x |
| far | 8,000 | 8,000 | 1.00x | 1.31x | 1.33x* | — | 1.33x |
| far | 20,000 | 20,000 | 1.00x | 1.11x | 1.11x* | — | 1.15x |
| far | 8,000 | 64,620 | 1.00x | 1.13x | 1.16x* | — | 1.18x |

(*far-field "rust_v2" column here is `stratton_chu_ff_v2` at default
config, since that kernel didn't exist before Iteration 3 — there's no
separate "v2 vs v3" split for far field the way there is for near field.)

**Where this leaves things:** near field tops out around 1.6-1.9x over
numba (best at the largest, most target-heavy problem; roughly a wash at
N_source=N_target=20,000, which is the one size where numba's own
throughput was already highest per Iteration 0's "throughput increases with
size" observation). Far field is a more consistent 1.1-2.7x. Given the
"both compile via LLVM" ceiling the maintainer flagged up front, further
large wins from here would more likely come from reducing redundant
per-pair work (e.g. the repeated `exp`/`sqrt`/`sin_cos` calls) or SIMD than
from more parallelization-strategy changes — see Iteration 4.

**Note on core count:** from this point on, benchmarks are capped to 10
cores (`benchmarks/_cap_cores.py`, imported first by every script in this
directory) per an explicit request to stop the benchmark suite from hogging
this shared 28-core machine. Iterations 0-3 above were measured at the
default (all 28 cores); iteration numbers from here on are not directly
comparable to those in absolute Mpairs/s, only in relative speedup-vs-numba
within the same run.

---

## Aside — does large `R` actually cause a `sin(k0*R)`/`cos(k0*R)` precision problem?

Before optimizing the Green's-function evaluation (`G = exp(-i*k0*R)/R`),
checked a specific concern raised by the maintainer: for `R` many
wavelengths away, does the `f64` phase `k0*R` get large enough that
`sin`/`cos` of it lose meaningful precision, and would a distance-based
"zone" dispatch (different kernels/precision strategies for near vs. very
far `R`) be needed for correctness?

`benchmarks/precision_check.py`: compared `f64` `sin(k0*R)`/`cos(k0*R)`
against an mpmath 50-digit reference, at a representative mm-wave frequency
(300 GHz, `k0`≈6288 rad/m):

| R | k0·R (rad) | \|sin error\| | \|cos error\| |
|---:|---:|---:|---:|
| 1 mm | 6.3 | 1.7e-16 | 0 |
| 1 m | 6,288 | 0 | 0 |
| 1 km | 6.29e6 | 5.2e-11 | 2.3e-10 |
| 100 km | 6.29e8 | 2.0e-8 | 7.8e-9 |
| 1,000 km | 6.29e9 | 1.8e-7 | 1.2e-7 |
| 1e9 m (~Earth-Moon) | 6.29e12 | 1.8e-4 | 5.0e-5 |
| 1e12 m (~6,700 AU) | 6.29e15 | 8.9e-2 | 5.5e-2 |

**Verdict: not a real problem for anything this library models.** Error
stays at or below 1e-7 out to 1,000 km — utterly negligible next to the
near-field kernel's ~5-9% discretization ceiling (Iteration 0/2) — and only
crosses the -60dB/0.1% threshold the maintainer separately mentioned (as a
target for the *interpolation* idea below) around R≈1e9 m, i.e. roughly the
Earth-Moon distance. No PO antenna/reflector/radome problem this codebase
targets gets anywhere near that. **No distance-zone dispatch was
implemented** — it would be solving a problem that doesn't occur at any
physically sensible scale, at the cost of extra branching/complexity in the
hot loop. (The error growth pattern above, roughly linear in `k0*R` until
argument reduction itself starts to degrade far past that, matches the
expected `~k0*R*eps` bound for correctly-rounded range reduction, which is
what Rust's `f64::sin_cos`/platform libm already does — there's no
"naive modulo" bug to fix here.)

---

## Iteration 4 — hoisted redundant per-pair work; tried and rejected a LUT Green's function

Two changes, benchmarked separately so each one's effect is isolated.

**4a. Hoisted `ie1*N_x_H[j]` / `ih1*N_x_E[j]` out of the inner loop.**
`ie1`/`ih1` are constants (depend only on `k0`); `N_x_H[j]`/`N_x_E[j]`
depend only on the source point `j`. Neither depends on the target. numba's
original kernel computes `ie1*NxHx[j]` etc. once per source point, because
its outer loop *is* the source loop (Iteration 0). Iteration 2 flipped the
near-field kernel to parallelize over *target* instead — which,
unnoticed at the time, meant these same per-source products were being
recomputed on **every single (source, target) pair** instead of once per
source: an O(N_target) blow-up of work that should have been O(1) per
source. Fixed by precomputing `ie1_nxh`/`ih1_nxe` arrays (size N_source)
once before the parallel loop. Also replaced the generic `Complex::exp`
Green's-function evaluation with a direct `sin_cos`-based construction
(skips an internal wasted `exp(0.0)` the generic complex-exponential path
computes) and decomposed `dG = 1/R*(1/R + i*k0)` into its real/imaginary
parts directly instead of via a complex multiply.

**Correctness:** matches numba to ~1e-14 relative error, same as every
prior iteration; re-verified against the analytic dipole (E: ~6.9% max
relerr, H: ~5.6e-5 max relerr — both exactly the values from every earlier
iteration, i.e. this change is a pure speed optimization with zero effect
on the actual numerics, as it should be).

| N_src | N_tgt | numba best | pre-4a (Iteration 2) best | post-4a best | 4a vs. pre-4a |
|---:|---:|---:|---:|---:|---:|
| 8,000 | 8,000 | 0.0859 s | 0.0827 s | 0.0811 s | 1.02x |
| 20,000 | 20,000 | 0.5231 s | 0.5157 s | 0.5123 s | 1.01x |
| 8,000 | 64,620 | 1.0535 s | 0.6666 s | 0.6615 s | 1.01x |

(28-core measurements, taken right before the 10-core cap below.) A real
but modest win (~1-2%) — the redundant work per pair was cheap (one
complex multiply), so eliminating N_target-fold repetition of a cheap
operation is a small effect in absolute terms, even though the *shape* of
the bug (an O(1)-per-source cost turned into an O(N_target)-per-source
cost by a parallelization-axis change) was worth catching on principle:
it would have gotten worse, not better, at larger N_target.

**4b. Tried an interpolated ("LUT") Green's-function phase evaluation —
rejected, but properly measured first.** The maintainer's proposal: since
`R` is real, `exp(-i*k0*R)` reduces to `sin`/`cos` of `k0*R mod 2*pi`, a
periodic function of one real variable — approximate it with a small
precomputed table + linear interpolation instead of calling `sin_cos`,
targeting "engineering" (~-60 dB) rather than machine precision. This is
exactly what a commented-out block in the original numba
`solvers/strattonchu.py` (`EXPIARRY`/`EXPOARRY`, `np.interp`) already tried
and apparently abandoned — worth understanding why before assuming a Rust
version would fare better.

Implementation: `greens::build_sincos_lut(n)` / `greens::lut_sin_cos`, a
*uniform* table (so lookup is direct index arithmetic, `O(1)`, not
`np.interp`'s generic binary search — plausibly why the numba version
wasn't a clear win), wired in via `KernelConfig::greens_lut_size` (`0` =
disabled/exact).

*Accuracy* (`benchmarks/lut_accuracy_check.py`, via the exact table code,
not a reimplementation — confirms the hand/Python-side sizing estimate):

| table size (n) | worst-case error | dB |
|---:|---:|---:|
| 16 | 1.9e-2 | -34.5 |
| 32 | 4.8e-3 | -46.4 |
| 64 | 1.2e-3 | -58.4 |
| **128** | **3.0e-4** | **-70.4** |
| 256 | 7.5e-5 | -82.5 |
| 512 | 1.9e-5 | -94.5 |

n=128 was chosen as the config example throughout: comfortably past -60 dB
with margin, and the resulting *field*-level error (not just the raw
sin/cos error) was confirmed at -72.5 dB against exact numba output, and
made **zero measurable difference** to the dipole-in-sphere physics test
(0.06890 vs. 0.06889 max relative error, exact vs. LUT) — exactly as
predicted, since -70 dB is nothing next to a ~7% discretization floor.

*Speed* (`benchmarks/bench_all.py`, 10-core cap — see note below):

| N_src | N_tgt | exact (no LUT) best | LUT (n=128) best | LUT vs. exact |
|---:|---:|---:|---:|---:|
| 2,000 | 2,000 | 0.0106 s | 0.0123 s | 0.86x (slower) |
| 8,000 | 8,000 | 0.1617 s | 0.1897 s | 0.85x (slower) |
| 20,000 | 20,000 | 1.0065 s | 1.1863 s | 0.85x (slower) |
| 8,000 | 64,620 | 1.2983 s | 1.5366 s | 0.84x (slower) |

**Verdict: the LUT is consistently ~15% *slower*, not faster.** The
overhead of the modulo, index computation, and interpolation arithmetic
outweighs the cost of the `sin_cos` call it's replacing — on this hardware
(Apple Silicon), the platform libm's `sin_cos` is evidently already very
well optimized, and a scalar table lookup can't beat it. This matches the
historical evidence already sitting in the codebase (the abandoned numba
`np.interp` attempt) closely enough that it's probably the same underlying
reason, not a coincidence.

**Kept anyway, disabled by default.** `KernelConfig::greens_lut_size`
stays in the API (default `0` = exact) because: it's correct and tested,
costs nothing when disabled, and remains a reasonable thing to try again if
this ever targets a platform/compiler where scalar `sin_cos` is *not*
well-optimized, or if it's ever extended to a properly SIMD-batched
lookup (computing 4-8 lanes of sin/cos at once) rather than one scalar
lookup per pair — that's a materially different and more promising
technique than what was tried here, not attempted in this session (would
need either nightly Rust `portable_simd` or a platform-specific/external
vectorized-math dependency, both adding real complexity/portability risk
for a `cibuildwheel`-distributed crate, and there was no evidence yet that
it was needed).

**Answering the maintainer's question directly:** for this piece
specifically (the Green's-function phase term), on this hardware,
literally evaluating `exp(-i*k0*R)` via `sin_cos` already appears to be
close to as good as a scalar implementation gets. The bigger, *proven*
levers so far have been parallelization strategy (Iteration 2) and
eliminating genuinely redundant work (4a above), not the transcendental
function itself.

---

## Cleanup — collapsed to one implementation per kernel, split into modules

With the experimentation above pointing at a clear winner for each kernel
(target-parallel + hoisted invariants + tunable tiling, LUT available but
off), the crate was restructured from one 1057-line `lib.rs` containing five
near-field generations (`_v1`..`_v5`) and two far-field generations
(`_v1`..`_v2`) down to **one implementation per kernel**, split across
focused files:

- `src/lib.rs` — module declarations + `#[pymodule]` registration only.
- `src/config.rs` — `KernelConfig`.
- `src/sources.rs` — physical constants, `MaskedSources`, `build_masked_sources`.
- `src/greens.rs` — the LUT Green's-function code (4b above) + its
  diagnostic accuracy hook.
- `src/near_field.rs` — `stratton_chu_xyz` (the former `_v5`, i.e. v2's
  parallelization + v3's tunable tiling + 4a's hoisting + 4b's optional LUT).
- `src/far_field.rs` — `stratton_chu_ff` (the former `_v2`).

The intermediate `_v1`..`_v4` near-field and `_v1` far-field functions were
**deleted from the source tree**, not kept behind flags or comments — their
reasoning and measurements are fully preserved above, and dead alternative
implementations left lying around a "final" version invite bit-rot (they'd
silently stop compiling or stop matching numba the next time something
shared changes) for no benefit, since nothing outside this crate's own
history needs them. `benchmarks/compare_rust_numba.py`, `bench_all.py`, and
`sweep_config*.py` were updated to call the renamed `stratton_chu_xyz`/
`stratton_chu_ff` (no more `_v1`/`_v2`/... suffixes anywhere).

Re-verified after the split: `benchmarks/compare_rust_numba.py` still
matches numba to ~1e-14 (default config), the full `pytest tests/` suite
(12 tests) still passes, and `bench_all.py`'s numbers are unchanged within
run-to-run noise from before the split — this was a pure reorganization,
not a behavior change.

**Note on core count (still in effect):** all measurements from Iteration 4
onward use the 10-core cap (`benchmarks/_cap_cores.py`) mentioned earlier;
Iterations 0-3's tables used the default all-28-cores configuration and
are not directly comparable in absolute Mpairs/s.

## Current final comparison (10-core cap, `benchmarks/bench_all.py`)

| kernel | N_src | N_tgt | numba | rust(default_cfg) | rust(swept_cfg) |
|---|---:|---:|---:|---:|---:|
| near | 2,000 | 2,000 | 1.00x | 1.17x | 1.18x |
| near | 8,000 | 8,000 | 1.00x | 1.04x | 1.04x |
| near | 20,000 | 20,000 | 1.00x | 1.04x | 1.05x |
| near | 8,000 | 64,620 | 1.00x | 1.06x | 1.05x |
| far | 2,000 | 2,000 | 1.00x | 1.43x | 1.54x |
| far | 8,000 | 8,000 | 1.00x | 1.15x | 1.15x |
| far | 20,000 | 20,000 | 1.00x | 1.06x | 1.06x |
| far | 8,000 | 64,620 | 1.00x | 1.05x | 1.05x |

At 10 cores the near-field speedup over numba is smaller and flatter
(1.04-1.18x) than the 28-core numbers in Iteration 2/3 (up to 1.6-1.9x) —
core count changes the numba-vs-Rayon comparison, not just the absolute
throughput, presumably because numba's `workqueue` threading layer and
Rayon's work-stealing scheduler have different overhead-vs-core-count
curves. Far field remains a more consistent 1.05-1.54x win. Both kernels
are correctness-verified (machine precision vs. numba, and against the
closed-form dipole for physical validity) at every point in this table.

## Iteration 5 — hyper-optimization pass: hoisting, FMA, division caching

A systematic line-by-line audit of both kernels' hot loops for classic
flop-reduction techniques: replacing multiply-then-add/subtract with fused
multiply-add (FMA) wherever one operand is a real scalar, hoisting anything
per-target-constant that was being recomputed per (source, target) pair,
and replacing division-by-a-constant with a precomputed reciprocal
multiply. New shared module `src/cmath.rs` implements the FMA-fused
primitives (`cmul`, `scaled_diff`, `triple_term`, `triple_sum`, `dot3`) used
throughout; see that file's doc comments for the exact instruction-count
accounting per helper.

**5a. Far-field `Z0` hoist (the big structural find).** The inner loop
computed `Z0 * (ry*NxH_z[j] - rz*NxH_y[j])` (and two analogous terms) fresh
on *every* (source, target) pair — but `Z0` is a global constant and
`rx`/`ry`/`rz` are per-*target* constants, loop-invariant across the entire
source loop for a fixed target. Precomputing `z0_rx = Z0*rx` etc. once per
target and folding it directly into the cross-product terms (`z0_ry*nxhz -
z0_rz*nxhy` instead of `Z0*(ry*nxhz - rz*nxhy)`) removes 2 real multiplies
per output component per pair — 6 real multiplies per (source, target)
pair, i.e. an O(N_source * N_target) reduction in the far-field kernel's
hottest loop. Structurally the same *shape* of bug as Iteration 4's
`ie1_nxh`/`ih1_nxe` hoist in the near-field kernel (something target- or
source-only being redone at pair granularity after a restructuring), just
not caught the first time because Iteration 3/4 focused on the near-field
kernel.

**5b. FMA fusion**, applied everywhere a multiply-then-add/subtract has a
real-scalar operand:
- The complex multiply operator itself (`cmul`): `num_complex::Complex`'s
  `Mul` is the generic 4-multiply/2-add formula with no FMA; `cmul` does
  the same computation as 2 multiplies + 2 fused multiply-adds (4
  instructions instead of 6).
- `R^2 = rx^2+ry^2+rz^2` (near field) and the phase dot product
  `kx*vx+ky*vy+kz*vz` (far field): 3-term real dot products, `dot3` turns
  each into 1 multiply + 2 FMAs instead of 3 multiplies + 2 adds.
- Every `N x V`-style cross-product term (`scaled_diff` for the 2-term
  far-field case, `triple_term` for the near-field kernel's 3-term
  cross-product-plus-radial-term, `triple_sum` for the sign-free `E.N`/`H.N`
  dot products in `sources.rs`): each collapses from `2n` multiplies +
  `n` add/subtracts (for an n-term combination) to `n` multiplies + `n+1`
  FMAs, matching the pattern above.

**5c. Redundant division caching**, exactly the pattern the maintainer
named directly:
- `far_field.rs`: `Z0_INV: f64 = 1.0 / Z0` — a `const`, evaluated at
  compile time — replaces the `/Z0` in the final H assembly with a
  multiply. This loop is only O(N_target), so the runtime effect is
  negligible, but it's the textbook version of the pattern and costs
  nothing to fix.
- `greens.rs` (the LUT path, off by default — see Iteration 4): the
  original `lut_sin_cos(lut, theta)` divided by `TAU` **inside the
  function**, i.e. once per (source, target) pair whenever the LUT path is
  active, even though the LUT's size (and therefore `n/TAU`) doesn't change
  during a kernel call. Restructured into a `SinCosLut` struct that
  precomputes `n_over_tau` once at construction; `SinCosLut::eval` now does
  one multiply, no division, per lookup.

**Correctness:** `benchmarks/compare_rust_numba.py` — near field now
matches numba to ~9e-15 (previously ~1e-14), far field to ~3.2e-15
(previously ~3.5e-15). Accuracy *improved slightly*, not just "stayed
equal" — expected, since FMA removes an intermediate rounding step per
fused operation rather than adding one. Full `pytest tests/` suite (12
tests, including the analytic-dipole physics test) still passes unchanged.
LUT accuracy is unaffected by 5c's division-caching fix (same table values,
same interpolation math, just computed with one fewer division per call) —
still ~-70 dB at n=128, still harmless to the physics test.

**Speed** (`benchmarks/bench_all.py`, 10-core cap). Far field's two changes
were also measured in isolation (temporarily reverting 5a while keeping 5b,
rebuilding, benchmarking, then restoring) to attribute the combined gain:

| N_src | N_tgt | numba | Iter. 4 (baseline) | 5b only (FMA) | 5a+5b (hoist+FMA) |
|---:|---:|---:|---:|---:|---:|
| 2,000 | 2,000 | 1.00x | 1.43x | 1.55x | 1.56x |
| 8,000 | 8,000 | 1.00x | 1.15x | 1.23x | 1.27x |
| 20,000 | 20,000 | 1.00x | 1.06x | 1.12x | 1.16x |
| 8,000 | 64,620 | 1.00x | 1.05x | 1.11x | 1.14x |

Reading this: FMA alone (5b) accounts for most of the jump at the smallest
problem size; the `Z0` hoist (5a) adds a further increment that *grows*
with problem size (+0.01 at N=2,000 vs. +0.04 at N=8,000-64,620x8,000) —
consistent with it removing O(N_source) work per target, so its relative
contribution should scale with how much source-side work there is to
amortize against. Near field (FMA only available, no analogous hoist
existed) moved more modestly, as expected:

| N_src | N_tgt | numba | Iteration 4 | Iteration 5 |
|---:|---:|---:|---:|---:|
| 2,000 | 2,000 | 1.00x | 1.17x | 1.22x |
| 8,000 | 8,000 | 1.00x | 1.04x | 1.08x |
| 20,000 | 20,000 | 1.00x | 1.04x | 1.08x |
| 8,000 | 64,620 | 1.00x | 1.06x | 1.11x |

The LUT path (still off by default) also benefited from 5c's division fix
enough to be worth re-noting: it went from a clear ~15% *slower* than exact
`sin_cos` (Iteration 4) to roughly break-even (0.96x-1.10x across sizes) —
the redundant division really had been a meaningful fraction of the LUT
path's own overhead. Still not a clear win, so the default stays off, but
the gap that made it an easy "reject" before is now much smaller.

**Portability caveat, noted but not resolved here:** `f64::mul_add`
compiles to a genuine hardware FMA instruction on this machine (Apple M3
Ultra / AArch64, where NEON mandates FMA) and is expected to do the same on
any AArch64 target and on x86_64 builds compiled with `target-feature=+fma`
(implied by `-C target-cpu=x86-64-v3` or newer). A *generic* x86_64 build
without that flag (plausible default for a `cibuildwheel` wheel aiming at
the widest compatibility) will still produce **correct** results — `mul_add`
always computes the right answer — but falls back to a software `fma()`
call instead of a hardware instruction, so 5b's measured speedup may not
fully carry over there. Worth re-benchmarking on an x86_64 CI runner (or
setting an explicit `target-feature=+fma` in the wheel build config) before
assuming these numbers generalize past this machine.

---

## Integration — Rust kernel is now the default in `Surface`

`Surface.expose_xyz`/`expose_thetaphi`/`expose_ff` (`src/optycal/surface.py`)
now call `optycal_kernels.stratton_chu_xyz`/`stratton_chu_ff` directly,
replacing the numba calls that used to live there. At this point (see
"numba deletion" below) the numba versions of these two kernels no longer
exist in the tree at all — they're not just unused, they're gone.
`stratton_chu_xyz_surface` (surface-to-surface, Fresnel R/T per source
point) is **not** part of this port, has no Rust equivalent, and still
runs on numba (kept in `solvers/strattonchu.py`).

Added `opt.KernelConfig` (re-exported from `optycal_kernels`) and
`opt.GLOBAL_SETTINGS.kernel_config` (default `KernelConfig()`, i.e. the
tuned defaults from Iteration 3/5) as the module-level API the maintainer
asked for, plus an optional `kernel_config=` kwarg on
`expose_xyz`/`expose_thetaphi`/`expose_ff` for a per-call override without
touching global state. `optycal_kernels` is now a declared (and enforced —
`surface.py` imports it unconditionally) dependency of `optycal`; see
`CLAUDE.md`'s "Dependency: `optycal_kernels`" section for the
`maturin develop` build step this requires.

Verified: full `pytest tests/` suite (12 tests, including the
closed-sphere dipole physics test — which now genuinely exercises the Rust
kernel end-to-end through `Surface`, not just the standalone benchmark
scripts) passes; per-call and global `KernelConfig` overrides checked
manually (default vs. a custom tile-size config on identical input give
bit-identical output; a global LUT-enabled override runs and produces
finite output through `expose_ff`).

---

## Bug found post-integration: cold-cache `import optycal` "hangs" — and the numba deletion that followed

Reported as "importing optycal takes forever". `python -X importtime -c
"import optycal"` traced it: `optycal.antennas.interpolator` (~19.4s),
`optycal.solvers.strattonchu` (~18.5s), and `optycal.solvers.strattonchuff`
(~4.5s) were the hot spots — ~42s of the ~52s total. Root cause: all three
modules define numba `@njit` functions with an **explicit type signature**
(e.g. `@njit(TupleType(...)(c16[:,:], ...), parallel=True, ...)`), which
makes numba compile them **eagerly at import time** rather than lazily on
first call. `strattonchu.py`/`strattonchuff.py` in particular have several
`parallel=True` functions, and numba's parallel accelerator is slow to
compile. A *second* `import optycal` in the same environment took 1.4s
(numba's `cache=True` had already written the compiled artifacts to disk)
— so this was specifically a cold-cache, first-import cost, but a ~50
second "hang" with zero progress indication is a real problem regardless
of whether it's one-time.

Two of those three modules were, by this point, no longer on any live code
path — `Surface` had already been switched to call `optycal_kernels.
stratton_chu_xyz`/`stratton_chu_ff` (the Integration section above), so the
numba `stratton_chu_xyz`/`stratton_chu_ff` were being eagerly compiled at
every cold import purely to sit unused. With the maintainer's go-ahead
("remove all unused numba code entirely, we're on a separate branch"),
**deleted** rather than just made lazy:

- `solvers/strattonchu.py::stratton_chu_xyz` — removed. `stratton_chu_xyz_surface`
  (still genuinely used, no Rust replacement) stays, unchanged, in the same
  file, still eagerly compiled (it's actually needed, so paying its
  compile cost at import is a reasonable tradeoff vs. surprising latency on
  first real use).
- `solvers/strattonchuff.py` — the file only ever contained `stratton_chu_ff`
  (used) plus an already-commented-out earlier attempt at the same
  function; deleting the live function left nothing but dead code, so the
  **whole file was deleted** and `solvers/__init__.py` now only exports
  `stratton_chu_xyz_surface`.
- `benchmarks/bench_stratton_chu.py` and `benchmarks/compare_rust_numba.py`
  existed specifically to benchmark/compare against the now-deleted numba
  functions — **deleted**. Their historical numbers are preserved in
  Iterations 0-5 above; the ongoing, numba-independent correctness gate is
  `tests/test_stratton_chu_dipole_sphere.py` (validates Rust directly
  against the closed-form analytic dipole, no numba involved). Their
  shared synthetic-data generators (`_random_source`, `_random_near_targets`,
  `_random_far_targets`, `PROBLEM_SIZES`, `K0`) moved to a new
  `benchmarks/_synthetic_data.py` so `bench_all.py`/`sweep_config*.py`
  (which don't depend on numba) keep working. `bench_all.py` itself was
  rewritten to compare `KernelConfig` settings against each other only, not
  against a numba baseline that no longer exists.

`antennas/interpolator.py`'s 12 explicit-signature `@njit` functions were
**not** deleted — `interpolation_pattern.py::AntennaPattern.from_function`/
`compute_interpolator_matrix` (which they implement) is still genuinely
needed, for the `Interpolated` gridded-antenna-pattern path (see
`claude_nodes/antenna_migration.md`). Instead, their explicit signatures
were removed, switching them from eager (compile at import) to lazy
(compile on first actual call, same `cache=True` persistence). Net effect:
a user who only ever uses native patterns (dipole/half-dipole/patch) now
pays *zero* cost for this machinery, ever — it only compiles if an
antenna actually needs gridded interpolation (Gaussian/triang generators,
or a genuinely custom user-defined pattern).

**Verified:** full `pytest tests/` suite (12 tests) passes; cold-ish
`import optycal` (no test in the current suite happens to exercise the
now-lazy `Interpolated` path, since patch became native in the same pass —
see the antenna migration log) now completes in ~1.2s.

## Status / next steps

Done: numba baseline, faithful Rust port, parallelization-axis fix,
tunable cache-blocking with an empirical sweep, a real redundant-work fix,
a properly-measured (and rejected, then partially rehabilitated)
Green's-function LUT experiment, a numba correctness bug found and fixed in
both implementations (both since deleted, see below), a clean
one-implementation-per-kernel module layout, a hyper-optimization pass
(hoisting + FMA + division caching) with each change individually
attributed, integration as `Surface`'s default (with a `KernelConfig`
API), and finally **deletion** of the superseded numba `stratton_chu_xyz`/
`stratton_chu_ff` (and their benchmark/comparison scripts) now that the
Rust port is the only implementation and has its own numba-independent
correctness gate (the physics test). `stratton_chu_xyz_surface`
(surface-to-surface) remains numba — no Rust port exists for it.

Not done: no `cibuildwheel` packaging configuration yet — a natural next
step once the maintainer wants to actually ship this. Possible further
speed work, in rough order of expected payoff based on everything measured
so far: SIMD-batched (not scalar) trig evaluation; verifying the FMA win
holds on a generic x86_64 build (see the portability caveat above) and
setting `target-feature=+fma` in the wheel build if not; re-checking
whether a properly warmed-up numba `tbb`/`omp` threading layer would have
changed the old numba-vs-Rust comparison (moot now that the numba code is
gone, but relevant if `stratton_chu_xyz_surface` is ever ported too);
revisiting per-core-count scaling now that it's known to matter; a GPU port
(discussed with the maintainer but not attempted — see the note on mixed
f64/f32 precision for phase range-reduction if that's ever pursued);
porting `stratton_chu_xyz_surface` itself, the one remaining numba
Stratton-Chu kernel.
