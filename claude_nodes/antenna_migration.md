# Antenna pattern evaluation: numba/Python → Rust migration log

Follow-up to `claude_nodes/kernel_optimization.md` (the Stratton-Chu surface
kernel port). This covers migrating `Antenna.expose_xyz`/`expose_thetaphi`
(and, automatically, `AntennaArray` and `expose_surface`/`normalize_power`,
which call them) to Rust: native hardcoded formulas for the common
patterns, 2D gridded interpolation for everything else.

## Design

**Pattern enum.** `optycal_kernels.AntennaPattern` (Rust:
`antenna_pattern.rs`), constructed via `.dipole()`, `.half_dipole()`, or
`.interpolated(theta_grid, phi_grid, full_matrix)`:

- `Dipole`/`HalfDipole`: exact closed-form formulas, transcribed directly
  from `antennas/patterns.py::dipole_pattern_ff`/`_nf` and
  `half_dipole_pattern_ff`/`_nf` (including the near-field's complex
  reactive-term structure, `F = 1/(i*k0*r)`). No approximation.
- `Interpolated`: bicubic-spline gridded interpolation, for every other
  pattern (parametrized generators like `generate_patch_pattern`/
  `generate_gaussian_pattern`/`generate_triang_pattern`, and arbitrary
  user-defined callables). The spline **coefficient grid is still built in
  Python**, reusing `antennas/interpolation_pattern.py`'s existing,
  unchanged `AntennaPattern.from_function(...).full_matrix(...)` (itself
  backed by `antennas/interpolator.py`'s numba Hermite-bicubic solve) --
  only the fast per-point *evaluation* (`c_interpolator_c16`'s bicubic
  polynomial, ported line-for-line) runs in Rust. Building a coefficient
  grid is a one-time O(grid size) cost per antenna (default grid: 181x361,
  1-degree resolution); reimplementing the spline *construction* in Rust
  was considered and rejected as unnecessary effort for an already-cheap,
  already-correct step.

**Grid convention, chosen deliberately, not inherited.**
`InterpolatingAntenna.__init__` (the existing numba-accelerated antenna,
`antennas/antenna.py`) builds its grid over `theta ∈ [-pi/2, pi/2]`, which
only works because the analytic pattern functions it's built from (e.g.
`dipole_pattern_ff`) happen to remap `theta = pi/2 - theta` internally --
that's not true of pattern functions in general, and tracing through how
`expose_xyz_single`/`expose_thetaphi_single` (`antennas/compiled/
antenna_single.py`) query that grid at evaluation time (`thetac =
arccos(lkz)`, which ranges over `[0, pi]`, against a grid built over
`[-pi/2, pi/2]`) surfaced what looks like a latent inconsistency in that
existing path. **Not investigated further or fixed** -- out of scope for
this migration, and `InterpolatingAntenna`/`EMergeAntenna` are untouched
(see "What was deliberately left alone" below). Instead, the new Rust path
uses its own clean, self-consistent convention throughout: sample and query
at the *physical* spherical angles the rest of the codebase already uses
everywhere else (`theta ∈ [0, pi]`, `phi ∈ [-pi, pi]`, matching e.g.
`Antenna.expose_xyz`'s own `thetac = arccos(lkz)`).

**Near-field approximation for `Interpolated`, inherited, not new.**
`InterpolatingAntenna` already evaluates its near field as "far-field
pattern shape x the standard `amplitude*exp(-ikR)/R` radial factor",
discarding any genuine `r`-dependence a hand-written `nf_pattern` might
define. Checking the actual pattern functions in `antennas/patterns.py`:
`patch_pattern_nf` and `generate_triang_pattern`'s near-field function
*already* ignore `r` in their bodies -- only `dipole`/`half_dipole` (and
the Gaussian generators) have genuine reactive near-field terms. The new
`Interpolated` variant keeps this same behavior (`eval_nf` just calls
`eval_ff`, ignoring `r`) rather than trying to build a 3D `(theta, phi, r)`
grid, which would be a much larger, unrequested undertaking for a case that
barely occurs in this codebase's own pattern library. Only `Dipole`/
`HalfDipole` get the exact reactive near field -- which matters, since
`dipole` is the default pattern and the one the closed-sphere physics test
depends on for its analytic ground truth.

**Coordinate transforms.** `CoordinateSystem.global_basis`/
`global_basis_inv` (a 3x3 rotation and its Python-computed
`np.linalg.pinv`) are passed into Rust as plain arrays; Rust only ever does
3x3 matrix-vector products, never a pseudo-inverse itself. This sidesteps
needing an SVD (or any general linear-algebra) dependency in the kernel
crate for a per-antenna, not per-point, one-time-per-call computation.

**Ported from the pure-Python `Antenna` class, not the numba
`InterpolatingAntenna` path.** Comparing `antennas/antenna.py::Antenna.
expose_thetaphi` against `antennas/compiled/antenna_single.py::
expose_thetaphi_single`, the numba version's array phase-steering term
dots a *local*-frame wavevector against the antenna's *global* position --
only correct for an unrotated antenna, and inconsistent with how
`Antenna.expose_thetaphi` itself computes the same term (local wavevector
dotted with local position, both same frame). The Rust port
(`antenna_expose.rs`) matches the pure-Python `Antenna` class's convention,
verified against it directly (see Correctness below), not the numba path's
convention.

## Wiring

`Antenna.__init__` now builds `self._rust_pattern` once (matching
`InterpolatingAntenna`'s existing "build once at construction" precedent,
including its existing limitation: reassigning `.frequency` after
construction doesn't rebuild the pattern):

```python
if nf_pattern is dipole_pattern_nf and ff_pattern is dipole_pattern_ff:
    AntennaPattern.dipole()
elif nf_pattern is half_dipole_pattern_nf and ff_pattern is half_dipole_pattern_ff:
    AntennaPattern.half_dipole()
else:
    # sample ff_pattern onto the 181x361 grid, build spline coefficients,
    # hand to Rust
    AntennaPattern.interpolated(...)
```

`Antenna.expose_xyz`/`expose_thetaphi` now call `optycal_kernels.
antenna_expose_xyz`/`antenna_expose_thetaphi` instead of the numpy-based
coordinate-transform-plus-pattern-call they used to inline. Since
`AntennaArray.add_1d_array`/`add_2d_array`/`add_2d_subarray` all construct
plain `Antenna` objects, and `Antenna.expose_surface`/`normalize_power`
both call `self.expose_xyz(...)` internally, **all of those get the Rust
path automatically** -- no changes needed in `array.py` or the
`expose_surface`/`normalize_power` bodies.

## What was deliberately left alone (scope)

- `Antenna.expose_kxyz` and `Antenna.receive_from`: call `ff_pattern`
  directly with a different calling convention (`kx,ky,kz` cartesian
  wavevector, not `theta,phi`) or bypass `expose_thetaphi` entirely. Left
  on the old pure-Python path -- `nf_pattern`/`ff_pattern` remain fully
  functional plain callables on every `Antenna`, unchanged, so nothing
  that calls them directly breaks.
- `InterpolatingAntenna`/`EMergeAntenna` (`antennas/antenna.py`,
  `antennas/compiled/antenna_single.py`): untouched, still numba-based.
  They're now functionally redundant with plain `Antenna`'s new default
  behavior (which covers native + arbitrary-interpolated patterns for
  every antenna, not just ones explicitly `.accelerate()`d), but weren't
  removed or refactored -- lower risk, and not asked for. Worth revisiting
  once the new default has some production mileage.
- `stratton_chu_xyz_surface` (surface-to-surface PO), `multilayer.py`'s
  Fresnel coefficient math: still numba, per `CLAUDE.md`'s Rust migration
  status notes -- unrelated to this antenna-specific pass.

## Correctness

`benchmarks/compare_antenna_rust.py`: Rust `antenna_expose_xyz`/
`antenna_expose_thetaphi` vs. the (unmodified) pure-Python `Antenna.
expose_xyz`/`expose_thetaphi`, same antenna (placed off-origin, in a
rotated+tilted coordinate system so the basis transforms are non-trivial),
same random points/directions:

| pattern | near field (E / H) | far field (E / H) |
|---|---:|---:|
| dipole (native) | 4.0e-16 / 2.5e-16 | 5.4e-6 / 4.9e-6 |
| half_dipole (native) | 1.3e-16 / 2.1e-16 | 4.1e-6 / 4.2e-6 |
| patch (interpolated) | 9.4e-6 / 1.2e-5 | 4.3e-4 / 6.5e-4 |

Reading these: native-pattern near field matches to machine precision
(both sides do the same complex arithmetic in f64, no approximation
anywhere). Native-pattern far field sits at ~1e-6, not machine precision --
traced to `Antenna.expose_thetaphi` itself downcasting `theta`/`phi` to
`float32` partway through (`gtheta = gtheta.astype(np.float32)`, an
existing line, unchanged); the Rust port stays in `f64` throughout, so it's
*more* accurate than the Python reference here, not less. The interpolated
(patch) pattern's larger residual (~1e-5 near field, ~1e-4 far field) is
genuine bicubic-spline interpolation error against patch's exact analytic
formula (which has a `sinc`-shaped null structure the spline approximates)
-- expected and inherent to gridded interpolation at a 181x361 grid
resolution, not a bug. Full `pytest tests/` suite (12 tests, unchanged
tolerances) passes with the new default, including the closed-sphere
dipole physics test, which now exercises the Rust dipole pattern +
Rust Stratton-Chu kernel end to end on both sides of the equivalence
comparison.

## Speed

`benchmarks/bench_antenna.py`, comparing against a reconstruction of the
pre-migration `Antenna.expose_xyz`/`expose_thetaphi` bodies (10-core cap):

| pattern | N | expose_thetaphi | expose_xyz |
|---|---:|---:|---:|
| dipole (native) | 1,000 | 1.65x | 3.63x |
| dipole (native) | 20,000 | 2.70x | 4.28x |
| dipole (native) | 64,620 | 3.15x | 4.76x |
| patch (interpolated) | 1,000 | 4.78x | 4.27x |
| patch (interpolated) | 20,000 | 1.45x | 1.80x |
| patch (interpolated) | 64,620 | 1.15x | 1.52x |

Native dipole speeds up consistently (2.7-4.8x at realistic sizes) --
avoiding per-call numba dispatch overhead plus the several intermediate
numpy array allocations the old inline implementation made. The
interpolated (patch) path speeds up less, and *less at larger N* --
numba's own JIT-compiled `patch_pattern_ff` is already reasonably fast
once vectorized over a big array, so the win here is mostly the removed
per-call Python/numpy overhead rather than a fundamentally faster
inner loop; unlike the Stratton-Chu kernels, this evaluation is O(N), not
O(N^2), so there was never going to be as much headroom.

## Bug found post-merge: redundant interpolation-grid rebuilding per array element

Reported as "`example_2_antenna_array.py` just takes forever to run" --
`example_2` builds a 20x10 = 200-element patch array via
`AntennaArray.add_2d_array(..., opt.patch_pattern_nf, opt.patch_pattern_ff)`.
`AntennaArray.add_1d_array`/`add_2d_array`/`add_2d_subarray` construct one
plain `Antenna` per element, and (per this migration) `Antenna.__init__`
now builds a Rust pattern eagerly -- for patch, that meant building a fresh
181x361-grid bicubic-spline coefficient matrix (`interpolator.py`'s numba
`_int_mat_c16`, several dense `np.linalg.solve` calls per grid row/column,
times 6 field components) **200 times**, once per element, even though all
200 elements share the exact same `patch_pattern_nf`/`patch_pattern_ff`
function objects and the same `k0`. What used to be "cheap, just store two
callables" at `Antenna.__init__` became "rebuild an expensive spline from
scratch" x200, redundantly, for identical inputs.

**Fix:** `_build_rust_pattern` (`antennas/antenna.py`) is now
`@lru_cache`d on `(nf_pattern, ff_pattern, k0)`. `AntennaPattern` objects
are stateless/read-only once built, so sharing one instance across every
antenna with matching pattern functions + k0 is safe. Verified: the
`example_2`-style 20x10 array construction went from "didn't finish in a
reasonable time" to 5.4s (with the interpolated patch path) and then 2.1s
once the native `Patch` pattern below removed the interpolation build
entirely -- the remaining 2.1s is the array's own (legitimate,
unavoidable) 200-element power-normalization sphere exposure in
`_normalize_power`, not pattern-construction overhead.

**Lesson for next time:** whenever a per-object "build once at
construction" cache is added to something that gets instantiated many
times with structurally-identical inputs (array elements, in this case),
check whether the *inputs* to that cache are actually shared before
shipping -- `InterpolatingAntenna` never hit this because nothing
constructed 200 of them at once with the same pattern; plain `Antenna`
becoming the default for every array element changed that assumption
silently.

## Native `Patch` pattern (on top of the above)

Per the maintainer's follow-up ("we should probably add a good patch
pattern equation in rust for this"): `patch_pattern_ff`/`_nf`
(`antennas/patterns.py`) and the parametrized
`generate_patch_pattern(Width, Length, k0)` are now a third native
`PatternKind::Patch { kw, kl, t_exp }` variant (`antenna_pattern.rs`),
alongside `Dipole`/`HalfDipole` -- not just a cache-fixed instance of
`Interpolated`. Transcribed exactly from both Python bodies, which turned
out to differ in more than just `kw`/`kl`: the plain functions use
`T = sqrt(|cos(theta)*cos(phi)|)` (exponent 0.5) while
`generate_patch_pattern`'s closures use exponent 0.02 -- a real formula
difference, not just a parameter, so `t_exp` is a third constructor
argument rather than assuming one fixed exponent.

**Detection.** The plain `patch_pattern_nf`/`_ff` are identity-matched
like `dipole`/`half_dipole` (`kw=kl=pi, t_exp=0.5`). `generate_patch_pattern`
returns a *fresh* closure on every call, so it can't be identity-matched --
instead it now tags its own returned closures with a
`_optycal_patch_params = (kW, kL, 0.02)` attribute (numba `@njit` dispatcher
objects accept arbitrary attribute assignment like any Python object,
confirmed directly rather than assumed), which `_build_rust_pattern` checks
for after the identity checks fail and before falling back to
`Interpolated`.

**Correctness:** direct comparison against both `patch_pattern_ff` and a
`generate_patch_pattern(...)` instance, same random (theta, phi) samples --
~4e-8 relative error for both (limited by the existing float32 downcast in
`antenna_expose_thetaphi`'s angle inputs, the same effect documented above
for dipole/half_dipole, not a new source of error). Full `pytest tests/`
suite (12 tests) still passes.

**Effect:** patch antennas (`example_2`, `example_4`, `test_reflector_basic.py`)
no longer pay any interpolation-grid construction cost at all (not even
once), and evaluate exactly rather than via spline approximation --
removing the ~1e-4/1e-5-level residual documented in the Correctness
section above for the interpolated case.

## Status / next steps

Done: `AntennaPattern` enum (native `Dipole`/`HalfDipole`/`Patch` + gridded
`Interpolated` for everything else), `antenna_expose_xyz`/
`antenna_expose_thetaphi`, wired as the default for every `Antenna` (and
therefore every `AntennaArray` element), with per-`(pattern, k0)` caching
so array construction doesn't redundantly rebuild identical interpolation
grids per element. Verified correct against the pure-Python reference and
against the full existing test suite; benchmarked speedup at multiple
sizes.

Not done: `expose_kxyz`/`receive_from` (different pattern calling
convention, out of scope here); `InterpolatingAntenna`/`EMergeAntenna`
left on the old numba path (now redundant but not broken); no
`cibuildwheel` packaging yet (same status as the kernel port). Possible
follow-up: since `Antenna`'s new default now does everything
`InterpolatingAntenna` did (and more, for native patterns), consider
deprecating/simplifying `InterpolatingAntenna` in a later pass once this
is proven out, rather than maintaining two parallel antenna-acceleration
mechanisms indefinitely.
