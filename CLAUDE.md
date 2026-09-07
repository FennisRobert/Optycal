# Optycal

A Python **Physical Optics (PO)** simulator for RF/antenna problems. Copyright Robert Fennis / EMerge Software.

## Defining design choice

Classical PO codes propagate **equivalent surface currents** (J, M). Optycal
instead propagates the **E/H fields directly** through the Stratton-Chu
integral. There is no `J`/`M` anywhere in the surface-propagation code path —
`Surface` objects store `E1/H1` and `E2/H2` (fields on the two sides of a
surface, in the direction of the surface normal), and the solvers integrate
those fields straight into an outgoing E/H field. **Keep it that way** — this
is the point of the library, not an implementation detail to "fix" during the
Rust port.

## Mental model / core objects

- **`Antenna`** (`antennas/antenna.py`) — a point source at `(x,y,z)` in a
  `CoordinateSystem`, defined by a near-field pattern function
  `nf_pattern(theta, phi, r, k0) -> (ex,ey,ez,hx,hy,hz)` and a far-field
  pattern function `ff_pattern(theta, phi, k0) -> (...)`. Antennas can
  `expose_xyz(...)`, `expose_thetaphi(...)`, or `expose_surface(surface)` —
  i.e. they *illuminate* points, angles, or a `Surface`. `expose_xyz`/
  `expose_thetaphi` run on Rust (`optycal_kernels.AntennaPattern` +
  `antenna_expose_xyz`/`antenna_expose_thetaphi`) by default — exact native
  formulas for `dipole`/`half_dipole`, gridded bicubic interpolation for
  any other `nf_pattern`/`ff_pattern` — see `claude_nodes/
  antenna_migration.md`. `nf_pattern`/`ff_pattern` remain plain, callable
  Python attributes throughout (used directly by `expose_kxyz`/
  `receive_from`, which are not Rust-backed).
  - `InterpolatingAntenna` / `EMergeAntenna` — same interface, still on the
    older numba interpolation path (`antennas/compiled/antenna_single.py`);
    now largely redundant with plain `Antenna`'s new default but left
    as-is (see the antenna migration log's "left alone" section).
  - `AntennaModel` (`antennas/antenna_model.py`, WIP) — antenna backed by a
    multi-frequency/multi-excitation far-field dataset (`FarfieldDataset`),
    e.g. imported from an EMerge FEM sweep, with msgpack (de)serialization.
- **`AntennaArray`** (`antennas/array.py`) — a collection of `Antenna`
  elements with taper/steering helpers (`add_1d_array`, `add_2d_array`,
  `set_scan_direction`, ...). Sums per-element fields; not numba-accelerated
  itself (loops in Python over antennas, each of which *is* accelerated).
- **`Surface`** (`surface.py`) — a `Mesh` + a `SurfaceRT` (Fresnel
  reflection/transmission model: `FRES_AIR`, `FRES_PEC`, or a `MultiLayer`
  dielectric stack). Field points live at mesh vertices (`polyorder=1`) or
  edge midpoints (`polyorder=2`, the default). A surface can be "exposed" by
  an `Antenna`/`AntennaArray`/another `Surface`, which writes `E1/H1` (side
  1) and `E2/H2` (side 2) via the Fresnel R/T split. It can then itself
  `expose_xyz` / `expose_thetaphi` / `expose_surface` onward — this is how PO
  chains work (source → reflector → far field, source → radome → far field,
  etc.). Optycal does **not** compute shadowing/blockage automatically —
  exposing a surface just evaluates fields from A to B regardless of what's
  in between.
- **`Mesh`** (`geo/mesh/`) — triangulated geometry. Built via
  `generate_sphere`/`generate_rectangle`/`generate_circle`, or generically
  via a `ParametricLine` swept by a `SweepFunction` (`revolve`, or a
  `Mapping` like `Mapping.parabolic_reflector(...)` composed with a sweep) —
  this is how reflectors/horns get built (see `example_4`).
- **`CoordinateSystem`** (`geo/cs.py`) — local/global frame tree, `GCS` is
  the global root. Antennas, arrays, and meshes all live in one.
- **`FarFieldSpace`/`NearFieldSpace`** (`samplespace.py`) — sample-point
  containers (`FF1D`, `FF2D`, `NF1D/2D/3D`) that a source's `.expose_ff(...)`
  writes results into.
- **Solvers** — the Stratton-Chu kernels `Surface.expose_xyz`/`expose_thetaphi`
  call. **Near-field (`Surface`→points) and far-field (`Surface`→angles) run
  on the Rust `optycal_kernels` extension** (`optycal_kernels.
  stratton_chu_xyz`/`stratton_chu_ff`, see "Dependency: `optycal_kernels`"
  below) — **the numba versions of these two have been deleted entirely**
  (not just superseded; see `claude_nodes/kernel_optimization.md`'s
  "numba deletion" section), since keeping unused `@njit` code with
  explicit type signatures around was making a cold-cache `import optycal`
  take ~50s (numba compiles explicit-signature functions eagerly at
  import, not lazily on first call). `solvers/strattonchu.py::
  stratton_chu_xyz_surface` (surface → surface, with Fresnel R/T per
  source point) is **still numba** — no Rust port exists for it, it's
  still genuinely needed. Also still numba/not yet ported:
  `antennas/compiled_functions.py` and `antennas/compiled/antenna_single.py`
  (used by `InterpolatingAntenna`/`EMergeAntenna`, now largely redundant
  with plain `Antenna`'s Rust-backed default but not removed) and the
  coefficient-matrix math in `multilayer.py`. `antennas/interpolator.py`
  (still used, for `Antenna`'s `Interpolated`/gridded custom-pattern path)
  had its `@njit` functions switched from explicit-signature (eager) to
  inferred (lazy) for the same import-time reason, without being deleted —
  it's genuinely needed, just shouldn't cost anything for antennas that
  never use it (native dipole/half-dipole/patch).
- **`Material`/`MultiLayer`** (`material.py`, `multilayer.py`) — dielectric
  material stacks and their angle-dependent Fresnel R/T curves
  (`SurfaceRT.rt_data()`), used by `Surface`. `lib.py` (CC0-licensed) is a
  big database of named material presets.
- **Display/plotting** (`viewer/`, `plot.py`) — PyVista-based 3D viewer and
  matplotlib far-field plots, all delegated to `emsutil`. **Never exercised
  in automated tests** — they open interactive windows / call `.show()` and
  will hang a test run.

## Dependency: `emsutil`

`emsutil` (EHField/EHFieldFF data containers, plotting, PyVista display
helpers) is a sibling package at `../emsutil` relative to this repo
(`/Users/robertfennis/EMerge/emsutil`), declared in `pyproject.toml` as a
`tool.uv.sources` **editable path dependency**, not a pinned PyPI release.
The current local checkout is v1.0.x — noticeably newer than the last
published PyPI version this repo used to pin to (0.8.3). The core `EHField`
/ `EHFieldFF` / `DataStructure` API that Optycal touches
(`surface.py`, `samplespace.py`, `antennas/antenna.py`, `antennas/array.py`)
has stayed compatible across that jump — confirmed by running the full
Stratton-Chu dipole-in-sphere validation against both versions with
identical results — but if you bump the `../emsutil` checkout again, re-run
the test suite before assuming nothing broke; `emsutil` is still evolving
and Optycal has needed refactors to track it before.

**Do not confuse `emsutil` with `emerge`** (a separate, much heavier FEM
solver package, and `emerge-aasds`, a git dependency of it). Those are only
needed for `examples/example_5_emerge_ant_in_radome.py` (FEM → Optycal
hand-off demo) and are **out of scope for backend work** — don't try to
install or fix them unless specifically asked to.

## Dependency: `optycal_kernels` (Rust)

`optycal_kernels` (`rust/optycal_kernels/`) is a PyO3 + Rayon extension
module, the Rust port of the Stratton-Chu near/far-field kernels — full
development history, every benchmarked iteration, and the correctness
methodology are in `claude_nodes/kernel_optimization.md`; read that before
touching kernel performance or adding kernel variants. **It's a hard
runtime dependency of `optycal` now** (`Surface.expose_xyz`/`expose_thetaphi`
import it directly and call it by default) — `import optycal` will fail
outright if it isn't built/installed. `pyproject.toml` declares it as a
`tool.uv.sources` editable path dependency (same pattern as `emsutil`), but
since it's a maturin project (not a plain Python package), building it
after a fresh clone or a Rust source change requires:

```
cd rust/optycal_kernels && maturin develop --release
```

into the `optycal` venv (see below) — plain `pip install -e .` /
`uv sync` alone won't rebuild the Rust side.

`opt.KernelConfig` (tile sizes for cache-blocking + an optional
Green's-function LUT, see the kernel log for what these do and why the
defaults are what they are) is exposed at the top level and via
`opt.GLOBAL_SETTINGS.kernel_config` (global default) or a `kernel_config=`
kwarg on `Surface.expose_xyz`/`expose_thetaphi`/`expose_ff` (per-call
override).

## Dev environment

- pyenv virtualenv **`optycal`** (Python 3.10) is the environment to use:
  `/Users/robertfennis/.pyenv/versions/3.10.19/envs/optycal/bin/python`.
- It was bootstrapped with `pip install -e .` (main deps only, skips the
  `emerge`/`emerge-aasds` dev-group deps) plus `pip install -e ../emsutil`,
  `maturin develop --release` inside `rust/optycal_kernels` (see above —
  **required**, `optycal` won't import without it), and `pytest`. If the
  venv ever looks broken/empty, or `import optycal` fails on
  `optycal_kernels`, that's the fix.
- The project also has a `uv.lock`, but note it currently reflects the old
  pinned-PyPI `emsutil==0.8.3` setup from before the path-dependency change
  above (and predates `optycal_kernels` entirely) — regenerate it
  (`uv lock`) before relying on `uv sync`, and remember `uv sync` still
  won't run `maturin develop` for you.

## Testing

- `tests/` (pytest, `--import-mode=importlib`, configured in
  `pyproject.toml`). Run with the `optycal` venv's pytest.
- **No plotting/display in tests.** Nothing may call `opt.plot_ff*`,
  `OptycalDisplay`, or `.show()` — they're interactive/blocking. Tests
  derived from `examples/*.py` reproduce the physics/data-flow of the
  example and assert on shapes/finiteness/expected symmetry instead of
  plotting the result. Example sizes are shrunk (fewer array elements,
  coarser meshes) purely for test runtime, not because larger cases are
  known to fail.
- **Why exact-value assertions are hard**: PO/Stratton-Chu results are
  numerically approximate by nature (finite mesh, vertex/edge-midpoint field
  quadrature, a magnitude-threshold cutoff (`LR` in `solvers/strattonchu.py`)
  that drops "weak" source points for speed). There's no independent
  ground truth for most configurations, so most tests can only check
  "did it run, are the numbers finite, do symmetric configurations look
  symmetric".
- **The one case with a real ground truth: the closed-sphere dipole
  equivalence-theorem test**
  (`tests/test_stratton_chu_dipole_sphere.py`). Put a `dipole_pattern_nf`
  point source at the origin, expose a closed sphere `Surface` (`FRES_AIR`,
  i.e. Tte=Ttm=1, no reflection) strictly enclosing it, then propagate the
  surface fields *back out* via `Surface.expose_xyz` (near-field
  Stratton-Chu) and `Surface.expose_thetaphi` (far-field Stratton-Chu).
  Because the dipole's near- and far-field patterns are also known
  analytically (`dipole_pattern_nf`/`_ff`, closed form, no PO approximation),
  the equivalence theorem says the two must match outside the sphere. This
  is the strongest available regression test for the Stratton-Chu kernels
  and is exactly what any Rust reimplementation of `solvers/` must keep
  reproducing.
  - Empirically (see that test's docstring/constants): the **far-field**
    kernel converges to the analytic answer very tightly (~1e-4 relative
    error) even on a coarse mesh. The **near-field** kernel plateaus around
    **~4-9% relative error** as the mesh is refined — this is a real,
    non-vanishing discretization ceiling of the vertex/edge-midpoint
    quadrature scheme (confirmed by a manual mesh-convergence sweep), not a
    bug. Near-field assertions use a generous tolerance for this reason;
    tightening it without changing the quadrature scheme will make the test
    flaky, not more correct.

## Rust migration status

Goal: port the numba kernels in `solvers/` (and the antenna-pattern
evaluation in `antennas/`) to Rust, published to PyPI via `cibuildwheel`.
The dipole-in-sphere test above is the primary correctness gate for every
step of this — a Rust kernel must reproduce the same near/far-field numbers
(within the same tolerances) as the numba implementation it replaces, not
just "be fast".

**Done:**
- `Surface`'s near/far-field Stratton-Chu kernels (see "Dependency:
  `optycal_kernels`" above).
- `Antenna`/`AntennaArray` field evaluation (`expose_xyz`/`expose_thetaphi`,
  and therefore `expose_surface`/`normalize_power` which call them) — see
  `claude_nodes/antenna_migration.md` for the full design/verification.
  `optycal_kernels.AntennaPattern` is a small enum: `Dipole`/`HalfDipole`/
  `Patch` are exact, hardcoded, closed-form (from `antennas/patterns.py`,
  including both the plain `patch_pattern_ff`/`_nf` and the parametrized
  `generate_patch_pattern(...)` — two genuinely different formulas, not
  just different parameters of one, transcribed separately);
  everything else (Gaussian generators, arbitrary user-defined
  patterns) goes through `Interpolated`, a bicubic-spline gridded
  interpolation whose coefficient grid is still built in Python (reusing
  `antennas/interpolation_pattern.py`/`interpolator.py` unchanged) with
  only the fast per-point evaluation ported to Rust. Applies automatically
  to every plain `Antenna` (including all `AntennaArray` elements) — no
  opt-in required, matching what `InterpolatingAntenna` already did for
  antennas explicitly `.accelerate()`d, just generalized to be the default.

**Not ported (still numba/Python, deliberately out of scope so far):**
`stratton_chu_xyz_surface` (surface-to-surface PO), the Fresnel
coefficient math in `multilayer.py`, `Antenna.expose_kxyz`/`receive_from`
(different pattern calling convention), and `InterpolatingAntenna`/
`EMergeAntenna` (now redundant with plain `Antenna`'s new default, but not
removed).
