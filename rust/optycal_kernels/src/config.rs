//! `KernelConfig`: externally-tunable performance/accuracy knobs for the
//! Stratton-Chu kernels, exposed to Python so they can be swept and tuned
//! from the outside without recompiling Rust for every trial. See
//! `claude_nodes/kernel_optimization.md` (Iterations 3-4) for how the
//! defaults below were chosen.

use pyo3::prelude::*;

/// Externally-tunable cache-blocking and accuracy parameters for the
/// Stratton-Chu kernels (`near_field::stratton_chu_xyz`,
/// `far_field::stratton_chu_ff`).
///
/// - `target_tile`: panel size handed to one Rayon work item (also the
///   granularity of parallelism -- too small wastes time on scheduling
///   overhead, too large starves other cores of work near the end of a
///   run; empirically the latter dominates by far on a many-core machine).
/// - `source_tile`: how many source points are processed before moving to
///   the next block of source points, for a fixed panel of targets -- the
///   point of blocking is to keep one source tile's data resident in
///   L1/L2 while it's reused across every target in the panel.
/// - `greens_lut_size`: size of a uniform sin/cos lookup table used to
///   evaluate the Green's function's phase term via linear interpolation
///   instead of an exact `f64::sin_cos` call. `0` (default) disables this
///   and uses exact `sin_cos`.
#[pyclass(from_py_object)]
#[derive(Clone, Copy, Debug)]
pub struct KernelConfig {
    #[pyo3(get, set)]
    pub target_tile: usize,
    #[pyo3(get, set)]
    pub source_tile: usize,
    #[pyo3(get, set)]
    pub greens_lut_size: usize,
}

#[pymethods]
impl KernelConfig {
    /// Defaults (32, 512, 0) come from an empirical sweep on a 28-core
    /// Apple M3 Ultra (`benchmarks/sweep_config.py`,
    /// `sweep_config_large.py`): `target_tile` needs to be small enough to
    /// give every core plenty of panels to work-steal (large tiles
    /// under-parallelize badly -- e.g. target_tile=4096 measured 5-8x
    /// *slower* than target_tile=16-64 at matched problem sizes);
    /// `source_tile` had only marginal effect even at 150,000 masked
    /// source points (~26 MB, well past typical per-core L2), so its
    /// default is chosen from the far-field kernel's slight preference for
    /// larger blocks rather than a sharp near-field optimum. `greens_lut_size`
    /// defaults to 0 (disabled/exact): benchmarking found the LUT is not
    /// actually faster than exact `sin_cos` on this hardware (Iteration 4)
    /// -- it's kept as an opt-in, verified-correct (accuracy tested down to
    /// -60 dB and past) knob for future hardware/compilers where it might
    /// pay off, not a recommended default. Re-run the sweep scripts before
    /// trusting any of these on different hardware or very different
    /// problem sizes.
    #[new]
    #[pyo3(signature = (target_tile=32, source_tile=512, greens_lut_size=0))]
    pub fn new(target_tile: usize, source_tile: usize, greens_lut_size: usize) -> Self {
        KernelConfig {
            target_tile: target_tile.max(1),
            source_tile: source_tile.max(1),
            greens_lut_size,
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "KernelConfig(target_tile={}, source_tile={}, greens_lut_size={})",
            self.target_tile, self.source_tile, self.greens_lut_size
        )
    }
}
