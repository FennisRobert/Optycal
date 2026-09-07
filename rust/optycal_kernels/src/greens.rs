//! Optional interpolated evaluation of the Green's function's phase term.
//!
//! `R` is always real and non-negative in these kernels, so
//! `exp(-i*k0*R) = cos(k0*R) - i*sin(k0*R)` reduces to a periodic function
//! of a single real variable, `theta = k0*R mod 2*pi`. Instead of an exact
//! `f64::sin_cos` call, `SinCosLut::eval` looks up a value from a small
//! uniform table over one period and linearly interpolates.
//!
//! Accuracy is not in question -- see `_sincos_lut_max_error` and
//! `claude_nodes/kernel_optimization.md` (Iteration 4): 128 entries gives
//! ~-70 dB (0.03%) worst-case error against exact `sin`/`cos`, comfortably
//! under a -60 dB target and utterly negligible next to the near-field
//! kernel's own ~5-9% mesh-discretization ceiling. What *did* turn out to
//! matter: benchmarking found this is not actually faster than exact
//! `sin_cos` on this hardware (Apple M3 Ultra) -- it's kept as a verified,
//! opt-in `KernelConfig` knob (default off) for future hardware/compilers
//! where it might pay off, not as a recommended default.

const TAU: f64 = std::f64::consts::TAU;

/// A uniform sin/cos lookup table over one period, with the per-lookup
/// scaling constant (`n/TAU`) precomputed once at construction instead of
/// recomputed on every call -- Iteration 5: the original free-function
/// version divided by `TAU` inside `lut_sin_cos` itself, i.e. once per
/// (source, target) pair when this path is active, instead of once per
/// kernel call.
pub struct SinCosLut {
    table: Vec<(f64, f64)>,
    n_over_tau: f64,
    n_f: f64,
}

impl SinCosLut {
    pub fn new(n: usize) -> Self {
        let n = n.max(4);
        let table = (0..n)
            .map(|i| {
                let theta = (i as f64) * TAU / (n as f64);
                theta.sin_cos()
            })
            .collect();
        SinCosLut {
            table,
            n_over_tau: (n as f64) / TAU,
            n_f: n as f64,
        }
    }

    /// Linearly interpolated `(sin, cos)`. `theta` may be any non-negative
    /// real value (callers pass `k0*R`, which is always >= 0 here).
    #[inline(always)]
    pub fn eval(&self, theta: f64) -> (f64, f64) {
        // theta * (n/TAU) directly scales into table-index units in one
        // multiply -- no runtime division by TAU.
        let scaled_raw = theta * self.n_over_tau;
        let scaled = scaled_raw.rem_euclid(self.n_f);
        let n = self.table.len();
        let i0 = scaled as usize % n; // guard f64 rounding landing exactly on n_f
        let i1 = (i0 + 1) % n;
        let frac = scaled - scaled.floor();
        let (s0, c0) = self.table[i0];
        let (s1, c1) = self.table[i1];
        (s0 + frac * (s1 - s0), c0 + frac * (c1 - c0))
    }
}

/// Diagnostic-only, exposed to Python for
/// `benchmarks/lut_accuracy_check.py`, so the accuracy claims in
/// `KernelConfig`'s docs and `claude_nodes/kernel_optimization.md` are
/// checked against the exact table-building/interpolation code the kernels
/// use, not a reimplementation. (`cargo test` doesn't work cleanly for a
/// pyo3 `extension-module` cdylib -- the test binary can't link against a
/// Python it isn't embedded in -- so this is validated from Python instead
/// of a `#[test]`.)
#[pyo3::pyfunction]
pub fn _sincos_lut_max_error(n: usize, test_points: usize) -> f64 {
    let lut = SinCosLut::new(n);
    let mut max_err = 0.0_f64;
    for i in 0..test_points {
        let theta = (i as f64) / (test_points as f64) * TAU;
        let (s_lut, c_lut) = lut.eval(theta);
        let (s_exact, c_exact) = theta.sin_cos();
        max_err = max_err
            .max((s_lut - s_exact).abs())
            .max((c_lut - c_exact).abs());
    }
    max_err
}
