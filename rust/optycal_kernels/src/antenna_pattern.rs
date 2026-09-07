//! Antenna radiation patterns: a small enum of natively-hardcoded analytic
//! patterns (exact, closed-form -- ported from `antennas/patterns.py`) plus
//! a `Interpolated` variant for arbitrary user-defined patterns, backed by
//! the same bicubic-spline gridded interpolation scheme already used by
//! `antennas/interpolation_pattern.py`/`antennas/compiled/antenna_single.py`
//! (`InterpolatingAntenna`/`EMergeAntenna`) -- construction of the spline
//! coefficient grid stays in Python (one-time cost, existing well-tested
//! code, reused as-is via `AntennaPattern.from_function(...).full_matrix(...)`
//! on the Python side); only the fast per-point evaluation is ported here.
//!
//! **Near-field approximation for `Interpolated`, inherited from the
//! existing Python design, not introduced here:** `InterpolatingAntenna`
//! already evaluates its near field as "far-field pattern shape x the
//! standard `amplitude*exp(-ikR)/R` radial factor", ignoring any genuine
//! `r`-dependence a hand-written `nf_pattern` might define. Of the patterns
//! in `antennas/patterns.py`, only `dipole`/`half_dipole` (and the
//! Gaussian-generator patterns) actually use `r` for reactive near-field
//! correction terms -- `patch`/`triang` already ignore it. `Interpolated`
//! here keeps that same behavior (`eval_nf` for `Interpolated` just calls
//! `eval_ff` and ignores `r`); only `Dipole`/`HalfDipole` get the exact
//! `dipole_pattern_nf` reactive-term formula, since that's the default
//! pattern and the one used by the closed-sphere physics test.

use numpy::ndarray::Array4;
use numpy::{Complex64, PyReadonlyArray1, PyReadonlyArray5};
use pyo3::prelude::*;

const Z0: f64 = 376.73031366857;
const Y0: f64 = 1.0 / Z0;

fn eomni() -> f64 {
    (3.0 * Z0 / (4.0 * std::f64::consts::PI)).sqrt()
}

/// Six field components: Ex, Ey, Ez, Hx, Hy, Hz.
pub type FieldSix = [Complex64; 6];

fn c(re: f64) -> Complex64 {
    Complex64::new(re, 0.0)
}

/// `antennas/patterns.py::dipole_pattern_ff`, ported exactly.
fn dipole_ff(theta_in: f64, phi_in: f64) -> FieldSix {
    let theta = std::f64::consts::FRAC_PI_2 - theta_in;
    let cst = theta.cos();
    let csp = phi_in.cos();
    let snp = phi_in.sin();
    let a = eomni() * 0.5 * (2.0 * theta).sin();

    let ex = -a * csp;
    let ey = -a * snp;
    let ez = 0.5 * eomni() * ((2.0 * theta).cos() + 1.0);
    let hx = eomni() * Y0 * cst * snp;
    let hy = -eomni() * Y0 * cst * csp;
    let hz = 0.0;
    [c(ex), c(ey), c(ez), c(hx), c(hy), c(hz)]
}

/// `antennas/patterns.py::dipole_pattern_nf`, ported exactly (including its
/// complex reactive-term structure: `F = 1/(i*k0*r)`).
fn dipole_nf(theta_in: f64, phi_in: f64, r: f64, k0: f64) -> FieldSix {
    let theta = std::f64::consts::FRAC_PI_2 - theta_in;
    let cst = theta.cos();
    let snt = theta.sin();
    let csp = phi_in.cos();
    let snp = phi_in.sin();
    let a = eomni();

    let f = Complex64::new(0.0, -1.0 / (k0 * r)); // 1/(i*k0*r)
    let qrr = f + f * f;
    let qrt = Complex64::new(1.0, 0.0) + qrr;

    let ex = c(a) * (-(qrt * c(snt * csp * cst)) - qrr * c(cst * csp * 2.0 * snt));
    let ey = c(a) * (-(qrt * c(snt * snp * cst)) - qrr * c(cst * snp * 2.0 * snt));
    let ez = c(a) * (qrt * c(cst * cst) - qrr * c(2.0 * snt * snt));
    let hx = (Complex64::new(1.0, 0.0) + f) * c(a * Y0 * cst * snp);
    let hy = -(Complex64::new(1.0, 0.0) + f) * c(a * Y0 * cst * csp);
    let hz = Complex64::new(0.0, 0.0);
    [ex, ey, ez, hx, hy, hz]
}

/// `antennas/patterns.py::half_dipole_pattern_ff` -- `dipole_ff` masked to
/// the front half-space (`|phi| < pi/2`).
fn half_dipole_ff(theta_in: f64, phi_in: f64) -> FieldSix {
    let active = if phi_in.abs() < std::f64::consts::FRAC_PI_2 { 1.0 } else { 0.0 };
    dipole_ff(theta_in, phi_in).map(|v| v * active)
}

/// `antennas/patterns.py::half_dipole_pattern_nf` -- `dipole_nf` masked to
/// the front half-space.
fn half_dipole_nf(theta_in: f64, phi_in: f64, r: f64, k0: f64) -> FieldSix {
    let active = if phi_in.abs() < std::f64::consts::FRAC_PI_2 { 1.0 } else { 0.0 };
    dipole_nf(theta_in, phi_in, r, k0).map(|v| v * active)
}

/// `numpy.sinc`-compatible normalized sinc: `sin(pi*x)/(pi*x)`, `sinc(0)=1`.
fn sinc(x: f64) -> f64 {
    if x == 0.0 {
        1.0
    } else {
        (std::f64::consts::PI * x).sin() / (std::f64::consts::PI * x)
    }
}

/// `antennas/patterns.py::patch_pattern_ff`/`_nf` (identical bodies -- no
/// `r`-dependence in either) and the parametrized
/// `generate_patch_pattern(Width, Length, k0)`'s returned closures, unified
/// into one native formula: the plain functions are exactly this with
/// `kw=kl=pi, t_exp=0.5`; `generate_patch_pattern` is this with
/// `kw=k0*Width, kl=k0*Length, t_exp=0.02` (note the *different* `T`
/// exponent between the two Python versions -- transcribed exactly, not
/// unified into one constant, since they really are two slightly
/// different formulas).
fn patch(theta_in: f64, phi_in: f64, kw: f64, kl: f64, t_exp: f64) -> FieldSix {
    let theta = std::f64::consts::FRAC_PI_2 - theta_in;
    let cst = theta.cos();
    let snt = theta.sin();
    let csp = phi_in.cos();
    let snp = phi_in.sin();
    let rx = csp * cst;
    let ry = snp * cst;
    let rz = snt;

    let t = (cst * csp).abs().powf(t_exp);
    let active = if phi_in.abs() < std::f64::consts::FRAC_PI_2 { 1.0 } else { 0.0 };

    let ex = -(kl / 2.0 * snt).cos() * snt * sinc(kw / 2.0 * snp) * active * t;
    let ey = 0.0_f64;
    let ez = (kl / 2.0 * snt).cos() * (csp * cst) * sinc(kw / 2.0 * snp) * active * t;

    let hx = Y0 * (ry * ez - rz * ey);
    let hy = Y0 * (rz * ex - rx * ez);
    let hz = Y0 * (rx * ey - ry * ex);
    [c(ex), c(ey), c(ez), c(hx), c(hy), c(hz)]
}

/// Bicubic-spline coefficient grid for one arbitrary (Python-defined)
/// far-field pattern, built on the Python side (`AntennaPattern.
/// from_function(...).full_matrix(...)`, unchanged existing code) and
/// handed over as a single `(6, 4, 4, N_theta-1, N_phi-1)` complex array
/// -- one 4x4 Hermite-bicubic coefficient block per grid cell, per field
/// component, in `[Ex,Ey,Ez,Hx,Hy,Hz]` order (matching
/// `interpolation_pattern.py::AntennaPattern.full_matrix`).
pub struct InterpolatedPattern {
    coeffs: Array4<[Complex64; 6]>, // shape (4, 4, N_theta-1, N_phi-1), one 6-vector per cell coefficient
    theta_grid: Vec<f32>,
    phi_grid: Vec<f32>,
}

impl InterpolatedPattern {
    fn eval(&self, theta: f64, phi: f64) -> FieldSix {
        let nx = self.theta_grid.len();
        let ny = self.phi_grid.len();
        let minx = self.theta_grid[0] as f64;
        let miny = self.phi_grid[0] as f64;
        let dx = (self.theta_grid[1] - self.theta_grid[0]) as f64;
        let dy = (self.phi_grid[1] - self.phi_grid[0]) as f64;

        let mut i = ((theta - minx) / dx).floor() as isize;
        i = i.max(0).min(nx as isize - 2);
        let mut j = ((phi - miny) / dy).floor() as isize;
        j = j.max(0).min(ny as isize - 2);
        let (i, j) = (i as usize, j as usize);

        let xi = (theta - self.theta_grid[i] as f64) / dx;
        let yi = (phi - self.phi_grid[j] as f64) / dy;
        let (ysq, xsq, yqb, xqb) = (yi * yi, xi * xi, yi * yi * yi, xi * xi * xi);

        let a = |p: usize, q: usize| self.coeffs[[p, q, i, j]];
        let mut out = [Complex64::new(0.0, 0.0); 6];
        for k in 0..6 {
            let a00 = a(0, 0)[k];
            let a01 = a(0, 1)[k];
            let a02 = a(0, 2)[k];
            let a03 = a(0, 3)[k];
            let a10 = a(1, 0)[k];
            let a11 = a(1, 1)[k];
            let a12 = a(1, 2)[k];
            let a13 = a(1, 3)[k];
            let a20 = a(2, 0)[k];
            let a21 = a(2, 1)[k];
            let a22 = a(2, 2)[k];
            let a23 = a(2, 3)[k];
            let a30 = a(3, 0)[k];
            let a31 = a(3, 1)[k];
            let a32 = a(3, 2)[k];
            let a33 = a(3, 3)[k];

            let m1 = a00 + a01 * yi + a02 * ysq + a03 * yqb;
            let m2 = a10 + a11 * yi + a12 * ysq + a13 * yqb;
            let m3 = a20 + a21 * yi + a22 * ysq + a23 * yqb;
            let m4 = a30 + a31 * yi + a32 * ysq + a33 * yqb;
            out[k] = m1 + m2 * xi + m3 * xsq + m4 * xqb;
        }
        out
    }
}

pub enum PatternKind {
    Dipole,
    HalfDipole,
    Patch { kw: f64, kl: f64, t_exp: f64 },
    Interpolated(InterpolatedPattern),
}

impl PatternKind {
    pub fn eval_ff(&self, theta: f64, phi: f64, _k0: f64) -> FieldSix {
        match self {
            PatternKind::Dipole => dipole_ff(theta, phi),
            PatternKind::HalfDipole => half_dipole_ff(theta, phi),
            PatternKind::Patch { kw, kl, t_exp } => patch(theta, phi, *kw, *kl, *t_exp),
            PatternKind::Interpolated(p) => p.eval(theta, phi),
        }
    }

    pub fn eval_nf(&self, theta: f64, phi: f64, r: f64, k0: f64) -> FieldSix {
        match self {
            PatternKind::Dipole => dipole_nf(theta, phi, r, k0),
            PatternKind::HalfDipole => half_dipole_nf(theta, phi, r, k0),
            // patch_pattern_nf's body is identical to patch_pattern_ff's
            // (no r-dependence) -- exact, not an approximation.
            PatternKind::Patch { kw, kl, t_exp } => patch(theta, phi, *kw, *kl, *t_exp),
            // Far-field shape only -- see module docs.
            PatternKind::Interpolated(p) => p.eval(theta, phi),
        }
    }
}

/// Python-facing antenna pattern handle. Construct via `AntennaPattern.
/// dipole()`, `.half_dipole()`, `.patch(kw, kl, t_exp)`, or
/// `.interpolated(theta_grid, phi_grid, full_matrix)`.
#[pyclass]
pub struct AntennaPattern {
    pub kind: PatternKind,
}

#[pymethods]
impl AntennaPattern {
    #[staticmethod]
    fn dipole() -> Self {
        AntennaPattern { kind: PatternKind::Dipole }
    }

    #[staticmethod]
    fn half_dipole() -> Self {
        AntennaPattern { kind: PatternKind::HalfDipole }
    }

    /// `kw`/`kl`: the sinc/cosine shape parameters (`patch_pattern_ff`'s
    /// plain, non-parametrized form is `kw=kl=pi`; `generate_patch_pattern
    /// (Width, Length, k0)` is `kw=k0*Width, kl=k0*Length`). `t_exp`: the
    /// exponent on the `|cos(theta)*cos(phi)|` roll-off term (plain
    /// `patch_pattern_ff` uses `0.5`; `generate_patch_pattern` uses
    /// `0.02` -- these are genuinely different formulas in
    /// `antennas/patterns.py`, not just different parameters of one).
    #[staticmethod]
    fn patch(kw: f64, kl: f64, t_exp: f64) -> Self {
        AntennaPattern { kind: PatternKind::Patch { kw, kl, t_exp } }
    }

    /// `theta_grid`/`phi_grid`: 1D float32 grid axes (physical spherical
    /// angles: theta in `[0, pi]`, phi in `[-pi, pi]`, uniform spacing).
    /// `full_matrix`: `(6, 4, 4, N_theta-1, N_phi-1)` complex128, exactly
    /// `interpolation_pattern.py::AntennaPattern.full_matrix(Precision.DOUBLE)`'s
    /// output when built from a grid with these axes (component order
    /// `[Ex,Ey,Ez,Hx,Hy,Hz]`).
    #[staticmethod]
    fn interpolated(
        theta_grid: PyReadonlyArray1<'_, f32>,
        phi_grid: PyReadonlyArray1<'_, f32>,
        full_matrix: PyReadonlyArray5<'_, Complex64>,
    ) -> PyResult<Self> {
        let fm = full_matrix.as_array();
        let shape = fm.shape();
        let (n4a, n4b, nx1, ny1) = (shape[1], shape[2], shape[3], shape[4]);
        if shape[0] != 6 || n4a != 4 || n4b != 4 {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "full_matrix must have shape (6, 4, 4, N_theta-1, N_phi-1), got {:?}",
                shape
            )));
        }
        let mut coeffs = Array4::from_elem((4, 4, nx1, ny1), [Complex64::new(0.0, 0.0); 6]);
        for p in 0..4 {
            for q in 0..4 {
                for i in 0..nx1 {
                    for j in 0..ny1 {
                        let mut v = [Complex64::new(0.0, 0.0); 6];
                        for k in 0..6 {
                            v[k] = fm[[k, p, q, i, j]];
                        }
                        coeffs[[p, q, i, j]] = v;
                    }
                }
            }
        }
        Ok(AntennaPattern {
            kind: PatternKind::Interpolated(InterpolatedPattern {
                coeffs,
                theta_grid: theta_grid.as_array().to_vec(),
                phi_grid: phi_grid.as_array().to_vec(),
            }),
        })
    }
}
