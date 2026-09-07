//! Antenna field evaluation: given a point source's position, orientation
//! (a 3x3 basis + its inverse, i.e. `CoordinateSystem.global_basis`/
//! `global_basis_inv`, computed on the Python side so this crate never
//! needs to implement a general matrix pseudo-inverse), complex excitation
//! amplitude, wavenumber, and an `AntennaPattern`, evaluate the radiated
//! E/H field at a set of points (`expose_xyz`, near field) or directions
//! (`expose_thetaphi`, far field).
//!
//! Ported to exactly match `antennas/antenna.py::Antenna.expose_xyz`/
//! `expose_thetaphi` (the plain, non-accelerated Python path) rather than
//! `antennas/compiled/antenna_single.py`'s numba `expose_xyz_single`/
//! `expose_thetaphi_single`: the numba path mixes a *local*-frame
//! wavevector with the antenna's *global* position when computing the
//! array phase-steering term, which is only correct for an unrotated
//! antenna -- not replicated here. This module's phase term instead
//! matches `Antenna.expose_thetaphi`'s (local wavevector dotted with the
//! antenna's local position, both in the same frame).

use numpy::{Complex64, IntoPyArray, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use pyo3::types::PyComplex;
use rayon::prelude::*;

use crate::antenna_pattern::{AntennaPattern, FieldSix};

type Basis<'a> = numpy::ndarray::ArrayView2<'a, f64>;

/// `numpy::Complex64` (a re-export of `num_complex::Complex<f64>`) has no
/// `FromPyObject` impl for use as a *scalar* `#[pyfunction]` argument (only
/// as an array element type) -- accept a Python `complex` and convert.
#[inline(always)]
fn complex_arg(c: &Bound<'_, PyComplex>) -> Complex64 {
    Complex64::new(c.real(), c.imag())
}

#[inline(always)]
fn matvec_real(m: &Basis, v: (f64, f64, f64)) -> (f64, f64, f64) {
    let (x, y, z) = v;
    (
        m[[0, 0]] * x + m[[0, 1]] * y + m[[0, 2]] * z,
        m[[1, 0]] * x + m[[1, 1]] * y + m[[1, 2]] * z,
        m[[2, 0]] * x + m[[2, 1]] * y + m[[2, 2]] * z,
    )
}

#[inline(always)]
fn matvec_complex(
    m: &Basis,
    v: (Complex64, Complex64, Complex64),
) -> (Complex64, Complex64, Complex64) {
    let (x, y, z) = v;
    (
        x * m[[0, 0]] + y * m[[0, 1]] + z * m[[0, 2]],
        x * m[[1, 0]] + y * m[[1, 1]] + z * m[[1, 2]],
        x * m[[2, 0]] + y * m[[2, 1]] + z * m[[2, 2]],
    )
}

fn scatter(results: &[FieldSix], n: usize) -> (Vec<Complex64>, Vec<Complex64>) {
    let mut eout = vec![Complex64::new(0.0, 0.0); 3 * n];
    let mut hout = vec![Complex64::new(0.0, 0.0); 3 * n];
    for (idx, r) in results.iter().enumerate() {
        eout[idx] = r[0];
        eout[n + idx] = r[1];
        eout[2 * n + idx] = r[2];
        hout[idx] = r[3];
        hout[n + idx] = r[4];
        hout[2 * n + idx] = r[5];
    }
    (eout, hout)
}

/// Near field: E/H at global points `(gx, gy, gz)` radiated by an antenna
/// at global position `ant_gxyz`, oriented by `global_basis`
/// (local->global rotation) with pseudo-inverse `global_basis_inv`
/// (global->local), complex excitation `amplitude`, wavenumber `k0`.
#[pyfunction]
pub fn antenna_expose_xyz<'py>(
    py: Python<'py>,
    gx: PyReadonlyArray1<'py, f64>,
    gy: PyReadonlyArray1<'py, f64>,
    gz: PyReadonlyArray1<'py, f64>,
    ant_gxyz: [f64; 3],
    global_basis: PyReadonlyArray2<'py, f64>,
    global_basis_inv: PyReadonlyArray2<'py, f64>,
    pattern: &AntennaPattern,
    amplitude: Bound<'py, PyComplex>,
    k0: f64,
) -> (Bound<'py, PyArray2<Complex64>>, Bound<'py, PyArray2<Complex64>>) {
    let gx = gx.as_array();
    let gy = gy.as_array();
    let gz = gz.as_array();
    let basis = global_basis.as_array();
    let basis_inv = global_basis_inv.as_array();
    let n = gx.len();
    let (sx, sy, sz) = (ant_gxyz[0], ant_gxyz[1], ant_gxyz[2]);
    let amplitude = complex_arg(&amplitude);

    let results: Vec<FieldSix> = (0..n)
        .into_par_iter()
        .map(|idx| {
            let dx = gx[idx] - sx;
            let dy = gy[idx] - sy;
            let dz = gz[idx] - sz;
            let r = (dx * dx + dy * dy + dz * dz).sqrt();
            let (kx, ky, kz) = (dx / r, dy / r, dz / r);

            let (lkx, lky, lkz) = matvec_real(&basis_inv, (kx, ky, kz));
            let thetac = lkz.acos();
            let phic = lky.atan2(lkx);

            let (s, cc) = (k0 * r).sin_cos();
            let b = amplitude * Complex64::new(cc, -s) / r;

            let [ex, ey, ez, hx, hy, hz] = pattern.kind.eval_nf(thetac, phic, r, k0);
            let (ex, ey, ez) = matvec_complex(&basis, (ex, ey, ez));
            let (hx, hy, hz) = matvec_complex(&basis, (hx, hy, hz));

            [ex * b, ey * b, ez * b, hx * b, hy * b, hz * b]
        })
        .collect();

    let (eout, hout) = scatter(&results, n);
    let eout = numpy::ndarray::Array2::from_shape_vec((3, n), eout).unwrap();
    let hout = numpy::ndarray::Array2::from_shape_vec((3, n), hout).unwrap();
    (eout.into_pyarray(py), hout.into_pyarray(py))
}

/// Far field: E/H radiated by an antenna towards global directions
/// `(gtheta, gphi)`. `local_xyz`: the antenna's position in its own local
/// frame (`Antenna.local_xyz`, i.e. the raw constructor `(x, y, z)`) --
/// used for the array phase-steering term, which must stay in the same
/// frame as the local wavevector it's dotted with (see module docs).
#[pyfunction]
pub fn antenna_expose_thetaphi<'py>(
    py: Python<'py>,
    gtheta: PyReadonlyArray1<'py, f32>,
    gphi: PyReadonlyArray1<'py, f32>,
    local_xyz: [f64; 3],
    global_basis: PyReadonlyArray2<'py, f64>,
    global_basis_inv: PyReadonlyArray2<'py, f64>,
    pattern: &AntennaPattern,
    amplitude: Bound<'py, PyComplex>,
    k0: f64,
) -> (Bound<'py, PyArray2<Complex64>>, Bound<'py, PyArray2<Complex64>>) {
    let gtheta = gtheta.as_array();
    let gphi = gphi.as_array();
    let basis = global_basis.as_array();
    let basis_inv = global_basis_inv.as_array();
    let n = gtheta.len();
    let (lx, ly, lz) = (local_xyz[0], local_xyz[1], local_xyz[2]);
    let amplitude = complex_arg(&amplitude);

    let results: Vec<FieldSix> = (0..n)
        .into_par_iter()
        .map(|idx| {
            let gth = gtheta[idx] as f64;
            let gph = gphi[idx] as f64;
            let (snt, cst) = gth.sin_cos();
            let (snp, csp) = gph.sin_cos();
            let xx = csp * snt;
            let yy = snp * snt;
            let zz = cst;

            // Global unit direction -> local frame. `(x2,y2,z2)` is
            // already the local unit vector -- deriving theta_local/
            // phi_local from it and then reconstructing kx/ky/kz via
            // trig (as the pure-Python `Antenna.expose_thetaphi` does)
            // is a mathematically-exact round trip; using (x2,y2,z2)
            // directly skips the redundant acos/atan2/sin/cos.
            let (x2, y2, z2) = matvec_real(&basis_inv, (xx, yy, zz));
            let theta_local = z2.acos();
            let phi_local = y2.atan2(x2);

            let (kx, ky, kz) = (k0 * x2, k0 * y2, k0 * z2);
            let phase = kx * lx + ky * ly + kz * lz;
            let (sph, cph) = phase.sin_cos();
            let b = amplitude * Complex64::new(cph, sph);

            let [ex, ey, ez, hx, hy, hz] = pattern.kind.eval_ff(theta_local, phi_local, k0);
            let (ex, ey, ez) = matvec_complex(&basis, (ex, ey, ez));
            let (hx, hy, hz) = matvec_complex(&basis, (hx, hy, hz));

            [ex * b, ey * b, ez * b, hx * b, hy * b, hz * b]
        })
        .collect();

    let (eout, hout) = scatter(&results, n);
    let eout = numpy::ndarray::Array2::from_shape_vec((3, n), eout).unwrap();
    let hout = numpy::ndarray::Array2::from_shape_vec((3, n), hout).unwrap();
    (eout.into_pyarray(py), hout.into_pyarray(py))
}
