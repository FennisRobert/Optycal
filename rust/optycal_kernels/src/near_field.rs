//! Near-field Stratton-Chu kernel: propagates E/H fields on a closed
//! surface out to arbitrary near-field points in space. Rust port of
//! `optycal/src/optycal/solvers/strattonchu.py::stratton_chu_xyz`.
//!
//! This is the kernel that arrived at, in order (full history and
//! benchmarks in `claude_nodes/kernel_optimization.md`):
//!
//! 1. A faithful structural port of numba's per-*source*-point `prange` +
//!    array-reduction strategy (Iteration 1) -- modest, inconsistent win.
//! 2. Restructured to parallelize over *target* points instead (Iteration
//!    2) -- no reduction needed, each target is one independent unit of
//!    work. Consistently faster than (1).
//! 3. Explicit, externally-tunable cache blocking via `KernelConfig`
//!    (Iteration 3) -- swept empirically; turned out `target_tile` (parallel
//!    granularity) matters enormously and `source_tile` (cache blocking)
//!    barely at all on the tested hardware/problem sizes.
//! 4. Hoisted `ie1*N_x_H[j]` / `ih1*N_x_E[j]` (constant per source point,
//!    was being recomputed on every (source, target) pair after step 2's
//!    restructuring) out of the inner loop, and replaced the generic
//!    `Complex::exp` Green's-function evaluation with a direct
//!    `sin_cos`-based construction (Iteration 4).
//! 5. Added an optional LUT-based Green's-function phase evaluation via
//!    `KernelConfig::greens_lut_size` (Iteration 4) -- verified accurate
//!    (well past a -60 dB target) but *not* faster than exact `sin_cos` on
//!    this hardware, so it stays off by default; see `greens` module.
//! 6. Hyper-optimization pass (Iteration 5, see
//!    `claude_nodes/kernel_optimization.md`): every multiply-then-add/
//!    subtract pattern in the inner loop where one operand is a real
//!    scalar (the `R^2` dot product, the `N x V + (V.N)*R` cross-product
//!    term) now goes through `cmath`'s FMA-fused helpers instead of plain
//!    `+`/`-`/`*`, trading 2 multiplies for 2 fused multiply-adds per call
//!    -- a real instruction-count reduction on FMA-capable hardware, not
//!    just fewer source characters.

use numpy::{Complex64, IntoPyArray, PyArray2, PyReadonlyArray2};
use pyo3::prelude::*;
use rayon::prelude::*;

use crate::cmath::{cmul, dot3, triple_term};
use crate::config::KernelConfig;
use crate::greens::SinCosLut;
use crate::sources::{build_masked_sources, row_to_vec, C0, EPS0, MU0, Q};

/// Propagates the E/H fields on a closed source surface (`ein`, `hin` at
/// points `vis` with area-weighted normals `wns`) out to the near-field
/// points `cout`, at free-space wavenumber `k0`.
///
/// `ein`/`hin`: `(3, N_source)` complex128. `vis`/`wns`: `(3, N_source)`
/// float64. `cout`: `(3, N_target)` float64. Returns `(E, H)`, each
/// `(3, N_target)` complex128.
#[pyfunction]
pub fn stratton_chu_xyz<'py>(
    py: Python<'py>,
    ein: PyReadonlyArray2<'py, Complex64>,
    hin: PyReadonlyArray2<'py, Complex64>,
    vis: PyReadonlyArray2<'py, f64>,
    wns: PyReadonlyArray2<'py, f64>,
    cout: PyReadonlyArray2<'py, f64>,
    k0: f64,
    config: &KernelConfig,
) -> (Bound<'py, PyArray2<Complex64>>, Bound<'py, PyArray2<Complex64>>) {
    let ein = ein.as_array();
    let hin = hin.as_array();
    let vis = vis.as_array();
    let wns = wns.as_array();
    let cout = cout.as_array();

    let src = build_masked_sources(&ein, &hin, &vis, &wns);
    let n_src = src.vx.len();
    let n_tgt = cout.shape()[1];

    let xo = row_to_vec(&cout, 0);
    let yo = row_to_vec(&cout, 1);
    let zo = row_to_vec(&cout, 2);

    let w0 = k0 * C0;
    let ie1 = Complex64::new(0.0, -w0 * MU0);
    let ih1 = Complex64::new(0.0, w0 * EPS0);

    // Target-independent per-source products, hoisted out of the
    // (source, target) inner loop -- see module docs, step 4.
    let ie1_nxh: [Vec<Complex64>; 3] =
        std::array::from_fn(|c| src.nx_h[c].iter().map(|&v| ie1 * v).collect());
    let ih1_nxe: [Vec<Complex64>; 3] =
        std::array::from_fn(|c| src.nx_e[c].iter().map(|&v| ih1 * v).collect());

    let lut = if config.greens_lut_size > 0 {
        Some(SinCosLut::new(config.greens_lut_size))
    } else {
        None
    };

    let mut eoutx = vec![Complex64::new(0.0, 0.0); n_tgt];
    let mut eouty = vec![Complex64::new(0.0, 0.0); n_tgt];
    let mut eoutz = vec![Complex64::new(0.0, 0.0); n_tgt];
    let mut houtx = vec![Complex64::new(0.0, 0.0); n_tgt];
    let mut houty = vec![Complex64::new(0.0, 0.0); n_tgt];
    let mut houtz = vec![Complex64::new(0.0, 0.0); n_tgt];

    eoutx
        .par_iter_mut()
        .zip(eouty.par_iter_mut())
        .zip(eoutz.par_iter_mut())
        .zip(houtx.par_iter_mut())
        .zip(houty.par_iter_mut())
        .zip(houtz.par_iter_mut())
        .enumerate()
        .for_each(|(o, (((((ex_o, ey_o), ez_o), hx_o), hy_o), hz_o))| {
            let (xo_, yo_, zo_) = (xo[o], yo[o], zo[o]);
            let mut ex = Complex64::new(0.0, 0.0);
            let mut ey = Complex64::new(0.0, 0.0);
            let mut ez = Complex64::new(0.0, 0.0);
            let mut hx = Complex64::new(0.0, 0.0);
            let mut hy = Complex64::new(0.0, 0.0);
            let mut hz = Complex64::new(0.0, 0.0);

            for j in 0..n_src {
                let rx = xo_ - src.vx[j];
                let ry = yo_ - src.vy[j];
                let rz = zo_ - src.vz[j];
                // R^2 via FMA: 1 multiply + 2 fused multiply-adds instead
                // of 3 multiplies + 2 adds.
                let r = dot3(rx, rx, ry, ry, rz, rz).sqrt();
                let ri = 1.0 / r;

                // Loop-invariant branch (same `lut` for every iteration of
                // every panel in this call) -- trivially predicted, not a
                // per-pair cost.
                let (s, c) = match &lut {
                    Some(table) => table.eval(r * k0),
                    None => (r * k0).sin_cos(),
                };
                let qri = Q * ri;
                let g = Complex64::new(c * qri, -s * qri);
                let dg = Complex64::new(ri * ri, ri * k0);

                let (nxhx, nxhy, nxhz) = (src.nx_h[0][j], src.nx_h[1][j], src.nx_h[2][j]);
                let (nxex, nxey, nxez) = (src.nx_e[0][j], src.nx_e[1][j], src.nx_e[2][j]);
                let edn = src.e_dot_n[j];
                let hdn = src.h_dot_n[j];

                // `triple_term(a,ra,b,rb,c,rc) = a*ra - b*rb + c*rc` via
                // FMA (3 multiplies + 4 FMAs instead of 6 multiplies + 3
                // add/subtracts); `cmul` fuses the two complex multiplies
                // per component (dg*cross_term, then g*(...)) the same way.
                ex += cmul(g, ie1_nxh[0][j] + cmul(dg, triple_term(nxey, rz, nxez, ry, edn, rx)));
                ey += cmul(g, ie1_nxh[1][j] + cmul(dg, triple_term(nxez, rx, nxex, rz, edn, ry)));
                ez += cmul(g, ie1_nxh[2][j] + cmul(dg, triple_term(nxex, ry, nxey, rx, edn, rz)));
                hx += cmul(g, ih1_nxe[0][j] + cmul(dg, triple_term(nxhy, rz, nxhz, ry, hdn, rx)));
                hy += cmul(g, ih1_nxe[1][j] + cmul(dg, triple_term(nxhz, rx, nxhx, rz, hdn, ry)));
                hz += cmul(g, ih1_nxe[2][j] + cmul(dg, triple_term(nxhx, ry, nxhy, rx, hdn, rz)));
            }

            *ex_o = ex;
            *ey_o = ey;
            *ez_o = ez;
            *hx_o = hx;
            *hy_o = hy;
            *hz_o = hz;
        });

    let mut eout = vec![Complex64::new(0.0, 0.0); 3 * n_tgt];
    let mut hout = vec![Complex64::new(0.0, 0.0); 3 * n_tgt];
    eout[0..n_tgt].copy_from_slice(&eoutx);
    eout[n_tgt..2 * n_tgt].copy_from_slice(&eouty);
    eout[2 * n_tgt..3 * n_tgt].copy_from_slice(&eoutz);
    hout[0..n_tgt].copy_from_slice(&houtx);
    hout[n_tgt..2 * n_tgt].copy_from_slice(&houty);
    hout[2 * n_tgt..3 * n_tgt].copy_from_slice(&houtz);

    let eout = numpy::ndarray::Array2::from_shape_vec((3, n_tgt), eout).unwrap();
    let hout = numpy::ndarray::Array2::from_shape_vec((3, n_tgt), hout).unwrap();
    (eout.into_pyarray(py), hout.into_pyarray(py))
}
