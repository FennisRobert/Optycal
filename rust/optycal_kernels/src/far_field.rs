//! Far-field Stratton-Chu kernel: propagates E/H fields on a closed source
//! surface out to far-field (theta, phi) directions via the plane-wave
//! phase kernel. Rust port of
//! `optycal/src/optycal/solvers/strattonchuff.py::stratton_chu_ff`.
//!
//! Structurally this already matched the "good" pattern from the start
//! (parallel over target, cache-tiled) -- numba's own implementation here
//! tiles source and target in hardcoded 128-element blocks. This version
//! replaces those hardcoded constants with the externally-tunable
//! `KernelConfig` (see `claude_nodes/kernel_optimization.md`, Iteration 3),
//! so the tile sizes can be swept and tuned instead of assumed.
//!
//! Also fixes a bug found while porting (Iteration 1/2): the original
//! numba kernel computed `Hout` from the *raw*, pre-`Q`-scaling `Eout`
//! accumulator instead of the actual (`Q`-scaled) `E` it returns, silently
//! dropping the complex factor `Q = -i*k0/(4*pi)` and leaving the far-field
//! H off by `1/Q` relative to the physical `H = r_hat x E / Z0` relation.
//! Fixed in both this Rust kernel and the numba source
//! (`solvers/strattonchuff.py`) identically; see
//! `tests/test_stratton_chu_dipole_sphere.py::test_farfield_H_matches_analytic_dipole`
//! for the regression test.
//!
//! Hyper-optimization pass (Iteration 5, see
//! `claude_nodes/kernel_optimization.md`): the original inner loop computed
//! `Z0 * (ry*NxH_z[j] - rz*NxH_y[j])` (and the two analogous terms) fresh on
//! *every* (source, target) pair, even though `Z0` and `ry`/`rz`/`rx` are
//! all per-*target* constants -- loop-invariant across the whole source
//! loop for a fixed target. Pre-scaling `r_hat` by `Z0` once per target
//! (`z0_rx`/`z0_ry`/`z0_rz` below) and folding it into the cross-product
//! terms directly removes 2 real multiplies per component per pair (6 per
//! pair total) that were pure repeated work. Also switched every
//! multiply-then-add/subtract with a real-scalar operand to `cmath`'s
//! FMA-fused helpers.

use numpy::{Complex64, IntoPyArray, PyArray2, PyReadonlyArray2};
use pyo3::prelude::*;
use rayon::prelude::*;

use crate::cmath::{cmul, dot3, scaled_diff};
use crate::config::KernelConfig;
use crate::sources::{build_masked_sources, row_to_vec};

const Z0: f64 = 376.73031366857;
const Z0_INV: f64 = 1.0 / Z0; // division-by-constant computed once, not per use

/// Propagates the E/H fields on a closed source surface (`ein`, `hin` at
/// points `vis` with area-weighted normals `wns`) out to the far-field
/// directions `tpout` (theta, phi), at free-space wavenumber `k0`.
///
/// `ein`/`hin`: `(3, N_source)` complex128. `vis`/`wns`: `(3, N_source)`
/// float64. `tpout`: `(2, N_target)` float64 (theta, phi in radians).
/// Returns `(E, H)`, each `(3, N_target)` complex128.
#[pyfunction]
pub fn stratton_chu_ff<'py>(
    py: Python<'py>,
    ein: PyReadonlyArray2<'py, Complex64>,
    hin: PyReadonlyArray2<'py, Complex64>,
    vis: PyReadonlyArray2<'py, f64>,
    wns: PyReadonlyArray2<'py, f64>,
    tpout: PyReadonlyArray2<'py, f64>,
    k0: f64,
    config: &KernelConfig,
) -> (Bound<'py, PyArray2<Complex64>>, Bound<'py, PyArray2<Complex64>>) {
    let ein = ein.as_array();
    let hin = hin.as_array();
    let vis = vis.as_array();
    let wns = wns.as_array();
    let tpout = tpout.as_array();

    let src = build_masked_sources(&ein, &hin, &vis, &wns);
    let n_src = src.vx.len();
    let n_tgt = tpout.shape()[1];

    let theta = row_to_vec(&tpout, 0);
    let phi = row_to_vec(&tpout, 1);

    let qff = Complex64::new(0.0, -k0 / (4.0 * std::f64::consts::PI));

    let target_tile = config.target_tile.max(1);
    let source_tile = config.source_tile.max(1);

    // E and r-hat per target, kept as panel-chunkable Vecs so each panel is
    // one contiguous, independently-writable slice for Rayon (no
    // reduction needed -- see module docs).
    let mut buf: Vec<[Complex64; 3]> = vec![[Complex64::new(0.0, 0.0); 3]; n_tgt];
    let mut rhat: Vec<[f64; 3]> = vec![[0.0; 3]; n_tgt];

    buf.par_chunks_mut(target_tile)
        .zip(rhat.par_chunks_mut(target_tile))
        .enumerate()
        .for_each(|(panel_idx, (panel, rpanel))| {
            let base = panel_idx * target_tile;

            for (local_o, (slot, rslot)) in panel.iter_mut().zip(rpanel.iter_mut()).enumerate() {
                let o = base + local_o;
                let (st, ct) = theta[o].sin_cos();
                let (sp, cp) = phi[o].sin_cos();
                let rx = st * cp;
                let ry = st * sp;
                let rz = ct;
                *rslot = [rx, ry, rz];
                let kx = k0 * rx;
                let ky = k0 * ry;
                let kz = k0 * rz;
                // Pre-scale r_hat by Z0 ONCE per target -- see module docs.
                // Folds the per-pair "* Z0" into the cross-product terms
                // themselves instead of applying it to their result.
                let z0_rx = Z0 * rx;
                let z0_ry = Z0 * ry;
                let z0_rz = Z0 * rz;

                let mut exa = Complex64::new(0.0, 0.0);
                let mut eya = Complex64::new(0.0, 0.0);
                let mut eza = Complex64::new(0.0, 0.0);

                let mut src_start = 0usize;
                while src_start < n_src {
                    let src_end = (src_start + source_tile).min(n_src);
                    for j in src_start..src_end {
                        // Phase dot product via FMA: 1 multiply + 2 fused
                        // multiply-adds instead of 3 multiplies + 2 adds.
                        let phase = dot3(kx, src.vx[j], ky, src.vy[j], kz, src.vz[j]);
                        let (sph, cph) = phase.sin_cos();
                        let g = Complex64::new(cph, sph);

                        let (nxhx, nxhy, nxhz) =
                            (src.nx_h[0][j], src.nx_h[1][j], src.nx_h[2][j]);
                        let (nxex, nxey, nxez) =
                            (src.nx_e[0][j], src.nx_e[1][j], src.nx_e[2][j]);

                        // z0_ry*nxhz - z0_rz*nxhy == Z0*(ry*nxhz - rz*nxhy),
                        // but without the extra per-pair "* Z0" multiply
                        // (folded into z0_ry/z0_rz above), plus FMA-fused.
                        let ie1x = cmul(nxex - scaled_diff(nxhz, z0_ry, nxhy, z0_rz), g);
                        let ie1y = cmul(nxey - scaled_diff(nxhx, z0_rz, nxhz, z0_rx), g);
                        let ie1z = cmul(nxez - scaled_diff(nxhy, z0_rx, nxhx, z0_ry), g);

                        exa += scaled_diff(ie1z, ry, ie1y, rz);
                        eya += scaled_diff(ie1x, rz, ie1z, rx);
                        eza += scaled_diff(ie1y, rx, ie1x, ry);
                    }
                    src_start = src_end;
                }

                // Store the actual (Q-scaled) E -- see the "fixes a bug"
                // note in the module docs.
                *slot = [qff * exa, qff * eya, qff * eza];
            }
        });

    let mut eout = vec![Complex64::new(0.0, 0.0); 3 * n_tgt];
    let mut hout = vec![Complex64::new(0.0, 0.0); 3 * n_tgt];
    for o in 0..n_tgt {
        let [ex, ey, ez] = buf[o];
        let [rx, ry, rz] = rhat[o];
        eout[o] = ex;
        eout[n_tgt + o] = ey;
        eout[2 * n_tgt + o] = ez;
        // H = r_hat x E / Z0 (far-zone plane-wave relation), from the
        // actual returned E. Multiply by the precomputed reciprocal
        // instead of dividing by Z0 (a compile-time constant) each time.
        hout[o] = scaled_diff(ez, ry, ey, rz) * Z0_INV;
        hout[n_tgt + o] = scaled_diff(ex, rz, ez, rx) * Z0_INV;
        hout[2 * n_tgt + o] = scaled_diff(ey, rx, ex, ry) * Z0_INV;
    }

    let eout = numpy::ndarray::Array2::from_shape_vec((3, n_tgt), eout).unwrap();
    let hout = numpy::ndarray::Array2::from_shape_vec((3, n_tgt), hout).unwrap();
    (eout.into_pyarray(py), hout.into_pyarray(py))
}
