//! Shared source-surface preprocessing used by both kernels: physical
//! constants, the `LR` magnitude mask (drops "weak" source points, mirrors
//! `solvers/strattonchu.py`'s `LR = 0.001`), and the per-source
//! cross-/dot-product terms that are invariant across all target points
//! (`N x H`, `N x E`, `E . N`, `H . N`) -- computed once here rather than
//! once per (source, target) pair.

use numpy::ndarray::{ArrayView2, Axis};
use numpy::Complex64;

use crate::cmath::{scaled_diff, triple_sum};

pub const LR: f64 = 0.001;
pub const C0: f64 = 299792458.0;
pub const MU0: f64 = 4.0 * std::f64::consts::PI * 1e-7;
pub const EPS0: f64 = 8.854187812813e-12;
pub const Q: f64 = 1.0 / (4.0 * std::f64::consts::PI);

/// One source point's worth of data, already filtered by the `LR` magnitude
/// mask and with the per-source cross-product / dot-product terms
/// precomputed once (mirrors what the numba kernel computes before its
/// `prange` loop).
pub struct MaskedSources {
    pub vx: Vec<f64>,
    pub vy: Vec<f64>,
    pub vz: Vec<f64>,
    pub nx_h: [Vec<Complex64>; 3], // N x H
    pub nx_e: [Vec<Complex64>; 3], // N x E
    pub e_dot_n: Vec<Complex64>,
    pub h_dot_n: Vec<Complex64>,
}

pub fn row_to_vec(arr: &ArrayView2<f64>, row: usize) -> Vec<f64> {
    arr.index_axis(Axis(0), row).to_vec()
}

fn crow_to_vec(arr: &ArrayView2<Complex64>, row: usize) -> Vec<Complex64> {
    arr.index_axis(Axis(0), row).to_vec()
}

pub fn build_masked_sources(
    ein: &ArrayView2<Complex64>,
    hin: &ArrayView2<Complex64>,
    vis: &ArrayView2<f64>,
    wns: &ArrayView2<f64>,
) -> MaskedSources {
    let n = ein.shape()[1];

    let ex = crow_to_vec(ein, 0);
    let ey = crow_to_vec(ein, 1);
    let ez = crow_to_vec(ein, 2);
    let hx = crow_to_vec(hin, 0);
    let hy = crow_to_vec(hin, 1);
    let hz = crow_to_vec(hin, 2);
    let vx = row_to_vec(vis, 0);
    let vy = row_to_vec(vis, 1);
    let vz = row_to_vec(vis, 2);
    let nx = row_to_vec(wns, 0);
    let ny = row_to_vec(wns, 1);
    let nz = row_to_vec(wns, 2);

    let emag: Vec<f64> = (0..n)
        .map(|j| (ex[j].norm_sqr() + ey[j].norm_sqr() + ez[j].norm_sqr()).sqrt())
        .collect();
    let emax = emag.iter().cloned().fold(0.0_f64, f64::max);
    let elevel = emax * LR;

    let mut out = MaskedSources {
        vx: Vec::new(),
        vy: Vec::new(),
        vz: Vec::new(),
        nx_h: [Vec::new(), Vec::new(), Vec::new()],
        nx_e: [Vec::new(), Vec::new(), Vec::new()],
        e_dot_n: Vec::new(),
        h_dot_n: Vec::new(),
    };

    for j in 0..n {
        if emag[j] <= elevel {
            continue;
        }
        let (ejx, ejy, ejz) = (ex[j], ey[j], ez[j]);
        let (hjx, hjy, hjz) = (hx[j], hy[j], hz[j]);
        let (njx, njy, njz) = (nx[j], ny[j], nz[j]);

        // N x H, N x E (note the numba kernel's sign convention: -(a x b),
        // folded directly into the FMA-fused `scaled_diff` argument order
        // below rather than negating the result afterwards).
        let nxhx = scaled_diff(hjz, njy, hjy, njz);
        let nxhy = scaled_diff(hjx, njz, hjz, njx);
        let nxhz = scaled_diff(hjy, njx, hjx, njy);
        let nxex = scaled_diff(ejz, njy, ejy, njz);
        let nxey = scaled_diff(ejx, njz, ejz, njx);
        let nxez = scaled_diff(ejy, njx, ejx, njy);

        let edn = triple_sum(ejx, njx, ejy, njy, ejz, njz);
        let hdn = triple_sum(hjx, njx, hjy, njy, hjz, njz);

        out.vx.push(vx[j]);
        out.vy.push(vy[j]);
        out.vz.push(vz[j]);
        out.nx_h[0].push(nxhx);
        out.nx_h[1].push(nxhy);
        out.nx_h[2].push(nxhz);
        out.nx_e[0].push(nxex);
        out.nx_e[1].push(nxey);
        out.nx_e[2].push(nxez);
        out.e_dot_n.push(edn);
        out.h_dot_n.push(hdn);
    }

    out
}
