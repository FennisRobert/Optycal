//! Rust port of the Stratton-Chu integral kernels (see
//! `optycal/src/optycal/solvers/strattonchu.py` and `strattonchuff.py`).
//!
//! This crate keeps only the current, best-known implementation of each
//! kernel -- `near_field::stratton_chu_xyz` and `far_field::stratton_chu_ff`
//! -- plus the shared `config`/`sources`/`greens` support code they're built
//! from. The full port history (five near-field iterations, two far-field
//! iterations, a numba bug found and fixed along the way, a precision
//! investigation, and a Green's-function LUT experiment that turned out
//! *not* to help) is preserved in `claude_nodes/kernel_optimization.md`,
//! not duplicated here as dead code.

mod antenna_expose;
mod antenna_pattern;
mod cmath;
mod config;
mod far_field;
mod greens;
mod near_field;
mod sources;

use pyo3::prelude::*;

pub use antenna_expose::{antenna_expose_thetaphi, antenna_expose_xyz};
pub use antenna_pattern::AntennaPattern;
pub use config::KernelConfig;
pub use far_field::stratton_chu_ff;
pub use greens::_sincos_lut_max_error;
pub use near_field::stratton_chu_xyz;

#[pymodule]
fn optycal_kernels(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<KernelConfig>()?;
    m.add_class::<AntennaPattern>()?;
    m.add_function(wrap_pyfunction!(stratton_chu_xyz, m)?)?;
    m.add_function(wrap_pyfunction!(stratton_chu_ff, m)?)?;
    m.add_function(wrap_pyfunction!(_sincos_lut_max_error, m)?)?;
    m.add_function(wrap_pyfunction!(antenna_expose_xyz, m)?)?;
    m.add_function(wrap_pyfunction!(antenna_expose_thetaphi, m)?)?;
    Ok(())
}
