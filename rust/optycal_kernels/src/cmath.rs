//! Hand-fused complex arithmetic for the Stratton-Chu kernels' hot loops.
//!
//! `num_complex::Complex`'s `Mul` impl is the textbook 4-multiply/2-add
//! formula (`(ac-bd) + (ad+bc)i`), written generically with no FMA. Every
//! multiply-then-add/subtract pattern below can instead use `f64::mul_add`
//! (a single hardware FMA instruction on any target with native FMA,
//! including all Apple Silicon / AArch64 -- NEON mandates it): one multiply
//! and one FMA in place of two multiplies and one add/subtract, e.g.
//! `a*c - b*d` becomes `a.mul_add(c, -(b*d))`.
//!
//! This is a genuine instruction-count reduction on FMA-capable hardware,
//! not just fewer source characters. On a target where the compiler can't
//! emit a real FMA instruction (e.g. a generic x86_64 build without
//! `target-feature=+fma`), `mul_add` still computes the correct result via
//! a (slower) software fallback -- correctness never depends on FMA being
//! native, only the speedup does. See `claude_nodes/kernel_optimization.md`
//! Iteration 5 for the measured effect on this machine (Apple M3 Ultra,
//! native FMA) and the x86_64 wheel-build caveat.

use numpy::Complex64;

/// Complex multiply via FMA: `a * b`, as 2 multiplies + 2 fused
/// multiply-adds instead of the naive 4 multiplies + 2 adds.
#[inline(always)]
pub fn cmul(a: Complex64, b: Complex64) -> Complex64 {
    Complex64::new(
        a.re.mul_add(b.re, -(a.im * b.im)),
        a.re.mul_add(b.im, a.im * b.re),
    )
}

/// `a*ra - b*rb` for complex `a`, `b` and real scalars `ra`, `rb` (the
/// shape of every "cross product with a real vector" term in both
/// kernels): 2 multiplies + 2 FMAs instead of 4 multiplies + 2 subtracts.
#[inline(always)]
pub fn scaled_diff(a: Complex64, ra: f64, b: Complex64, rb: f64) -> Complex64 {
    Complex64::new(
        a.re.mul_add(ra, -(b.re * rb)),
        a.im.mul_add(ra, -(b.im * rb)),
    )
}

/// `a*ra - b*rb + c*rc` for complex `a`,`b`,`c` and real scalars
/// `ra`,`rb`,`rc` (the near-field kernel's `N x V` cross-product-plus-
/// radial-term pattern): 3 multiplies + 4 FMAs instead of 6 multiplies +
/// 3 add/subtracts.
#[inline(always)]
pub fn triple_term(a: Complex64, ra: f64, b: Complex64, rb: f64, c: Complex64, rc: f64) -> Complex64 {
    Complex64::new(
        a.re.mul_add(ra, (-b.re).mul_add(rb, c.re * rc)),
        a.im.mul_add(ra, (-b.im).mul_add(rb, c.im * rc)),
    )
}

/// `a*ra + b*rb + c*rc` for complex `a`,`b`,`c` and real scalars
/// `ra`,`rb`,`rc` (no sign flips -- used for `E.N`/`H.N` dot products in
/// `sources.rs`): 3 multiplies + 4 FMAs instead of 6 multiplies + 3 adds.
#[inline(always)]
pub fn triple_sum(a: Complex64, ra: f64, b: Complex64, rb: f64, c: Complex64, rc: f64) -> Complex64 {
    Complex64::new(
        a.re.mul_add(ra, b.re.mul_add(rb, c.re * rc)),
        a.im.mul_add(ra, b.im.mul_add(rb, c.im * rc)),
    )
}

/// Real 3-term dot product `ax*bx + ay*by + az*bz` via FMA: 1 multiply +
/// 2 FMAs instead of 3 multiplies + 2 adds. Used for `R^2` and the
/// far-field phase dot product.
#[inline(always)]
pub fn dot3(ax: f64, bx: f64, ay: f64, by: f64, az: f64, bz: f64) -> f64 {
    az.mul_add(bz, ay.mul_add(by, ax * bx))
}
