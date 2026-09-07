"""Empirical check of the maintainer's question: does evaluating
`exp(-i*k0*R)` via f64 `sin_cos(k0*R)` actually lose meaningful precision
once `R` gets very large (many wavelengths), the way it would for a naive
argument-reduction implementation?

Ground truth from mpmath at 50 decimal digits; f64 error measured against
that. k0 chosen for a representative mm-wave/near-THz frequency so R can be
pushed to physically implausible distances before we run out of float64
range.
"""
import mpmath as mp
import numpy as np

mp.mp.dps = 50

C0 = 299792458.0


def max_abs_err(freq_hz: float, distances_m):
    k0 = 2 * np.pi * freq_hz / C0
    print(f"\nfreq={freq_hz:.3g} Hz, k0={k0:.6g} rad/m")
    print(f"{'R (m)':>14s} {'k0*R (rad)':>16s} {'|sin err|':>12s} {'|cos err|':>12s} {'phase err est (rad)':>20s}")
    for r in distances_m:
        theta = k0 * r
        s_f64, c_f64 = np.sin(theta), np.cos(theta)

        theta_mp = mp.mpf(k0) * mp.mpf(r)
        s_mp = float(mp.sin(theta_mp))
        c_mp = float(mp.cos(theta_mp))

        s_err = abs(s_f64 - s_mp)
        c_err = abs(c_f64 - c_mp)
        # theoretical estimate: relative error in theta (~ eps) times theta itself
        phase_err_est = theta * np.finfo(np.float64).eps
        print(f"{r:14.4g} {theta:16.6g} {s_err:12.3e} {c_err:12.3e} {phase_err_est:20.3e}")


if __name__ == "__main__":
    # Representative PO scales: mm to a few km, at a mm-wave frequency.
    print("=== Realistic-to-generous PO distance range, 300 GHz ===")
    max_abs_err(300e9, [1e-3, 1.0, 10.0, 1e3, 1e5, 1e6])

    # Push far beyond anything physically sensible for this codebase, to
    # find where f64 sin/cos actually starts to break down.
    print("\n=== Deliberately extreme distances, 300 GHz ===")
    max_abs_err(300e9, [1e9, 1e12, 1e15, 1e18])

    # And at a much higher (sub-mm / THz-ish) k0 to see if frequency alone
    # (independent of distance) changes the picture.
    print("\n=== Extreme distances, 3 THz ===")
    max_abs_err(3e12, [1e3, 1e9, 1e15])
