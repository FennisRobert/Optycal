"""Validates the sin/cos LUT sizing claims in KernelConfig's docs / Iteration
4 of claude_nodes/kernel_optimization.md, using the exact table-building and
interpolation code the kernels use (via the diagnostic-only
`_sincos_lut_max_error`, not a reimplementation).
"""
import numpy as np

import optycal_kernels

if __name__ == "__main__":
    print(f"{'n':>6s} {'max_err':>12s} {'dB':>8s}")
    for n in [16, 32, 64, 128, 256, 512]:
        err = optycal_kernels._sincos_lut_max_error(n, 200_000)
        db = 20 * np.log10(err)
        print(f"{n:6d} {err:12.3e} {db:8.1f}")
