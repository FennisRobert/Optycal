from enum import Enum

from optycal_kernels import KernelConfig

class Precision(Enum):
    SINGLE = 1
    DOUBLE = 2

class Settings:

    def __init__(self):
        self.precision: Precision = Precision.SINGLE
        self.geometry_discretization: int = 1_000_000
        self.integration_limit: float = 1e-4
        # Tuning knobs for the Rust Stratton-Chu kernels (target/source
        # tile size, optional Green's-function LUT) -- see
        # claude_nodes/kernel_optimization.md for what these do and how the
        # defaults were chosen. Override globally via
        # `opt.GLOBAL_SETTINGS.kernel_config = opt.KernelConfig(...)`, or
        # pass `kernel_config=...` to an individual `expose_*` call.
        self.kernel_config: KernelConfig = KernelConfig()


GLOBAL_SETTINGS = Settings()