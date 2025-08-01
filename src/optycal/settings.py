from enum import Enum

class Precision(Enum):
    SINGLE = 1
    DOUBLE = 2

class Settings:

    def __init__(self):
        self.precision: Precision = Precision.SINGLE
        self.geometry_discretization: int = 1_000_000
        self.integration_limit: float = 1e-4


GLOBAL_SETTINGS = Settings()