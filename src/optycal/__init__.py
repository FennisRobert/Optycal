from loguru import logger
import sys

logger.remove()
logger.add(sys.stderr, level="WARNING")

from .antennas.antenna import Antenna, EMergeAntenna
from .antennas.array import AntennaArray, taper
from .antennas.patterns import Eiso, Eomni, dipole_pattern_ff, dipole_pattern_nf, half_dipole_pattern_ff, half_dipole_pattern_nf, patch_pattern_nf, patch_pattern_ff, generate_gaussian_pattern, generate_gaussian_pattern_z, generate_patch_pattern, generate_triang_pattern
from .geo.cs import CoordinateSystem, GCS
from .settings import Settings, GLOBAL_SETTINGS
from .samplespace import FF1D, FF2D, FarFieldSpace, NearFieldSpace
from .geo.mesh import Mesh
from .geo.mesh.generators import generate_circle, generate_rectangle, generate_sphere
from .geo.mesh.parametric import ParametricLine, SweepFunction, Mapping
from .surface import Surface
from .multilayer import MultiLayer, MAT_AIR, FRES_AIR, FRES_PEC
from .viewer import OptycalDisplay
from .plot import plot, plot_ff, plot_ff_polar
from .geo.align import AlignOrigin, AlignX, AlignY, AlignZ

#from .tapers import chebwin, cosine, taylor, bartlett, blackman, blackmanharris, boxcar, dpss, flattop, gaussian, general_gaussian, hamming, hann, kaiser, nuttall, triang, tukey
from . import tapers as taper