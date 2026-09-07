# Optycal is an open source Python based PO Solver.
# Copyright (C) 2025  Robert Fennis.

# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation; either version 2
# of the License, or (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program; if not, see
# <https://www.gnu.org/licenses/>.
from __future__ import annotations
from ..geo.cs import CoordinateSystem, GCS
import numpy as np
import optycal_kernels
from .patterns import (
    dipole_pattern_ff, dipole_pattern_nf, half_dipole_pattern_ff, half_dipole_pattern_nf,
    patch_pattern_ff, patch_pattern_nf,
)
from emsutil.emdata import EHField, EHFieldFF
from emsutil.const import Z0

from ..surface import Surface
from ..samplespace import FarFieldSpace
from .compiled_functions import _c_cross_comp, _c_dot_comp
from ..multilayer import FRES_AIR
from .interpolation_pattern import AntennaPattern
from ..settings import GLOBAL_SETTINGS, Precision
from functools import reduce, lru_cache
from loguru import logger
from .compiled.antenna_single import expose_surface_single, expose_thetaphi_single, expose_xyz_single
from typing import Callable

# Angle grid used to auto-sample any custom (non-native) far-field pattern
# function into the bicubic-spline interpolation data
# `optycal_kernels.AntennaPattern.interpolated(...)` evaluates in Rust.
# Physical spherical convention (theta in [0,pi], phi in [-pi,pi]) --
# deliberately NOT the same [-pi/2,pi/2] theta convention
# `InterpolatingAntenna.__init__` uses (see claude_nodes/antenna_migration.md
# for why): that convention only works because `dipole_pattern_ff` happens to
# remap theta internally, which isn't true of pattern functions in general.
_CUSTOM_PATTERN_THETA_GRID = np.linspace(0, np.pi, 181, dtype=np.float32)
_CUSTOM_PATTERN_PHI_GRID = np.linspace(-np.pi, np.pi, 361, dtype=np.float32)


@lru_cache(maxsize=256)
def _build_rust_pattern(nf_pattern: Callable, ff_pattern: Callable, k0: float) -> optycal_kernels.AntennaPattern:
    """Maps an (nf_pattern, ff_pattern) callable pair onto a Rust
    `AntennaPattern`: the native, exact `Dipole`/`HalfDipole` variants for
    the two patterns that have one, or -- for anything else, including
    parametrized patterns from `generate_gaussian_pattern`/
    `generate_patch_pattern`/`generate_triang_pattern` and arbitrary
    user-defined callables -- a bicubic-spline gridded-interpolation
    approximation built by sampling `ff_pattern` once here in Python
    (reusing the existing, unchanged `AntennaPattern.from_function`/
    `compute_interpolator_matrix` machinery) and handing the coefficient
    grid to Rust for fast evaluation.

    Matches `InterpolatingAntenna`'s existing, already-accepted behavior:
    the near field of a non-native pattern is evaluated as the far-field
    pattern shape times the standard `amplitude*exp(-ikR)/R` radial
    falloff, ignoring any genuine `r`-dependence `nf_pattern` might define
    (only `dipole`/`half_dipole` keep their exact reactive near-field
    terms). See `claude_nodes/antenna_migration.md`.

    `@lru_cache`d on `(nf_pattern, ff_pattern, k0)` identity/value --
    critical for `AntennaArray`, which builds one `Antenna` per element
    (e.g. 200 for a 20x10 array) that all share the exact same pattern
    functions and k0: without this cache, every element would redundantly
    rebuild the same 181x361-grid bicubic spline (a handful of dense
    `np.linalg.solve` calls per grid row/column, times 6 field components)
    from scratch, which is slow enough to make array construction look
    like a hang. `AntennaPattern` is stateless/read-only once built, so
    sharing the same instance across antennas is safe. Only pushes work
    away, never staleness: `Antenna.frequency` reassignment after
    construction already doesn't rebuild the pattern regardless of this
    cache (see this function's caller).
    """
    if nf_pattern is dipole_pattern_nf and ff_pattern is dipole_pattern_ff:
        return optycal_kernels.AntennaPattern.dipole()
    if nf_pattern is half_dipole_pattern_nf and ff_pattern is half_dipole_pattern_ff:
        return optycal_kernels.AntennaPattern.half_dipole()
    if nf_pattern is patch_pattern_nf and ff_pattern is patch_pattern_ff:
        return optycal_kernels.AntennaPattern.patch(np.pi, np.pi, 0.5)
    # generate_patch_pattern(...) returns fresh closures each call, so they
    # can't be identity-matched like the plain patterns above -- it tags
    # its own closures with `_optycal_patch_params` instead (see
    # patterns.py) for this to detect.
    patch_params = getattr(ff_pattern, '_optycal_patch_params', None)
    if patch_params is not None:
        kw, kl, t_exp = patch_params
        return optycal_kernels.AntennaPattern.patch(kw, kl, t_exp)

    pattern = AntennaPattern.from_function(
        ff_pattern, _CUSTOM_PATTERN_THETA_GRID, _CUSTOM_PATTERN_PHI_GRID, k0
    )
    full_matrix = pattern.full_matrix(Precision.DOUBLE)
    return optycal_kernels.AntennaPattern.interpolated(
        _CUSTOM_PATTERN_THETA_GRID, _CUSTOM_PATTERN_PHI_GRID, full_matrix
    )

class Antenna:
    
    def __init__(self, x: float, y: float, z: float, frequency: float,  cs: CoordinateSystem = None,
                 nf_pattern = None, ff_pattern = None, name: str = 'Antenna'):
        self.x: float = x
        self.y: float = y
        self.z: float = z

        if cs is None:
            cs = GCS

        self._physical_cs: CoordinateSystem = cs
        self._phase_cs: CoordinateSystem = None
        self._deformed_cs: CoordinateSystem = None
        self.name: str = name
        self.frequency: float = frequency
        self.taper_coefficient: complex = 1
        self.array_compensation: complex = 1
        self.scan_coefficient: complex = 1
        self.correction_coefficient: complex = 1
        self.aux_coefficients: list[complex] = [1.0,]
        self.active: float = 1

        self.power: float | None = None
        
        if ff_pattern is None:
            logger.debug('Defaulting to dipole farfield pattern')
            self.ff_pattern: Callable = dipole_pattern_ff
        else:
            self.ff_pattern: Callable = ff_pattern
        if nf_pattern is None:
            logger.debug('Default to dipole near field pattern.')
            self.nf_pattern: Callable = dipole_pattern_nf
        else:
            self.nf_pattern: Callable = nf_pattern

        # Rust-backed pattern evaluation for expose_xyz/expose_thetaphi --
        # built once here (like InterpolatingAntenna's interp_pattern is),
        # so it goes stale the same pre-existing way if `.frequency` is
        # reassigned after construction. See _build_rust_pattern's docstring
        # and claude_nodes/antenna_migration.md.
        self._rust_pattern: optycal_kernels.AntennaPattern = _build_rust_pattern(
            self.nf_pattern, self.ff_pattern, self.k0
        )

    @property
    def cs(self):
        if self._deformed_cs is not None:
            return self._deformed_cs
        else:
            return self._physical_cs
      
    def __str__(self) -> str:
        return f"Antenna[{self.name}]"
    
    @property
    def k0(self) -> float:
        """The antennas propagation constant

        Returns:
            float: The propagation constant
        """
        return 2 * np.pi * self.frequency / 299792458
    
    @property
    def amplitude(self) -> float:
        """ The antenna excitation amplitude"""
        return self.taper_coefficient * self.scan_coefficient * self.correction_coefficient * self.array_compensation * self.active * reduce(lambda x,y: x*y, self.aux_coefficients)

    @property
    def camp(self) -> np.complex64:
        """The complex amplitude in c64 format

        Returns:
            np.complex64: _description_
        """
        return np.complex64(self.amplitude)
    
    @property
    def local_xyz(self) -> tuple:
        """The position in local XYZ coordinates

        Returns:
            tuple: _description_
        """
        return self.x, self.y, self.z
    
    @property
    def gxyz(self) -> tuple[float, float, float]:
        """The Global XYZ position

        Returns:
            tuple[float, float, float]: _description_
        """
        return self.cs.in_global_cs(self.x, self.y, self.z)
    
    @property
    def phase_gxyz(self) -> tuple[float, float, float]:
        """The XYZ coordinates for computing the required antenna phase

        Returns:
            tuple[float, float, float]: _description_
        """
        if self._phase_cs is not None:
            return self._phase_cs.in_global_cs(self.x,self.y,self.z)
        return self._physical_cs.in_global_cs(self.x, self.y, self.z)
    
    @property
    def physical_gxyz(self) -> np.ndarray:
        return self.cs.in_global_cs(self.x, self.y, self.z)
    
    @property
    def gx(self) -> float:
        """The global X-coordinate
        """
        return self.gxyz[0]

    @property
    def gy(self) -> float:
        """ The global Y-coordinate"""
        return self.gxyz[1]

    @property
    def gz(self) -> float:
        """ The global Z-coordinate"""
        return self.gxyz[2]

    def __repr__(self) -> str:
        return f"Antenna(x={self.x}, y={self.y}, z={self.z})"
    
    def __expose__(self, target: Surface | FarFieldSpace, **options):
        if isinstance(target, Surface):
            return self.expose_surface(target)
        elif isinstance(target, FarFieldSpace):
            return self.compute(target)
        else:
            raise TypeError('Target must be either a Surface or a SampleSpace')
    
    def reset_deformation(self):
        self._deformed_cs = None

    def deform(self, T: Callable) -> None:
        """Deforms the antenna by placing it physically in a different spot.
        The transofrmation T(x,y,z) -> (x,y,z) determins the displacement
        
        Args:
            T (Callable): The displacement function
        """
        gxyz = np.array(self.gxyz)
        xh = self.cs.gxhat
        yh = self.cs.gyhat
        zh = self.cs.gzhat
        e = 1e-6
        dgxyz = T(*gxyz)

        # Compute tiny basis vectors to make sure the panel experiences basis vector transformation
        cx = gxyz+e*xh
        cy = gxyz+e*yh
        cz = gxyz+e*zh

        # Transform the basis vector coordinates
        dxt = T(*cx)
        dyt = T(*cy)
        dzt = T(*cz)

        # Compute the new XYZ unit vectors
        xhn = (dxt-dgxyz)/np.linalg.norm(dxt-dgxyz)
        yhn = (dyt-dgxyz)/np.linalg.norm(dyt-dgxyz)
        zhn = (dzt-dgxyz)/np.linalg.norm(dzt-dgxyz)
        xhn = np.cross(yhn, zhn)
        newcs = CoordinateSystem(dgxyz, xhn, yhn, zhn, parent=GCS)
        logger.debug(newcs)
        self._deformed_cs = newcs

    def expose_xyz(self, gx: np.ndarray, gy: np.ndarray, gz: np.ndarray) -> EHField:
        """
        Compute the nearfield of the antenna at the points (gx, gy, gz)
        """
        basis = np.ascontiguousarray(self.cs.global_basis, dtype=np.float64)
        basis_inv = np.ascontiguousarray(self.cs.global_basis_inv, dtype=np.float64)
        E, H = optycal_kernels.antenna_expose_xyz(
            np.ascontiguousarray(gx, dtype=np.float64),
            np.ascontiguousarray(gy, dtype=np.float64),
            np.ascontiguousarray(gz, dtype=np.float64),
            list(self.gxyz),
            basis,
            basis_inv,
            self._rust_pattern,
            complex(self.amplitude),
            self.k0,
        )

        return EHField(_E=np.asarray(E), _H=np.asarray(H), x=gx, y=gy, z=gz, freq=self.frequency, aux={'creator': self.name})
    

    def expose_thetaphi(self, gtheta: np.ndarray, gphi: np.ndarray) -> EHFieldFF:
        """
        Compute the farfield of the antenna at the points (theta, phi)
        """
        gtheta = gtheta.astype(np.float32)
        gphi = gphi.astype(np.float32)
        basis = np.ascontiguousarray(self.cs.global_basis, dtype=np.float64)
        basis_inv = np.ascontiguousarray(self.cs.global_basis_inv, dtype=np.float64)
        E, H = optycal_kernels.antenna_expose_thetaphi(
            np.ascontiguousarray(gtheta, dtype=np.float32),
            np.ascontiguousarray(gphi, dtype=np.float32),
            list(self.local_xyz),
            basis,
            basis_inv,
            self._rust_pattern,
            complex(self.amplitude),
            self.k0,
        )
        return EHFieldFF(_E=np.asarray(E), _H=np.asarray(H), theta=gtheta, phi=gphi, Ptot=self.power)
    

    def expose_kxyz(self, kx: np.ndarray, ky: np.ndarray, kz: np.ndarray) -> EHFieldFF:
        """
        Compute the farfield of the antenna at the points (theta, phi)
        """
        kxl, kyl, kzl = self.cs.from_global_basis(kx, ky, kz)
        kx, ky, kz = self.k0*kxl, self.k0*kyl, self.k0*kzl
        B = self.amplitude * np.exp(1j * (kx * self.x + ky * self.y + kz * self.z))
        [ex, ey, ez, hx, hy, hz] = self.ff_pattern(kxl, kyl, kzl, self.k0)
        
        E1 = np.array(self.cs.in_global_basis(ex, ey, ez))
        H1 = np.array(self.cs.in_global_basis(hx, hy, hz))
        E = B * E1
        H = B * H1

        #logger.debug("Field Computation Complete")
        theta = np.arccos(kzl)
        phi = np.arctan2(kyl, kxl)
        return EHFieldFF(_E=E, _H=H, theta=theta, phi=phi, Ptot=self.power)
    
    def expose_surface(self, surface: Surface, add_field: bool = True) -> EHField:
        """Expose a surface object with EM energy

        Args:
            surface (Surface): The surface to expose
            add_field (bool, optional): If the field should be added instead of overwritten. Defaults to True.

        Returns:
            Field: _description_
        """
        refang = surface.fresnel.angles
        
        E1 = np.zeros(surface.fieldshape, dtype=np.complex64)
        E2 = np.zeros_like(E1, dtype=np.complex64)
        H1 = np.zeros_like(E1, dtype=np.complex64)
        H2 = np.zeros_like(E1, dtype=np.complex64)
        xyz = surface.field_coordinates().astype(np.float32)
        x = np.float32(xyz[0,:])
        y = np.float32(xyz[1,:])
        z = np.float32(xyz[2,:])
        fr = self.expose_xyz(x, y, z)
        E = fr.E
        H = fr.H
        Ex = E[0,:]
        Ey = E[1,:]
        Ez = E[2,:]
        Hx = H[0,:]
        Hy = H[1,:]
        Hz = H[2,:]
        
        gx, gy, gz = self.gxyz
        rsx = x-gx
        rsy = y-gy
        rsz = z-gz
        tn = surface.field_normals()
        tn = tn + np.random.rand(*tn.shape)*1e-8
        tn = tn / np.linalg.norm(tn, axis=0)
        R = np.sqrt(rsx**2 + rsy**2 + rsz**2)
        tnx = tn[0,:]
        tny = tn[1,:]
        tnz = tn[2,:]
        
        rdotn = (rsx*tnx + rsy*tny + rsz*tnz)/R
        sphx, sphy, sphz = _c_cross_comp(rsx, rsy, rsz, tnx, tny, tnz)
        S = np.sqrt(sphx**2 + sphy**2 + sphz**2)
        
        sphx = sphx/S
        sphy = sphy/S
        sphz = sphz/S
        
        pphx, pphy, pphz = _c_cross_comp(rsx, rsy, rsz, sphx, sphy, sphz)
        P = np.sqrt(pphx**2 + pphy**2 + pphz**2)
        pphx = pphx/P
        pphy = pphy/P
        pphz = pphz/P
        
        pdn = pphx*tnx + pphy*tny + pphz*tnz
        pprhx = 2*pdn*tnx - pphx
        pprhy = 2*pdn*tny - pphy
        pprhz = 2*pdn*tnz - pphz
        
        angin = np.arccos(np.clip(np.abs(rdotn), a_min=0, a_max=1))
        
        Rte1 = np.interp(angin, refang, surface.fresnel.Rte1)
        Rtm1 = np.interp(angin, refang, surface.fresnel.Rtm1)
        Rte2 = np.interp(angin, refang, surface.fresnel.Rte2)
        Rtm2 = np.interp(angin, refang, surface.fresnel.Rtm2)
        Tte = np.interp(angin, refang, surface.fresnel.Tte)
        Ttm = np.interp(angin, refang, surface.fresnel.Ttm)
        
        same = (rdotn>0).astype(np.float32)
        other = 1-same

        Rte = Rte1*same + Rte2*other
        Rtm = Rtm1*same + Rtm2*other
        
        Es = _c_dot_comp(Ex, Ey, Ez, sphx, sphy, sphz)
        Ep = _c_dot_comp(Ex, Ey, Ez, pphx, pphy, pphz)
        Hs = _c_dot_comp(Hx, Hy, Hz, sphx, sphy, sphz)
        Hp = _c_dot_comp(Hx, Hy, Hz, pphx, pphy, pphz)
        
        Erefx = Rte*(Es*sphx) + Rtm*(Ep*pprhx)
        Erefy = Rte*(Es*sphy) + Rtm*(Ep*pprhy)
        Erefz = Rte*(Es*sphz) + Rtm*(Ep*pprhz)
        Etransx = Tte*(Es*sphx) + Ttm*(Ep*pphx)
        Etransy = Tte*(Es*sphy) + Ttm*(Ep*pphy)
        Etransz = Tte*(Es*sphz) + Ttm*(Ep*pphz)
        
        Hrefx = Rtm*(Hs*sphx) + Rte*(Hp*pprhx)
        Hrefy = Rtm*(Hs*sphy) + Rte*(Hp*pprhy)
        Hrefz = Rtm*(Hs*sphz) + Rte*(Hp*pprhz)
        Htransx = Ttm*(Hs*sphx) + Tte*(Hp*pphx)
        Htransy = Ttm*(Hs*sphy) + Tte*(Hp*pphy)
        Htransz = Ttm*(Hs*sphz) + Tte*(Hp*pphz)
        
        
        E1[0,:] = Erefx*same + Etransx*other
        E1[1,:] = Erefy*same + Etransy*other
        E1[2,:] = Erefz*same + Etransz*other
        H1[0,:] = Hrefx*same + Htransx*other
        H1[1,:] = Hrefy*same + Htransy*other
        H1[2,:] = Hrefz*same + Htransz*other
        E2[0,:] = Erefx*other + Etransx*same
        E2[1,:] = Erefy*other + Etransy*same
        E2[2,:] = Erefz*other + Etransz*same
        H2[0,:] = Hrefx*other + Htransx*same
        H2[1,:] = Hrefy*other + Htransy*same
        H2[2,:] = Hrefz*other + Htransz*same

        fr1 = EHField(_E=E1, _H=H1, x=x, y=y, z=z, freq=self.frequency)
        fr2 = EHField(_E=E2, _H=H2, x=x, y=y, z=z, freq=self.frequency)
        
        if add_field:
            surface.add_field(1, E=E1, H=H1, k0=self.k0)
            surface.add_field(2, E=E2, H=H2, k0=self.k0)
        return fr1, fr2
    
    def expose_ff(self, target: FarFieldSpace) -> EHFieldFF:
        """Expose a FarFieldSpace object

        Args:
            target (FarFieldSpace): _description_

        Returns:
            Field: The resultant Field
        """
        fr = self.expose_thetaphi(target.theta, target.phi)
        target.field = fr
        return fr
    
    def receive_from(self, other: Antenna | Surface) -> complex:
        """Compute the complex received signal/voltage from another Antenna or Surface.

        Uses Lorentz Reciprocity (Reaction Concept) to couple incoming fields 
        with this antenna's receiving response and complex array excitation.

        Args:
            other (Antenna | Surface): Source of the incoming EM field.

        Returns:
            complex: The complex received signal value.
        """
    
        # 1. Compute the incident field generated by 'other' at this antenna's position
        gx = np.array([self.gx], dtype=np.float32)
        gy = np.array([self.gy], dtype=np.float32)
        gz = np.array([self.gz], dtype=np.float32)
        
        field = other.expose_xyz(gx, gy, gz)
        E_inc = field.E[:, 0]  # Global complex (Ex, Ey, Ez) at self position

        # 2. Vector pointing from self towards 'other'
        d_global = np.array(other.gxyz) - np.array(self.gxyz)
        dist = np.linalg.norm(d_global)
        
        if dist == 0:
            return 0.0 + 0.0j
            
        k_global = d_global / dist  # Unit arrival direction vector

        # 3. Convert arrival direction to self's local coordinate system
        lkx, lky, lkz = self.cs.from_global_basis(k_global[0], k_global[1], k_global[2])
        theta_local = np.arccos(np.clip(lkz, -1.0, 1.0))
        phi_local = np.arctan2(lky, lkx)

        # 4. Evaluate self's farfield radiation/reception pattern in that arrival direction
        [ex, ey, ez, _, _, _] = self.ff_pattern(theta_local, phi_local, self.k0)
        E_pat_global = np.array(self.cs.in_global_basis(ex, ey, ez)).flatten()

        # 5. Reciprocal coupling (E_inc . E_pat) scaled by element amplitude/phase weighting
        received_signal = np.dot(E_inc, E_pat_global) * self.amplitude
        return complex(received_signal)
    
    def reset_aux(self):
        """Resets any auxilliary scan coefficients.
        """
        self.aux_coefficients = [1,]

    def normalize_power(self, power: float = 1.0):
        """Normalizes the total radiated power to the desired amount

        Args:
            power (float): The power to radiate
        """
        from ..geo.mesh.generators import generate_sphere
        
        _lambda = 2 * np.pi / self.k0
        Rmax = 5*_lambda
        p0 = np.array(self.gxyz)
        mesh = generate_sphere(p0, 1.5 * Rmax, _lambda/2, self.cs.get_global())
        surf = Surface(mesh, FRES_AIR)
        self.expose_surface(surf)

        Po = sum(surf.powerflux())
        
        logger.debug(f"Measured power: {Po} W")
        self.correction_coefficient = self.correction_coefficient * np.sqrt(power / Po)
        logger.debug(f"Compensation factor: {self.correction_coefficient}")
        self.power = power
        
    def accelerate(self) -> InterpolatingAntenna:
        """Return an antenna that uses an interpolation function instead of the antenna function (midly faster)

        Returns:
            InterpolatingAntenna: _description_
        """
        return InterpolatingAntenna(
            self.x, self.y, self.z, 
            self.frequency, 
            self.cs, self.nf_pattern, self.ff_pattern, self.name + "_Accelerated")


class InterpolatingAntenna(Antenna):

    def __init__(self, x: float, y: float, z: float, frequency: float,  cs: CoordinateSystem,
                 nf_pattern = None, ff_pattern = None, name: str = 'Antenna'):
        super().__init__(x, y, z, frequency, cs, nf_pattern, ff_pattern, name)
        th = np.linspace(-np.pi/2, np.pi/2, 51)
        ph = np.linspace(-np.pi, np.pi, 101)
        self.interp_pattern: AntennaPattern = AntennaPattern.from_function(self.ff_pattern,th, ph, self.k0)

    
    def expose_thetaphi(self, gtheta, gphi) -> EHFieldFF:
        gtheta = gtheta.astype(np.float32)
        gphi = gphi.astype(np.float32)
        gxyz = np.array(self.gxyz).astype(np.float64)
        E, H = expose_thetaphi_single(gtheta, 
                                      gphi, 
                                      gxyz, 
                                      self.interp_pattern.full_matrix(Precision.SINGLE),
                                      self.interp_pattern.theta_grid, 
                                      self.interp_pattern.phi_grid,
                                      self.cs.global_basis,
                                      self.camp,
                                      self.k0)
        return EHFieldFF(_E=E, _H=H, theta=gtheta, phi=gphi)
    
    def expose_xyz(self, gx, gy, gz) -> EHField:
        gxyz = np.array(self.gxyz)
        E, H = expose_xyz_single(gx,
                                 gy, 
                                 gz,  
                                 gxyz, 
                                 self.interp_pattern.full_matrix(Precision.SINGLE),
                                 self.interp_pattern.theta_grid, 
                                 self.interp_pattern.phi_grid,
                                 self.cs.global_basis,
                                 self.camp,
                                 self.k0)
        return EHField(_E=E, _H=H, x=gx, y=gy, z=gz, freq=self.frequency)
    
    def expose_surface(self, surface: Surface, add_field = True) -> tuple[EHField, EHField]:
        gxyz = np.array(self.gxyz)
        E1, H1, E2, H2 = expose_surface_single(surface.gxyz, 
                                               surface.field_normals(),
                                               surface.fresnel.rt_data,
                                               gxyz, 
                                               self.interp_pattern.full_matrix(Precision.SINGLE),
                                               self.interp_pattern.theta_grid,
                                               self.interp_pattern.phi_grid,
                                               self.cs.global_basis,
                                               self.camp,
                                               self.k0)
        surface.add_field(1, E=E1, H=H1, k0=self.k0)
        surface.add_field(2, E=E2, H=H2, k0=self.k0)
        return (EHField(x=surface.gx, y=surface.gy, z=surface.gz, _E=E1, _H=H1, freq=self.frequency), 
               EHField(x=surface.gx, y=surface.gy, z=surface.gz, _E=E2, _H=H2, freq=self.frequency))

class EMergeAntenna(Antenna):
    
    def __init__(self, x: float, y: float, z: float, emdata: dict, 
                 cs: CoordinateSystem | None = GCS, name: str = 'Antenna', angle_step: float = 5.0):
        frequency = emdata['freq']
        th = np.linspace(0, np.pi, int(np.ceil(180/angle_step)))
        ph = np.linspace(-np.pi, np.pi, int(np.ceil(360/angle_step)))
        self.interp_pattern: AntennaPattern = AntennaPattern.from_function(emdata['ff_function'], th, ph, 2*np.pi*frequency/299792458)
        nf_pattern = self.interp_pattern.nf_pattern
        ff_pattern = self.interp_pattern.ff_pattern
        
        super().__init__(x, y, z, frequency, cs, nf_pattern, ff_pattern, name)

    
    def expose_thetaphi(self, gtheta, gphi) -> EHFieldFF:
        gtheta = gtheta.astype(np.float32)
        gphi = gphi.astype(np.float32)
        gxyz = np.array(self.gxyz).astype(np.float64)
        E, H = expose_thetaphi_single(gtheta, 
                                      gphi, 
                                      gxyz, 
                                      self.interp_pattern.full_matrix(Precision.SINGLE),
                                      self.interp_pattern.theta_grid, 
                                      self.interp_pattern.phi_grid,
                                      self.cs.global_basis,
                                      self.camp,
                                      self.k0)
        return EHFieldFF(_E=E, _H=H, theta=gtheta, phi=gphi)
    
    def expose_xyz(self, gx, gy, gz) -> EHField:
        gxyz = np.array(self.gxyz)
        E, H = expose_xyz_single(gx,
                                 gy, 
                                 gz,  
                                 gxyz, 
                                 self.interp_pattern.full_matrix(Precision.SINGLE),
                                 self.interp_pattern.theta_grid, 
                                 self.interp_pattern.phi_grid,
                                 self.cs.global_basis,
                                 self.camp,
                                 self.k0)
        return EHField(_E=E, _H=H, x=gx, y=gy, z=gz, freq=self.frequency)
    
    def expose_surface(self, surface: Surface, add_field = True) -> tuple[EHField, EHField]:
        gxyz = np.array(self.gxyz)
        E1, H1, E2, H2 = expose_surface_single(surface.gxyz, 
                                               surface.field_normals(),
                                               surface.fresnel.rt_data,
                                               gxyz, 
                                               self.interp_pattern.full_matrix(Precision.SINGLE),
                                               self.interp_pattern.theta_grid,
                                               self.interp_pattern.phi_grid,
                                               self.cs.global_basis,
                                               self.camp,
                                               self.k0)
        
        surface.add_field(1, E=E1, H=H1, k0=self.k0)
        surface.add_field(2, E=E2, H=H2, k0=self.k0)
        return (EHField(_E=E1, _H=H1, x=surface.gx, y=surface.gy, z=surface.gz, freq=self.frequency),
                EHField(_E=E2, _H=H2, x=surface.gx, y=surface.gy, z=surface.gz, freq=self.frequency))