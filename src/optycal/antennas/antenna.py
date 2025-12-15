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
from .patterns import dipole_pattern_ff, dipole_pattern_nf
from ..field import Field
from ..surface import Surface
from ..samplespace import FarFieldSpace
from .compiled_functions import _c_cross_comp, _c_dot_comp
from ..multilayer import FRES_AIR
from .interpolation_pattern import AntennaPattern
from ..settings import GLOBAL_SETTINGS, Precision
from functools import reduce
from loguru import logger
from .compiled.antenna_single import expose_surface_single, expose_thetaphi_single, expose_xyz_single
from typing import Callable

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
        return 2 * np.pi * self.frequency / 299792458
    
    @property
    def amplitude(self) -> float:
        return self.taper_coefficient * self.scan_coefficient * self.correction_coefficient * self.array_compensation * self.active * reduce(lambda x,y: x*y, self.aux_coefficients)

    @property
    def camp(self) -> np.complex64:
        return np.complex64(self.amplitude.real)
    
    @property
    def local_xyz(self) -> tuple:
        return self.x, self.y, self.z
    
    @property
    def gxyz(self) -> tuple[float, float, float]:
        return self.cs.in_global_cs(self.x, self.y, self.z)
    
    @property
    def phase_gxyz(self) -> tuple[float, float, float]:
        if self._phase_cs is not None:
            return self._phase_cs.in_global_cs(self.x,self.y,self.z)
        return self._physical_cs.in_global_cs(self.x, self.y, self.z)
    
    @property
    def physical_gxyz(self) -> np.ndarray:
        return self.cs.in_global_cs(self.x, self.y, self.z)
    
    @property
    def gx(self) -> float:
        return self.gxyz[0]

    @property
    def gy(self) -> float:
        return self.gxyz[1]

    @property
    def gz(self) -> float:
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
    
    def restore_cs(self):
        self._deformed_cs = None

    def deform(self, T: Callable) -> None:
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

    def expose_xyz(self, gx: np.ndarray, gy: np.ndarray, gz: np.ndarray) -> Field:
        """
        Compute the nearfield of the antenna at the points (gx, gy, gz)
        """
        sx, sy, sz = self.gxyz

        dx = gx - sx
        dy = gy - sy
        dz = gz - sz

        E = np.zeros((3, gx.shape[0]), dtype=np.complex128)
        H = np.zeros((3, gx.shape[0]), dtype=np.complex128)
        
        R = np.sqrt(dx**2 + dy**2 + dz**2)
        kx = dx/R
        ky = dy/R
        kz = dz/R
        
        lkx, lky, lkz = self.cs.from_global_basis(kx, ky, kz)
        thetac = np.arccos(lkz)
        phic = np.arctan2(lky, lkx)
        
        B = self.amplitude * np.exp(-1j * self.k0 * R) / R

        [ex, ey, ez, hx, hy, hz] = self.nf_pattern(thetac, phic, R, self.k0)

        ex, ey, ez = self.cs.in_global_basis(ex, ey, ez)
        hx, hy, hz = self.cs.in_global_basis(hx, hy, hz)
        
        E[0,:] = ex*B
        E[1,:] = ey*B
        E[2,:] = ez*B
        H[0,:] = hx*B
        H[1,:] = hy*B
        H[2,:] = hz*B
        return Field(E=E, H=H)
    

    def expose_thetaphi(self, gtheta: np.ndarray, gphi: np.ndarray) -> Field:
        """
        Compute the farfield of the antenna at the points (theta, phi)
        """
        gtheta = gtheta.astype(np.float32)
        gphi = gphi.astype(np.float32)
        theta_local, phi_local = self.cs.ae_from_global_cs(gtheta, gphi)
        cst = np.cos(theta_local)
        csp = np.cos(phi_local)
        snt = np.sin(theta_local)
        snp = np.sin(phi_local)
        kxh = snt * csp
        kyh = snt * snp
        kzh = cst
        
        theta_local = np.arccos(kzh)
        phi_local = np.arctan2(kyh, kxh)
        x0, y0, z0 = [0,0,0]
        kx, ky, kz = self.k0*kxh, self.k0*kyh, self.k0*kzh
        gx, gy, gz = self.local_xyz
        B = self.amplitude * np.exp(1j * (kx * gx + ky * gy + kz * gz))
        [ex, ey, ez, hx, hy, hz] = self.ff_pattern(theta_local, phi_local, self.k0)
        E1 = np.array(self.cs.in_global_basis(ex, ey, ez))
        H1 = np.array(self.cs.in_global_basis(hx, hy, hz))

        E = B * E1
        H = B * H1
        return Field(E=E, H=H, theta=gtheta, phi=gphi)
    

    def expose_kxyz(self, kx: np.ndarray, ky: np.ndarray, kz: np.ndarray) -> Field:
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
        return Field(E=E, H=H)
    
    def expose_surface(self, surface: Surface, add_field: bool = True) -> Field:
        
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
        
        fr1 = Field(E=E1, H=H1)
        fr2 = Field(E=E2, H=H2)
        if add_field:
            surface.add_field(1, fr1, self.k0)
            surface.add_field(2, fr2, self.k0)
        return fr1, fr2
    
    def expose_ff(self, target: FarFieldSpace) -> Field:
        fr = self.expose_thetaphi(target.theta, target.phi)
        target.field = fr
        return fr
    
    def reset_aux(self):
        self.aux_coefficients = [1,]

    def normalize_power(self, power: float):
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
    
    def accelerate(self) -> InterpolatingAntenna:
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

    
    def expose_thetaphi(self, gtheta, gphi) -> Field:
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
        return Field(E=E, H=H, theta=gtheta, phi=gphi)
    
    def expose_xyz(self, gx, gy, gz) -> Field:
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
        return Field(E=E, H=H, x=gx, y=gy, z=gz)
    
    def expose_surface(self, surface: Surface, add_field = True) -> tuple[Field, Field]:
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
        surface.add_field(1, Field(E=E1, H=H1), self.k0)
        surface.add_field(2, Field(E=E2, H=H2), self.k0)
        return Field(E=E1, H=H1), Field(E=E2, H=H2)
    
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

    
    def expose_thetaphi(self, gtheta, gphi) -> Field:
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
        return Field(E=E, H=H, theta=gtheta, phi=gphi)
    
    def expose_xyz(self, gx, gy, gz) -> Field:
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
        return Field(E=E, H=H, x=gx, y=gy, z=gz)
    
    def expose_surface(self, surface: Surface, add_field = True) -> tuple[Field, Field]:
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
        surface.add_field(1, Field(E=E1, H=H1), self.k0)
        surface.add_field(2, Field(E=E2, H=H2), self.k0)
        return Field(E=E1, H=H1), Field(E=E2, H=H2)