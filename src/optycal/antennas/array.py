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
from ..geo.cs import CoordinateSystem, GCS
import numpy as np
from .patterns import dipole_pattern_ff, dipole_pattern_nf
from ..field import Field
from ..surface import Surface
from ..samplespace import FarFieldSpace
from .compiled_functions import _c_cross_comp, _c_dot_comp
from rich.progress import Progress
from loguru import logger
from .antenna import Antenna
from ..multilayer import FRES_AIR

def taper(N: int):
    return np.ones((N,))

def compute_spacing(fmax: float, scan_max: float, degrees: bool = True) -> float:
    if degrees:
        scan_max = scan_max*np.pi/180
    lmax = 299792458/fmax
    return lmax/(1+np.sin(scan_max))


class AntennaArray:
    def __init__(
        self,
        frequency: float,
        cs: CoordinateSystem = None,
        power: float = 1,
        name: str = "AntennaArray",
    ):
        if cs is None:
            cs = GCS
        self.frequency: float = frequency
        self.name: str = name
        self.cs: CoordinateSystem = cs
        self.antennas: list[Antenna] = []
        self.power_compensation: float = 1
        self.scan_theta: float | None = None
        self.scan_phi: float | None = None
        self.power: float = power
        self.k0: float = 2*np.pi*frequency/299792458
        self.arraygrids = None
        self.skip_phase_steering: bool = False
    
    def __str__(self) -> str:
        return f"AntennaArray[{self.name}]"
    
    @property
    def nantennas(self) -> int:
        return len(self.antennas)
    
    def add_antenna(self, antenna: Antenna) -> None:
        self.antennas.append(antenna)

    def set_scan_direction(self, theta, phi, degree: bool = True, auto_update: bool = True) -> None:
        if degree:
            theta = theta * np.pi / 180
            phi = phi * np.pi / 180
        self.scan_theta = theta
        self.scan_phi = phi
        if auto_update:
            self._update_antennas()

    def reset_aux_coefficients(self) -> None:
        for a in self.antennas:
            a.reset_aux()

    def displaced(self, displacement_function):
        xyz_old = [(a.x, a.y, a.z) for a in self.antennas]
        try:
            logger.debug('Changing coordinates.')
            for a in self.antennas:
                dxyz_new = displacement_function(a.gx, a.gy, a.gz)
                a.x += dxyz_new[0]
                a.y += dxyz_new[1]
                a.z += dxyz_new[2]
            yield self
        finally:
            for a, xyz in zip(self.antennas, xyz_old):
                a.x, a.y, a.z = xyz
            logger.debug('Restored xyz coordinates')

    def expose_thetaphi(self, gtheta: np.ndarray, gphi: np.ndarray) -> Field:
        E = np.zeros((3,len(gtheta)), dtype=np.complex64)
        H = np.zeros((3,len(gtheta)), dtype=np.complex64)
        logger.debug("Iterating over antennas")
        with Progress() as p:
            task1 = p.add_task("[red]Processing antennas...", total=self.nantennas)
            for ant in self.antennas:
                p.update(task1, advance=1)
                fr = ant.expose_thetaphi(gtheta, gphi)
                E += fr.E
                H += fr.H
        logger.debug("Field Computation Complete")
        return Field(E=E, H=H, theta=gtheta, phi=gphi)

    def expose_xyz(self, gx, gy, gz) -> Field:
        E = np.zeros((3,len(gx)), dtype=np.complex64)
        H = np.zeros((3,len(gx)), dtype=np.complex64)
        logger.debug("Iterating over antennas")
        with Progress() as p:
            task1 = p.add_task("[red]Processing antennas...", total=self.nantennas)
            for ant in self.antennas:
                p.update(task1, advance=1)
                fr = ant.expose_xyz(gx, gy, gz)
                E += fr.E
                H += fr.H
            logger.debug("Field Computation Complete")
        return Field(E=E, H=H, x=gx, y=gy, z=gz)

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
        
        tn = surface.field_normals()
        tn = tn + np.random.rand(*tn.shape)*1e-12
        tn = tn / np.linalg.norm(tn)
        tnx = tn[0,:]
        tny = tn[1,:]
        tnz = tn[2,:]

        with Progress() as p:
            task = p.add_task("[red]Exposing surface...", total=self.nantennas)
            for i in range(self.nantennas):
                p.update(task, advance=1)
                fr = self.antennas[i].expose_xyz(x, y, z)
                E = fr.E
                H = fr.H
                Ex = E[0,:]
                Ey = E[1,:]
                Ez = E[2,:]
                Hx = H[0,:]
                Hy = H[1,:]
                Hz = H[2,:]
                
                gx, gy, gz = self.antennas[i].gxyz
                rsx = x-gx
                rsy = y-gy
                rsz = z-gz
                
                R = np.sqrt(rsx**2 + rsy**2 + rsz**2)
                
                rdotn = rsx/R*tnx + rsy/R*tny + rsz/R*tnz
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
                
                angin = np.arccos(np.abs(rdotn))
                
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
                
                
                E1[0,:] += Erefx*same + Etransx*other
                E1[1,:] += Erefy*same + Etransy*other
                E1[2,:] += Erefz*same + Etransz*other
                H1[0,:] += Hrefx*same + Htransx*other
                H1[1,:] += Hrefy*same + Htransy*other
                H1[2,:] += Hrefz*same + Htransz*other
                E2[0,:] += Erefx*other + Etransx*same
                E2[1,:] += Erefy*other + Etransy*same
                E2[2,:] += Erefz*other + Etransz*same
                H2[0,:] += Hrefx*other + Htransx*same
                H2[1,:] += Hrefy*other + Htransy*same
                H2[2,:] += Hrefz*other + Htransz*same
            
        fr1 = Field(E=E1, H=H1, x=xyz[0,:], y=xyz[1,:], z=xyz[2,:])
        fr2 = Field(E=E2, H=H2, x=xyz[0,:], y=xyz[1,:], z=xyz[2,:])
        if add_field:
            surface.add_field(1, fr1, self.k0)
            surface.add_field(2, fr2, self.k0)

        return fr1, fr2
    
    def expose_ff(self, target: FarFieldSpace) -> Field:
        fr = self.expose_thetaphi(target.theta, target.phi)
        target.field = fr
        return fr
    
    def deform(self, T: callable, auto_update: bool = True):
        for a in self.antennas:
            a.deform(T)
        if auto_update:
            self._update_antennas()

    def restore_cs(self):
        for a in self.antennas:
            a.restore_cs()

    def aux_scan(self, theta, phi, deg: bool=True, grouping: tuple = (1,1)):
        if len(self.arraygrids.shape)==1:
            Nx = self.arraygrids.shape[0]
            Ny = 1
        else:
            Nx, Ny = self.arraygrids.shape
        if deg:
            theta = theta * np.pi/180
            phi = phi * np.pi/180
        gx, gy = grouping
        nxg = Nx//gx
        nyg = Ny//gy
        
        kx = self.k0 * np.cos(theta) * np.cos(phi)
        ky = self.k0 * np.cos(theta) * np.sin(phi)
        kz = self.k0 * np.sin(theta)
        k = np.array([kx, ky, kz])

        for ix in range(nxg):
            for iy in range(nyg):
                antennas = list(self.arraygrids[ix*gx:(ix+1)*gx, iy*gy:(iy+1)*gy].flatten())
                N = len(antennas)
                gxyz = sum([a.physical_gxyz/N for a in antennas])
                for a in antennas:
                    a.aux_coefficients.append(np.exp(-1j * (k @ gxyz)))
        

    def _update_antennas(self):
        
        for ant in self.antennas:
            ant.frequency = self.k0*299792458/(2*np.pi)

        skip_phase = self.skip_phase_steering
        
        if self.scan_phi is not None and self.scan_theta is not None:
            kx = self.k0 * np.sin(self.scan_theta) * np.cos(self.scan_phi)
            ky = self.k0 * np.sin(self.scan_theta) * np.sin(self.scan_phi)
            kz = self.k0 * np.cos(self.scan_theta)
            k = np.array([kx, ky, kz])
        else:
            skip_phase = True
        
        
        for ant in self.antennas:
            gxyz = ant.phase_gxyz
            ant.array_compensation = 1/np.sqrt(self.nantennas)
            if not skip_phase:
                ant.scan_coefficient = np.exp(-1j * (k @ gxyz))
            else:
                ant.scan_coefficient = 1.0
            ant.correction_coefficient = self.power_compensation*np.sqrt(self.power)

    def _normalize_power(self, value=None, resolution=0.5):

        from ..geo.mesh.generators import generate_sphere

        logger.debug('Normalizing pattern')
        self._update_antennas()
        if value is not None and isinstance(value, (float, int)):
            logger.debug('Setting to provided value.')
            self.power_compensation=value
            self._update_antennas()
            return
        
        activation = []
        for a in self.antennas:
            activation.append(a.active)
            a.active = 1

        NA = len(self.antennas)

        gxyz = np.zeros((3,NA))
        for i, ant in enumerate(self.antennas):
            gxyz[0,i] = ant.gxyz[0]
            gxyz[1,i] = ant.gxyz[1]
            gxyz[2,i] = ant.gxyz[2]
            
        p0 = np.mean(gxyz, axis=1)
        pr = gxyz - p0.reshape((3,1))
        
        Rmax = np.max(np.sqrt(pr[0,:]**2 + pr[1,:]**2 + pr[2,:]**2))
        logger.debug(f'Generating sphere at {p0*1000}mm, with radius {Rmax*1000}mm')
        _lambda = 2 * np.pi / self.k0
        Rmax = max(Rmax, 5*_lambda)
        
        mesh = generate_sphere(p0, 1.5 * Rmax, _lambda/2, self.cs.get_global())
        surf = Surface(mesh, FRES_AIR)
        self.expose_surface(surf)

        Po = sum(surf.powerflux())
        
        logger.debug(f"Measured power: {Po} W")
        self.power_compensation = self.power_compensation * np.sqrt(1 / Po)
        logger.debug(f"Compensation factor: {self.power_compensation}")
        self._update_antennas()
        for a, active in zip(self.antennas, activation):
            a.active = active
        
    def add_1d_array(self, taper, ds, axis, nf_pattern = dipole_pattern_nf, ff_pattern = dipole_pattern_ff):
        N = len(taper)
        axis = np.array(axis)
        self.arraygrids = np.empty((N,), dtype=object)
        print(f'Detected N antennas: {N}, arraygrid={self.arraygrids.shape}')
        for i in range(N):
            xyz = (i-(N-1)/2)*axis*ds
            ant = Antenna(xyz[0], xyz[1], xyz[2], self.frequency, self.cs, nf_pattern=nf_pattern, ff_pattern=ff_pattern)
            ant.taper_coefficient = taper[i]
            self.arraygrids[i] = ant

        self.antennas = list(self.arraygrids.flatten())
        #self._normalize_power(power_compensation)

    def add_2d_array(self, xtaper, ytaper, dx: np.ndarray, dy: np.ndarray, offset=0, nf_pattern = dipole_pattern_nf, ff_pattern = dipole_pattern_ff, power_compensation: float = None):
        Nx = len(xtaper)
        Ny = len(ytaper)
        dx = np.array(dx)
        dy = np.array(dy)
        self.arraygrids = np.empty((Nx,Ny), dtype=object)
        for i in range(Nx):
            for j in range(Ny):
                xyz = (i-(Nx-1)/2)*dx + ((j%2)*offset*dx) + (j-(Ny-1)/2)*dy
                ant = Antenna(xyz[0], xyz[1], xyz[2], self.frequency, self.cs, nf_pattern=nf_pattern, ff_pattern=ff_pattern)
                ant.taper_coefficient = xtaper[i]*ytaper[j]
                self.add_antenna(ant)
                self.arraygrids[i,j] = ant
        self.antennas = list(self.arraygrids.flatten())
        
        self._normalize_power(power_compensation)

    def add_2d_subarray(self, xtaper, ytaper, dx, dy, 
                        offset: float = 0, 
                        nf_pattern = dipole_pattern_nf, 
                        ff_pattern = dipole_pattern_ff, 
                        power_compensation: float = None, 
                        copies: tuple = (1,1),
                        xcopy_taper: list[float] = None,
                        ycopy_taper: list[float] = None):
        if xcopy_taper is None:
            xcopy_taper = [1 for _ in range(copies[0])]
        
        if ycopy_taper is None:
            ycopy_taper = [1 for _ in range(copies[1])]

        Nx = len(xtaper)
        Ny = len(ytaper)
        dx = np.array(dx)
        dy = np.array(dy)
        NSx = copies[0]
        NSy = copies[1]
        ixys = []
        self.arraygrids = np.empty((Nx*NSx, Ny*NSy), dtype=object)
        for ix in range(NSx):
            for iy in range(NSy):
                ixys.append((ix,iy))
        ijs = []
        for i in range(Nx):
            for j in range(Ny):
                ijs.append((i,j))
        DX = Nx*dx
        DY = Ny*dy
        logger.debug(f'Creating {NSx*NSy} subarray copies ({NSx}x{NSy}).')
        logger.debug(f'Each subarray is {Nx} x {Ny} ({Nx*Ny} elements)')
        for ix, iy in ixys:
            xyzc = (ix-(NSx-1)/2)*DX + (iy-(NSy-1)/2)*DY
            coeff = xcopy_taper[ix]*ycopy_taper[iy]
            #logger.debug(f'Subarray at {xyzc}')
            CSsub = self.cs.displace(xyzc[0], xyzc[1], xyzc[2])
            for i,j in ijs:
                xyz = (i-(Nx-1)/2)*dx + ((j%2)*offset*dx) + (j-(Ny-1)/2)*dy
                ant = Antenna(xyz[0], xyz[1], xyz[2], self.frequency, CSsub, nf_pattern=nf_pattern, ff_pattern=ff_pattern)
                ant._phase_cs = CSsub
                ant.taper_coefficient = xtaper[i]*ytaper[j]*coeff
                self.add_antenna(ant)
                self.arraygrids[ix*Nx+i, iy*Ny+j] = ant

        self._normalize_power(power_compensation)
