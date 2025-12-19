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
from typing import Iterable, Callable


def taper(N: int):
    return np.ones((N,))

def compute_spacing(fmax: float, scan_max: float, degrees: bool = True) -> float:
    if degrees:
        scan_max = scan_max*np.pi/180
    lmax = 299792458/fmax
    return lmax/(1+np.sin(scan_max))

def _cast_vector(arry: Iterable[float], shape: tuple[float,...]) -> np.ndarray:
    arry_out = np.array(arry)
    if arry_out.shape != shape:
        raise ValueError(f'Provided array {arry_out} has shape {arry.shape}. A shape {shape} is expected.')
    return arry_out

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
        """Adds an antenna to the array

        Args:
            antenna (Antenna): The antenna element to add.
        """
        self.antennas.append(antenna)

    def set_scan_direction(self, theta: float, phi: float, degree: bool = True, auto_update: bool = True) -> None:
        """Compute phase settings to make all antennas scan in the desired scan direction.

        Args:
            theta (float): The theta angle
            phi (float): The phi angle
            degree (bool, optional): If the angles are provided in degrees. Defaults to True.
            auto_update (bool, optional): If the antenna settings should be automatically updated. Defaults to True.
        """
        if degree:
            theta = theta * np.pi / 180
            phi = phi * np.pi / 180
        self.scan_theta = theta
        self.scan_phi = phi
        if auto_update:
            self._update_antennas()

    def reset_aux_coefficients(self) -> None:
        """Resets all auxilliary scan coefficients.
        """
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
        """Exposes a set of far field theta/phi coordinates

        X: theta=90, phi=0
        Y: theta=90, phi=90
        Z: theta=0
        
        Args:
            gtheta (np.ndarray): The theta coordinates
            gphi (np.ndarray): The phi coordinates

        Returns:
            Field: The resultant EH field object
        """
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

    def expose_xyz(self, gx: np.ndarray, gy: np.ndarray, gz: np.ndarray) -> Field:
        """Exposes a set of XYZ coordinates defined in the global space.

        Args:
            gx (np.ndarray): The global x-coordinates
            gy (np.ndarray): The global y-coordinates
            gz (np.ndarray): The global z-coordinates

        Returns:
            Field: The resultat Field data.
        """
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
        """Exposes a Surface

        Args:
            surface (Surface): The surface to expose
            add_field (bool, optional): If the field shoeld be added. If false its overwritten. Defaults to True.

        Returns:
            Field: Returns the computed field.
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
        
        tn = surface.field_normals()
        tn = tn + np.random.rand(*tn.shape)*1e-12
        tn = tn / np.linalg.norm(tn)
        tnx = tn[0,:]
        tny = tn[1,:]
        tnz = tn[2,:]

        with Progress() as p:
            task = p.add_task("[red] Exposing surface...", total=self.nantennas)
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
        """Exposes a FarFieldSpace object

        Args:
            target (FarFieldSpace): The target FarFieldSpace object

        Returns:
            Field: _description_
        """
        fr = self.expose_thetaphi(target.theta, target.phi)
        target.field = fr
        return fr
    
    def deform(self, T: callable, auto_update: bool = True):
        """Deforms the array by displacing the antenna elements using a mapping T(x,y,z) -> (x,y,z)
        
        Antennas with a deformation applied with base their phase steering on their original (not deformed) position.

        Args:
            T (callable): The displacement map: ℝ3 -> ℝ3
            auto_update (bool, optional): _description_. Defaults to True.
        """
        for a in self.antennas:
            a.deform(T)
        if auto_update:
            self._update_antennas()

    def reset_deformation(self):
        """ Removes any deformed coordinate systems from the antennas.
        """
        for a in self.antennas:
            a.reset_deformation()

    def set_scan_direction_groups(self, theta: float, phi: float, deg: bool=True, grouping: tuple = (1,1), auto_update: bool = True):
        """Compute phase settings to make groups of antennas scan in the desired scan direction.

        The group size is defined by the grouping tuple. 
        Args:
            theta (float): The theta angle
            phi (float): The phi angle
            degree (bool, optional): If the angles are provided in degrees. Defaults to True.
            grouping (tuple[float, float], optional): The groupings of antennas to scan.
            auto_update (bool, optional) If the antenna settings should be automatically updated. Defaults to True.
        """
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
        
        kx = self.k0 * np.sin(theta) * np.cos(phi)
        ky = self.k0 * np.sin(theta) * np.sin(phi)
        kz = self.k0 * np.cos(theta)
        k = np.array([kx, ky, kz])

        for ix in range(nxg):
            for iy in range(nyg):
                antennas = list(self.arraygrids[ix*gx:(ix+1)*gx, iy*gy:(iy+1)*gy].flatten())
                N = len(antennas)
                gxyz = sum([a.physical_gxyz/N for a in antennas])
                for a in antennas:
                    a.aux_coefficients.append(np.exp(-1j * (k @ gxyz)))
        
        if auto_update:
            self._update_antennas()

    def _update_antennas(self):
        """Update the antenna objects with the proper phase, taper and power normalization settings.
        """
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

    def _normalize_power(self, value: float | None = None, resolution: float = 0.33):
        """Normalizes the total radiated power in the current setting by performing a surface integral of a sphere around it

        Args:
            value (float, optional): The power normalisation constant. Defaults to None.
            resolution (float, optional): The integration surface sample resolution. Defaults to 0.33.
        """
        from ..geo.mesh.generators import generate_sphere

        logger.debug('Normalizing pattern')
        self._update_antennas()
        if value is not None and isinstance(value, (float, int)):
            logger.debug('Setting to provided value.')
            self.power_compensation = value
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
        
        mesh = generate_sphere(p0, 1.5 * Rmax, _lambda*resolution, self.cs.get_global())
        surf = Surface(mesh, FRES_AIR)
        self.expose_surface(surf)

        Po = sum(surf.powerflux())
        
        logger.debug(f"Measured power: {Po} W")
        self.power_compensation = self.power_compensation * np.sqrt(1 / Po)
        logger.debug(f"Compensation factor: {self.power_compensation}")
        self._update_antennas()
        for a, active in zip(self.antennas, activation):
            a.active = active
        
    def add_1d_array(self, taper: Iterable[float | complex], ds: tuple[float, float, float], nf_pattern: Callable = dipole_pattern_nf, ff_pattern: Callable = dipole_pattern_ff):
        """Adds a 1D array of antennas to the antenna array object
        

        Args:
            taper (Iterable[float | complex]): An iterable of antenna element amplitudes (comple)
            axis (tuple[float, float, float]): The vector describing the element separation
            nf_pattern (Callable, optional): The nearfield atenna pattern to use. Defaults to dipole_pattern_nf.
            ff_pattern (Callable, optional): The farfield antenna pattern to use. Defaults to dipole_pattern_ff.
        """
        ds = _cast_vector(ds, (3,))
        N = len(taper)
        ds = np.array(ds)
        self.arraygrids = np.empty((N,), dtype=object)
        for i in range(N):
            xyz = (i-(N-1)/2)*ds
            ant = Antenna(xyz[0], xyz[1], xyz[2], self.frequency, self.cs, nf_pattern=nf_pattern, ff_pattern=ff_pattern)
            ant.taper_coefficient = taper[i]
            self.arraygrids[i] = ant

        self.antennas = list(self.arraygrids.flatten())
        #self._normalize_power(power_compensation)

    def add_2d_array(self, xtaper: Iterable[float], ytaper: Iterable[float], dx: np.ndarray, dy: np.ndarray, offset=0, nf_pattern: Callable = dipole_pattern_nf, ff_pattern: Callable = dipole_pattern_ff, power_compensation: float = None):
        """Adds a 2D array of antenna elements to the antenna array object.

        Args:
            xtaper (Iterable[float]): A list of amplitudes for the X-axis taper
            ytaper (Iterable[float]): A list of amplitudes for the Y-axis taper
            dx (np.ndarray): A 3D vector that defines the local lattice X-vector
            dy (np.ndarray): A 3D vector that defines the local lattice Y-vector
            offset (int, optional): A row offset parameter. 0.5 creates a triangular array. Defaults to 0.
            nf_pattern (Callable, optional): The Near-field pattern to use for the antennas. Defaults to dipole_pattern_nf.
            ff_pattern (Callable, optional): The Far-field patter to use for the antennas. Defaults to dipole_pattern_ff.
            power_compensation (float, optional): The power compensation coefficient to use.. Defaults to None.
            
        The nf_pattern and ff_pattern callables should be of function (theta, phi, r) and (theta, phi) for nf and ff respectively.
        """
        dx = _cast_vector(dx, (3,))
        dy = _cast_vector(dy, (3,))
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
        """Adds a 2D subarray of antenna elements to the antenna array object.

        Args:
            xtaper (Iterable[float]): A list of amplitudes for the X-axis taper
            ytaper (Iterable[float]): A list of amplitudes for the Y-axis taper
            dx (np.ndarray): A 3D vector that defines the local lattice X-vector
            dy (np.ndarray): A 3D vector that defines the local lattice Y-vector
            offset (float, optional): A row offset parameter. 0.5 creates a triangular array. Defaults to 0.
            nf_pattern (Callable, optional): The Near-field pattern to use for the antennas. Defaults to dipole_pattern_nf.
            ff_pattern (Callable, optional): The Far-field pattern to use for the antennas. Defaults to dipole_pattern_ff.
            power_compensation (float, optional): The power compensation coefficient to use. Defaults to None.
            copies (tuple, optional): The number of copies to create in the x and y directions. Defaults to (1,1).
            xcopy_taper (list[float], optional): The tapering coefficients for the x copies. Defaults to None.
            ycopy_taper (list[float], optional): The tapering coefficients for the y copies. Defaults to None.
        """
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
