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
import numpy as np
from .field import Field
from typing import Literal


class FarFieldSpace:
    pass

    def __str__(self) -> str:
        return f'FarFieldSpace[{self.theta.shape}]'
    
    def catch(self, field: Field, k0: float):
        self.field = field


class FF1D(FarFieldSpace):
    
    def __init__(self, theta: np.ndarray, phi: np.ndarray):
        self.theta = theta
        self.phi = phi
        self.field = None
    
    @staticmethod
    def aziele(dangle: float = 1, phirange = (-180, 180), degree: bool = True, phi0: float=0.0) -> tuple[FF1D, FF1D]:
        """Creates two FF1D objects for an azimuth and elevation cut

        Args:
            dangle (float, optional): The angle step size. Defaults to 1.
            phirange (tuple, optional): Phi angular range. Defaults to (-180, 180).
            degree (bool, optional): If the provided range/step is in degrees. Defaults to True.
            phi0 (float, optional): The phi offset. Defaults to 0.0.

        Returns:
            tuple[FF1D, FF1D]: The Azimuth and Elevation FF1D objects
        """
        N = int(1 + 360/dangle)
        theta = np.linspace(0, 180, N)
        phis = np.linspace(phirange[0], phirange[1], N)
        if degree:
            theta = theta*np.pi/180
            phis = phis*np.pi/180
            phi0 = phi0*np.pi/180
        return FF1D(np.pi/2*np.ones_like(phis), phis+phi0), FF1D(theta, np.ones_like(theta)*phi0)


class FF2D(FarFieldSpace):
    
    def __init__(self, theta: np.ndarray, phi: np.ndarray):
        self._theta = theta
        self._phi = phi
        self.Theta, self.Phi = np.meshgrid(theta, phi)
        self.theta = self.Theta.flatten()
        self.phi = self.Phi.flatten()
        self.field = None
    
    @staticmethod
    def sphere(dangle: float = 1, degree=True, sector: float = 1.0) -> FF2D:
        """Creates an FF2D object of a sphere
        

        Args:
            dangle (float, optional): The angle discretization size. Defaults to 1.
            degree (bool, optional): If the angle size is in degrees.. Defaults to True.
            sector (float, optional): The fraction of the sphere. 1.0 is a full sphere 0.5 is a half sphere.. Defaults to 1.0.

        Returns:
            FF2D: The resultant FF2D object.
        """
        if degree:
            dangle = dangle * np.pi/180
        th = np.linspace((1-sector)/2*np.pi,((1-sector)/2+sector)*np.pi, int(np.ceil(sector*np.pi/dangle)))
        ph = np.linspace(-sector*np.pi,sector*np.pi, int(np.ceil(sector*2*np.pi/dangle)))
        return FF2D(th, ph)
    
    @staticmethod
    def halfsphere(dangle: float = 1, degree=True):
        """Creates an FF2D object of a half sphere.
        equivalent to sphere with sector==0.5

        Args:
            dangle (float, optional): The angle discretization size. Defaults to 1.
            degree (bool, optional): If the angle size is in degrees.. Defaults to True.

        Returns:
            FF2D: The resultant FF2D object.
        """
        if degree:
            dangle = dangle * np.pi/180
        th = np.linspace(-np.pi/2,np.pi/2, int(np.ceil(np.pi/dangle)))
        ph = np.linspace(-np.pi/2,np.pi/2, int(np.ceil(np.pi/dangle)))
        return FF2D(th, ph)
    
    def reshape(self, data: np.ndarray) -> np.ndarray:
        """Reshapes a numpy array of data to be conforming to the 2D grid shape.

        Args:
            data (np.ndarray): The input dataset

        Returns:
            np.ndarray: The output dataset with the same shape as this field.
        """
        return data.reshape(self.Theta.shape)
    
    def __getattr__(self, item):
        return self.reshape(getattr(self.field, item))

    def surfplot(self, 
             polarization: Literal['Ex','Ey','Ez','Etheta','Ephi','normE'], 
             quantity: Literal['abs','real','imag','angle'] = 'abs',
             isotropic: bool = True, dB: bool = False, dBfloor: float = -30, rmax: float = None,
             offset: tuple[float, float, float] = (0,0,0)) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Returns the parameters to be used as positional arguments for the display.add_surf() function.

        Example:
        >>> model.display.add_surf(*dataset.field[n].farfield_3d(...).surfplot())

        Args:
            polarization ('Ex','Ey','Ez','Etheta','Ephi','normE'): What quantity to plot
            isotropic (bool, optional): Whether to look at the ratio with isotropic antennas. Defaults to True.
            dB (bool, optional): Whether to plot in dB's. Defaults to False.
            dBfloor (float, optional): The dB value to take as R=0. Defaults to -10.

        Returns:
            tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: The X, Y, Z, F values
        """
        if polarization == "Ex":
            F = self.Ex
        elif polarization == "Ey":
            F = self.Ey
        elif polarization == "Ez":
            F = self.Ez
        elif polarization == "normE":
            F = self.normE
        elif polarization == "Etheta":
            F = self.Etheta
        elif polarization == "Ephi":
            F = self.Ephi
        else:
            F = self.normE
        if isotropic:
            F = F/np.sqrt(376.730313412/(2*np.pi))
        if dB:
            F = 20*np.log10(np.clip(np.abs(F), a_min=10**(dBfloor/20), a_max = 1e9))-dBfloor
        else:
            if quantity=='abs':
                F = np.abs(F)
            elif quantity=='real':
                F = F.real
            elif quantity=='imag':
                F = F.imag
            elif quantity=='angle':
                F = np.angle(F)
        if rmax is not None:
            F = rmax * F/np.max(F)
        
        F = self.reshape(F)
        xs = F*np.sin(self.Theta)*np.cos(self.Phi) + offset[0]
        ys = F*np.sin(self.Theta)*np.sin(self.Phi) + offset[1]
        zs = F*np.cos(self.Theta) + offset[2]

        return xs, ys, zs, F

   
class NearFieldSpace:
    def __str__(self) -> str:
        return f'NearFieldSpace[{self.x.shape}]'
    
    def catch(self, field: Field, k0: float):
        self.field = field


class NF1D(NearFieldSpace):
    
    def __init__(self, x: np.ndarray, y: np.ndarray, z: np.ndarray):
        self.x = x
        self.y = y
        self.z = z
        self.field


class NF2D(NearFieldSpace):
    
    def __init__(self, x: np.ndarray, y: np.ndarray, z: np.ndarray):
        self.x = x
        self.y = y
        self.z = z
        self.field


class NF3D(NearFieldSpace):
    
    def __init__(self, x: np.ndarray, y: np.ndarray, z: np.ndarray):
        self.x = x
        self.y = y
        self.z = z
        self.field