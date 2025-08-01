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

import numpy as np
from .field import Field

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
    def aziele(dangle: float = 1, range = (-180, 180), degree: bool = True):
        theta = np.linspace(range[0], range[1], int((range[1]-range[0])/dangle)+1)
        phi = 0*theta
        if degree:
            theta = theta*np.pi/180
            phi = phi*np.pi/180
        return FF1D(phi, theta), FF1D(theta, phi)

class FF2D(FarFieldSpace):
    
    def __init__(self, theta: np.ndarray, phi: np.ndarray):
        self._theta = theta
        self._phi = phi
        self.Theta, self.Phi = np.meshgrid(theta, phi)
        self.theta = self.Theta.flatten()
        self.phi = self.Phi.flatten()
        self.field = None
    
    @staticmethod
    def sphere(dangle: float = 1, degree=True):
        if degree:
            dangle = dangle * np.pi/180
        th = np.linspace(-np.pi/2,np.pi/2, int(np.ceil(np.pi/dangle)))
        ph = np.linspace(-np.pi,np.pi, int(np.ceil(np.pi/dangle)))
        return FF2D(th, ph)
    
    @staticmethod
    def halfsphere(dangle: float = 1, degree=True):
        if degree:
            dangle = dangle * np.pi/180
        th = np.linspace(-np.pi/2,np.pi/2, int(np.ceil(np.pi/dangle)))
        ph = np.linspace(-np.pi/2,np.pi/2, int(np.ceil(np.pi/dangle)))
        return FF2D(th, ph)
    
    def reshape(self, data: np.ndarray):
        return data.reshape(self.Theta.shape)
    
    def __getattr__(self, item):
        return self.reshape(getattr(self.field, item))
    
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