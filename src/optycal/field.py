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
from typing import Literal

class Field:
    
    def __init__(self, 
                 x: np.ndarray = None,
                 y: np.ndarray = None,
                 z: np.ndarray = None,
                 theta: np.ndarray = None,
                 phi: np.ndarray = None,
                 E: np.ndarray = None,
                 H: np.ndarray = None,
                 creator: str = 'Unknown'):
        self.theta: np.ndarray = theta
        self.phi: np.ndarray = phi
        self.x: np.ndarray = x
        self.y: np.ndarray = y
        self.z: np.ndarray = z
        self.E: np.ndarray = E
        self.H: np.ndarray = H
        self.creator = creator
    
    @property
    def isff(self) -> bool:
        if self.theta is not None and self.phi is not None:
            return True
        return False
    
    @property
    def isnf(self) -> bool:
        if self.x is not None and self.y is not None and self.z is not None:
            return True
        return False
    
    @property
    def Ex(self):
        return self.E[0,:]

    @property
    def Ey(self):
        return self.E[1,:]

    @property
    def Ez(self):
        return self.E[2,:]
    
    @property
    def Etheta_X(self):
        ux = np.cos(self.theta)*np.cos(self.phi)
        uy = np.sin(self.phi)*np.cos(self.theta)
        uz = np.sin(self.theta)
        thl = np.arccos(ux)
        phl = np.arctan2(uz,uy)
        thx = -np.sin(thl)
        thy = np.cos(thl)*np.cos(phl)
        thz = np.cos(thl)*np.sin(phl)
        return self.E[0,:]*thx + self.E[1,:]*thy + self.E[2,:]*thz
    
    @property
    def Etheta(self):
        thz = np.sin(self.theta)
        thx = -np.cos(self.theta)*np.cos(self.phi)
        thy = -np.cos(self.theta)*np.sin(self.phi)
        return self.E[0,:]*thx + self.E[1,:]*thy + self.E[2,:]*thz

    @property
    def Ephi(self):
        phx = -np.sin(self.phi)
        phy = np.cos(self.phi)
        phz = 0*self.phi
        return self.E[0,:]*phx + self.E[1,:]*phy + self.E[2,:]*phz
    
    @property
    def Hx(self):
        return self.H[0,:]

    @property
    def Hy(self):
        return self.H[1,:]

    @property
    def Hz(self):
        return self.H[2,:]
    
    @property
    def Sx(self):
        return 0.5 * (self.E[1, :] * np.conj(self.H[2, :]) - self.E[2, :] * np.conj(self.H[1, :]))
    
    @property
    def Sy(self):
        return 0.5 * (self.E[2, :] * np.conj(self.H[0, :]) - self.E[0, :] * np.conj(self.H[2, :]))
    
    @property
    def Sz(self):
        return 0.5 * (self.E[0, :] * np.conj(self.H[1, :]) - self.E[1, :] * np.conj(self.H[0, :]))
    
    @property
    def S(self):
        return np.vstack((self.Sx, self.Sy, self.Sz))
    
    @property
    def normE(self):
        return np.sqrt(np.abs(self.Ex)**2 + np.abs(self.Ey)**2 + np.abs(self.Ez)**2)
    
    @property
    def normH(self):
        return np.sqrt(np.abs(self.Hx)**2 + np.abs(self.Hy)**2 + np.abs(self.Hz)**2)
    
    @property
    def normS(self):
        return np.sqrt(np.abs(self.Sx)**2 + np.abs(self.Sy)**2 + np.abs(self.Sz)**2)
    
    def __add__(self, other):
        return Field(E=self.E + other.E, H=self.H+other.H, x=other.x, y=other.y, z=other.z, theta=other.theta,phi=other.phi)

    def __sub__(self, other):
        return Field(E=self.E - other.E, H=self.H - other.H, x=other.x, y=other.y, z=other.z, theta=other.theta,phi=other.phi)
    
    def __mul__(self, other):
        if isinstance(other,Field):
            return Field(E=self.E * other.E, H=self.H * other.H)
        else:
            return Field(E=self.E * other, H=self.H * other)
    
    def __rmul__(self, other):
        return self.__mul__(other)
    
    def shaped_like(self, field):
        shape = field.shape
        nf = Field()
        props = ['theta', 'phi', 'x','y','z','E','H']
        for p in props:
            if self.__dict__[p] is None:
                continue
            if p in ['E','H']:
                nf.__dict__[p] = self.__dict__[p].reshape((3,shape[0],shape[1]))
                continue
            nf.__dict__[p] = self.__dict__[p].reshape(shape)
        
        return nf

    