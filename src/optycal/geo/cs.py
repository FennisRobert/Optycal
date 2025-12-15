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
from typing import Tuple
from abc import ABC
from numba import njit, float32, f8, f4, c8, c16
from numba.types import Tuple as TypeTuple
from loguru import logger

@njit(cache=True, nogil=True)
def matmul(a: np.ndarray, b: np.ndarray):
    out = np.zeros((3,b.shape[1]), dtype=b.dtype)
    out[0,:] = a[0,0]*b[0,:] + a[0,1]*b[1,:] + a[0,2]*b[2,:]
    out[1,:] = a[1,0]*b[0,:] + a[1,1]*b[1,:] + a[1,2]*b[2,:]
    out[2,:] = a[2,0]*b[0,:] + a[2,1]*b[1,:] + a[2,2]*b[2,:]
    return out

@njit(cache=True, fastmath=True, parallel=True, nogil=True)
def _tp_from_global(invbasis, th, ph):
    snt = np.sin(th)
    xx = np.cos(ph) * snt
    yy = np.sin(ph) * snt
    zz = np.cos(th)
    x2 = xx*invbasis[0,0] + yy*invbasis[0,1] + zz*invbasis[0,2]
    y2 = xx*invbasis[1,0] + yy*invbasis[1,1] + zz*invbasis[1,2]
    z2 = xx*invbasis[2,0] + yy*invbasis[2,1] + zz*invbasis[2,2]
    phi2 = np.arctan2(y2, x2)
    theta2 = np.arccos(z2)
    return theta2, phi2

class NamedDescriptor(ABC):

    def __set_name__(self, owner, name):
        self.name = f"_{name}"
        self._original_name = name

    def __get__(self, obj, objtype=None):
        if obj is None:
            return self
        return getattr(obj, self.name)

    def __set__(self, obj, value):
        setattr(obj, self.name, value)
        
class CoordinateArray(NamedDescriptor):
    pass

class VectorArray(NamedDescriptor):
    pass
        
class CoordinateTuple(NamedDescriptor):
    pass

class VectorTuple(NamedDescriptor):
    pass

class XScalarArray(NamedDescriptor):
    pass

class YScalarArray(NamedDescriptor):
    pass

class ZScalarArray(NamedDescriptor):
    pass


class CoordinateSystem:

    def __init__(
        self,
        origin: list,
        x: list,
        y: list,
        z: list,
        parent: "CoordinateSystem" = None,
        is_global: bool = False,
    ):
        origin = np.array(origin, dtype=np.float32)
        x = np.array(x, dtype=np.float32)
        y = np.array(y, dtype=np.float32)
        z = np.array(z, dtype=np.float32)
        
        self.origin: np.ndarray = origin.astype(np.float32)
        self.xhat: np.ndarray = x.astype(np.float32)
        self.yhat: np.ndarray = y.astype(np.float32)
        self.zhat: np.ndarray = z.astype(np.float32)
        
        self.parent_cs: CoordinateSystem = parent
        self.is_global: bool = is_global

        self.children: list = []

        if parent is not None:
            self.parent_cs.children.append(self)
        else:
            self.is_global = True

        self.basis: np.ndarray = None
        self.basis_inv: np.ndarray  = None
        self.global_basis: np.ndarray  = None
        self.global_basis_inv: np.ndarray  = None
        self.global_origin: np.ndarray  = None
        self.x: float = None
        self.y: float = None
        self.z: float = None
        self.gx: float = None
        self.gy: float = None
        self.gz: float = None
        self.gxhat: np.ndarray = None
        self.gyhat: np.ndarray = None
        self.gzhat: np.ndarray = None

        self._calculate_properties()

    def __repr__(self) -> str:
        if self.is_global:
            return f"GlobalCS(@{list(self.origin.squeeze())}, x={list(self.xhat.squeeze())}, y={list(self.yhat.squeeze())}, z={list(self.zhat.squeeze())})"
        return f"CoordinateSystem(@{list(self.origin.squeeze())}, x={list(self.xhat.squeeze())}, y={list(self.yhat.squeeze())}, z={list(self.zhat.squeeze())})"

    def __str__(self) -> str:
        return self.__repr__()

    def _calculate_properties(self):
        self.basis = np.array([self.xhat, self.yhat, self.zhat], dtype=np.float32).T.squeeze()
        self.basis_inv = np.linalg.pinv(self.basis)
        self.x, self.y, self.z = self.origin

        if not self.is_global:
            self.global_basis = self.parent_cs.global_basis @ self.basis
            self.global_origin = (
                self.parent_cs.global_basis @ self.origin + self.parent_cs.global_origin
            )
        else:
            self.global_basis = self.basis
            self.global_origin = self.origin

        self.gx, self.gy, self.gz = self.global_origin

        self.global_basis_inv = np.linalg.pinv(self.global_basis)
        self.gxhat = self.global_basis[:, 0]
        self.gyhat = self.global_basis[:, 1]
        self.gzhat = self.global_basis[:, 2]

        for child in self.children:
            child._calculate_properties()

    def add_child(self, object) -> None:
        self.children.append(object)

    def rotate_basis(
        self, axis: np.ndarray, angle: float, degrees: bool = True
    ) -> CoordinateSystem:
        """Rotates the coordinate system around an axis.

        Args:
            axis (np.ndarray): The Axis as numpy array
            angle (float): The angle
            degrees (bool, optional): If the angle is in degrees. Defaults to False.
        """
        if self.is_global:
            raise RuntimeError('Cannot modify global coordinate systems')
        axis = np.array(axis)
        if degrees:
            angle_rad = angle*np.pi/180
        else:
            angle_rad = angle

        ux, uy, uz = axis
        cs = np.cos(angle_rad)
        sn = np.sin(angle_rad)
        mcs = 1 - cs
        R = np.array(
            [
                [cs + ux**2 * mcs, ux * uy * mcs - uz * sn, ux * uz * mcs + uy * sn],
                [uy * ux * mcs + uz * sn, cs + uy**2 * mcs, uy * uz * mcs - ux * sn],
                [uz * ux * mcs - uy * sn, uz * uy * mcs + ux * sn, cs + uz**2 * mcs],
            ]
        )
        self.xhat = R @ self.xhat
        self.yhat = R @ self.yhat
        self.zhat = R @ self.zhat
        self._calculate_properties()
        return self
        
    def translate(self, dx: float, dy: float, dz: float) -> None:
        """Translates the coordinate system

        Args:
            dx (float): The X-displacement
            dy (float): The Y-displacement
            dz (float): The Z-displacement
        """
        self.origin += np.array([dx, dy, dz]).reshape((3, 1))
        self._calculate_properties()
        
    def in_global_cs(self, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        x2 = self.global_basis[0,0] * x + self.global_basis[0,1] * y + self.global_basis[0,2] * z
        y2 = self.global_basis[1,0] * x + self.global_basis[1,1] * y + self.global_basis[1,2] * z
        z2 = self.global_basis[2,0] * x + self.global_basis[2,1] * y + self.global_basis[2,2] * z
        x2 = x2 + self.global_origin[0]
        y2 = y2 + self.global_origin[1]
        z2 = z2 + self.global_origin[2]
        return x2, y2, z2

    def from_global_cs(self, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        xl = x - self.global_origin[0]
        yl = y - self.global_origin[1]
        zl = z - self.global_origin[2]
        x2 = self.global_basis_inv[0,0] * xl + self.global_basis_inv[0,1] * yl + self.global_basis_inv[0,2] * zl
        y2 = self.global_basis_inv[1,0] * xl + self.global_basis_inv[1,1] * yl + self.global_basis_inv[1,2] * zl
        z2 = self.global_basis_inv[2,0] * xl + self.global_basis_inv[2,1] * yl + self.global_basis_inv[2,2] * zl
        return x2, y2, z2
    
    def ae_in_global_cs(self, theta: np.ndarray, phi: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.is_global:
            return theta, phi
        xx = np.cos(phi) * np.sin(theta)
        yy = np.sin(phi) * np.sin(theta)
        zz = np.cos(theta)
        x2, y2, z2 = self.in_global_basis(xx, yy, zz)
        phi2 = np.arctan2(y2, x2)
        theta2 = np.arccos(z2)
        return theta2, phi2

    def ae_from_global_cs(self, theta, phi) -> tuple[np.ndarray, np.ndarray]:
        if self.is_global:
            return theta, phi
        th, ph = _tp_from_global(self.global_basis_inv, theta, phi)
        return th, ph

    def in_global_basis(self, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> Tuple[np.ndarray]:
        x2 = self.global_basis[0,0] * x + self.global_basis[0,1] * y + self.global_basis[0,2] * z
        y2 = self.global_basis[1,0] * x + self.global_basis[1,1] * y + self.global_basis[1,2] * z
        z2 = self.global_basis[2,0] * x + self.global_basis[2,1] * y + self.global_basis[2,2] * z
        return x2, y2, z2

    def from_global_basis(self, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> Tuple[np.ndarray]:
        xl, yl, zl = x, y, z
        x2 = self.global_basis_inv[0,0] * xl + self.global_basis_inv[0,1] * yl + self.global_basis_inv[0,2] * zl
        y2 = self.global_basis_inv[1,0] * xl + self.global_basis_inv[1,1] * yl + self.global_basis_inv[1,2] * zl
        z2 = self.global_basis_inv[2,0] * xl + self.global_basis_inv[2,1] * yl + self.global_basis_inv[2,2] * zl
        return x2, y2, z2

    def displace(self, dx: float = 0, dy: float = 0, dz: float = 0) -> CoordinateSystem:
        return CoordinateSystem(
            origin= np.array([dx, dy, dz]),
            x=np.array([1,0,0]),
            y=np.array([0,1,0]),
            z=np.array([0,0,1]),
            parent=self,
        )

    def copy(self) -> CoordinateSystem:
        return CoordinateSystem(
            origin= np.array([0,0,0]),
            x=np.array([1,0,0]),
            y=np.array([0,1,0]),
            z=np.array([0,0,1]),
            parent=self,
        )
    
    def get_global(self) -> CoordinateSystem:
        """Returns the global cordinate system"""
        cs = self
        while not cs.is_global:
            cs = cs.parent_cs
        return cs
    
    @staticmethod
    def from_dir_pol(direction: np.ndarray, polarization: np.ndarray, parent: CoordinateSystem) -> CoordinateSystem:
        if not isinstance(direction, np.ndarray):
            direction = np.array(direction)
        if not isinstance(polarization, np.ndarray):
            polarization = np.array(polarization)
        
        X = direction/np.linalg.norm(direction)
        Z = polarization/np.linalg.norm(polarization)
        Y = np.cross(Z,X)
        Z = np.cross(X,Y)
        Y = Y/np.linalg.norm(Y)
        Z = Z/np.linalg.norm(Z)
        
        return CoordinateSystem(parent.origin, X, Y, Z, parent)


def sph_to_cart(R: np.ndarray, Theta: np.ndarray, Phi: np.ndarray) -> Tuple[np.ndarray]:
    X = R*np.cos(Phi)*np.sin(Theta)
    Y = R*np.sin(Phi)*np.sin(Theta)
    Z = R*np.cos(Theta)
    return X,Y,Z

def cart_to_sph(X: np.ndarray, Y: np.ndarray, Z: np.ndarray) -> Tuple[np.ndarray]:
    R = np.sqrt(X**2+Y**2+Z**2)
    theta = np.arccos(Z)
    phi = np.arctan2(Y,X)
    return R, theta, phi

class ToGlobalTransformer:
    def __init__(self, obj):
        self._obj: CoordinateSystem = obj
        
    @property
    def _basis(self) -> np.ndarray:
        return self._obj.cs.global_basis
    
    @property
    def _origin(self) -> np.ndarray:
        return self._obj.cs.global_origin
    
    def __getattr__(self, _name):
        data = getattr(self._obj, _name)
        dtype = self._obj.__class__.__dict__[_name]
        if isinstance(dtype, CoordinateArray):
            return matmul(self._basis, data) + self._origin[:,np.newaxis]
        elif isinstance(dtype, VectorArray):
            return matmul(self._basis, data)
        elif isinstance(dtype, CoordinateTuple):
            data2 = np.array(data)
            data2 = matmul(self._basis, data2) + self._origin[:,np.newaxis]
            return (data2[0,:], data2[1,:], data2[2,:])
        elif isinstance(dtype, VectorTuple):
            data2 = np.array(data)
            data2 = matmul(self._basis, data2)
            return (data2[0,:], data2[1,:], data2[2,:])
        else:
            logger.warning(f'Unrecognized type, returning {type(dtype)}')
            return data
        
        
CS = CoordinateSystem

GCS = CoordinateSystem([0,0,0],[1,0,0],[0,1,0],[0,0,1], is_global=True)