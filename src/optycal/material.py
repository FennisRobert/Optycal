# EMerge is an open source Python based FEM EM simulation module.
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
from typing import Callable
import inspect

C0 = 299792458


def num_args(func):
    sig = inspect.signature(func)
    return sum(
        1
        for p in sig.parameters.values()
        if p.default is inspect._empty and p.kind in (
            p.POSITIONAL_ONLY,
            p.POSITIONAL_OR_KEYWORD,
            p.KEYWORD_ONLY
        )
    )
    
def _to_mat(value: float | complex | int | np.ndarray) -> np.ndarray:
    if np.isscalar(value):
        return np.eye(3)*value
    if value.shape in ((3,), (3,1), (1,3)):
        return np.diag(np.ravel(value))
    if value.shape == (3,3):
        return value
    else:
        return ValueError(f'Trying to parse {value} as a material property tensor but it cant be identified as scalar, vector or matrix')

class MatProperty:
    _freq_dependent: bool = False
    _coord_dependent: bool = False
    _pickle_exclude = {"_func","_fmax"}
    """The MatProperty class is an interface for EMerge to deal with frequency and coordinate dependent material properties
    """
    
    def __init__(self, value: float | complex | int | np.ndarray):
        self.value: np.ndarray = _to_mat(value)
        
        self._apply_to: np.ndarray = np.array([], dtype=np.int64)
        self._x: np.ndarray = np.array([], dtype=np.float64)
        self._y: np.ndarray = np.array([], dtype=np.float64)
        self._z: np.ndarray = np.array([], dtype=np.float64)
        
        self._fmax = lambda f: value
        
    def initialize(self, x: np.ndarray, y: np.ndarray, z: np.ndarray, ids: np.ndarray) -> None:
        self._apply_to = np.concatenate([self._apply_to, ids])
        self._x = np.concatenate([self._x, x])
        self._y = np.concatenate([self._y, y])
        self._z = np.concatenate([self._z, z])
        
    def __call__(self, f: float, data: np.ndarray) -> np.ndarray:
        data[:,:,self._apply_to] = np.repeat(self.value[:,:,np.newaxis], self._apply_to.shape[0], axis=2)
        return data

    def scalar(self, f: float):
        return self.value[0,0]

    def reset(self) -> None:
        self._apply_to: np.ndarray = np.array([], dtype=np.int64)
        self._x: np.ndarray = np.array([], dtype=np.float64)
        self._y: np.ndarray = np.array([], dtype=np.float64)
        self._z: np.ndarray = np.array([], dtype=np.float64)
    
    def __getstate__(self):
        state = self.__dict__.copy()
        for k in self._pickle_exclude:
            state.pop(k, None)
        
        return state
    
    def __setstate__(self, state):
        self.__dict__.update(state)
        for k in self._pickle_exclude:
            setattr(self, k, None)
    
class FreqDependent(MatProperty):
    _freq_dependent: bool = True
    _coord_dependent: bool = False

    def __init__(self, 
                 scalar: Callable | None = None,
                 vector: Callable | None = None,
                 matrix: Callable | None = None):
        """Creates a frequency dependent property object.
        
        If the property is defined as a scalar value, use the "scalar" argument
        If the property is a diagonal rank-2 tensor, use the "vector" argument                   
        If the property is a full rank-2 tensor, use the "matrix" argument

        The max_value property must be set to tell EMerge how height this value can get 
        as it will be used to define the discretization of the mesh.
        
        Args:
            scalar (Callable | None, optional): The scalar value function returning a float/complex. Defaults to None.
            vector (Callable | None, optional): The diagonal rank-2 tensor function returning a (3,) array. Defaults to None.
            matrix (Callable | None, optional): The rank-2 tensor function returning a (3,3) array. Defaults to None.

        Returns:
            _type_: _description_
        """
        if scalar is not None:
            def _func(f: float) -> np.ndarray:
                return np.eye(3)*scalar(f)
        if vector is not None:
            def _func(f: float) -> np.ndarray:
                return np.diag(np.ravel(vector(f)))
        
        if matrix is not None:
            _func = matrix

        self._func: Callable = _func
        
        self._apply_to: np.ndarray = np.array([], dtype=np.int64)
        self._x: np.ndarray = np.array([], dtype=np.float64)
        self._y: np.ndarray = np.array([], dtype=np.float64)
        self._z: np.ndarray = np.array([], dtype=np.float64)
        
        self._fmax: Callable = lambda f: np.max(np.ravel(self._func(f)))

    def initialize(self, x: np.ndarray, y: np.ndarray, z: np.ndarray, ids: np.ndarray) -> None:
        self._apply_to = np.concatenate([self._apply_to, ids])
        
    def __call__(self, f: float, data: np.ndarray) -> np.ndarray:
        data[:,:,self._apply_to] = np.repeat(self._func(f)[:,:,np.newaxis], self._apply_to.shape[0], axis=2)
        return data
    
    def scalar(self, f: float):
        return self._func(f)[0,0]
    
class CoordDependent(MatProperty):
    _freq_dependent: bool = False
    _coord_dependent: bool = True
    def __init__(self, 
                 max_value: float,
                 scalar: Callable | None = None,
                 vector: Callable | None = None,
                 matrix: Callable | None = None,
                 ):
        """Creates a coordinate dependent property object.
        
        If the property is defined as a scalar value, use the "scalar" argument.
        
        If the property is a diagonal rank-2 tensor, use the "vector" argument.
        
        If the property is a full rank-2 tensor, use the "matrix" argument.
        

        The max_value property must be set to tell EMerge how height this value can get 
        as it will be used to define the discretization of the mesh.
        
        Args:
            max_value (float): The heighest value of the material property
            scalar (Callable | None, optional): The scalar value function returning a float/complex. Defaults to None.
            vector (Callable | None, optional): The diagonal rank-2 tensor function returning a (3,) array. Defaults to None.
            matrix (Callable | None, optional): The rank-2 tensor function returning a (3,3) array. Defaults to None.

        Returns:
            _type_: _description_
        """
        
        if scalar is not None:
            def _func(x, y, z) -> np.ndarray:
                return np.eye(3)[:, :, None] * scalar(x,y,z)[None, None, :]
        
        if vector is not None:
            def _func(x, y, z) -> np.ndarray:
                N = x.shape[0]
                out = np.zeros((3, 3, N), dtype=vector(0,0,0).dtype)
                idx = np.arange(3)
                out[idx, idx, :] = vector(x,y,z)
                return out
        if matrix is not None:
            _func = matrix

        self._func: Callable = _func
        self._apply_to: np.ndarray = np.array([], dtype=np.int64)
        self._x: np.ndarray = np.array([], dtype=np.float64)
        self._y: np.ndarray = np.array([], dtype=np.float64)
        self._z: np.ndarray = np.array([], dtype=np.float64)
        
        self._values: np.ndarray = None
        self._fmax: Callable = lambda f: max_value

    def initialize(self, x: np.ndarray, y: np.ndarray, z: np.ndarray, ids: np.ndarray) -> None:
        self._apply_to = np.concatenate([self._apply_to, ids])
        self._x = np.concatenate([self._x, x])
        self._y = np.concatenate([self._y, y])
        self._z = np.concatenate([self._z, z])
        
    def __call__(self, f: float, data: np.ndarray) -> np.ndarray:
        data[:,:,self._apply_to] = self._func(self._x, self._y, self._z)
        return data

    def scalar(self, f: float):
        return self._func(0,0,0)[0,0]
    
class FreqCoordDependent(MatProperty):
    _freq_dependent: bool = True
    _coord_dependent: bool = True
    
    def __init__(self, 
                 max_value: float,
                 scalar: Callable | None = None,
                 vector: Callable | None = None,
                 matrix: Callable | None = None):
        """Creates a frequency and coordinate dependent property object.
        
        If the property is defined as a scalar value, use the "scalar" argument.
        
        If the property is a diagonal rank-2 tensor, use the "vector" argument.
        
        If the property is a full rank-2 tensor, use the "matrix" argument.

        The max_value property must be set to tell EMerge how height this value can get 
        as it will be used to define the discretization of the mesh.
        
        Args:
            max_value (float): The heighest value of the material property
            scalar (Callable | None, optional): The scalar value function returning a float/complex. Defaults to None.
            vector (Callable | None, optional): The diagonal rank-2 tensor function returning a (3,) array. Defaults to None.
            matrix (Callable | None, optional): The rank-2 tensor function returning a (3,3) array. Defaults to None.

        Returns:
            _type_: _description_
        """
        if scalar is not None:
            def _func(f, x, y, z) -> np.ndarray:
                return np.eye(3)[:, :, None] * scalar(f,x,y,z)[None, None, :]
        
        if vector is not None:
            def _func(f,x, y, z) -> np.ndarray:
                N = x.shape[0]
                out = np.zeros((3, 3, N), dtype=vector(1e9,0,0,0).dtype)
                idx = np.arange(3)
                out[idx, idx, :] = vector(f,x,y,z)
                return out
        
        if matrix is not None:
            _func = matrix

        self._func: Callable = _func
        
        self._apply_to: np.ndarray = np.array([], dtype=np.int64)
        self._x: np.ndarray = np.array([], dtype=np.float64)
        self._y: np.ndarray = np.array([], dtype=np.float64)
        self._z: np.ndarray = np.array([], dtype=np.float64)
        
        self._fmax: Callable = lambda f: max_value

    def initialize(self, x: np.ndarray, y: np.ndarray, z: np.ndarray, ids: np.ndarray) -> None:
        self._apply_to = np.concatenate([self._apply_to, ids])
        self._x = np.concatenate([self._x, x])
        self._y = np.concatenate([self._y, y])
        self._z = np.concatenate([self._z, z])
        
    def __call__(self, f: float, data: np.ndarray) -> np.ndarray:
        data[:,:,self._apply_to] = self._func(f,self._x, self._y,self._z)
        return data
    
    def scalar(self, f: float):
        return self._func(f, 0,0,0)[0,0]

class Material:
    """The Material class generalizes a material in the EMerge FEM environment.

    If a scalar value is provided for the relative permittivity or the relative permeability
    it will be used as multiplication entries for the material property diadic as identity matrix.

    Additionally, a frequency, coordinate or both frequency and coordinate dependent material property
    may be supplied for the properties: er, ur, tand and cond.
    
    To supply a frequency-dependent property use: emerge.FreqDependent()
    To supply a coordinate-dependent property use: emerge.CoordDependent()
    to supply a frequency and coordinate dependent property use: emerge.FreqCoordDependent()
    
    """
    _pickle_exclude = {"_neff"}
    
    def __init__(self,
                 er: float | complex | np.ndarray | MatProperty = 1.0,
                 ur: float | complex | np.ndarray | MatProperty = 1.0,
                 tand: float | MatProperty = 0.0,
                 cond: float | MatProperty = 0.0,
                 _neff: float | None = None,
                 color: str ="#BEBEBE",
                 opacity: float = 1.0,
                 _metal: bool = False,
                 name: str = 'unnamed'):
        
        if not isinstance(er, MatProperty):
            er = MatProperty(er)
        if not isinstance(ur, MatProperty):
            ur = MatProperty(ur)
        if not isinstance(tand, MatProperty):
            tand = MatProperty(tand)
        if not isinstance(cond, MatProperty):
            cond = MatProperty(cond)
        
        self.name: str = name
        self.er: MatProperty = er
        self.ur: MatProperty = ur
        self.tand: MatProperty = tand
        self.cond: MatProperty = cond

        self.color: str = color
        self.opacity: float = opacity
        self._hash_key: int = -1
        
        if _neff is None:
            self._neff: Callable = lambda f: np.sqrt(self.ur._fmax(f)*self.er._fmax(f))
        else:
            self._neff: Callable = lambda f: _neff
        hex_str = self.color.lstrip('#')
        self._color_rgb = tuple(int(hex_str[i:i+2], 16)/255.0 for i in (0, 2, 4))
        self._metal: bool = _metal
    
    def __getstate__(self):
        state = self.__dict__.copy()
        for k in self._pickle_exclude:
            state.pop(k, None)
        
        return state
    
    def __setstate__(self, state):
        self.__dict__.update(state)
        for k in self._pickle_exclude:
            setattr(self, k, None)

    def __hash__(self):
        return self._hash_key
    
    def __str__(self) -> str:
        return f'Material({self.name}, {self._hash_key})'
    
    def __repr__(self):
        return f'Material({self.name}, {self._hash_key})'
    
    def initialize(self, xs: np.ndarray, ys: np.ndarray, zs: np.ndarray, ids: np.ndarray):
        """Initializes the Material properties to be evaluated at xyz-coordinates for
        a given set of tetrahedral ids.

        Args:
            xs (np.ndarray): The tet-centroid x-coordinates
            ys (np.ndarray): The tet-centroid y-coordinates
            zs (np.ndarray): The tet-centroid z-coordinates
            ids (np.ndarray): The tet-indices
        """
        self.er.initialize(xs, ys, zs, ids)
        self.ur.initialize(xs, ys, zs, ids)
        self.tand.initialize(xs, ys, zs, ids)
        self.cond.initialize(xs, ys, zs, ids)
    
    def reset(self) -> None:
        """Resets assignment of material properties to coordiantes and tetrahedral indices.
        """
        self.er.reset()
        self.ur.reset()
        self.tand.reset()
        self.cond.reset()
        self._hash_key = -1
        
    @property
    def frequency_dependent(self) -> bool:
        """If The material property are at all frequency dependent.

        Returns:
            bool: _description_
        """
        return self.er._freq_dependent or self.ur._freq_dependent or self.tand._freq_dependent or self.cond._freq_dependent
    
    @property
    def coordinate_dependent(self) -> bool:
        """If the material properties are at all coordinate dependent

        Returns:
            bool: _description_
        """
        return self.er._coord_dependent or self.ur._coord_dependent or self.tand._coord_dependent or self.cond._coord_dependent
    
    
    def neff(self, f: float):
        """ Computes the maximum occuring effective refractive index for this material."""
        return self._neff(f)
    
    @property
    def color_rgb(self) -> tuple[float,float,float]:
        return self._color_rgb
     
    @staticmethod
    def drude_model(conductivity: float, 
                    colission_time: float, 
                    er: float = 1.0,
                    ur: float = 1.0, 
                    color: str = "#aaaaaa", 
                    opacity: float = 0.3,
                    metal: bool = True) -> Material:
        """Creates a Material using the Drume model for conductivity
        Requires at least the DC bulk condutivity σ₀ [S/m] and the
        collision time τ.

        Args:
            conductivity (float): The DC bulk conductivity σ₀ in S/m
            colission_time (float): The collision time.
            er (float, optional): The dielectric constant. Defaults to 1.0.
            ur (float, optional): The relative permeability. Defaults to 1.0.
            color (str, optional): The material rendering color. Defaults to "#aaaaaa".
            opacity (float, optional): The material rendering opacity. Defaults to 0.3.
            metal (bool, optional): If it should be rendered as a metal.. Defaults to True.

        Returns:
            Material: The resultant material.
        """
        colission_dist = colission_time/C0
        fsigma = FreqDependent(scalar = lambda f: conductivity/(1 - 1j*2*np.pi*f*colission_dist))
        return Material(er, ur, 0.0, fsigma, color=color, opacity=opacity, _metal=metal)
    
AIR = Material(color="#4496f3", opacity=0.05, name='Air')
COPPER = Material(cond=5.8e7, color="#62290c", _metal=True, name='Copper')
PEC = Material(color="#ff78aa", opacity=1.0, cond=1e30, _metal=True, name="PEC")