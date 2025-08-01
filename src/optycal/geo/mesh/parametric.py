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
from typing import Callable
from loguru import logger

class ParLine:

    def __init__(self,
        fx: Callable = lambda t: 0*t,
        fy: Callable = lambda t: 0*t,
        fz: Callable = lambda t: 0*t,
        trange: tuple[float, float] = (0,1),
        Nsteps: int = 10_000):
        
        self.fx: Callable = fx
        self.fy: Callable = fy
        self.fz: Callable = fz

    def segment(self, ds: float) -> np.ndarray:
        t = np.linspace(self.trange[0], self.trange[1], self.Nsteps)
        x = self.fx(t)
        y = self.fy(t)
        z = self.fz(t)
        dx = np.diff(x)
        dy = np.diff(y)
        dz = np.diff(z)
        dl = np.sqrt(dx**2, dy**2, dz**2)
        LM = np.sum(dl)
        L = np.cumsum(dl)
        L = np.concatenate(([0,], L))
        tint = np.interp(np.linspace(0, LM, int(np.ceil(LM/ds))), L, t)
        xs = self.fx(tint)
        ys = self.fy(tint)
        zs = self.fz(tint)
        ps = np.array([xs, ys, zs])
        return ps

class SweepFuncion:

    def __init__(self,
        fx: Callable = lambda x,y,z,t: x*np.ones_like(t),
        fy: Callable = lambda x,y,z,t: x*np.ones_like(t),
        fz: Callable = lambda x,y,z,t: x*np.ones_like(t),
        prange: tuple = (0.0, 1.0),
        Nsteps: int = 10_000):
        self.fx: Callable = fx
        self.fy: Callable = fy
        self.fz: Callable = fz
        self.prange: tuple[float, float] = prange
        self.Nsteps = Nsteps

    def _transform_coordinates(self, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
        p = np.linspace(*self.prange, self.Nsteps)
        x = self.fx(p)
        y = self.fy(p)
        z = self.fz(p)
        return np.ndarray([x,y,z])

    @staticmethod
    def revolve(axis: np.ndarray,
                angle: float = 2*np.pi,
                origin: tuple[float, float, float] = (0., 0., 0.)):
        ux, uy, uz = axis
        
        