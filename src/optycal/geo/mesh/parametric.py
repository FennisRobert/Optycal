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
        Rxx = lambda p: np.cos(p*angle) + ux**2*(1-np.cos(p*angle))
        Rxy = lambda p: ux*uy*(1-np.cos(p*angle)) - uz*np.sin(p*angle)
        Rxz = lambda p: ux*uz*(1-np.cos(p*angle)) + uy*np.sin(p*angle)
        
        Ryx = lambda p: uy*ux*(1-np.cos(p*angle)) + uz*np.sin(p*angle)
        Ryy = lambda p: np.cos(p*angle) + uy**2*(1-np.cos(p*angle))
        Ryz = lambda p: uy*uz*(1-np.cos(p*angle)) - ux*np.sin(p*angle)
        
        Rzx = lambda p: uz*ux*(1-np.cos(p*angle)) - uy*np.sin(p*angle)
        Rzy = lambda p: uz*uy*(1-np.cos(p*angle)) + ux*np.sin(p*angle)
        Rzz = lambda p: np.cos(p*angle) + uz**2*(1-np.cos(p*angle))

        x0, y0, z0 = origin
        
        fx = lambda x,y,z,p: Rxx(p)*(x-x0) + Rxy(p)*(y-y0) + Rxz(p)*(z-z0) + x0
        fy = lambda x,y,z,p: Ryx(p)*(x-x0) + Ryy(p)*(y-y0) + Ryz(p)*(z-z0) + y0
        fz = lambda x,y,z,p: Rzx(p)*(x-x0) + Rzy(p)*(y-y0) + Rzz(p)*(z-z0) + z0
        
        return SweepFunction(fx, fy, fz)

    def __call__(self, path: ParLine, ds: float):
        T = self
        
        ps = path.segment(ds)
        xs, ys, zs = ps
        xsa = np.array([])
        ysa = np.array([])
        zsa = np.array([])
        #t = np.linspace(self.trange[0], self.trange[1], self.Nsteps)
        TColumns = []
        i = 0
        is_closed = []
        p = np.linspace(T.prange[0], T.prange[1], T.Nsteps)
            
        for x0, y0, z0 in zip(xs, ys, zs):
            x = T.fx(x0,y0,z0,p)
            y = T.fy(x0,y0,z0,p)
            z = T.fz(x0,y0,z0,p)
            #print(x,y,z,p,T.prange, T.Nsteps)
            if np.linalg.norm(np.array([x[0], y[0], z[0]]) - np.array([x[-1], y[-1], z[-1]])) < 1e-6:
                is_closed.append(True)
            else:
                is_closed.append(False)
            dx = np.diff(x)
            dy = np.diff(y)
            dz = np.diff(z)
            dl = np.sqrt(dx**2 + dy**2 + dz**2)
            LM = sum(dl)
            if LM == 0:
                TColumns.append(np.array([x[0], y[0], z[0], i]).reshape((4,1)))
                is_closed[-1] = False
                i = i + 1
                continue
            L = np.cumsum(dl)
            L = np.concatenate(([0,], L))
            pint = np.interp(np.linspace(0, LM, int(np.ceil(LM/ds))), L, p)
            xsi = T.fx(x0,y0,z0,pint)
            ysi = T.fy(x0,y0,z0,pint)
            zsi = T.fz(x0,y0,z0,pint)
            arry = np.array([xsi, ysi, zsi, np.arange(i, i+len(xsi))])
            TColumns.append(arry)
            i = i + len(xsi)
        triangles = []
        nrm = np.linalg.norm
        for row1, row2 in zip(TColumns[:-1], TColumns[1:]):
            
            if row1.shape==(4,1) and row2.shape[1]>1:
                ind2 = row2[3,:]
                for i2, i3 in zip(ind2[:-1], ind2[1:]):
                    triangles.append([row1[3,0], i2, i3])
            elif row2.shape==(4,1) and row1.shape[1]>1:
                ind1 = row1[3,:]
                for i1, i2 in zip(ind1[:-1], ind1[1:]):
                    triangles.append([i1, i2, row2[3,0]])
            else:
                ileft = 0
                iright = 0
                ilmax = row1.shape[1]-1
                irmax = row2.shape[1]-1

                while ileft < ilmax or iright < irmax:
                    vleft = row1[0:3,ileft]
                    vright = row2[0:3,iright]
                    if ileft == ilmax:
                        triangles.append((row1[3,ileft], row2[3,iright], row2[3,iright+1]))
                        iright = iright + 1
                        continue
                    if iright == irmax:
                        triangles.append((row1[3,ileft], row2[3,iright], row1[3,ileft+1]))
                        ileft = ileft + 1
                        continue
                    vltop = row1[0:3,ileft+1]
                    vrtop = row2[0:3,iright+1]
                    if nrm(vltop-vright)+nrm(vleft-vltop) < nrm(vrtop-vleft)+nrm(vright-vrtop):
                        triangles.append((row1[3,ileft], row2[3,iright], row1[3,ileft+1]))
                        ileft = ileft + 1
                    else:
                        triangles.append((row1[3,ileft], row2[3,iright], row2[3,iright+1]))
                        iright = iright + 1
        xs = np.array([])
        ys = np.array([])
        zs = np.array([])
        ids = np.array([])
        sub = dict()
        for xyzind, IC in zip(TColumns, is_closed):
            for i in xyzind[3,:]:
                sub[int(i)] = int(i)

            xs = np.concatenate((xs, xyzind[0,:]))
            ys = np.concatenate((ys, xyzind[1,:]))
            zs = np.concatenate((zs, xyzind[2,:]))
            ids = np.concatenate((ids, xyzind[3,:]))
            if IC:
                #print(f'{int(xyzind[3,-1])} -> {int(xyzind[3,0])}')
                sub[int(xyzind[3,-1])] = int(xyzind[3,0])

            
        vertices = np.array([xs, ys, zs])
        #for key, value in sub.items():
        #    print(f'Mapping {vertices[:,key]} to {vertices[:,value]}')
        triangles = [(sub[i1],sub[i2],sub[i3]) for i1, i2, i3 in triangles]
        
        vertices, triangles = remove_unmeshed_vertices(vertices, triangles)
        return vertices, triangles

class Mapping:
    def __init__(self,
                 fx: Callable = lambda x,y,z: x, 
                 fy: Callable = lambda x,y,z: y, 
                 fz: Callable = lambda x,y,z: z):
        self.fx = fx
        self.fy = fy
        self.fz = fz
        
    
    def __rmul__(self, other: SweepFunction) -> SweepFunction:
        return SweepFunction(fx = lambda x,y,z,p: other.fx(self.fx(x,y,z), self.fy(x,y,z), self.fz(x,y,z), p),
                             fy = lambda x,y,z,p: other.fy(self.fx(x,y,z), self.fy(x,y,z), self.fz(x,y,z), p),
                             fz = lambda x,y,z,p: other.fz(self.fx(x,y,z), self.fy(x,y,z), self.fz(x,y,z), p),
                                prange = other.prange,
                                Nsteps = other.Nsteps)
    
    def __mul__(self, other: SweepFunction) -> SweepFunction:
        return SweepFunction(fx = lambda x,y,z,p: self.fx(other.fx(x,y,z,p), other.fy(x,y,z,p), other.fz(x,y,z,p)),
                             fy = lambda x,y,z,p: self.fy(other.fx(x,y,z,p), other.fy(x,y,z,p), other.fz(x,y,z,p)),
                             fz = lambda x,y,z,p: self.fz(other.fx(x,y,z,p), other.fy(x,y,z,p), other.fz(x,y,z,p)),
                                prange = other.prange,
                                Nsteps = other.Nsteps)
    
    def pmap(self, x, y, z):
        return self.fx(x,y,z), self.fy(x,y,z), self.fz(x,y,z)
    
    @staticmethod
    def parabolic_reflector(origin: np.ndarray,
                            focal_length: float,
                            direction: np.ndarray):
        p0point = origin - focal_length*direction
        _, ax2, ax3 = orthonormalize(direction)
        ox, oy, oz = p0point
        dx, dy, dz = direction
        Fd = lambda x,y,z: np.sqrt((x-ox)**2 + (y-oy)**2 + (z-oz)**2)
        fxflat = lambda x,y,z: x-((x-ox)*dx + (y-oy)*dy + (z-oz)*dz)*dx
        fyflat = lambda x,y,z: y-((x-ox)*dx + (y-oy)*dy + (z-oz)*dz)*dy
        fzflat = lambda x,y,z: z-((x-ox)*dx + (y-oy)*dy + (z-oz)*dz)*dz
        
        #fxflat = lambda x,y,z: x-(x-p0point[0])*direction[0]
        #fyflat = lambda x,y,z: y-(y-p0point[1])*direction[1]
        #fzflat = lambda x,y,z: z-(z-p0point[2])*direction[2]
        
        fx = lambda x,y,z: fxflat(x,y,z) + Fd(fxflat(x,y,z),fyflat(x,y,z),fzflat(x,y,z))**2/(4*focal_length)*direction[0]
        fy = lambda x,y,z: fyflat(x,y,z) + Fd(fxflat(x,y,z),fyflat(x,y,z),fzflat(x,y,z))**2/(4*focal_length)*direction[1]
        fz = lambda x,y,z: fzflat(x,y,z) + Fd(fxflat(x,y,z),fyflat(x,y,z),fzflat(x,y,z))**2/(4*focal_length)*direction[2]
        #fx = lambda x,y,z: fxflat(x,y,z) + Fd(fxflat(x,y,z),fyflat(x,y,z),fzflat(x,y,z))*direction[0]
        #fy = lambda x,y,z: fyflat(x,y,z) + Fd(fxflat(x,y,z),fyflat(x,y,z),fzflat(x,y,z))*direction[1]
        #fz = lambda x,y,z: fzflat(x,y,z) + Fd(fxflat(x,y,z),fyflat(x,y,z),fzflat(x,y,z))*direction[2]
        return Mapping(fx, fy, fz)
        
        
