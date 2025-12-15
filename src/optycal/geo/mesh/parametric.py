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
from typing import Callable, overload
from loguru import logger
from ..mesh import Mesh
from ..cs import CoordinateSystem, GCS
def orthonormalize(axis: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generates a set of orthonormal vectors given an input vector X

    Args:
        axis (np.ndarray): The X-axis

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: The X, Y and Z axis (orthonormal)
    """
    Xaxis = axis/np.linalg.norm(axis)
    V = np.array([0,1,0])
    if np.abs(np.dot(Xaxis, V)) < 1e-12:
        V = np.array([0,0,1])
    Yaxis = np.cross(Xaxis, V)
    Yaxis = np.abs(Yaxis/np.linalg.norm(Yaxis))
    Zaxis = np.cross(Xaxis, Yaxis)
    Zaxis = np.abs(Zaxis/np.linalg.norm(Zaxis))
    return Xaxis, Yaxis, Zaxis

def remove_unmeshed_vertices(vertices: np.ndarray, triangulation: list[tuple[int,int,int]]) -> tuple[np.ndarray, np.ndarray]:
    """Updates the vertices and triangles to a set with no unmeshed vertices.

    Args:
        vertices (np.ndarray): The input vertices (3,N)
        triangulation (list): The triangulation 

    Returns:
        tuple[np.ndarray, np.ndarray]: The vertices + triangulation
    """
    unique_id = np.sort(np.unique(np.array(triangulation).flatten()))
    replace_id = np.arange(unique_id.shape[0])
    mapping = {idx: i for idx, i in zip(unique_id, replace_id)}
    tri_out = [(mapping[i1], mapping[i2], mapping[i3]) for i1, i2, i3 in triangulation]
    tri_out = np.array(tri_out)

    return vertices[:, unique_id], tri_out

class ParametricLine:

    def __init__(self,
        fx: Callable = lambda t: 0*t,
        fy: Callable = lambda t: 0*t,
        fz: Callable = lambda t: 0*t,
        trange: tuple[float, float] = (0,1),
        Nsteps: int = 10_000):
        """The parametric line represents an object that draws a path through 3D space

        The coordinates of the line are defined by a parametrization
            x = fx(t) ∀ t ∈ trange
            y = fy(t) ∀ t ∈ trange
            z = fz(t) ∀ t ∈ trange

        Args:
            fx (Callalbe, optional): The parametric function for X. Defaults to lambda t:0*t.
            fy (Callalbe, optional): The parametric function for Y. Defaults to lambda t:0*t.
            fz (Callalbe, optional): The parametric function for Z. Defaults to lambda t:0*t.
            trange (tuple[float, float], optional): The parametric sweep range. Defaults to (0,1).
            Nsteps (int, optional): The number of steps. Defaults to 10_000.
        """
        
        self.fx: Callable = fx
        self.fy: Callable = fy
        self.fz: Callable = fz
        self.trange: tuple[float, float] = trange
        self.Nsteps: int = Nsteps

    def _segment(self, ds: float) -> np.ndarray:
        """Discretizes the parametric line into a set of discrete vertices connected by line segments

        Args:
            ds (float): The discretization step size ds

        Returns:
            np.ndarray: The (3,N) output coordinates.
        """
        t = np.linspace(self.trange[0], self.trange[1], self.Nsteps)
        x = self.fx(t)
        y = self.fy(t)
        z = self.fz(t)
        dx = np.diff(x)
        dy = np.diff(y)
        dz = np.diff(z)
        dl = np.sqrt(dx**2 + dy**2 + dz**2)
        LM = np.sum(dl)
        L = np.cumsum(dl)
        L = np.concatenate(([0,], L))
        tint = np.interp(np.linspace(0, LM, int(np.ceil(LM/ds))), L, t)
        xs = self.fx(tint)
        ys = self.fy(tint)
        zs = self.fz(tint)
        ps = np.array([xs, ys, zs])
        return ps

class SweepFunction:

    def __init__(self,
        fx: Callable = lambda x,y,z,t: x*np.ones_like(t),
        fy: Callable = lambda x,y,z,t: y*np.ones_like(t),
        fz: Callable = lambda x,y,z,t: z*np.ones_like(t),
        trange: tuple = (0.0, 1.0),
        Nsteps: int = 10_000):
        """A SweepFunctin object represents a parametric transformation of a 3D coordinate

        Let:
         - (x, y, z)         be the input 3D point
         - t ∈ [t0, t1]      the sweep parameter
         - N                 number of steps

        Define the sweep function T as:

        T(x, y, z) : [t0, t1] → ℝ³
        T(x, y, z)(t) = (
            fx(x, y, z, t),
            fy(x, y, z, t),
            fz(x, y, z, t)
        )

        That is, T maps a point (x, y, z) to a parameterized curve in 3D space.

        Args:
            fx (Callable, optional): The fx transformation. Defaults to lambdax.
            fy (Callable, optional): The fy transformation. Defaults to lambdax.
            fz (Callable, optional): The fz transformation. Defaults to lambdax.
            trange (tuple, optional): The parameter range. Defaults to (0.0, 1.0).
            Nsteps (int, optional): The number of discretization steps used in the derivation. Defaults to 10_000.
        """
        self.fx: Callable = fx
        self.fy: Callable = fy
        self.fz: Callable = fz
        self.trange: tuple[float, float] = trange
        self.Nsteps = Nsteps

    def _transform_coordinates(self, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
        """Applies the transformation mapping onto a coordinate set

        Args:
            x (np.ndarray): The X-coordinates (N,)
            y (np.ndarray): The Y-coordinates (N,)
            z (np.ndarray): The Z-coordinates (N,)

        Returns:
            np.ndarray: the XYZ coordinates (3,N)
        """
        p = np.linspace(*self.trange, self.Nsteps) # ty: ignore
        x = self.fx(p)
        y = self.fy(p)
        z = self.fz(p)
        return np.ndarray([x,y,z])

    @staticmethod
    def revolve(axis: tuple[float, float, float],
                angle: float = 2*np.pi,
                origin: tuple[float, float, float] = (0., 0., 0.)) -> SweepFunction:
        """Creates a revolution SweepFunction

        Args:
            axis (tuple[float, float, float]): The axis of rotation
            angle (float, optional): The angle of rotation (in radians). Defaults to 2*np.pi.
            origin (tuple[float, float, float], optional): The origin of the rotation axis. Defaults to (0., 0., 0.).

        Returns:
            SweepFunction: A sweep function
        """
        ux, uy, uz = axis

        def Rxx(p):
            return np.cos(p*angle) + ux**2*(1-np.cos(p*angle))
        def Rxy(p):
            return ux*uy*(1-np.cos(p*angle)) - uz*np.sin(p*angle)
        def Rxz(p):
            return ux*uz*(1-np.cos(p*angle)) + uy*np.sin(p*angle)
        
        def Ryx(p):
            return uy*ux*(1-np.cos(p*angle)) + uz*np.sin(p*angle)
        def Ryy(p):
            return np.cos(p*angle) + uy**2*(1-np.cos(p*angle))
        def Ryz(p):
            return uy*uz*(1-np.cos(p*angle)) - ux*np.sin(p*angle)
        
        def Rzx(p):
            return uz*ux*(1-np.cos(p*angle)) - uy*np.sin(p*angle)
        def Rzy(p):
            return uz*uy*(1-np.cos(p*angle)) + ux*np.sin(p*angle)
        def Rzz(p):
            return np.cos(p*angle) + uz**2*(1-np.cos(p*angle))

        x0, y0, z0 = origin
        
        def fx(x, y, z, p):
            return Rxx(p)*(x-x0) + Rxy(p)*(y-y0) + Rxz(p)*(z-z0) + x0
        def fy(x, y, z, p):
            return Ryx(p)*(x-x0) + Ryy(p)*(y-y0) + Ryz(p)*(z-z0) + y0
        def fz(x, y, z, p):
            return Rzx(p)*(x-x0) + Rzy(p)*(y-y0) + Rzz(p)*(z-z0) + z0
        
        return SweepFunction(fx, fy, fz)

    def __call__(self, path: ParametricLine, ds: float) -> tuple[np.ndarray, np.ndarray]:
        """Sweeps the input ParametricLine accordint to this sweep function

        Args:
            path (ParametricLine): The ParametricLine to sweep
            ds (float): The maximum step size

        Returns:
            (np.ndarray, np.ndarray): The Vertices and Triangulation of the output Mesh
        """
        T = self
        ps = path._segment(ds)
        xs, ys, zs = ps
        TColumns = []
        i = 0
        is_closed = []
        p = np.linspace(T.trange[0], T.trange[1], T.Nsteps)
            
        for x0, y0, z0 in zip(xs, ys, zs):
            x = T.fx(x0,y0,z0,p)
            y = T.fy(x0,y0,z0,p)
            z = T.fz(x0,y0,z0,p)
            
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
                sub[int(xyzind[3,-1])] = int(xyzind[3,0])

            
        vertices = np.array([xs, ys, zs])
        
        triangles = [(sub[i1],sub[i2],sub[i3]) for i1, i2, i3 in triangles]
        
        vertices, triangles = remove_unmeshed_vertices(vertices, triangles)
        return vertices, triangles

    def apply(self, line: ParametricLine, ds: float) -> tuple[np.ndarray, np.ndarray]:
        """Applies the Sweep function to a parametric line object to create a 3D surface

        Args:
            line (ParLine): The parametric line
            ds (float): The discretization step

        Returns:
            tuple[np.ndarray, np.ndarray]: The resultant vertices and triangulation
        """
        return self(line, ds)
    
    def mesh(self, line: ParametricLine, ds: float, cs: CoordinateSystem = GCS, 
             alignment_function: Callable | None = None,
             alignment_origin: tuple[float, float, float] | None = None) -> Mesh:
        v, t = self.apply(line, ds)
        mesh = Mesh(v, cs, alignment_function)
        if alignment_origin is not None:
            mesh.align_from_origin(*alignment_origin)
        mesh.set_triangles(t)
        return mesh
        
 
class Mapping:
    def __init__(self,
                 fx: Callable = lambda x,y,z: x, 
                 fy: Callable = lambda x,y,z: y, 
                 fz: Callable = lambda x,y,z: z):
        """Implements a generic coordinate transformation T(x,y,z) : ℝ³ → ℝ³
            fx (Callable, optional): The X-coordinate as a function of (x,y,z). Defaults to f(x,y,z)=x.
            fy (Callable, optional): The Y-coordinate as a function of (x,y,z). Defaults to f(x,y,z)=y.
            fz (Callable, optional): The Z-coordinate as a function of (x,y,z). Defaults to f(x,y,z)=z.
        """
        self.fx = fx
        self.fy = fy
        self.fz = fz
    
    @overload
    def map(self, other: SweepFunction) -> SweepFunction: ...
    
    @overload
    def map(self, other: Mapping) -> Mapping: ...
    
    def map(self, other: SweepFunction | Mapping) -> SweepFunction | Mapping:
        """Combines this coordinate mapping with another mapping or SweepFunction object

        Args:
            other (SweepFunction | Mapping): The transformation to apply this mapping to

        Returns:
            SweepFunction | Mapping: The resultant sweepfunctin or mapping
        """
        if isinstance(other, Mapping):
            return Mapping(fx = lambda x,y,z: self.fx(other.fx(x,y,z), other.fy(x,y,z), other.fz(x,y,z)),
                           fy = lambda x,y,z: self.fy(other.fx(x,y,z), other.fy(x,y,z), other.fz(x,y,z)),
                           fz = lambda x,y,z: self.fz(other.fx(x,y,z), other.fy(x,y,z), other.fz(x,y,z)))
        else:
            return SweepFunction(fx = lambda x,y,z,p: other.fx(self.fx(x,y,z), self.fy(x,y,z), self.fz(x,y,z), p),
                                fy = lambda x,y,z,p: other.fy(self.fx(x,y,z), self.fy(x,y,z), self.fz(x,y,z), p),
                                fz = lambda x,y,z,p: other.fz(self.fx(x,y,z), self.fy(x,y,z), self.fz(x,y,z), p),
                                    trange = other.trange,
                                    Nsteps = other.Nsteps)

    def __rmul__(self, other: SweepFunction) -> SweepFunction:
        return self.map(other)
    
    def __mul__(self, other: SweepFunction) -> SweepFunction:
        return self.map(other)
    
    def pmap(self, x, y, z):
        return self.fx(x,y,z), self.fy(x,y,z), self.fz(x,y,z)
    
    @staticmethod
    def parabolic_reflector(focal_point: tuple[float, float, float],
                            focal_length: float,
                            direction: tuple[float, float, float],) -> Mapping:
        """Creates the coordinate transformation to map a plane to a parabola.
        
        The mapping takes an input plane normal to the direction vector and displaces it such that
        the origin is in the focal point of the parabola

        The transformation only executes a displacement parallel to the direction to map any point in 3D space
        to be on the parabolic plane. 
        
        Args:
            origin (tuple[float, float, float]): The focal point
            focal_length (float): The focal length
            direction (tuple[float, float, float]): The pointing direction

        Returns:
            Mapping: The coordinate transformation
        """
        origin = np.array(focal_point)
        direction = np.array(direction)
        p0point = origin - focal_length*direction
        _, ax2, ax3 = orthonormalize(direction)
        ox, oy, oz = p0point
        dx, dy, dz = direction
        def Fd(x, y, z):
            return np.sqrt((x-ox)**2 + (y-oy)**2 + (z-oz)**2)
        def fxflat(x, y, z):
            return x-((x-ox)*dx + (y-oy)*dy + (z-oz)*dz)*dx
        def fyflat(x, y, z):
            return y-((x-ox)*dx + (y-oy)*dy + (z-oz)*dz)*dy
        def fzflat(x, y, z):
            return z-((x-ox)*dx + (y-oy)*dy + (z-oz)*dz)*dz
        
        def fx(x, y, z):
            return fxflat(x,y,z) + Fd(fxflat(x,y,z),fyflat(x,y,z),fzflat(x,y,z))**2/(4*focal_length)*direction[0]
        def fy(x, y, z):
            return fyflat(x,y,z) + Fd(fxflat(x,y,z),fyflat(x,y,z),fzflat(x,y,z))**2/(4*focal_length)*direction[1]
        def fz(x, y, z):
            return fzflat(x,y,z) + Fd(fxflat(x,y,z),fyflat(x,y,z),fzflat(x,y,z))**2/(4*focal_length)*direction[2]
        
        return Mapping(fx, fy, fz)
        
        
