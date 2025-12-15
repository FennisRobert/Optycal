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
from enum import Enum
from typing import Tuple, Union

import numpy as np

from .solvers import stratton_chu_xyz_surface, stratton_chu_ff, stratton_chu_xyz
from .samplespace import FarFieldSpace
from .geo import Mesh, CoordinateSystem
from .geo.cs import GCS
from .multilayer import SurfaceRT, FRES_AIR
from .field import Field
from .settings import GLOBAL_SETTINGS
from numba_progress import ProgressBar
from loguru import logger

class SurfaceType(Enum):
    CONVEX = 0
    CONCAVE = 1

def fortran_array(x):
    return np.asfortranarray(x)

class Surface:
    mesh: Mesh = Mesh
    fresnel: SurfaceRT = SurfaceRT
    polyorder: int = 2
    surfacetype: SurfaceType = SurfaceType
    cs: CoordinateSystem = CoordinateSystem

    def __init__(
        self,
        mesh: Mesh,
        fresnel: SurfaceRT,
        polyorder: int = 2,
        name: str = "UnnamedSurface",
    ):

        self.mesh: Mesh = mesh
        self.polyorder = polyorder
        self.name = name
        self.fresnel: SurfaceRT = fresnel
        self.n_points = 0
        if self.polyorder == 1:
            self.n_points = self.mesh.nvertices
        elif self.polyorder == 2:
            self.n_points = self.mesh.nedges

        self.E1: np.ndarray = np.zeros(self.fieldshape, dtype=np.complex128)
        self.E2: np.ndarray = np.zeros(self.fieldshape, dtype=np.complex128)
        self.H1: np.ndarray = np.zeros(self.fieldshape, dtype=np.complex128)
        self.H2: np.ndarray = np.zeros(self.fieldshape, dtype=np.complex128)
        self.k0 = None

        self.__post_init__()

    def __str__(self) -> str:
        return f'Surface[{self.name}]'
    
    @property
    def fieldshape(self) -> tuple:
        return (3, self.npoints)
    
    def __post_init__(self):
        pass
    
    def __gt__(self, other) -> tuple:
        return (self, other)
    
    def catch(self, fr12: tuple, k0: float):
        fr1, fr2 = fr12
        self.add_field(1, fr1, k0)
        self.add_field(2, fr2, k0)
        
    def clear_fields(self):
        self.E1: np.ndarray = np.zeros(self.fieldshape, dtype=np.complex128)
        self.E2: np.ndarray = np.zeros(self.fieldshape, dtype=np.complex128)
        self.H1: np.ndarray = np.zeros(self.fieldshape, dtype=np.complex128)
        self.H2: np.ndarray = np.zeros(self.fieldshape, dtype=np.complex128)
        self.k0 = None
        
    def trimap(self, field: np.ndarray) -> np.ndarray:
        if self.polyorder == 1:
            tris = self.mesh.triangles
            return (field[tris[0,:]] + field[tris[1,:]] + field[tris[2,:]]) / 3
        if self.polyorder == 2:
            e2t = np.array([self.mesh.t2e[i] for i in range(self.mesh.ntriangles)])
            edges = (field[e2t[0,:]] + field[e2t[1,:]] + field[e2t[2,:]]) / 3

    def field_coordinates(self) -> np.ndarray:
        if self.polyorder == 1:
            return self.mesh.g.vertices
        elif self.polyorder == 2:
            return self.mesh.g.edge_centers

    def field_normals(self) -> np.ndarray:
        if self.polyorder == 1:
            return self.mesh.g.vertex_normals
        elif self.polyorder == 2:
            return self.mesh.g.edge_normals

    def trianglewise_indices(self) -> np.ndarray:
        if self.polyorder == 1:
            return self.mesh.triangles
        elif self.polyorder == 2:
            return self.mesh.t2e
        
    def normals(self) -> np.ndarray:
        return self.mesh.g.normals
    
    @property
    def xyz(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self.mesh.g.xyz
    
    @property
    def gxyz(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self.mesh.g.xyz

    def write_field(self, side: int, field: Field, k0: float):
        if side == 1:
            self.E1 = field.E
            self.H1 = field.H
            self.k0 = k0
        elif side == 2:
            self.E2 = field.E
            self.H2 = field.H
            self.k0 = k0
    
    def add_field(self, side: int, field: Field, k0: float):
        if side == 1:
            self.E1 += field.E
            self.H1 += field.H
            self.k0 = k0
        elif side == 2:
            self.E2 += field.E
            self.H2 += field.H
            self.k0 = k0

    def vertex_field(self, side: int = 0) -> Field:

        E = self.E1 + self.E2
        H = self.H1 + self.H2
        if side == 1:
            E = self.E1
            H = self.H1
        if side == 2:
            E = self.E2
            H = self.H2

        gv = self.mesh.g.vertices
        if self.polyorder == 1:
            return Field(E=E, H=H, x=gv[0,:], y=gv[1,:], z=gv[2,:], creator=self.name)
        
        if self.polyorder == 2:
            
            E2 = np.zeros((3, self.mesh.nvertices), dtype=np.complex128)
            H2 = np.zeros((3, self.mesh.nvertices), dtype=np.complex128)
            for iv in range(self.mesh.nvertices):
                ies = np.array(self.mesh.v2e[iv])
                E2[:, iv] = E[:, ies].mean(axis=1)
                H2[:, iv] = H[:, ies].mean(axis=1)
            return Field(E=E2, H=H2, x=gv[0,:], y=gv[1,:], z=gv[2,:], creator=self.name)

    def integration_fields(
        self,
    ) -> Union[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if self.polyorder == 1:
            tris = self.mesh.triangles
            E1 = (
                self.E1[:, tris[0,:]],
                self.E1[:, tris[1,:]],
                self.E1[:, tris[2,:]],
            )
            E2 = (
                self.E2[:, tris[0,:]],
                self.E2[:, tris[1,:]],
                self.E2[:, tris[2,:]],
            )
            H1 = (
                self.H1[:, tris[0,:]],
                self.H1[:, tris[1,:]],
                self.H1[:, tris[2,:]],
            )
            H2 = (
                self.H2[:, tris[0,:]],
                self.H2[:, tris[1,:]],
                self.H2[:, tris[2,:]],
            )
            return (E1, H1, E2, H2)
        if self.polyorder == 2:
            t2e = self.mesh.t2e
            E1 = (self.E1[:, t2e[0, :]], self.E1[:, t2e[1, :]], self.E1[:, t2e[2, :]])
            E2 = (self.E2[:, t2e[0, :]], self.E2[:, t2e[1, :]], self.E2[:, t2e[2, :]])
            H1 = (self.H1[:, t2e[0, :]], self.H1[:, t2e[1, :]], self.H1[:, t2e[2, :]])
            H2 = (self.H2[:, t2e[0, :]], self.H2[:, t2e[1, :]], self.H2[:, t2e[2, :]])
            return (E1, H1, E2, H2)
        return None

    def expose_xyz(self, gx: np.ndarray, gy: np.ndarray, gz: np.ndarray, side=-1) -> Field:
        return sc_expose_xyz(self, gx, gy, gz, side=side)
    
    def expose_thetaphi(self, gtheta: np.ndarray, gphi: np.ndarray, side=-1) -> Field:
        return sc_expose_thetaphi(self, gtheta, gphi, side=side)
    
    def expose_surface(self, target, side=-1) -> Field:
        fr1, fr2 = sc_expose_surface(self, target, side=side)
        target.add_field(1, fr1, self.k0)
        target.add_field(2, fr2, self.k0)
    
    def expose_ff(self, target: FarFieldSpace, side=-1) -> Field:
        fr = sc_expose_thetaphi(self, target.theta, target.phi, side=side)
        target.field = fr
        return fr
        
    @property
    def fieldshape(self):
        return (3, self.n_points)

    @property
    def side1(self) -> Field:
        return Field(
            E=self.E1, H=self.H1, creator=self.name, x=self.mesh.g.xs, y=self.mesh.g.ys, z=self.mesh.g.zs
        )

    @property
    def side2(self) -> Field:
        return Field(
            E=self.E2, H=self.H2, creator=self.name, x=self.mesh.g.xs, y=self.mesh.g.ys, z=self.mesh.g.zs
        )

    def intersects(self, x0, y0, z0, x2, y2, z2):
        raise NotImplementedError
    
    def powerflux(self):
        Ptot1 = 0
        Ptot2 = 0
        if self.polyorder==2:
            ne = self.mesh.edge_normals
            for ie in range(self.mesh.nedges):
                ts = tuple(self.mesh.e2t[ie])
                avgarea = np.sum(self.mesh.areas[np.ix_(ts)])/3
                S1 = 0.5*np.cross(self.E1[:,ie], np.conj(self.H1[:,ie])).real
                S2 = 0.5*np.cross(self.E2[:,ie], np.conj(self.H2[:,ie])).real

                Ptot1 += avgarea*(S1[0]*ne[0,ie] + S1[1]*ne[1,ie] + S1[2]*ne[2,ie])
                Ptot2 += avgarea*(S2[0]*ne[0,ie] + S2[1]*ne[1,ie] + S2[2]*ne[2,ie])
            return Ptot1, Ptot2
        else:
            nv = self.mesh.vertex_normals
            for iv in range(self.mesh.nedges):
                ts = tuple(self.mesh.v2t[iv])
                avgarea = np.sum(self.mesh.areas[np.ix_(ts)])/3
                S1 = 0.5*np.cross(self.E1[:,ie], np.conj(self.H1[:,ie])).real
                S2 = 0.5*np.cross(self.E2[:,ie], np.conj(self.H2[:,ie])).real

                Ptot1.append(avgarea*(S1[0]*ne[0,ie] + S1[1]*ne[1,ie] + S1[2]*ne[2,ie]))
                Ptot2.append(avgarea*(S2[0]*ne[0,ie] + S2[1]*ne[1,ie] + S2[2]*ne[2,ie]))
            
            return Ptot1, Ptot2
    
    def powerflux_Eonly(self):
        from emerge.library.lib import Z0
        tris = self.mesh.triangles
        Ptot1 = 0
        Ptot2 = 0
        E1 = self.E1
        E2 = self.E2
        if self.polyorder==2:
            ne = self.mesh.edge_normals
            for ie in range(self.mesh.nedges):
                ts = tuple(self.mesh.e2t[ie])
                avgarea = np.sum(self.mesh.areas[np.ix_(ts)])/3
                n = ne[:,ie]
                E1n = E1[:,ie] - (E1[0,ie]*n[0] + E1[1,ie]*n[1] + E1[2,ie]*n[2])*n
                E2n = E2[:,ie] - (E2[0,ie]*n[0] + E2[1,ie]*n[1] + E2[2,ie]*n[2])*n
                
                Ptot1 += avgarea*0.5*np.linalg.norm(E1n)**2/Z0
                Ptot2 += avgarea*0.5*np.linalg.norm(E2n)**2/Z0
            
            return Ptot1, Ptot2
        else:
            nv = self.mesh.vertex_normals
            for iv in range(self.mesh.nedges):
                ts = tuple(self.mesh.v2t[iv])
                avgarea = np.sum(self.mesh.areas[np.ix_(ts)])/3
                n = nv[:,iv]
                E1n = E1[:,iv] - (E1[0,iv]*n[0] + E1[1,iv]*n[1] + E1[2,iv]*n[2])*n
                E2n = E2[:,iv] - (E2[0,iv]*n[0] + E2[1,iv]*n[1] + E2[2,iv]*n[2])*n
                
                Ptot1 += avgarea*0.5*np.linalg.norm(E1n)**2/Z0
                Ptot2 += avgarea*0.5*np.linalg.norm(E2n)**2/Z0
            
            return Ptot1, Ptot2
        
        E1 = (self.E1[:,tris[0,:]] + self.E1[:,tris[1,:]] + self.E1[:,tris[2,:]])/3
        E2 = (self.E2[:,tris[0,:]] + self.E2[:,tris[1,:]] + self.E2[:,tris[2,:]])/3
        
        n = self.mesh.normals
        A = self.mesh.areas
        E1 = E1-(E1[0,:]*n[0,:] + E1[1,:]*n[1,:] + E1[2,:]*n[2,:])*n
        E2 = E2-(E2[0,:]*n[0,:] + E2[1,:]*n[1,:] + E2[2,:]*n[2,:])*n

        S1n = 0.5*np.linalg.norm(E1,axis=0)**2/Z0
        S2n = 0.5*np.linalg.norm(E2,axis=0)**2/Z0

        Po1 = np.sum(S1n * A)
        Po2 = np.sum(S2n * A)
        return Po1, Po2
    
    def generate_antenna_patterns(self, NSamples: int):
        from scipy.interpolate import RegularGridInterpolator
        phis = np.linspace(-np.pi,np.pi,NSamples)
        thetas = np.linspace(-np.pi,np.pi,NSamples)
        T,P = np.meshgrid(thetas,phis,indexing='ij')
        Tf = T.flatten()
        Pf = P.flatten()

        fr = self.expose_thetaphi(Tf,Pf,side=2)

        Ex = fr.Ex.reshape(T.shape)
        Ey = fr.Ey.reshape(T.shape)
        Ez = fr.Ez.reshape(T.shape)
        Hx = fr.Hx.reshape(T.shape)
        Hy = fr.Hy.reshape(T.shape)
        Hz = fr.Hz.reshape(T.shape)

        fex = RegularGridInterpolator((thetas, phis), Ex, bounds_error=False, fill_value=0, method='cubic')
        fey = RegularGridInterpolator((thetas, phis), Ey, bounds_error=False, fill_value=0, method='cubic')
        fez = RegularGridInterpolator((thetas, phis), Ez, bounds_error=False, fill_value=0, method='cubic')
        fhx = RegularGridInterpolator((thetas, phis), Hx, bounds_error=False, fill_value=0, method='cubic')
        fhy = RegularGridInterpolator((thetas, phis), Hy, bounds_error=False, fill_value=0, method='cubic')
        fhz = RegularGridInterpolator((thetas, phis), Hz, bounds_error=False, fill_value=0, method='cubic')

        def ffpat(theta, phi, k0):
            ex = fex((theta, phi))
            ey = fey((theta, phi))
            ez = fez((theta, phi))
            hx = fhx((theta, phi))
            hy = fhy((theta, phi))
            hz = fhz((theta, phi))
            
            return ex, ey, ez, hx, hy, hz

        # Near-field pattern function (similar to ffpat, but includes 'r')
        def nfpat(theta, phi, r, k0):

            ex = fex((theta, phi))
            ey = fey((theta, phi))
            ez = fez((theta, phi))
            hx = fhx((theta, phi))
            hy = fhy((theta, phi))
            hz = fhz((theta, phi))
            return ex, ey, ez, hx, hy, hz

        return nfpat, ffpat

    @staticmethod
    def import_model(
        vertices: np.ndarray,
        triangles: np.ndarray,
        E: np.ndarray,
        H: np.ndarray,
        origin: np.ndarray,
        k0: float,
        cs: CoordinateSystem = GCS) -> Surface:
        
        mesh = Mesh(vertices, cs)
        mesh.align_from_origin(origin[0], origin[1], origin[2])
        mesh.set_triangles(triangles)
        surface = Surface(mesh, FRES_AIR, 2, "ImportedSurface")
        surface.write_field(2, Field(E=E, H=H), k0)
        return surface

class OrientableSurface(Surface):
    surfacetype: SurfaceType = SurfaceType

    def __post_init__(self):
        self.surfacetype = SurfaceType.CONVEX

    def set_convex(self):
        self.surfacetype = SurfaceType.CONVEX

    def set_concave(self):
        self.surfacetype = SurfaceType.CONCAVE

    def inside(self, x, y, z):
        raise NotImplementedError


class Sphere(OrientableSurface):
    def __init__(
        self,
        origin: np.ndarray,
        radius: float,
        ds: float,
        fresnel: SurfaceRT,
        cs: CoordinateSystem,
        polyorder: int = 1,
        name: str = "UnnamedSphere",
    ):
        self.radius = radius
        self.origin = origin
        mesh = Mesh.generate_sphere(origin, radius, ds, cs)
        OrientableSurface.__init__(
            self, mesh, fresnel, polyorder=polyorder, name=name, cs=cs
        )
        self.set_convex()

def sc_expose_surface(
    source: Surface, target: Surface, side: int = -1
) -> Field:
    fres = target.fresnel
    E1in = source.E1
    E2in = source.E2
    H1in = source.H1
    H2in = source.H2
    
    Ein = np.array(E2in)
    Hin = np.array(H2in)


    Ein = fortran_array(np.array(Ein))
    Hin = fortran_array(np.array(Hin))

    Emag = np.sqrt(np.abs(Ein[0,:])**2 + np.abs(Ein[1,:])**2 + np.abs(Ein[2,:])**2)
    Ntot = np.argwhere(Emag>GLOBAL_SETTINGS.integration_limit*np.max(Emag)).shape[0]
    logger.debug(f'Percentage Included: {Ntot/Emag.shape[0]*100:.0f}%')
    mesh = source.mesh
    areas = fortran_array(mesh.areas)
    vis = fortran_array(source.field_coordinates())
    wns = np.zeros_like(vis).astype(np.float64)
    tri_normals = source.normals()
    tri_ids = source.trianglewise_indices()

    for i in range(mesh.ntriangles):
        n = tri_normals[:,i]
        i1, i2, i3 = tri_ids[:,i]
        wns[:,i1] += n*areas[i]/3
        wns[:,i2] += n*areas[i]/3
        wns[:,i3] += n*areas[i]/3
    
    cout = target.field_coordinates()
    normals_out = target.field_normals()
    normals_out = normals_out + np.random.rand(*normals_out.shape)*1e-15
    if side == 1:
        Ein = E1in
        Hin = H1in
    if side == 2:
        wns = -wns
        Ein = E2in
        Hin = H2in
    if side == -1:
        Ein = E2in - E1in
        Hin = H2in - H1in
        
    fresnel = np.array(
        [fres.angles, fres.Rte1, fres.Rtm1, fres.Rte2, fres.Rtm2, fres.Tte, fres.Ttm]
    ).T

    with ProgressBar(total=Ntot, ncols=100, dynamic_ncols=False) as pgb:
        # numba_function(num_iterations, progress)

        Eout1, Hout1, Eout2, Hout2 = stratton_chu_xyz_surface(
            Ein.astype(np.complex128),
            Hin.astype(np.complex128),
            vis.astype(np.float64),
            wns.astype(np.float64),
            cout.astype(np.float64),
            normals_out.astype(np.float64),
            fresnel.astype(np.complex128),
            source.k0,
            pgb,
        )
    fr1 = Field(E=Eout1, H=Hout1, x=cout[0,:], y=cout[1,:], z=cout[2,:])
    fr2 = Field(E=Eout2, H=Hout2, x=cout[0,:], y=cout[1,:], z=cout[2,:])
    return fr1, fr2



def sc_expose_xyz(source: Surface, x: np.ndarray, y: np.ndarray, z: np.ndarray, side: int = -1) -> Field:

    E1in = source.E1
    E2in = source.E2
    H1in = source.H1
    H2in = source.H2


    mesh = source.mesh
    areas = fortran_array(mesh.areas)
    vis = fortran_array(source.field_coordinates())
    wns = np.zeros_like(vis).astype(np.float64)
    tri_normals = source.normals()
    tri_ids = source.trianglewise_indices()
    
    for i in range(mesh.ntriangles):
        n = tri_normals[:,i]
        i1, i2, i3 = tri_ids[:,i]
        wns[:,i1] += n*areas[i]/3
        wns[:,i2] += n*areas[i]/3
        wns[:,i3] += n*areas[i]/3
    
    Eout = None
    Hout = None
    cout = np.array([x,y,z]).astype(np.float64)
    
    if side == 1:
        wns = -wns
        Ein = E1in
        Hin = H1in
    if side == 2:
        wns = wns
        Ein = E2in
        Hin = H2in
    if side == -1:
        Ein = E2in - E1in
        Hin = H2in - H1in
    Emag = np.sqrt(np.abs(Ein[0,:])**2 + np.abs(Ein[1,:])**2 + np.abs(Ein[2,:])**2)
    Ntot = np.argwhere(Emag>GLOBAL_SETTINGS.integration_limit*np.max(Emag)).shape[0]
    logger.debug(f'Percentage Included: {Ntot/Emag.shape[0]*100:.0f}%')
    with ProgressBar(total=Ntot, ncols=100, dynamic_ncols=False) as pgb:
        Eout, Hout = stratton_chu_xyz(
            Ein.astype(np.complex128),
            Hin.astype(np.complex128),
            vis.astype(np.float64),
            wns.astype(np.float64),
            cout.astype(np.float64),
            source.k0,
            pgb,
        )
    return Field(E=Eout, H=Hout, x=x, y=y, z=z)


def sc_expose_thetaphi(source: Surface, theta: np.ndarray, phi: np.ndarray, side: int):
    #(E1in, H1in, E2in, H2in) = source.integration_fields()
    E1in = source.E1
    E2in = source.E2
    
    H1in = source.H1
    H2in = source.H2
    
    if side == 1:
        Ein = E1in
        Hin = H1in
    elif side == 2:
        Ein = E2in
        Hin = H2in
    else:
        Ein = E1in - E2in
        Hin = H1in - H2in

    Ein = np.array(Ein)
    Hin = np.array(Hin)

    Emag = np.sqrt(np.abs(Ein[0,:])**2 + np.abs(Ein[1,:])**2 + np.abs(Ein[2,:])**2)
    Ntot = np.argwhere(Emag>GLOBAL_SETTINGS.integration_limit*np.max(Emag)).shape[0]
    logger.debug(f'Percentage Included: {Ntot/Emag.shape[0]*100:.0f}%')
    mesh = source.mesh
    areas = mesh.areas
    vis = source.field_coordinates()
    wns = np.zeros_like(vis).astype(np.float64)
    tri_normals = source.normals()
    tri_ids = source.trianglewise_indices()
    
    for i in range(mesh.ntriangles):
        n = tri_normals[:,i]
        i1, i2, i3 = tri_ids[:,i]
        wns[:,i1] += n*areas[i]/3
        wns[:,i2] += n*areas[i]/3
        wns[:,i3] += n*areas[i]/3
    
    Eout = None
    Hout = None
    tpout = np.array([theta, phi])
    with ProgressBar(total=Ntot, ncols=100, dynamic_ncols=False) as pgb:
        Eout, Hout = stratton_chu_ff(
            Ein.astype(np.complex128),
            Hin.astype(np.complex128),
            vis.astype(np.float64),
            wns.astype(np.float64),
            tpout.astype(np.float64),
            np.float64(source.k0),
            pgb,
        )
    return Field(E=Eout, H=Hout, theta=theta, phi=phi)