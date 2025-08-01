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
from .mesh import Mesh
from ..cs import CoordinateSystem, GCS
from ..space import Polygon, Point
from .triangulate import advancing_front_triangulation
from scipy.spatial import Delaunay
import numpy as np
from loguru import logger

def generate_rectangle(
    width: float,
    height: float,
    ax1: np.ndarray,
    ax2: np.ndarray,
    ds: float,
    cs: CoordinateSystem = GCS,
    origin: np.ndarray = np.array([0, 0, 0])
    ) -> Mesh:
    """Generates a rectangular plate mesh

    Args:
        width (float): The width of the plate
        height (float): The height of th eplate
        ax1 (np.ndarray): The width 3D axis
        ax2 (np.ndarray): The height 3D axis
        ds (float): The discretization size
        cs (CoordinateSystem): The coordinate system to define the plate in
        origin (np.ndarray, optional): The origin of the center of the plate. Defaults to np.array([0, 0, 0]).

    Returns:
        Mesh: The mesh object
    """
    x1 = np.linspace(-width / 2, width / 2, int(max(2, np.ceil(width / ds))))
    x2 = np.linspace(-height / 2, height / 2, int(max(2, np.ceil(height / ds))))
    [X, Y] = np.meshgrid(x1, x2)
    x = X.flatten()
    y = Y.flatten()
    X = x * ax1[0] + y * ax2[0]
    Y = x * ax1[1] + y * ax2[1]
    Z = x * ax1[2] + y * ax2[2]

    tri = Delaunay(np.array([x, y]).T)
    triangles = tri.simplices
    X = X + origin[0]
    Y = Y + origin[1]
    Z = Z + origin[2]
    
    mesh = Mesh(np.array([X, Y, Z]), cs)
    mesh.set_triangles(triangles)
    mesh._fill_complete()
    return mesh

def generate_sphere(
    center: np.ndarray,
    radius: float,
    ds: float,
    cs: CoordinateSystem = GCS,
) -> Mesh:
    """Generates a spherical surface mesh

    Args:
        center (np.ndarray): The center of the sphere
        radius (float): The radius of the sphere
        ds (float): The discretization size
        cs (CoordinateSystem, optional): The coordinate system in which to place the sphere. Defaults to GCS.

    Returns:
        Mesh: The sphere mesh
    """
    def f(x):
        return int(4 * np.round((x + 1e-8) / 4))
    
    center = np.array(center)

    Ntheta = f(np.ceil(np.pi * radius / ds))
    Atri = 0.5 * ds**2
    Asphere = 4 * np.pi * radius**2
    Ntri = int(Asphere / Atri)
    Nvert = 1.5 * Ntri / 2
    vertices = []
    ths = np.linspace(-np.pi / 2, np.pi / 2, Ntheta)
    vbot = center + np.array([0, 0, -radius])
    vtop = center + np.array([0, 0, radius])
    vertices.append(vbot)
    vertices.append(vtop)
    logger.debug("Generating sphere vertices.")
    for i, th in enumerate(ths):
        circ = 2 * np.pi * radius * np.cos(th)
        Nphi = f(np.ceil(circ / ds))
        phis = np.linspace(0, 2 * np.pi, Nphi, endpoint=False)
        xs = radius * np.cos(phis) * np.cos(th)
        ys = radius * np.sin(phis) * np.cos(th)
        zs = radius * np.sin(th) * np.ones_like(phis)
        vertices.extend(
            [center + np.array([x, y, z]) for x, y, z in zip(xs, ys, zs)]
        )
    N = len(vertices)
    vertices = np.array(vertices)
    logger.debug("Vertices generated")
    mesh = Mesh(vertices, cs)
    mesh.align_from_origin(*center)
    mesh.tri_convexhull()
    mesh.update()
    logger.debug("Mesh generated")
    return mesh

def generate_circle(
    center: np.ndarray,
    radius: float,
    ds: float,
    cs: CoordinateSystem,
    ax1: np.ndarray = np.array([1, 0, 0]),
    ax2: np.ndarray = np.array([0, 1, 0]),
) -> Mesh:
    """Generates a circular disc

    Args:
        center (np.ndarray): The center of the disc
        radius (float): The radius of the disc
        ds (float): The discretization size of the disc
        cs (CoordinateSystem): The coordinate system of the disc
        ax1 (np.ndarray, optional): The first axis defining the Plane. Defaults to np.array([1, 0, 0]).
        ax2 (np.ndarray, optional): The second axis defining the Plane. Defaults to np.array([0, 1, 0]).

    Returns:
        Mesh: The circular disc mesh
    """
    radii = np.linspace(0, radius, int(np.ceil(radius / ds)))
    vertices = [np.array([0,0,0])]
    mesh_vertices = [np.array([0,0])]
    for r in radii:
        N = int(np.ceil(2 * np.pi * r / ds))
        phis = np.linspace(0, 2 * np.pi, N, endpoint=False)
        xs = r * np.cos(phis)
        ys = r * np.sin(phis)
        zs = np.zeros_like(xs)
        vertices.extend(
            [center + x * ax1 + y * ax2 for x, y in zip(xs, ys)]
        )
        mesh_vertices.extend([x*np.array([1,0]) + y*np.array([0,1]) for x, y in zip(xs, ys)])
    vertices = np.array(vertices).T
    xys = vertices[:2,:]
    mesh = Mesh(np.array(vertices), cs)
    mesh.set_triangles(Delaunay(np.array(mesh_vertices)).simplices)
    mesh.alignment_function = lambda x, y, z: np.cross(ax1, ax2)
    mesh.update()
    return mesh

def from_polygon(poly: Polygon, ds:float, cs: CoordinateSystem, origin: np.ndarray = None) -> Mesh:
    """Generes a mesh from a polygon using an advancing front triangulation algorithm

    Args:
        poly (Polygon): The Polygon object
        ds (float): The discretiation size
        cs (CoordinateSystem): The coordinate system 
        origin (np.ndarray, optional): The origin of the polygon. Defaults to None.

    Returns:
        Mesh: The resultant mesh
    """
    poly2, cs2 = poly.local_2d()
    vertices, tris, boundary = advancing_front_triangulation(poly2, ds, 0.6, 5)
    zax = cs2.global_basis @ np.array([0,0,1]) + cs2.global_origin
    x2, y2, z2 = cs2.in_global_cs(vertices[0, :], vertices[1, :], 0*vertices[2, :])
    zx, zy, zz = cs2.global_basis_inv @ zax
    x2, y2, z2 = cs.from_global_cs(x2, y2, z2)
    zx, zy, zz = cs.from_global_cs(zx, zy, zz)
    
    mesh = Mesh(np.array((x2, y2, z2)), cs)
    
    if origin is None:
        mesh.alignment_function = lambda x, y, z: np.array([zx, zy, zz]).T
    else:
        mesh.align_from_origin(origin[0], origin[1], origin[2])
    mesh.set_triangles(tris)
    mesh.update()
    mesh.boundary_vertices = boundary
    return mesh

def from_2d_mesh(
    points: np.ndarray,
    triangles: np.ndarray,
    xhat: np.ndarray,
    yhat: np.ndarray,
    cs: CoordinateSystem,
):
    N = len(points)
    xyz = xhat * points[0,:] + yhat * points[1, :]
    vertices = [Point(x, y, z) for x, y, z in zip(xyz[0,:], xyz[1,:], xyz[2,:])]
    mesh = Mesh(vertices, cs)
    mesh.triangles = triangles
    mesh._fill_complete()
    return mesh
