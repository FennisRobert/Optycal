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
import pickle
from abc import ABC
from collections import defaultdict
from functools import reduce
from pathlib import Path
from typing import Callable, List
from numba import int32, float32, njit
from numba.types import Tuple as TypeTuple
import numpy as np
from scipy.spatial import ConvexHull, Delaunay

from ..cs import CoordinateSystem, CoordinateArray, VectorArray, XScalarArray, YScalarArray, ZScalarArray, CoordinateTuple, ToGlobalTransformer, GCS
from ...util.check import mustbe
from ...util.printing import summarize
from ..space import Point, Polygon
from loguru import logger

def pair_edge(x, y):
    return (min(x, y), max(x, y))

_SIG = "MESH"

MESHDIR = Path("Meshes")
MESHDIR.mkdir(parents=True, exist_ok=True)

def _fix_dimensions(arr: np.ndarray):
    if arr.shape[0] == 3:
        return arr
    if arr.shape[1] == 3:
        return arr.transpose()

def remove_unmeshed_vertices(vertices: np.ndarray, triangulation: list) -> tuple[np.ndarray, list]:
    unique_id = np.sort(np.unique(np.array(triangulation).flatten()))
    replace_id = np.arange(unique_id.shape[0])
    mapping = {idx: i for idx, i in zip(unique_id, replace_id)}
    tri_out = [(mapping[i1], mapping[i2], mapping[i3]) for i1, i2, i3 in triangulation]
    tri_out = np.array(tri_out)
    
    return vertices[:,unique_id], tri_out

@njit(cache=True, nogil=True)
def matmul(M: np.ndarray, vecs: np.ndarray):
    """Executes a basis transformation of vectors (3,N) with a basis matrix M

    Args:
        M (np.ndarray): A (3,3) basis matrix
        vec (np.ndarray): A (3,N) set of coordinates

    Returns:
        np.ndarray: The transformed (3,N) set of vectors
    """
    out = np.empty((3,vecs.shape[1]), dtype=vecs.dtype)
    out[0,:] = M[0,0]*vecs[0,:] + M[0,1]*vecs[1,:] + M[0,2]*vecs[2,:]
    out[1,:] = M[1,0]*vecs[0,:] + M[1,1]*vecs[1,:] + M[1,2]*vecs[2,:]
    out[2,:] = M[2,0]*vecs[0,:] + M[2,1]*vecs[1,:] + M[2,2]*vecs[2,:]
    return out
    
@njit(TypeTuple((float32[:, :], float32[:], float32[:,:], int32[:]))(float32[:,:], int32[:,:]), cache=True)
def _c_gen_normals_areas_centroids(vertices, tris):
    normals = np.zeros(tris.shape, dtype=np.float32)
    areas = np.zeros((tris.shape[1],), dtype=np.float32)
    centroids = np.zeros_like(normals, dtype=np.float32)
    bad_triangles = np.zeros_like(areas, dtype=np.int32)

    for i in range(tris.shape[1]):
        
        v1 = vertices[:, tris[0, i]]
        v2 = vertices[:, tris[1, i]]
        v3 = vertices[:, tris[2, i]]
        centroids[:, i] = (v1 + v2 + v3) / 3
        e1 = v2 - v1
        e2 = v3 - v1
        nx = e1[1] * e2[2] - e1[2] * e2[1]
        ny = e1[2] * e2[0] - e1[0] * e2[2]
        nz = e1[0] * e2[1] - e1[1] * e2[0]
        nn = np.sqrt(nx**2 + ny**2 + nz**2)
        areas[i] = nn / 2
        if nn==0:
            bad_triangles[i] = 1
        else:
            normals[0, i] = nx/nn
            normals[1, i] = ny/nn
            normals[2, i] = nz/nn
    return normals, areas, centroids, bad_triangles

def edge_iter(lst: list):
    # lst = sorted(lst)
    lst = list(lst)
    for i1, elem1 in enumerate(lst[:-1]):
        elem2 = lst[i1 + 1]
        yield sorted((elem1, elem2))
    yield sorted((lst[0], lst[-1]))


def pick_opposite(lst: list, item):
    i1, i2 = lst
    if i1 == item:
        return i2
    return i1

class Mesh:
    vertices: np.ndarray = CoordinateArray()
    centroids: np.ndarray = CoordinateArray()
    normals: np.ndarray = VectorArray()
    edge_centers: np.ndarray = CoordinateArray()
    xs: np.ndarray = XScalarArray()
    ys: np.ndarray = YScalarArray()
    zs: np.ndarray = ZScalarArray()
    xyz: np.ndarray = CoordinateTuple()
    edge_normals: np.ndarray = VectorArray()
    vertex_normals: np.ndarray = VectorArray()
    
    def __init__(
        self,
        vertices: np.ndarray,
        cs: CoordinateSystem = GCS,
        alignment_function: Callable = None,
    ):
        mustbe(vertices, np.ndarray)
        
        self.alignment_function = alignment_function
        self.cs = cs
        self.vertices: np.ndarray = _fix_dimensions(vertices).astype(np.float32)
        self.edges = []
        self.triangles: np.ndarray = []

        self.boundary_vertices = []

        self.v2e: defaultdict = None
        self.v2t: defaultdict = None
        self.e2v: defaultdict = None
        self.e2t: defaultdict = None
        self.t2v: np.ndarray = None
        self.t2e: np.ndarray = None
        self.I1 = None
        self.I2 = None
        self.I3 = None
        
        self.centroids: np.ndarray = None
        self.normals: np.ndarray = None
        self.areas: np.ndarray = None
        self.edge_centers: np.ndarrays = None
        self.edge_normals: np.ndarray = None
        
        self.xs = self.vertices[0,:]
        self.ys = self.vertices[1,:]
        self.zs = self.vertices[2,:]
        self.xyz = (self.vertices[0,:], self.vertices[1,:], self.vertices[2,:])
        

    def _get_picklable_attributes(self):
        picklable_attributes = {}
        for name, attr in self.__dict__.items():
            try:
                # Attempt to pickle the attribute
                pickle.dumps(attr)
                picklable_attributes[name] = attr
            except AttributeError:
                logger.debug(f"Pickling error on attribute {name}")
                pass
            except TypeError:
                logger.debug(f"Pickling error on attribute {name}")
                pass
        return picklable_attributes
    
    def _save(self, filename):
        filepath = MESHDIR / (filename + ".pkl")
        logger.debug(f"Saving file as: {filepath}")
        atributes = self._get_picklable_attributes()
        with open(str(filepath), "wb") as f:
            pickle.dump(atributes, f)
        logger.debug("Saving successful!")

    @staticmethod
    def _load(filename, cs: CoordinateSystem):
        filepath = MESHDIR / (filename + ".pkl")
        logger.debug(f"Attempting to load: {filepath}")
        mesh = None
        with open(str(filepath), "rb") as f:
            attributes = pickle.load(f)
            mesh = Mesh(np.zeros((3, 1)), cs, None)
            for field, value in attributes.items():
                setattr(mesh, field, value)
            mesh.cs = cs
        logger.debug("Loading file successful!")
        return mesh

    def _report(self):
        logger.debug("Mesh:")
        logger.debug("  nvertices: ", self.nvertices)
        logger.debug("  nedges:    ", self.nedges)
        logger.debug("  ntriangles:", self.ntriangles)

        logger.debug("vertex 2 edge:", len(self.v2e), list(self.v2e.items())[0:5], "...")
        logger.debug("vertex 2 tri:", len(self.v2t), list(self.v2t.items())[0:5], "...")
        logger.debug("edge 2 vertex:", len(self.e2v), list(self.e2v.items())[0:5], "...")
        logger.debug("edge 2 tri:", len(self.e2t), list(self.e2t.items())[0:5], "...")
        logger.debug("tri 2 vertex:", len(self.t2v), list(self.t2v.items())[0:5], "...")
        logger.debug("tri 2 edge:", len(self.t2e), list(self.t2e.items())[0:5], "...")

    @property
    def nedges(self) -> int:
        return len(self.edges)

    @property
    def nvertices(self) -> int:
        return self.vertices.shape[1]

    @property
    def ntriangles(self) -> int:
        return self.triangles.shape[1]
    
    @property
    def g(self) -> Mesh:
        return ToGlobalTransformer(self)
    
    def merge(self, other):
        logger.debug(f'Merging {self} with {other}')
        N = self.vertices.shape[1]
        self.vertices = np.hstack((self.vertices, other.vertices))
        self.triangles = np.hstack((self.triangles, other.triangles+N))
    
    def constrain_centroids(self, constrain_f: Callable) -> Mesh:
        ids = np.squeeze(np.argwhere(constrain_f(self.centroids[0,:], self.centroids[1,:], self.centroids[2,:])==True))
        triangles = self.triangles[:,ids]
        vids = np.sort(np.unique(np.array(self.triangles)))
        new_mesh = Mesh(self.vertices[:,vids], self.cs)
        new_mesh.set_triangles(triangles)
        new_mesh._update_mesh()
        return new_mesh

    def displace(self, dx, dy, dz) -> Mesh:
        newMesh = Mesh(np.zeros((3,1)), None, None)
        for key, value in self.__dict__.items():
            newMesh.__dict__[key] = value
        newMesh.vertices[0,:] = newMesh.vertices[0,:] + dx
        newMesh.vertices[1,:] = newMesh.vertices[1,:] + dy
        newMesh.vertices[2,:] = newMesh.vertices[2,:] + dz
        return newMesh

    def align_from_origin(self, x0, y0, z0):
        self.alignment_function = lambda x, y, z: np.array([x - x0, y - y0, z - z0]).T

    def iter_triangle_ids(self):
        for Is in zip(self.triangles[0,:], self.triangles[1,:], self.triangles[2,:]):
            yield Is

    def add_vertex(self, vertex: Point) -> None:
        mustbe(vertex, Point)
        logger.debug(f'Adding vertex {vertex} to {self}')
        self.vertices = np.concatenate(self.vertices, vertex.numpy)

    def get_neighbors(self, index):
        ids = set()
        for t in self.v2t[index]:
            ids.update(self.triangles[:, t])
        ids = list(ids)
        ids.remove(index)
        return ids

    def add_polygon_id(self, *ids) -> None:
        logger.debug(f'Adding polygon {ids} to {self}')
        self.triangles = np.concatenate(self.triangles, np.array(ids))

    def set_boundary(self, ids: List[int]) -> None:
        self.boundary_vertices = ids
        
    def project_to_sphere(self, radius: float, center: np.ndarray):
        logger.debug("Projecting to sphere.")
        center = np.array(center, dtype=np.float32)
        vn = self.vertices.T - center
        self.vertices = (center + ((vn.T/np.linalg.norm(vn, axis=1) * radius)).T).astype(np.float32).T

    def set_triangles(self, triangles: np.ndarray,
                      auto_update: bool = True) -> Mesh:

        logger.debug("Setting triangles.")
        if isinstance(triangles, np.ndarray):
            if triangles.shape[1] == 3:
                triangles = triangles.transpose()
            self.triangles = triangles.astype(np.int32)
        else:
            mustbe(triangles, list)
            mustbe(triangles[0], tuple)
            mustbe(triangles[0][0], int)
            N = self.nvertices
            logger.debug("Checking triangle validity.")
            [I1, I2, I3] = tuple(zip(*triangles))
            self.triangles = _fix_dimensions(np.array([I1, I2, I3], dtype=np.int32))
        logger.debug("Triangles set")
        if auto_update:
            self.update()
        return self


    def __getitem__(self, index: int) -> np.ndarray:
        return self.vertices[index, :]

    def _gen_edges(self):
        logger.debug('Generating mesh edge data.')
        edges = set()
        for it in range(self.ntriangles):
            i1, i2, i3 = self.triangles[:,it]
            edges.add(pair_edge(i1, i2))
            edges.add(pair_edge(i2, i3))
            edges.add(pair_edge(i1, i3))
        self.edges = list(edges)
    
    
    def _gen_normals(self):
        logger.debug("Generating mesh normals")

        tris = [I for I in self.iter_triangle_ids()]
        logger.debug('Calling normals C-routine.')
        logger.debug(f'Vertices {summarize(self.vertices)}')
        self.normals, self.areas, self.centroids, bad_tris = _c_gen_normals_areas_centroids(self.vertices, self.triangles)
        logger.debug(f'Area data: {summarize(self.areas)}')
        logger.debug(f'Normals data: {summarize(self.normals)}')
        logger.debug(f'Centroids data: {summarize(self.centroids)}')
        
        logger.debug('Aligning mesh normals.')
        for ii, (i1, i2, i3) in enumerate(tris):
            if self.alignment_function is not None:
                # c = self.centroids[ii, :]
                v = self.alignment_function(*self.centroids[:,ii])
                if np.dot(v, self.normals[:, ii]) < 0:
                    self.normals[:, ii] = -self.normals[:,ii]
                    self.triangles[:, ii] = np.array([i1, i3, i2], dtype=np.int32)
        self.areas = np.array(self.areas)
        self.normals = _fix_dimensions(self.normals)
        self.centroids = _fix_dimensions(self.centroids)
        logger.debug("Normals aligned")

    def update(self):
        self._update_mesh()
        
    def _update_mesh(self):
        self._fill_complete()
    
    def _fill_complete(self):
        logger.debug("Updating mesh properties")
        self._remove_unmeshed_vertices()
        self.v2e = defaultdict(set)
        self.v2t = defaultdict(set)
        self.e2v = defaultdict(set)
        self.e2t = defaultdict(set)

        self.t2v = np.zeros((3, self.ntriangles), dtype=np.int32)
        self.t2e = np.zeros((3, self.ntriangles), dtype=np.int32)
        self._gen_normals()
        self._gen_edges()

        tris = [I for I in self.iter_triangle_ids()]
        edgedict = {edge: i for i, edge in enumerate(self.edges)}
        logger.debug("Defining edge, vertex, triangle mappings.")
        for it, (iv1, iv2, iv3) in enumerate(tris):
            self.t2v[:, it] = [iv1, iv2, iv3]
            self.v2t[iv1].add(it)
            self.v2t[iv2].add(it)
            self.v2t[iv3].add(it)

            IE1 = edgedict[pair_edge(iv1, iv2)]
            IE2 = edgedict[pair_edge(iv2, iv3)]
            IE3 = edgedict[pair_edge(iv1, iv3)]
            
            self.v2e[iv1].update({IE1, IE3})
            self.v2e[iv2].update({IE1, IE2})
            self.v2e[iv3].update({IE2, IE3})

            self.t2e[0, it] = IE1
            self.t2e[1, it] = IE2
            self.t2e[2, it] = IE3
            self.e2t[IE1].add(it)
            self.e2t[IE2].add(it)
            self.e2t[IE3].add(it)

        for key in self.v2t.keys():
            self.v2t[key] = list(self.v2t[key])
        
        self.v2e = {key: list(value) for key, value in self.v2e.items()}
        #for ie, (i1, i2) in enumerate(self.edges):
        self.e2v = {i: e for e, i in edgedict.items()}
        

        logger.debug("Updating centroids and edge centers")
        
        # Computing Edge Centers
        logger.debug('Computing edge centers.')
        ie1, ie2 = list(zip(*self.edges))
        ve1 = self.vertices[:, ie1]
        ve2 = self.vertices[:, ie2]
        self.edge_centers = _fix_dimensions((ve1 + ve2) / 2)

        # Filling Coordinate Arrays
        logger.debug("Filling coordinate arrays.")
        self.xs = self.vertices[0,:]
        self.ys = self.vertices[1,:]
        self.zs = self.vertices[2,:]
        self.xyz = (self.vertices[0,:], self.vertices[1,:], self.vertices[2,:])
        
        # Edge Normals
        logger.debug("Computing edge normals.")
        normals = np.zeros((3, self.nedges))
        counter = np.zeros((self.nedges,))
        for it in range(self.ntriangles):
            i1, i2, i3 = self.t2e[:, it]
            counter[i1] += 1
            counter[i2] += 1
            counter[i3] += 1
            normals[:, i1] += self.normals[:, it]
            normals[:, i2] += self.normals[:, it]
            normals[:, i3] += self.normals[:, it]
        normals = normals / counter
        self.edge_normals = matmul(self.cs.global_basis, normals)
        
        # Vertex Normals
        logger.debug('Computing vertex normals')
        normals = np.zeros((3, self.nvertices))
        for iv in range(self.nvertices):
            ts = np.array(list(self.v2t[iv])).astype(np.int32)
            if ts.shape[0] == 0:
                logger.warning(f'Empty vertex to triangle binding for vertex {iv} -> {ts}' )
                continue
            normals[:, iv] = np.mean(self.normals[:, ts], axis=1)
        self.vertex_normals = matmul(self.cs.global_basis, normals)
        logger.debug(f'Vertex normal data: {summarize(self.vertex_normals)}')
        logger.debug("Update complete")

    def tri_convexhull(self):
        logger.debug("Calling ConvexHull.")
        tri = ConvexHull(self.vertices.T)
        logger.debug("ConvexHull finished.")
        self.set_triangles(tri.simplices)
        logger.debug("Triangles set.")
        self._fill_complete()

    def tri_delaunay(self):
        logger.debug("Calling Delaunay.")
        tri = Delaunay(self.vertices.T)
        logger.debug("Delaunay finished.")
        self.set_triangles(tri.simplices)
        logger.debug("Triangles set.")
        self._fill_complete()

    def _remove_unmeshed_vertices(self) -> None:
        meshed_vertices = list(set(list(self.triangles.flatten())))
        Nmeshed = len(meshed_vertices)
        Noriginal = self.nvertices
        logger.debug('Reducing mesh.')
        logger.debug(f'Total vertices = {Noriginal}, total meshed = {Nmeshed}')
        if Nmeshed < Noriginal:
            logger.debug(f'Removing {Noriginal-Nmeshed} vertices')
            vertex_mapping = {ind: i for i, ind in enumerate(meshed_vertices)}
            for it in range(self.ntriangles):
                self.triangles[0,it] = vertex_mapping[self.triangles[0,it]]
                self.triangles[1,it] = vertex_mapping[self.triangles[1,it]]
                self.triangles[2,it] = vertex_mapping[self.triangles[2,it]]
            self.vertices = self.vertices[:,meshed_vertices]
            logger.debug('Reduction successful')
    
        

class Meshable(ABC):
    def generate_mesh(self, ds: float) -> Mesh:
        pass