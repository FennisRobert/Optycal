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
from dataclasses import dataclass
from typing import List, Tuple
import numpy as np
from ..settings import GLOBAL_SETTINGS
from ..util import check, iterators

@dataclass
class Point:
    x: float
    y: float
    z: float = 0
    scale: int = GLOBAL_SETTINGS.precision
    data: any = None

    def __post_init__(self):
        self.ix = int(self.x * self.scale)
        self.iy = int(self.y * self.scale)
        self.iz = int(self.z * self.scale)
        self.ituple = (self.ix, self.iy, self.iz)

    def __str__(self) -> str:
        return f"Point({self.x:.2f},{self.y:.2f},{self.z:.2f})"

    def __repr__(self) -> str:
        return str(self)

    @property
    def vector(self):
        return Vector(self.x, self.y, self.z)
    
    @property
    def numpy(self) -> np.array:
        return np.array([self.x, self.y, self.z])

    @property
    def X(self) -> float:
        return float(self.ix / self.scale)

    @property
    def Y(self) -> float:
        return float(self.iy / self.scale)

    @property
    def Z(self) -> float:
        return float(self.iz / self.scale)

    @property
    def complex(self) -> complex:
        return complex(self.x, self.y)

    def __add__(self, other) -> Point:
        if not isinstance(other, Point):
            raise TypeError("Can only add a point with a point object")
        return Point(
            self.x + other.x,
            self.y + other.y,
            self.z + other.z,
            min(self.scale, other.scale),
        )

    def __hash__(self) -> int:
        return hash((self.ix, self.iy, self.iz))

    def __radd__(self, other) -> Point:
        return self + other

    def __sub__(self, other) -> Point:
        if not isinstance(other, Point):
            raise TypeError("Can only subtract a point from a point object")
        return Vector(
            self.x - other.x,
            self.y - other.y,
            self.z - other.z,
            min(self.scale, other.scale),
        )

    def __rsub__(self, other) -> Point:
        return self - other

    def __matmul__(self, other) -> float:
        if not isinstance(other, Point):
            raise TypeError("Can only multiply a point with a point object")
        return self.x * other.x + self.y * other.y + self.z * other.z
    
    def __mul__(self, other) -> Point:
        if isinstance(other, Point):
            return Point(self.x*other.x, self.y*other.y, self.z*other.z, self.scale)
        elif isinstance(other, (float, int, complex)):
            return Point(self.x * other, self.y * other, self.z * other, self.scale)
        else:
            raise TypeError("Can only multiply a point with a point object or float")

    def __truediv__(self, other):
        if isinstance(other, Point):
            return Point(self.x / other.x, self.y / other.y, self.z / other.z, self.scale)
        elif isinstance(other, (float, int, complex)):
            return Point(self.x / other, self.y / other, self.z / other, self.scale)
        else:
            raise TypeError("Can only divide a point with a point object or a float")

    def __rmul__(self, other) -> Point:
        return self * other

    def __eq__(self, other) -> bool:
        check.mustbe(other, Point)
        return self.ituple == other.ituple

    def __req__(self, other) -> bool:
        return self.ituple == other.ituple

    def angle(self, other) -> float:
        return np.angle(other.complex / self.complex)

    @property
    def magnitude(self) -> float:
        return np.sqrt(self.x**2 + self.y**2 + self.z**2)

    @property
    def mag(self) -> float:
        return self.magnitude

    @property
    def hat(self) -> Point:
        return self / self.magnitude

    def rotate2d(self, angle: float) -> Point:
        n2 = self.complex * np.exp(1j * angle)
        return Point(n2.real, n2.imag, 0)

    def mean(self, other: Point) -> Point:
        check.mustbe(other, Point)
        return Point(
            0.5 * (self.x + other.x), 0.5 * (self.y + other.y), 0.5 * (self.z + other.z)
        )

    def distance(self, other: Point) -> float:
        check.mustbe(other, Point)
        return (self - other).magnitude


class Vector(Point):
    def __init__(self, x: float, y: float, z: float = 0, scale: int = 1_000_000):
        super().__init__(x, y, z, scale)
    
    @property
    def vector(self) -> Vector:
        return self

    def dot(self, other: Vector) -> float:
        check.mustbe(other, Vector)
        return self.x * other.x + self.y * other.y + self.z * other.z

    def cross(self, other: Vector):
        check.mustbe(other, Vector)
        return Vector(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x,
            self.scale,
        )

    def normalize(self) -> Vector:
        M = self.magnitude
        if M == 0:
            raise ValueError(f"Cannot normalize a zero vector {self}")
        return self / self.magnitude

    def __repr__(self) -> str:
        return f"Vector({self.x}, {self.y}, {self.z})"

    def __str__(self) -> str:
        return f"Vector({self.x}, {self.y}, {self.z})"
    
    def __mul__(self, other) -> float:
        if isinstance(other, Point):
            return Vector(self.x*other.x, self.y*other.y, self.z*other.z, self.scale)
        elif isinstance(other, (float, int, complex)):
            return Vector(self.x * other, self.y * other, self.z * other, self.scale)
        else:
            raise TypeError("Can only multiply a point with a point object or float")

    def __matmul__(self, other: Vector | Point | float) -> Vector:
        if not isinstance(other, (Point, Vector, float, int, complex)):
            raise TypeError("Can only multiply a vector with a vector or Point object")
        if isinstance(other, Point):
            return self.x * other.x + self.y * other.y + self.z * other.z
        elif isinstance(other, (float, int, complex)):
            return Vector(self.x * other, self.y * other, self.z * other, self.scale)
        else:
            raise TypeError("Can only multiply a vector with a vector or Point object")
    
    def __div__(self, other: Vector | Point | float) -> Vector:
        if isinstance(other, Point):
            return Vector(self.x / other.x, self.y / other.y, self.z / other.z, self.scale)
        elif isinstance(other, (float, int, complex)):
            return Vector(self.x / other, self.y / other, self.z / other, self.scale)
        else:
            raise TypeError("Can only divide a point with a point object or a float")
        
    def __truediv__(self, other: Vector | Point | float) -> Vector:
        return self.__div__(other)

    @property
    def is_normalized(self):
        return self.magnitude == 1


@dataclass
class Edge:
    p1: Point
    p2: Point

    def __post_init__(self):
        if self.p1 == self.p2:
            raise ValueError("An edge cannot have the same points")
        # Implement dictionary ordering
        if self.p1.ituple > self.p2.ituple:
            self.p1, self.p2 = self.p2, self.p1
    
    def __hash__(self):
        return hash((self.p1.ituple, self.p2.ituple))

    @property
    def vector(self) -> Vector:
        return self.p2-self.p1
    
    @property
    def center(self) -> Point:
        return Point(
            (self.p1.x + self.p2.x) / 2,
            (self.p1.y + self.p2.y) / 2,
            (self.p1.z + self.p2.z) / 2,
        )

    @property
    def xs(self) -> tuple[float, float]:
        return self.p1.x, self.p2.x

    def ys(self) -> tuple[float, float]:
        return self.p1.y, self.p2.y

    def zs(self) -> tuple[float, float]:
        return self.p1.z, self.p2.z

    @property
    def length(self) -> float:
        return self.p1.distance(self.p2)

    @property
    def np(self) -> np.ndarray:
        return np.array([self.p2.x-self.p1.x,self.p2.y-self.p1.y,self.p2.z-self.p1.z])


class Polygon:
    def __init__(self, vertices: List[Point]):
        self.vertices = vertices
        if vertices[0] != vertices[-1]:
            self.vertices.append(vertices[0])

        self.edges = None
        self.len = None
        self._process()

    def __str__(self) -> str:
        return f"Polygon({self.vertices})"
    
    def _process(self) -> None:
        self.edges = [Edge(*point_pairs) for point_pairs in iterators.loop_iter(self.vertices)]

    def copy(self) -> Polygon:
        return Polygon(self.vertices)

    def circumcenter(self) -> Point:
        return self.centroid

    @property
    def center(self) -> Point:
        return self.centroid

    @property
    def centroid(self) -> Point:
        return sum(self.vertices) / len(self.vertices)

    def drop(self, index: int) -> Polygon:
        self.vertices.pop(index)
        self._process()
        return self

    def int_angle(self, index: int) -> float:
        v0 = self[index - 1]
        v1 = self[index]
        v2 = self[index + 1]
        return np.pi - (v1 - v0).angle(v2 - v1)

    def iter_from_point(self, point: Point):
        istart = self.vertices.index(point)
        yielded = []
        for i in range(istart, len(self.vertices)):
            v1 = self[istart + i]
            if v1 not in yielded:
                yield v1
                yielded.append(v1)
            if len(yielded) == len(self.vertices):
                break
            v2 = self[istart - i]
            if v2 not in yielded:
                yield v2
                yielded.append(v2)
            if len(yielded) == len(self.vertices):
                break

    def return_closest_point(self, point: Point, distance: float) -> Tuple[int, Point]:
        for v in self.iter_from_point(point):
            if (v - point).magnitude < distance:
                return self.vertices.index(v), v
        return None, None

    def insert(self, location: int, point: Point) -> Polygon:
        self.vertices.insert(location, point)
        self._process()
        return self

    def replace(self, location: int, point: Point) -> Polygon:
        self.vertices[location] = point
        self._process()
        return self

    @property
    def signed_area(self) -> float:
        return np.sum(self.xs[0:-1] * self.ys[1:] - self.ys[:-1] * self.xs[1:]) / 2.0

    def refine_edges(self, dsmax: float) -> Polygon:
        vertices = []
        for v1, v2 in iterators.loop_iter(self.vertices):
            L = (v1 - v2).magnitude
            Nchop = max(2,int(np.ceil(L / dsmax)))
            xs = np.linspace(v1.x, v2.x, Nchop)
            ys = np.linspace(v1.y, v2.y, Nchop)
            for x, y in zip(xs[:-1], ys[:-1]):
                vertices.append(Point(x, y))

        return Polygon(vertices)
    
    def local_2d(self) -> Polygon:
        raise NotImplementedError("This method shouldn't be implemented here.")
        e1 = self.edges[0]
        e2 = self.edges[1]
        for e in self.edges[1:]:
            e2 = e
            if e1.vector.cross(e2.vector).magnitude > 1e-10:
                break
        be1 = e1.vector.normalize()
        ce = e1.vector.cross(e2.vector).normalize()
        be2 = be1.vector.cross(ce).normalize()
        p0 = self.vertices[0]
        #B = np.array([be1, be2, ce]).T
        #origin
        cs = None#COORDINATE_SYSTEM(p0.numpy, be1.numpy, be2.numpy, ce.numpy)
        x, y, z = self.xyzs
        x2, y2, z2 = cs.from_global_cs(x, y, z)
        return Polygon([Point(x,y,z) for x,y,z in zip(x2, y2, z2)]), cs
    
    def __getitem__(self, index: int) -> Point:
        # print('Referencing:', index, index % len(self.vertices), self.len, len(self.vertices))
        return self.vertices[index % len(self.vertices)]

    @property
    def xs(self) -> np.ndarray:
        return np.array([p.x for p in self.vertices])

    @property
    def ys(self) -> np.ndarray:
        return np.array([p.y for p in self.vertices])

    @property
    def zs(self) -> np.ndarray:
        return np.array([p.z for p in self.vertices])

    @property
    def xyzs(self) -> Tuple[np.ndarray]:
        return self.xs, self.ys, self.zs

    @staticmethod
    def fromxyz(xyz):
        vertices = [Point(x,y,z) for x,y,z in zip(xyz[0,:], xyz[1,:], xyz[2,:])]
        return Polygon(vertices)

class Poly2D:
    def __init__(self, points: List[Point], startid: int = 0):
        if points[0] is not points[-1]:
            points = points + [points[0]]

        self.points = points
        self.ids = [i for i in range(startid, startid + len(points))]

    def refine(self, dsmax):
        points = []
        for a, b in zip(self.points[:-1], self.points[1:]):
            d = a.distance(b)
            N = int(np.ceil(d / dsmax))
            xs = np.linspace(a.x, b.x, N)
            ys = np.linspace(a.y, b.y, N)
            zs = np.linspace(a.z, b.z, N)
            points = points + Point.points(xs[:-1], ys[:-1], zs[:-1])
        return Poly2D(points)



    
    