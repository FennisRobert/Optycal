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

    def angle(self, other: Point) -> float:
        """Calculates the angle between this point and another point.

        Args:
            other (Point): The other point to calculate the angle with.

        Returns:
            float: The angle in radians.
        """
        return np.angle(other.complex / self.complex)

    @property
    def magnitude(self) -> float:
        """ Calculates the magnitude of the point vector."""
        return np.sqrt(self.x**2 + self.y**2 + self.z**2)

    @property
    def mag(self) -> float:
        """Calculates the magnitude of the point vector.

        Returns:
            float: The magnitude of the vector.
        """
        return self.magnitude

    @property
    def hat(self) -> Point:
        """Calculates the unit vector (hat vector) in the direction of the point.

        Returns:
            Point: The unit vector in the direction of the point.
        """
        return self / self.magnitude

    def rotate2d(self, angle: float) -> Point:
        """Rotates the point around the origin in 2D space.

        Args:
            angle (float): The angle in radians to rotate the point.

        Returns:
            Point: The rotated point.
        """
        n2 = self.complex * np.exp(1j * angle)
        return Point(n2.real, n2.imag, 0)

    def mean(self, other: Point) -> Point:
        """Calculates the mean point between this point and another point.

        Args:
            other (Point): The other point to calculate the mean with.

        Returns:
            Point: The mean point.
        """
        check.mustbe(other, Point)
        return Point(
            0.5 * (self.x + other.x), 0.5 * (self.y + other.y), 0.5 * (self.z + other.z)
        )

    def distance(self, other: Point) -> float:
        """Calculates the distance between this point and another point.

        Args:
            other (Point): The other point to calculate the distance with.

        Returns:
            float: The distance between the two points.
        """
        check.mustbe(other, Point)
        return (self - other).magnitude


class Vector(Point):
    def __init__(self, x: float, y: float, z: float = 0, scale: int = 1_000_000):
        super().__init__(x, y, z, scale)
    
    @property
    def vector(self) -> Vector:
        """Returns the vector representation of the point.

        Returns:
            Vector: The vector representation of the point.
        """
        return self

    def dot(self, other: Vector) -> float:
        """Calculates the dot product of this vector with another vector.

        Args:
            other (Vector): The other vector to calculate the dot product with.

        Returns:
            float: The dot product of the two vectors.
        """
        check.mustbe(other, Vector)
        return self.x * other.x + self.y * other.y + self.z * other.z

    def cross(self, other: Vector):
        """Calculates the cross product of this vector with another vector.

        Args:
            other (Vector): The other vector to calculate the cross product with.

        Returns:
            Vector: The cross product of the two vectors.
        """
        check.mustbe(other, Vector)
        return Vector(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x,
            self.scale,
        )

    def normalize(self) -> Vector:
        """Normalizes the vector to have a magnitude of 1.

        Raises:
            ValueError: If the vector is zero-length.

        Returns:
            Vector: The normalized vector.
        """
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
        """Checks if the vector is normalized (magnitude == 1).

        Returns:
            bool: True if the vector is normalized, False otherwise.
        """
        return self.magnitude == 1


@dataclass
class Edge:
    p1: Point
    p2: Point
    """Represents an edge between two points in 2D space.
    """
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
        """Returns the vector representation of the edge.

        Returns:
            Vector: The vector representation of the edge.
        """
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
        """Returns the x-coordinates of the edge.

        Returns:
            tuple[float, float]: The x-coordinates of the edge.
        """
        return self.p1.x, self.p2.x

    def ys(self) -> tuple[float, float]:
        """Returns the y-coordinates of the edge.

        Returns:
            tuple[float, float]: The y-coordinates of the edge.
        """
        return self.p1.y, self.p2.y

    def zs(self) -> tuple[float, float]:
        """Returns the z-coordinates of the edge.

        Returns:
            tuple[float, float]: The z-coordinates of the edge.
        """
        return self.p1.z, self.p2.z

    @property
    def length(self) -> float:
        """Calculates the length of the edge.

        Returns:
            float: The length of the edge.
        """
        return self.p1.distance(self.p2)

    @property
    def np(self) -> np.ndarray:
        """Returns the vector representation of the edge as a NumPy array.

        Returns:
            np.ndarray: The vector representation of the edge.
        """
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
        """Creates a copy of the polygon.

        Returns:
            Polygon: A new polygon instance with the same vertices.
        """
        return Polygon(self.vertices)

    def circumcenter(self) -> Point:
        """Calculates the circumcenter of the polygon.

        Returns:
            Point: The circumcenter of the polygon.
        """
        return self.centroid

    @property
    def center(self) -> Point:
        """Calculates the center of the polygon.

        Returns:
            Point: The center of the polygon.
        """
        return self.centroid

    @property
    def centroid(self) -> Point:
        """Calculates the centroid of the polygon.

        Returns:
            Point: The centroid of the polygon.
        """
        return sum(self.vertices) / len(self.vertices)

    def drop(self, index: int) -> Polygon:
        """Drops a vertex from the polygon.

        Args:
            index (int): The index of the vertex to drop.

        Returns:
            Polygon: The modified polygon with the vertex dropped.
        """
        self.vertices.pop(index)
        self._process()
        return self

    def int_angle(self, index: int) -> float:
        """Calculates the internal angle at a vertex of the polygon.

        Args:
            index (int): The index of the vertex.

        Returns:
            float: The internal angle at the vertex.
        """
        v0 = self[index - 1]
        v1 = self[index]
        v2 = self[index + 1]
        return np.pi - (v1 - v0).angle(v2 - v1)

    def iter_from_point(self, point: Point):
        """Yields the vertices of the polygon starting from a given point.

        Args:
            point (Point): The point to start from.

        Yields:
            Point: The next vertex in the polygon.
        """
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
        """Returns the index and the closest point within a certain distance from the given point.

        Args:
            point (Point): The point to check against.
            distance (float): The maximum distance to consider.

        Returns:
            Tuple[int, Point]: The index and the closest point, or (None, None) if no point is found.
        """
        for v in self.iter_from_point(point):
            if (v - point).magnitude < distance:
                return self.vertices.index(v), v
        return None, None

    def insert(self, location: int, point: Point) -> Polygon:
        """Inserts a vertex into the polygon.

        Args:
            location (int): The index at which to insert the vertex.
            point (Point): The vertex to insert.

        Returns:
            Polygon: The modified polygon with the new vertex.
        """
        self.vertices.insert(location, point)
        self._process()
        return self

    def replace(self, location: int, point: Point) -> Polygon:
        """Replaces a vertex in the polygon.

        Args:
            location (int): The index of the vertex to replace.
            point (Point): The new vertex.

        Returns:
            Polygon: The modified polygon with the vertex replaced.
        """
        self.vertices[location] = point
        self._process()
        return self

    @property
    def signed_area(self) -> float:
        """Calculates the signed area of the polygon.

        Returns:
            float: The signed area of the polygon.
        """
        return np.sum(self.xs[0:-1] * self.ys[1:] - self.ys[:-1] * self.xs[1:]) / 2.0

    def refine_edges(self, dsmax: float) -> Polygon:
        """Refines the edges of the polygon by adding vertices.

        Args:
            dsmax (float): The maximum distance between vertices.

        Returns:
            Polygon: The modified polygon with refined edges.
        """
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
        """Transforms the polygon into a 2D representation.

        Raises:
            NotImplementedError: This method shouldn't be implemented here.

        Returns:
            Polygon: The 2D representation of the polygon.
        """
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
        """Returns the x-coordinates of the polygon's vertices.

        Returns:
            np.ndarray: The x-coordinates of the vertices.
        """
        return np.array([p.x for p in self.vertices])

    @property
    def ys(self) -> np.ndarray:
        """Returns the y-coordinates of the polygon's vertices.

        Returns:
            np.ndarray: The y-coordinates of the vertices.
        """
        return np.array([p.y for p in self.vertices])

    @property
    def zs(self) -> np.ndarray:
        """Returns the z-coordinates of the polygon's vertices.

        Returns:
            np.ndarray: The z-coordinates of the vertices.
        """
        return np.array([p.z for p in self.vertices])

    @property
    def xyzs(self) -> Tuple[np.ndarray]:
        """Returns the x, y, and z coordinates of the polygon's vertices.
        """
        return self.xs, self.ys, self.zs

    @staticmethod
    def fromxyz(xyz):
        """Creates a polygon from x, y, and z coordinates.
        """
        vertices = [Point(x,y,z) for x,y,z in zip(xyz[0,:], xyz[1,:], xyz[2,:])]
        return Polygon(vertices)

class Poly2D:
    def __init__(self, points: List[Point], startid: int = 0):
        if points[0] is not points[-1]:
            points = points + [points[0]]

        self.points = points
        self.ids = [i for i in range(startid, startid + len(points))]

    def refine(self, dsmax: float) -> Poly2D:
        """Refines the polygon by adding vertices.

        Args:
            dsmax (float): The maximum distance between vertices.  

        Returns:
            Poly2D: The modified polygon with added vertices.
        """
        points = []
        for a, b in zip(self.points[:-1], self.points[1:]):
            d = a.distance(b)
            N = int(np.ceil(d / dsmax))
            xs = np.linspace(a.x, b.x, N)
            ys = np.linspace(a.y, b.y, N)
            zs = np.linspace(a.z, b.z, N)
            points = points + Point.points(xs[:-1], ys[:-1], zs[:-1])
        return Poly2D(points)



    
    