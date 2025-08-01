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
from numba import njit, f8

@njit(cache=True, fastmath=True)
def _c_cross_comp(ax, ay, az, bx, by, bz):
    cx = ay * bz - az * by
    cy = az * bx - ax * bz
    cz = ax * by - ay * bx
    return cx, cy, cz

@njit(cache=True, fastmath=True)
def _c_dot_comp(ax, ay, az, bx, by, bz):
    return ax * bx + ay * by + az * bz

@njit(cache=True, fastmath=True)
def _c_dot_vec(a, b):
    return a[:, 0] * b[:, 0] + a[:, 1] * b[:, 1] + a[:, 2] * b[:, 2]

@njit(cache=True, fastmash=True)
def _c_cross_vec(a, b):
    c = np.zeros_like(a)
    c[:, 0] = a[:, 1] * b[:, 2] - a[:, 2] * b[:, 1]
    c[:, 1] = a[:, 2] * b[:, 0] - a[:, 0] * b[:, 2]
    c[:, 2] = a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]
    return c

@njit(f8[:, ::1](f8[:, ::1], f8[:, ::1]), cache=True, parallel=True, fastmath=True)
def orthogonalize(v1: np.ndarray, v2: np.ndarray):
    if (v1.shape[0] != v2.shape[0]) or (v1.shape[1] != v2.shape[1]):
        raise ValueError
    if v1.shape[0] != 3:
        raise ValueError

    v1x = v1[0, :]
    v1y = v1[1, :]
    v1z = v1[2, :]
    v2x = v2[0, :]
    v2y = v2[1, :]
    v2z = v2[2, :]
    v1n = np.sqrt(v1x**2 + v1y**2 + v1z**2)
    v2n = np.sqrt(v2x**2 + v2y**2 + v2z**2)

    v1x = v1x / v1n
    v1y = v1y / v1n
    v1z = v1z / v1n
    v2x = v2x / v2n
    v2y = v2y / v2n
    v2z = v2z / v2n

    v3 = np.zeros_like(v1).astype(np.float64)

    dotp = 1 - (v1x * v2x + v1y * v2y + v1z * v2z)
    v2x[dotp < 1e-10] = 1
    dotp = 1 - (v1x * v2x + v1y * v2y + v1z * v2z)
    v2y[dotp < 1e-10] = 1

    v3x = v1y * v2z - v1z * v2y
    v3y = v1z * v2x - v1x * v2z
    v3z = v1x * v2y - v1y * v2x

    v3n = np.sqrt(v3x**2 + v3y**2 + v3z**2)
    v3[0, :] = v3x / v3n
    v3[1, :] = v3y / v3n
    v3[2, :] = v3z / v3n

    return v3

