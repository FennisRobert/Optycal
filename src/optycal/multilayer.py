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

from typing import List, Tuple
from .material import Material
import numpy as np
from numba import c16, njit
from .util import check
from abc import ABC

@njit(c16[:,:](c16[:,:], c16[:,:]), cache=True, nogil=True)
def mat_mul(A, B):
    #Multiply a 2x2 matrix A with 2xN matrix B
    out = np.empty((2,2), dtype=B.dtype)
    out[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0]
    out[0,1] = A[0,0]*B[0,1] + A[0,1]*B[1,1]
    out[1,0] = A[1,0]*B[0,0] + A[1,1]*B[1,0]
    out[1,1] = A[1,0]*B[0,1] + A[1,1]*B[1,1]
    return out


@njit(c16[:, :, :](c16[:, :, :], c16[:, :, :]), cache=True)
def matmultiply(A, B):
    shape = A.shape
    C = np.zeros(shape, dtype=np.complex128)
    for i in range(shape[2]):
        C[:, :, i] = mat_mul(A[:, :, i], B[:, :, i])
    return C


class SurfaceRT(ABC):
    pass


def StoT(S: np.ndarray) -> np.ndarray:
    tiny = 1e-15
    T = np.zeros_like(S)
    DET = S[:, 0, 0, :] * S[:, 1, 1, :] - S[:, 0, 1, :] * S[:, 1, 0, :]
    T[:, 0, 0, :] = -DET / (S[:, 1, 0, :] + tiny)
    T[:, 0, 1, :] = S[:, 0, 0, :] / (S[:, 1, 0, :] + tiny)
    T[:, 1, 0, :] = -S[:, 1, 1, :] / (S[:, 1, 0, :] + tiny)
    T[:, 1, 1, :] = 1 / (S[:, 1, 0, :] + tiny)
    return T


def TtoS(T: np.ndarray) -> np.ndarray:
    tiny = 1e-15
    S = np.zeros_like(T)
    DET = T[0, 0, :] * T[1, 1, :] - T[0, 1, :] * T[1, 0, :]
    S[0, 0, :] = T[0, 1, :] / (T[1, 1, :] + tiny)
    S[0, 1, :] = DET / (T[1, 1, :] + tiny)
    S[1, 0, :] = 1 / (T[1, 1, :] + tiny)
    S[1, 1, :] = -T[1, 0, :] / (T[1, 1, :] + tiny)
    return S


class MultiLayer(SurfaceRT):
    def __init__(
        self, k0: float, materials: List[Material], ds: List[float], nangles: int = 100
    ):
        """Creates a multilayer surface with given materials and thicknesses.

        Args:
            k0 (float): The propagation constant to evaluate the multilayer at
            materials (List[Material]): The list of materials in the multilayer
            ds (List[float]): The list of thicknesses for each layer
            nangles (int, optional): The number of angles to sample. Defaults to 100.
        """
        self.materials = materials
        self.ds = ds
        self.k0 = k0
        self.freq = k0 / (2 * np.pi) * 299792458

        angs = np.linspace(0, np.pi / 2, nangles)
        self.angles = angs
        (
            self.Rte1,
            self.Rte2,
            self.Rtm1,
            self.Rtm2,
            self.Tte,
            self.Ttm,
        ) = self._compute_coefficients(angs)

    @property
    def color(self) -> str:
        return self.materials[0].color
    
    @property
    def opacity(self) -> float:
        return sum([mat.opacity for mat in self.materials])/len(self.materials)
    
    def rt_data(self) -> np.ndarray:
        Npts = len(self.angles)
        data = np.zeros((7, Npts), dtype=np.complex64)
        data[0, :] = self.angles.astype(np.complex64)
        data[1, :] = self.Rte1
        data[2, :] = self.Rtm1
        data[3, :] = self.Rte2
        data[4, :] = self.Rtm2
        data[5, :] = self.Tte
        data[6, :] = self.Ttm
        return data
    
    def _compute_coefficients(self, angin: np.ndarray) -> Tuple[np.ndarray]:
        k0 = self.k0
        #check.mustbeequal(self.materials, self.ds)
        #check.mustbe(angin, np.ndarray)
        Nlayers = len(self.ds)
        ds = self.ds
        ers, tands, urs = zip(*[(m.er.scalar(self.freq), m.tand.scalar(self.freq), m.ur.scalar(self.freq)) for m in self.materials])
        ns = [
            np.sqrt(er * ur * (1 - 1j * tand)) for er, ur, tand in zip(ers, urs, tands)
        ]
        Ste = np.zeros((Nlayers, 2, 2, len(angin)), dtype=np.complex128)
        Stm = np.zeros((Nlayers, 2, 2, len(angin)), dtype=np.complex128)
        thi = angin
        cosi = np.cos(thi)
        n1 = 1
        for i in range(Nlayers):
            d = ds[i]
            n2 = ns[i]
            tht = np.arcsin(np.sin(angin) / n2)
            cost = np.cos(tht)
            kz = k0 * n2 * np.cos(tht)
            P = np.exp(-1j * kz * d)

            Rtm  = (n2 * cosi - n1 * cost) / (n2 * cosi + n1 * cost)
            Rtmp = (n1 * cost - n2 * cosi) / (n1 * cost + n2 * cosi)
            Ttm  = (2 * n1 * cosi) / (n2 * cosi + n1 * cost)
            Ttmp = (2 * n2 * cost) / (n1 * cost + n2 * cosi)

            Rte  = (n1 * cosi - n2 * cost) / (n1 * cosi + n2 * cost)
            Rtep = (n2 * cost - n1 * cosi) / (n2 * cost + n1 * cosi)
            Tte  = (2 * n1 * cosi) / (n1 * cosi + n2 * cost)
            Ttep = (2 * n2 * cost) / (n2 * cost + n1 * cosi)

            Ttmm = Ttm * Ttmp * P / (1 - (Rtmp * P) ** 2)
            Rtmm = Rtm + (P**2 * Ttm * Ttmp * Rtmp) / (1 - (P * Rtmp) ** 2)

            Ttem = Tte * Ttep * P / (1 - (Rtep * P) ** 2)
            Rtem = Rte + (P**2 * Tte * Ttep * Rtep) / (1 - (P * Rtep) ** 2)

            Stm[i, 0, 0, :] = Rtmm
            Stm[i, 0, 1, :] = Ttmm
            Stm[i, 1, 0, :] = Ttmm
            Stm[i, 1, 1, :] = Rtmm

            Ste[i, 0, 0, :] = Rtem
            Ste[i, 0, 1, :] = Ttem
            Ste[i, 1, 0, :] = Ttem
            Ste[i, 1, 1, :] = Rtem

        Ttm = StoT(Stm)
        Tte = StoT(Ste)

        Ttmtot = np.squeeze(Ttm[0, :, :, :])
        Ttetot = np.squeeze(Tte[0, :, :, :])
        for i in range(1, Nlayers):
            Ttmtot = matmultiply(Ttmtot, np.squeeze(Ttm[i, :, :, :]))
            Ttetot = matmultiply(Ttetot, np.squeeze(Tte[i, :, :, :]))

        Stm = TtoS(Ttmtot)
        Ste = TtoS(Ttetot)
        # D = sum(ds)
        Rte1 = Ste[0, 0, :].squeeze()
        Rte2 = Ste[1, 1, :].squeeze()
        Rtm1 = Stm[0, 0, :].squeeze()
        Rtm2 = Stm[1, 1, :].squeeze()
        Tte = Ste[1, 0, :].squeeze()
        Ttm = Stm[1, 0, :].squeeze()
        return Rte1, Rte2, Rtm1, Rtm2, Tte, Ttm


class Custom(SurfaceRT):
    def __init__(self, R: complex, T: complex, nangles: int = 1000):
        self.angin = np.linspace(0, np.pi / 2, nangles)
        self.Rte1 = -R * np.ones((nangles,))
        self.Rte2 = -R * np.ones((nangles,))
        self.Rtm1 = R * np.ones((nangles,))
        self.Rtm2 = R * np.ones((nangles,))
        self.Ttm = T * np.ones((nangles,))
        self.Tte = T * np.ones((nangles,))


class PEC(SurfaceRT):
    def __init__(self, nangles: int = 1000):
        self.angles = np.linspace(0, np.pi / 2, nangles)
        self.Rte1 = -np.ones((nangles,))
        self.Rte2 = -np.ones((nangles,))
        self.Rtm1 = np.ones((nangles,))
        self.Rtm2 = np.ones((nangles,))
        self.Ttm = np.zeros((nangles,))
        self.Tte = np.zeros((nangles,))
        self.color: str = "#aaaaaa"
        self.opacity = 1.0
        
    def rt_data(self) -> np.ndarray:
        Npts = len(self.angles)
        data = np.zeros((7, Npts), dtype=np.complex64)
        data[0, :] = self.angles.astype(np.complex64)
        data[1, :] = self.Rte1
        data[2, :] = self.Rtm1
        data[3, :] = self.Rte2
        data[4, :] = self.Rtm2
        data[5, :] = self.Tte
        data[6, :] = self.Ttm
        return data

class AIR(SurfaceRT):
    def __init__(self, nangles: int = 1000):
        self.angles = np.linspace(0, np.pi / 2, nangles)
        self.Rte1 = np.zeros((nangles,))
        self.Rte2 = np.zeros((nangles,))
        self.Rtm1 = np.zeros((nangles,))
        self.Rtm2 = np.zeros((nangles,))
        self.Ttm = np.ones((nangles,))
        self.Tte = np.ones((nangles,))
        self.color: str = "#56b9ff"
        self.opacity: float = 0.1
        
    def rt_data(self) -> np.ndarray:
        Npts = len(self.angles)
        data = np.zeros((7, Npts), dtype=np.complex64)
        data[0, :] = self.angles.astype(np.complex64)
        data[1, :] = self.Rte1
        data[2, :] = self.Rtm1
        data[3, :] = self.Rte2
        data[4, :] = self.Rtm2
        data[5, :] = self.Tte
        data[6, :] = self.Ttm
        return data


FRES_AIR = AIR()
FRES_PEC = PEC()