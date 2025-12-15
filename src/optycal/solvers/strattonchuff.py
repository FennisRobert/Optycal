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
from numba import c16, c16, f8, i8, njit, prange, typeof, f8
from numba.types import Tuple as TupleType
from numba_progress import ProgressBar
from numba_progress.progress import ProgressBarType

LR = 0.001
@njit(
    TupleType((c16[:, :], c16[:, :]))(
        c16[:, :],
        c16[:, :],
        f8[:, :],
        f8[:, :],
        f8[:, :],
        f8,
        ProgressBarType,
    ),
    parallel=True,
    fastmath=True,
    cache=True,
    nogil=True,
)
def stratton_chu_ff(Ein, Hin, vis, wns, tpout, k0, pgb):
    
    Ex = Ein[0, :].flatten()
    Ey = Ein[1, :].flatten()
    Ez = Ein[2, :].flatten()
    Hx = Hin[0, :].flatten()
    Hy = Hin[1, :].flatten()
    Hz = Hin[2, :].flatten()
    vx = vis[0, :].flatten()
    vy = vis[1, :].flatten()
    vz = vis[2, :].flatten()
    nx = wns[0, :].flatten()
    ny = wns[1, :].flatten()
    nz = wns[2, :].flatten()
    
    Emag = np.sqrt(np.abs(Ex)**2 + np.abs(Ey)**2 + np.abs(Ez)**2)
    
    Elevel = np.max(Emag) * LR
    NT = wns.shape[1]
    ids = np.argwhere(Emag > Elevel)
    Nids = ids.shape[0]
    #iadd = NT // Nids
    Ex = Ex[Emag > Elevel]
    Ey = Ey[Emag > Elevel]
    Ez = Ez[Emag > Elevel]
    Hx = Hx[Emag > Elevel]
    Hy = Hy[Emag > Elevel]
    Hz = Hz[Emag > Elevel]
    vx = vx[Emag > Elevel]
    vy = vy[Emag > Elevel]
    vz = vz[Emag > Elevel]
    nx = nx[Emag > Elevel]
    ny = ny[Emag > Elevel]
    nz = nz[Emag > Elevel]

    thout = tpout[0, :]
    phout = tpout[1, :]

    rx = np.sin(thout) * np.cos(phout)
    ry = np.sin(thout) * np.sin(phout)
    rz = np.cos(thout)

    kx = k0 * rx
    ky = k0 * ry
    kz = k0 * rz

    N = tpout.shape[1]

    Eout = np.zeros((3, N)).astype(np.complex128)
    Hout = np.zeros((3, N)).astype(np.complex128)

    Eoutx = np.zeros((N,)).astype(np.complex128)
    Eouty = np.zeros((N,)).astype(np.complex128)
    Eoutz = np.zeros((N,)).astype(np.complex128)

    Z0 = np.float64(376.73031366857)

    Q = np.complex128(-1j * k0 / (4 * np.pi))
    ii = np.complex128(1j)
    

    NxHx = ny * Hz - nz * Hy
    NxHy = nz * Hx - nx * Hz
    NxHz = nx * Hy - ny * Hx

    NxEx = ny * Ez - nz * Ey
    NxEy = nz * Ex - nx * Ez
    NxEz = nx * Ey - ny * Ex

    for j in prange(Nids):
        xi = vx[j]
        yi = vy[j]
        zi = vz[j]
        G = np.exp(ii * (kx * xi + ky * yi + kz * zi))

        RxNxHx = ry * NxHz[j] - rz * NxHy[j]
        RxNxHy = rz * NxHx[j] - rx * NxHz[j]
        RxNxHz = rx * NxHy[j] - ry * NxHx[j]

        ie1x = (NxEx[j] - Z0 * RxNxHx) * G
        ie1y = (NxEy[j] - Z0 * RxNxHy) * G
        ie1z = (NxEz[j] - Z0 * RxNxHz) * G

        Eoutx += Q * (ry * ie1z - rz * ie1y)
        Eouty += Q * (rz * ie1x - rx * ie1z)
        Eoutz += Q * (rx * ie1y - ry * ie1x)

        # ii += iadd
        pgb.update(1)
        
    Eout[0, :] = Eoutx
    Eout[1, :] = Eouty
    Eout[2, :] = Eoutz

    Hout[0, :] = (ry * Eoutz - rz * Eouty) / Z0
    Hout[1, :] = (rz * Eoutx - rx * Eoutz) / Z0
    Hout[2, :] = (rx * Eouty - ry * Eoutx) / Z0

    return Eout, Hout