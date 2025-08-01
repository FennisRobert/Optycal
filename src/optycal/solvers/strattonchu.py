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
from numba import c16, c16, f8, i8, njit, prange, typeof, vectorize, f8, c16
from numba.types import Tuple as TupleType
from numba_progress.progress import ProgressBarType

LR = 0.001

EXPIARRY = np.linspace(0,2*np.pi,100).astype(np.float64)
EXPOARRY = np.exp(-1j*EXPIARRY).astype(np.complex128)

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
def stratton_chu_xyz(Ein, Hin, vis, wns, cout, k0, pgb):
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

    xo = cout[0,:]
    yo = cout[1,:]
    zo = cout[2,:]
    
    N = cout.shape[1]

    Eout = np.zeros((3, N)).astype(np.complex128)
    Hout = np.zeros((3, N)).astype(np.complex128)

    Eoutx = np.zeros((N,)).astype(np.complex128)
    Houtx = np.zeros((N,)).astype(np.complex128)
    Eouty = np.zeros((N,)).astype(np.complex128)
    Houty = np.zeros((N,)).astype(np.complex128)
    Eoutz = np.zeros((N,)).astype(np.complex128)
    Houtz = np.zeros((N,)).astype(np.complex128)

    NO = cout.shape[0]
    w0 = np.float64(k0 * 299792458)
    u0 = np.float64(4 * np.pi * 1e-7)
    eps0 = np.float64(8.854187812813e-12)
    
    Q = np.float64(1 / (4 * np.pi))
    ii = np.complex128(1j)
    
    NxHx = -(Hy * nz - Hz * ny)
    NxHy = -(Hz * nx - Hx * nz)
    NxHz = -(Hx * ny - Hy * nx)

    NxEx = -(Ey * nz - Ez * ny)
    NxEy = -(Ez * nx - Ex * nz)
    NxEz = -(Ex * ny - Ey * nx)

    EdN = Ex * nx + Ey * ny + Ez * nz
    HdN = Hx * nx + Hy * ny + Hz * nz
    ie1 = -ii * w0 * u0
    ih1 = ii * w0 * eps0
    
    for j in prange(Nids):
        Rx = xo-vx[j]
        Ry = yo-vy[j] 
        Rz = zo-vz[j]

        R = np.sqrt(Rx**2 + Ry**2 + Rz**2)
        Ri = 1/R
        G = Q*np.exp(-ii * R * k0)*Ri
        #G = Q*np.interp(k0*R % PI2, EXPIARRY, EXPOARRY) * Ri
        dG = Ri*(Ri + ii * k0)

        Eoutx += G * ((ie1 * NxHx[j]) + dG * ((NxEy[j] * Rz - NxEz[j] * Ry) + EdN[j] * Rx))
        Eouty += G * ((ie1 * NxHy[j]) + dG * ((NxEz[j] * Rx - NxEx[j] * Rz) + EdN[j] * Ry))
        Eoutz += G * ((ie1 * NxHz[j]) + dG * ((NxEx[j] * Ry - NxEy[j] * Rx) + EdN[j] * Rz))
        Houtx += G * ((ih1 * NxEx[j]) + dG * ((NxHy[j] * Rz - NxHz[j] * Ry) + HdN[j] * Rx))
        Houty += G * ((ih1 * NxEy[j]) + dG * ((NxHz[j] * Rx - NxHx[j] * Rz) + HdN[j] * Ry))
        Houtz += G * ((ih1 * NxEz[j]) + dG * ((NxHx[j] * Ry - NxHy[j] * Rx) + HdN[j] * Rz))
        # ii += iadd
        pgb.update(1)
    Eout[0, :] = Eoutx
    Eout[1, :] = Eouty
    Eout[2, :] = Eoutz
    Hout[0, :] = Houtx
    Hout[1, :] = Houty
    Hout[2, :] = Houtz
    return Eout, Hout


@njit(
    TupleType((c16[:, :], c16[:, :], c16[:, :], c16[:, :]))(
        c16[:, :],
        c16[:, :],
        f8[:, :],
        f8[:, :],
        f8[:, :],
        f8[:, :],
        c16[:, :],
        f8,
        ProgressBarType,
    ),
    parallel=True,
    fastmath=True,
    cache=True,
    nogil=True,
)
def stratton_chu_xyz_surface(
    Ein, Hin, vis, wns, cout, nout, fresnel, k0, pgb,
):
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
    ids = np.argwhere(Emag > Elevel)
    Nids = ids.shape[0]
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

    xo = cout[0, :]
    yo = cout[1, :]
    zo = cout[2, :]
    nox = nout[0, :]
    noy = cout[1, :]
    noz = cout[2, :]
    nox = nox + np.random.rand(nox.shape[0])*1e-12
    noy = noy + np.random.rand(nox.shape[0])*1e-12
    noz = noz + np.random.rand(nox.shape[0])*1e-12
    N = cout.shape[1]

    Eout1x = np.zeros((N,)).astype(np.complex128)
    Hout1x = np.zeros((N,)).astype(np.complex128)
    Eout1y = np.zeros((N,)).astype(np.complex128)
    Hout1y = np.zeros((N,)).astype(np.complex128)
    Eout1z = np.zeros((N,)).astype(np.complex128)
    Hout1z = np.zeros((N,)).astype(np.complex128)
    
    Eout2x = np.zeros((N,)).astype(np.complex128)
    Hout2x = np.zeros((N,)).astype(np.complex128)
    Eout2y = np.zeros((N,)).astype(np.complex128)
    Hout2y = np.zeros((N,)).astype(np.complex128)
    Eout2z = np.zeros((N,)).astype(np.complex128)
    Hout2z = np.zeros((N,)).astype(np.complex128)

    Eout1 = np.zeros((3, N), dtype=np.complex128)
    Eout2 = np.zeros((3, N), dtype=np.complex128)
    Hout1 = np.zeros((3, N), dtype=np.complex128)
    Hout2 = np.zeros((3, N), dtype=np.complex128)
    
    w0 = np.float64(k0 * 299792458)
    u0 = np.float64(4 * np.pi * 1e-7)
    eps0 = np.float64(8.854187812813e-12)
    
    Q = np.float64(1 / (4 * np.pi))
    ii = np.complex128(1j)
    refangle = np.real(fresnel[:,0])
    
    NxHx = -(Hy * nz - Hz * ny)
    NxHy = -(Hz * nx - Hx * nz)
    NxHz = -(Hx * ny - Hy * nx)

    NxEx = -(Ey * nz - Ez * ny)
    NxEy = -(Ez * nx - Ex * nz)
    NxEz = -(Ex * ny - Ey * nx)

    EdN = Ex * nx + Ey * ny + Ez * nz
    HdN = Hx * nx + Hy * ny + Hz * nz
    
    for j in prange(Nids):

        xi = vx[j]
        yi = vy[j]
        zi = vz[j]

        Rx = xo-xi
        Ry = yo-yi 
        Rz = zo-zi

        R = np.sqrt(Rx**2 + Ry**2 + Rz**2)
        
        rx = Rx / R
        ry = Ry / R
        rz = Rz / R

        NxExRx = NxEy[j] * rz - NxEz[j] * ry
        NxExRy = NxEz[j] * rx - NxEx[j] * rz
        NxExRz = NxEx[j] * ry - NxEy[j] * rx

        NxHxRx = NxHy[j] * rz - NxHz[j] * ry
        NxHxRy = NxHz[j] * rx - NxHx[j] * rz
        NxHxRz = NxHx[j] * ry - NxHy[j] * rx


        G = np.exp(-ii * R * k0) / R
        dG = (1 / R + ii * k0) * G

        ie1 = -ii * w0 * u0 * G
        ie1x = ie1 * NxHx[j]
        ie1y = ie1 * NxHy[j]
        ie1z = ie1 * NxHz[j]

        ih1 = ii * w0 * eps0 * G
        ih1x = ih1 * NxEx[j]
        ih1y = ih1 * NxEy[j]
        ih1z = ih1 * NxEz[j]

        Enewx = Q * (ie1x + dG * (NxExRx + EdN[j] * rx))
        Enewy = Q * (ie1y + dG * (NxExRy + EdN[j] * ry))
        Enewz = Q * (ie1z + dG * (NxExRz + EdN[j] * rz))
        Hnewx = Q * (ih1x + dG * (NxHxRx + HdN[j] * rx))
        Hnewy = Q * (ih1y + dG * (NxHxRy + HdN[j] * ry))
        Hnewz = Q * (ih1z + dG * (NxHxRz + HdN[j] * rz))

        kdotn = rx * nox + ry * noy + rz * noz

        angkn = np.abs(kdotn)
        angkn[angkn>1.0] = 1.0

        angin = np.arccos(angkn)

        sphatx = ry * noz - rz * noy
        sphaty = rz * nox - rx * noz
        sphatz = rx * noy - ry * nox
        normsphat = np.sqrt(sphatx**2 + sphaty**2 + sphatz**2)

        sphatx = sphatx / normsphat
        sphaty = sphaty / normsphat
        sphatz = sphatz / normsphat

        pphatx = ry * sphatz - rz * sphaty
        pphaty = rz * sphatx - rx * sphatz
        pphatz = rx * sphaty - ry * sphatx
        normpphat = np.sqrt(pphatx**2 + pphaty**2 + pphatz**2)

        pphatx = pphatx / normpphat
        pphaty = pphaty / normpphat
        pphatz = pphatz / normpphat

        pdn = (pphatx * nox + pphaty * noy + pphatz * noz)
        pprhatx = 2 * pdn * nox - pphatx
        pprhaty = 2 * pdn * noy - pphaty
        pprhatz = 2 * pdn * noz - pphatz
        
        Rte1 = np.interp(angin, refangle, fresnel[:, 1])
        Rtm1 = np.interp(angin, refangle, fresnel[:, 2])
        Rte2 = np.interp(angin, refangle, fresnel[:, 3])
        Rtm2 = np.interp(angin, refangle, fresnel[:, 4])
        Tte = np.interp(angin, refangle, fresnel[:, 5])
        Ttm = np.interp(angin, refangle, fresnel[:, 6])
        
        same = (kdotn>=0).astype(np.float64)
        other = 1 - same
        
        Rte = Rte1*same + Rte2*other
        Rtm = Rtm1*same + Rtm2*other

        Edots = sphatx * Enewx + sphaty * Enewy + sphatz * Enewz
        Edotp = pphatx * Enewx + pphaty * Enewy + pphatz * Enewz
        
        Hdots = sphatx * Hnewx + sphaty * Hnewy + sphatz * Hnewz
        Hdotp = pphatx * Hnewx + pphaty * Hnewy + pphatz * Hnewz
        
        Erefx = Rte * Edots * sphatx + Rtm * Edotp * pprhatx
        Erefy = Rte * Edots * sphaty + Rtm * Edotp * pprhaty
        Erefz = Rte * Edots * sphatz + Rtm * Edotp * pprhatz
        Etransx = Tte * Edots * sphatx + Ttm * Edotp * pphatx
        Etransy = Tte * Edots * sphaty + Ttm * Edotp * pphaty
        Etransz = Tte * Edots * sphatz + Ttm * Edotp * pphatz

        Hrefx = Rtm * Hdots * sphatx + Rte * Hdotp * pprhatx
        Hrefy = Rtm * Hdots * sphaty + Rte * Hdotp * pprhaty
        Hrefz = Rtm * Hdots * sphatz + Rte * Hdotp * pprhatz
        Htransx = Ttm * Hdots * sphatx + Tte * Hdotp * pphatx
        Htransy = Ttm * Hdots * sphaty + Tte * Hdotp * pphaty
        Htransz = Ttm * Hdots * sphatz + Tte * Hdotp * pphatz
        
        same, other = other, same
        Eout1x += Erefx * same + Etransx * other
        Eout1y += Erefy * same + Etransy * other
        Eout1z += Erefz * same + Etransz * other
        Hout1x += Hrefx * same + Htransx * other
        Hout1y += Hrefy * same + Htransy * other
        Hout1z += Hrefz * same + Htransz * other
        
        Eout2x += Erefx * other + Etransx * same
        Eout2y += Erefy * other + Etransy * same
        Eout2z += Erefz * other + Etransz * same
        Hout2x += Hrefx * other + Htransx * same
        Hout2y += Hrefy * other + Htransy * same
        Hout2z += Hrefz * other + Htransz * same
        
        # ii += iadd
        pgb.update(1)
    Eout1[0, :] = Eout1x
    Eout1[1, :] = Eout1y
    Eout1[2, :] = Eout1z
    Hout1[0, :] = Hout1x
    Hout1[1, :] = Hout1y
    Hout1[2, :] = Hout1z
    
    Eout2[0, :] = Eout2x
    Eout2[1, :] = Eout2y
    Eout2[2, :] = Eout2z
    Hout2[0, :] = Hout2x
    Hout2[1, :] = Hout2y
    Hout2[2, :] = Hout2z
    return Eout1, Hout1, Eout2, Hout2