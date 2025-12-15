from numba import njit, f4, c8, c16, f8
from numba.types import Tuple as TypeTuple
from ..interpolation_pattern import pattern_interp_single
import numpy as np

@njit(TypeTuple((f4[:], f4[:], f4[:]))(f4[:,:], f4[:], f4[:], f4[:]), cache=True, fastmath=True, parallel=False, nogil=True)
def _xyz_basis_4(basis, x, y, z):
    x2 = x*basis[0,0] + y*basis[0,1] + z*basis[0,2]
    y2 = x*basis[1,0] + y*basis[1,1] + z*basis[1,2]
    z2 = x*basis[2,0] + y*basis[2,1] + z*basis[2,2]
    return x2, y2, z2

@njit(TypeTuple((f8[:], f8[:], f8[:]))(f4[:,:], f8[:], f8[:], f8[:]), cache=True, fastmath=True, parallel=False, nogil=True)
def _xyz_basis_8(basis, x, y, z):
    x2 = x*basis[0,0] + y*basis[0,1] + z*basis[0,2]
    y2 = x*basis[1,0] + y*basis[1,1] + z*basis[1,2]
    z2 = x*basis[2,0] + y*basis[2,1] + z*basis[2,2]
    return x2, y2, z2

@njit(TypeTuple((c8[:], c8[:], c8[:]))(f4[:,:], c8[:], c8[:], c8[:]), cache=True, fastmath=True, parallel=False, nogil=True)
def _cxyz_basis_8(basis, x, y, z):
    x2 = x*basis[0,0] + y*basis[0,1] + z*basis[0,2]
    y2 = x*basis[1,0] + y*basis[1,1] + z*basis[1,2]
    z2 = x*basis[2,0] + y*basis[2,1] + z*basis[2,2]
    return x2, y2, z2

@njit(TypeTuple((c16[:], c16[:], c16[:]))(f4[:,:], c16[:], c16[:], c16[:]), cache=True, fastmath=True, parallel=False, nogil=True)
def _cxyz_basis_16(basis, x, y, z):
    x2 = x*basis[0,0] + y*basis[0,1] + z*basis[0,2]
    y2 = x*basis[1,0] + y*basis[1,1] + z*basis[1,2]
    z2 = x*basis[2,0] + y*basis[2,1] + z*basis[2,2]
    return x2, y2, z2

@njit(TypeTuple((f4[:], f4[:]))(f4[:], f4[:], f4[:,:]), cache=True, fastmath=True, parallel=False, nogil=True)
def _theta_phi_transform(theta, phi, Basis):
    ux = np.cos(theta)*np.cos(phi)
    uy = np.cos(theta)*np.sin(phi)
    uz = np.sin(theta)
    ux2 = ux*Basis[0,0] + uy*Basis[0,1] + uz*Basis[0,2]
    uy2 = ux*Basis[1,0] + uy*Basis[1,1] + uz*Basis[1,2]
    uz2 = ux*Basis[2,0] + uy*Basis[2,1] + uz*Basis[2,2]
    theta2 = np.arctan2(uz2, np.sqrt(ux2**2 + uy2**2))
    phi2 = np.arctan2(uy2, ux2)
    return theta2, phi2

@njit(TypeTuple((f8[:], f8[:], f8[:]))(f8[:], f8[:], f8[:], f8[:], f8[:], f8[:]), cache=True, fastmath=True, nogil=True)
def _c_cross_comp(x1, y1, z1, x2, y2, z2):
    x = y1*z2 - z1*y2
    y = z1*x2 - x1*z2
    z = x1*y2 - y1*x2
    return x, y, z



@njit(TypeTuple((c8[:,:], c8[:,:]))(f8[:], f8[:], f8[:], f8[:,:], c8[:,:,:,:,:], f4[:], f4[:], f4[:,:], c8, f8), cache=True, fastmath=True, parallel=False, nogil=True)
def expose_xyz_single(gx, gy, gz, ant_gxyz, Amats, thetagrid, phigrid, Gbasis, amplitude, k0):
    """
    Compute the nearfield of the antenna at the points (gx, gy, gz)
    """
    sx, sy, sz = ant_gxyz #double

    dx = gx - sx #double
    dy = gy - sy #double
    dz = gz - sz #double

    E = np.zeros((3, gx.shape[0]), dtype=np.complex64)
    H = np.zeros((3, gx.shape[0]), dtype=np.complex64)
    
    R = np.sqrt(dx**2 + dy**2 + dz**2) #double
    kx = dx/R #double
    ky = dy/R #double
    kz = dz/R #double
    
    lkx, lky, lkz = _xyz_basis_8(Gbasis, kx, ky, kz) #double

    thetac = np.arccos(lkz).astype(np.float32)#, np.sqrt(lkx**2 + lky**2)).astype(np.float32)
    phic = np.arctan2(lky, lkx).astype(np.float32) #single
    

    B = amplitude * np.exp(-1j * k0 * R) / R # double

    [ex, ey, ez, hx, hy, hz] = pattern_interp_single(thetac, phic, thetagrid, phigrid, Amats)
    
    Binv = np.linalg.pinv(Gbasis) #single

    ex, ey, ez = _cxyz_basis_8(Binv, ex, ey, ez) #single
    hx, hy, hz = _cxyz_basis_8(Binv, hx, hy, hz) #signle

    E[0,:] = ex*B
    E[1,:] = ey*B
    E[2,:] = ez*B
    H[0,:] = hx*B
    H[1,:] = hy*B
    H[2,:] = hz*B
    return E, H


@njit(TypeTuple((c8[:,:], c8[:,:]))(f4[:], f4[:], f8[:], c8[:,:,:,:,:], f4[:], f4[:], f4[:,:], c8, f8), cache=True, fastmath=True, parallel=False, nogil=True)
def expose_thetaphi_single(gtheta, gphi, ant_gxyz, Amats, thetagrid, phigrid, Gbasis, amplitude, k0):
    """
    Compute the farfield of the antenna at the points (theta, phi)
    """
    gtheta = gtheta.astype(np.float32)
    gphi = gphi.astype(np.float32)

    Binv = np.linalg.pinv(Gbasis)

    E = np.zeros((3, gtheta.shape[0]), dtype=np.complex64)
    H = np.zeros((3, gtheta.shape[0]), dtype=np.complex64)

    uxg = np.sin(gtheta)*np.cos(gphi)
    uyg = np.sin(gtheta)*np.sin(gphi)
    uzg = np.cos(gtheta)

    uxl = uxg*Binv[0,0] + uyg*Binv[0,1] + uzg*Binv[0,2]
    uyl = uxg*Binv[1,0] + uyg*Binv[1,1] + uzg*Binv[1,2]
    uzl = uxg*Binv[2,0] + uyg*Binv[2,1] + uzg*Binv[2,2]
    
    theta_local = np.arccos(uzl)#, np.sqrt(uxl**2 + uyl**2))
    phi_local = np.arctan2(uyl, uxl)
    
    x0, y0, z0 = [0,0,0]
    kx, ky, kz = k0*uxl, k0*uyl, k0*uzl
    
    gx, gy, gz = ant_gxyz
    
    B = amplitude * np.exp(1j * (kx * (gx-x0) + ky * (gy-y0) + kz * (gz-z0)))
    [ex, ey, ez, hx, hy, hz] = pattern_interp_single(theta_local, phi_local, thetagrid, phigrid, Amats)
    
    ex, ey, ez = _cxyz_basis_8(Binv, ex, ey, ez)
    hx, hy, hz = _cxyz_basis_8(Binv, hx, hy, hz)

    E[0,:] = ex*B
    E[1,:] = ey*B
    E[2,:] = ez*B
    H[0,:] = hx*B
    H[1,:] = hy*B
    H[2,:] = hz*B
    return E, H

@njit(TypeTuple((c8[:,:], c8[:,:], c8[:,:], c8[:,:]))(
        f8[:,:],
        f8[:,:],
        c8[:,:],
        f8[:,:],
        c8[:,:,:,:,:],
        f4[:],
        f4[:],
        f4[:,:],
        c8,
        f8), cache=True, fastmath=True, parallel=False, nogil=True)
def expose_surface_single(gxyz, tn, fresnel, ant_gxyz, Amats, thetagrid, phigrid, Gbasis, amplitude, k0):
    
    gx = gxyz[0,:] #double
    gy = gxyz[1,:]
    gz = gxyz[2,:]
    
    fshape = gxyz.shape
    E1 = np.zeros(fshape, dtype=np.complex64)
    E2 = np.zeros(fshape, dtype=np.complex64)
    H1 = np.zeros(fshape, dtype=np.complex64)
    H2 = np.zeros(fshape, dtype=np.complex64)

    x = gxyz[0,:] #double
    y = gxyz[1,:]
    z = gxyz[2,:]

    E, H = expose_xyz_single(gx, gy, gz, ant_gxyz, Amats, thetagrid, phigrid, Gbasis, amplitude, k0)

    Ex = E[0,:] #single
    Ey = E[1,:]
    Ez = E[2,:]
    Hx = H[0,:]
    Hy = H[1,:]
    Hz = H[2,:]
    
    gx, gy, gz = ant_gxyz #double
    rsx = x-gx
    rsy = y-gy # double
    rsz = z-gz

    tn = tn + np.random.rand(3, tn.shape[1])*1e-8 #double
    tn = tn / np.sqrt(tn[0,:]**2 + tn[1,:]**2 + tn[2,:]**2)

    R = np.sqrt(rsx**2 + rsy**2 + rsz**2) #double
    tnx = tn[0,:] #double
    tny = tn[1,:]
    tnz = tn[2,:]
    
    rdotn = (rsx*tnx + rsy*tny + rsz*tnz)/R #double
    sphx, sphy, sphz = _c_cross_comp(rsx, rsy, rsz, tnx, tny, tnz)
    S = np.sqrt(sphx**2 + sphy**2 + sphz**2)
    
    sphx = sphx/S #double
    sphy = sphy/S
    sphz = sphz/S
    
    pphx, pphy, pphz = _c_cross_comp(rsx, rsy, rsz, sphx, sphy, sphz) #double
    P = np.sqrt(pphx**2 + pphy**2 + pphz**2)
    pphx = pphx/P
    pphy = pphy/P #double
    pphz = pphz/P
    
    pdn = pphx*tnx + pphy*tny + pphz*tnz
    pprhx = 2*pdn*tnx - pphx
    pprhy = 2*pdn*tny - pphy #double
    pprhz = 2*pdn*tnz - pphz
    
    angin = np.arccos(np.clip(np.abs(rdotn), a_min=0, a_max=1)) #double
    refang = fresnel[0,:]
    Rte1_data = fresnel[1,:]
    Rtm1_data = fresnel[2,:] #single
    Rte2_data = fresnel[3,:]
    Rtm2_data = fresnel[4,:]
    Tte_data = fresnel[5,:]
    Ttm_data = fresnel[6,:]

    refang_float = refang.real.astype(np.float32)
    angin_float = angin.astype(np.float32)
    
    Rte1 = np.interp(angin_float, refang_float, Rte1_data)
    Rtm1 = np.interp(angin_float, refang_float, Rtm1_data)
    Rte2 = np.interp(angin_float, refang_float, Rte2_data) #single
    Rtm2 = np.interp(angin_float, refang_float, Rtm2_data)
    Tte = np.interp(angin_float, refang_float, Tte_data)
    Ttm = np.interp(angin_float, refang_float, Ttm_data)
    
    same = (rdotn>0).astype(np.float32) # single
    other = 1-same #signle

    Rte = Rte1*same + Rte2*other #Single
    Rtm = Rtm1*same + Rtm2*other
    

    Es = Ex*sphx + Ey*sphy + Ez*sphz
    Ep = Ex*pphx + Ey*pphy + Ez*pphz
    Hs = Hx*sphx + Hy*sphy + Hz*sphz
    Hp = Hx*pphx + Hy*pphy + Hz*pphz
    
    Erefx = Rte*(Es*sphx) + Rtm*(Ep*pprhx)
    Erefy = Rte*(Es*sphy) + Rtm*(Ep*pprhy)
    Erefz = Rte*(Es*sphz) + Rtm*(Ep*pprhz)
    Etransx = Tte*(Es*sphx) + Ttm*(Ep*pphx)
    Etransy = Tte*(Es*sphy) + Ttm*(Ep*pphy)
    Etransz = Tte*(Es*sphz) + Ttm*(Ep*pphz)
    
    Hrefx = Rtm*(Hs*sphx) + Rte*(Hp*pprhx)
    Hrefy = Rtm*(Hs*sphy) + Rte*(Hp*pprhy)
    Hrefz = Rtm*(Hs*sphz) + Rte*(Hp*pprhz)
    Htransx = Ttm*(Hs*sphx) + Tte*(Hp*pphx)
    Htransy = Ttm*(Hs*sphy) + Tte*(Hp*pphy)
    Htransz = Ttm*(Hs*sphz) + Tte*(Hp*pphz)
    
    
    E1[0,:] = Erefx*same + Etransx*other
    E1[1,:] = Erefy*same + Etransy*other
    E1[2,:] = Erefz*same + Etransz*other
    H1[0,:] = Hrefx*same + Htransx*other
    H1[1,:] = Hrefy*same + Htransy*other
    H1[2,:] = Hrefz*same + Htransz*other
    E2[0,:] = Erefx*other + Etransx*same
    E2[1,:] = Erefy*other + Etransy*same
    E2[2,:] = Erefz*other + Etransz*same
    H2[0,:] = Hrefx*other + Htransx*same
    H2[1,:] = Hrefy*other + Htransy*same
    H2[2,:] = Hrefz*other + Htransz*same
    
    return E1, H1, E2, H2
