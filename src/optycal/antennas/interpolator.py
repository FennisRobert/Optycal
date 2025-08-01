import numpy as np
from numba import njit, f8, prange, i8, c8, c16, f4
from enum import Enum


class intmode(Enum):
    NaturalSpline = 0
    ParabolicRunout = 1
    CubicRunout = 2

@njit(f4[:](f4[:], f4[:], i8), cache=True, fastmath=True, nogil=True)
def _diff_f4(x, y, mode):
    N = len(x)
    Bs = 6*(y[:-2] - 2*y[1:-1] + y[2:])
    A = np.zeros((N-2,N-2), dtype=np.float32)

    np.fill_diagonal(A, 4)
    np.fill_diagonal(A[1:,:-1], 1)
    np.fill_diagonal(A[:-1,1:], 1)
    
    if mode == 0:
        mfill = 4.
    elif mode == 1:
        mfill = 5.
    elif mode == 2:
        mfill = 6.
    
    A[0,0] = mfill
    A[-1,-1] = mfill

    Ms = np.zeros((N,), dtype=np.float32)
    Ms[1:-1] = np.linalg.solve(A,Bs)
    if mode == 0:
        Ms[0] = 0.
        Ms[-1] = 0.
    elif mode == 1:
        Ms[0] = Ms[1]
        Ms[-1] = Ms[-2]
    elif mode == 2:
        Ms[0] = 2*Ms[1]-Ms[2]
        Ms[-1] = 2*Ms[-2]-Ms[-3]
    
    ais = (Ms[1:]-Ms[:-1])/(6)
    bis = Ms[:-1]/2
    cis = (y[1:]-y[:-1]) - ((Ms[1:] + 2*Ms[:-1])/6)
    
    dys = 0*y
    dys[:-1] = cis
    dys[-1] = 3*ais[-1] + 2*bis[-1] + cis[-1]
    return dys

@njit(f8[:](f4[:], f8[:], i8), cache=True, fastmath=True, nogil=True)
def _diff_f8(x,y, mode):
    N = len(x)
    Bs = 6*(y[:-2] - 2*y[1:-1] + y[2:])
    A = np.zeros((N-2,N-2), dtype=np.float64)

    np.fill_diagonal(A, 4)
    np.fill_diagonal(A[1:,:-1], 1)
    np.fill_diagonal(A[:-1,1:], 1)
    
    if mode == 0:
        mfill = 4.
    elif mode == 1:
        mfill = 5.
    elif mode == 2:
        mfill = 6.
    
    A[0,0] = mfill
    A[-1,-1] = mfill

    Ms = np.zeros((N,), dtype=np.float64)
    Ms[1:-1] = np.linalg.solve(A,Bs)
    if mode == 0:
        Ms[0] = 0.
        Ms[-1] = 0.
    elif mode == 1:
        Ms[0] = Ms[1]
        Ms[-1] = Ms[-2]
    elif mode == 2:
        Ms[0] = 2*Ms[1]-Ms[2]
        Ms[-1] = 2*Ms[-2]-Ms[-3]
    
    ais = (Ms[1:]-Ms[:-1])/(6)
    bis = Ms[:-1]/2
    cis = (y[1:]-y[:-1]) - ((Ms[1:] + 2*Ms[:-1])/6)
    
    dys = 0*y
    dys[:-1] = cis
    dys[-1] = 3*ais[-1] + 2*bis[-1] + cis[-1]
    return dys

@njit(c8[:](f4[:], c8[:], i8), cache=True, fastmath=True, nogil=True)
def _diff_c8(x,y, mode):
    N = len(x)
    Bs = 6*(y[:-2] - 2*y[1:-1] + y[2:])
    A = np.zeros((N-2,N-2), dtype=np.complex64)

    np.fill_diagonal(A, 4)
    np.fill_diagonal(A[1:,:-1], 1)
    np.fill_diagonal(A[:-1,1:], 1)
    
    if mode == 0:
        mfill = 4.
    elif mode == 1:
        mfill = 5.
    elif mode == 2:
        mfill = 6.
    
    A[0,0] = mfill
    A[-1,-1] = mfill

    Ms = np.zeros((N,), dtype=np.complex64)
    Ms[1:-1] = np.linalg.solve(A,Bs)
    if mode == 0:
        Ms[0] = 0.
        Ms[-1] = 0.
    elif mode == 1:
        Ms[0] = Ms[1]
        Ms[-1] = Ms[-2]
    elif mode == 2:
        Ms[0] = 2*Ms[1]-Ms[2]
        Ms[-1] = 2*Ms[-2]-Ms[-3]
    
    ais = (Ms[1:]-Ms[:-1])/(6)
    bis = Ms[:-1]/2
    cis = (y[1:]-y[:-1]) - ((Ms[1:] + 2*Ms[:-1])/6)
    
    dys = 0*y
    dys[:-1] = cis
    dys[-1] = 3*ais[-1] + 2*bis[-1] + cis[-1]
    return dys

@njit(c16[:](f4[:], c16[:], i8), cache=True, fastmath=True, nogil=True)
def _diff_c16(x,y, mode):
    N = len(x)
    Bs = 6*(y[:-2] - 2*y[1:-1] + y[2:])
    A = np.zeros((N-2,N-2), dtype=np.complex128)

    np.fill_diagonal(A, 4)
    np.fill_diagonal(A[1:,:-1], 1)
    np.fill_diagonal(A[:-1,1:], 1)
    
    if mode == 0:
        mfill = 4.0
    elif mode == 1:
        mfill = 5.0
    elif mode == 2:
        mfill = 6.0
    
    A[0,0] = mfill
    A[-1,-1] = mfill

    Ms = np.zeros((N,), dtype=np.complex128)
    Ms[1:-1] = np.linalg.solve(A,Bs)
    if mode == 0:
        Ms[0] = 0.
        Ms[-1] = 0.
    elif mode == 1:
        Ms[0] = Ms[1]
        Ms[-1] = Ms[-2]
    elif mode == 2:
        Ms[0] = 2*Ms[1]-Ms[2]
        Ms[-1] = 2*Ms[-2]-Ms[-3]
    
    ais = (Ms[1:]-Ms[:-1])/(6)
    bis = Ms[:-1]/2
    cis = (y[1:]-y[:-1]) - ((Ms[1:] + 2*Ms[:-1])/6)
    
    dys = 0*y
    dys[:-1] = cis
    dys[-1] = 3*ais[-1] + 2*bis[-1] + cis[-1]
    return dys

@njit(f4[:, :, :, :](f4[:], f4[:], f4[:, :], i8), cache=True, fastmath=True, nogil=True, parallel=True)
def _int_mat_f4(x_ax, y_ax, zs, mode):
    Nx = len(x_ax)
    Ny = len(y_ax)
    a_coeffs = np.zeros((4,4,Nx-1,Ny-1), dtype=np.float32)

    dzdx = 0*zs
    dzdy = 0*zs
    dzdxy = 0*zs
    # Compute the derivatives
    for j in prange(Ny-1):
        dzdx[:,j] = _diff_f4(x_ax,zs[:,j],mode)
    for i in prange(Nx-1):
        dzdy[i,:] = _diff_f4(y_ax,zs[i,:],mode)
    for i in prange(Nx-1):
        dzdxy[i,:] = _diff_f4(y_ax,dzdx[i,:],mode)

    M1 = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [-3.0, 3.0, -2.0, -1.0], [2.0, -2.0, 1.0, 1.0]]).astype(np.float32)
    M2 = np.array([[1.0, 0.0, -3.0, 2.0], [0.0, 0.0, 3.0, -2.0], [0.0, 1.0, -2.0, 1.0], [0.0, 0.0, -1.0, 1.0]]).astype(np.float32)

    for i in prange(Nx-1):
        for j in range(Ny-1):
            F = np.array([[zs[i,j], zs[i,j+1],dzdy[i,j], dzdy[i,j+1]],
                          [zs[i+1,j],zs[i+1,j+1],dzdy[i+1,j],dzdy[i+1,j+1]],
                          [dzdx[i,j], dzdx[i,j+1],dzdxy[i,j],dzdxy[i,j+1]],
                          [dzdx[i+1,j],dzdx[i+1,j+1],dzdxy[i+1,j], dzdxy[i+1,j+1]]])
            a_coeffs[:,:,i,j] = M1 @ (F @ M2)
    
    return a_coeffs

@njit(f8[:, :, :, :](f4[:], f4[:], f8[:, :], i8), cache=True, fastmath=True, nogil=True, parallel=True)
def _int_mat_f8(x_ax, y_ax, zs, mode):
    Nx = len(x_ax)
    Ny = len(y_ax)
    a_coeffs = np.zeros((4,4,Nx-1,Ny-1), dtype=np.float64)

    dzdx = 0*zs
    dzdy = 0*zs
    dzdxy = 0*zs
    # Compute the derivatives
    for j in prange(Ny-1):
        dzdx[:,j] = _diff_f8(x_ax,zs[:,j],mode)
    for i in prange(Nx-1):
        dzdy[i,:] = _diff_f8(y_ax,zs[i,:],mode)
    for i in prange(Nx-1):
        dzdxy[i,:] = _diff_f8(y_ax,dzdx[i,:],mode)

    M1 = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [-3.0, 3.0, -2.0, -1.0], [2.0, -2.0, 1.0, 1.0]]).astype(np.float64)
    M2 = np.array([[1.0, 0.0, -3.0, 2.0], [0.0, 0.0, 3.0, -2.0], [0.0, 1.0, -2.0, 1.0], [0.0, 0.0, -1.0, 1.0]]).astype(np.float64)

    for i in prange(Nx-1):
        for j in range(Ny-1):
            F = np.array([[zs[i,j], zs[i,j+1],dzdy[i,j], dzdy[i,j+1]],
                          [zs[i+1,j],zs[i+1,j+1],dzdy[i+1,j],dzdy[i+1,j+1]],
                          [dzdx[i,j], dzdx[i,j+1],dzdxy[i,j],dzdxy[i,j+1]],
                          [dzdx[i+1,j],dzdx[i+1,j+1],dzdxy[i+1,j], dzdxy[i+1,j+1]]])
            a_coeffs[:,:,i,j] = M1 @ (F @ M2)
    
    return a_coeffs

@njit(c8[:, :, :, :](f4[:], f4[:], c8[:, :], i8), cache=True, fastmath=True, nogil=True, parallel=True)
def _int_mat_c8(x_ax, y_ax, zs, mode):
    Nx = len(x_ax)
    Ny = len(y_ax)
    a_coeffs = np.zeros((4,4,Nx-1,Ny-1), dtype=np.complex64)

    dzdx = 0*zs
    dzdy = 0*zs
    dzdxy = 0*zs
    # Compute the derivatives
    for j in prange(Ny-1):
        dzdx[:,j] = _diff_c8(x_ax,zs[:,j],mode)
    for i in prange(Nx-1):
        dzdy[i,:] = _diff_c8(y_ax,zs[i,:],mode)
    for i in prange(Nx-1):
        dzdxy[i,:] = _diff_c8(y_ax,dzdx[i,:],mode)

    M1 = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [-3.0, 3.0, -2.0, -1.0], [2.0, -2.0, 1.0, 1.0]]).astype(np.complex64)
    M2 = np.array([[1.0, 0.0, -3.0, 2.0], [0.0, 0.0, 3.0, -2.0], [0.0, 1.0, -2.0, 1.0], [0.0, 0.0, -1.0, 1.0]]).astype(np.complex64)

    for i in prange(Nx-1):
        for j in range(Ny-1):
            F = np.array([[zs[i,j], zs[i,j+1],dzdy[i,j], dzdy[i,j+1]],
                          [zs[i+1,j],zs[i+1,j+1],dzdy[i+1,j],dzdy[i+1,j+1]],
                          [dzdx[i,j], dzdx[i,j+1],dzdxy[i,j],dzdxy[i,j+1]],
                          [dzdx[i+1,j],dzdx[i+1,j+1],dzdxy[i+1,j], dzdxy[i+1,j+1]]])
            a_coeffs[:,:,i,j] = M1 @ (F @ M2)
    
    return a_coeffs

@njit(c16[:, :, :, :](f4[:], f4[:], c16[:, :], i8), cache=True, fastmath=True, nogil=True, parallel=True)
def _int_mat_c16(x_ax, y_ax, zs, mode):
    Nx = len(x_ax)
    Ny = len(y_ax)
    a_coeffs = np.zeros((4,4,Nx-1,Ny-1), dtype=np.complex128)

    dzdx = 0*zs
    dzdy = 0*zs
    dzdxy = 0*zs
    # Compute the derivatives
    for j in prange(Ny-1):
        dzdx[:,j] = _diff_c16(x_ax,zs[:,j],mode)
    for i in prange(Nx-1):
        dzdy[i,:] = _diff_c16(y_ax,zs[i,:],mode)
    for i in prange(Nx-1):
        dzdxy[i,:] = _diff_c16(y_ax,dzdx[i,:],mode)

    M1 = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [-3.0, 3.0, -2.0, -1.0], [2.0, -2.0, 1.0, 1.0]]).astype(np.complex128)
    M2 = np.array([[1.0, 0.0, -3.0, 2.0], [0.0, 0.0, 3.0, -2.0], [0.0, 1.0, -2.0, 1.0], [0.0, 0.0, -1.0, 1.0]]).astype(np.complex128)

    for i in prange(Nx-1):
        for j in range(Ny-1):
            F = np.array([[zs[i,j], zs[i,j+1],dzdy[i,j], dzdy[i,j+1]],
                          [zs[i+1,j],zs[i+1,j+1],dzdy[i+1,j],dzdy[i+1,j+1]],
                          [dzdx[i,j], dzdx[i,j+1],dzdxy[i,j],dzdxy[i,j+1]],
                          [dzdx[i+1,j],dzdx[i+1,j+1],dzdxy[i+1,j], dzdxy[i+1,j+1]]])
            a_coeffs[:,:,i,j] = M1 @ (F @ M2)
    
    return a_coeffs

def compute_interpolator_matrix(x_ax, y_ax, zs, mode: intmode = intmode.CubicRunout):
    x_ax = x_ax.astype(np.float32)
    y_ax = y_ax.astype(np.float32)

    if zs.dtype == np.float32:
        return _int_mat_f4(x_ax, y_ax, zs, mode.value)
    elif zs.dtype == np.float64:
        return _int_mat_f8(x_ax, y_ax, zs, mode.value)
    elif zs.dtype == np.complex64:
        return _int_mat_c8(x_ax, y_ax, zs, mode.value)
    elif zs.dtype == np.complex128:
        return _int_mat_c16(x_ax, y_ax, zs, mode.value)
    else:
        raise ValueError("Invalid dtype for zs")    
    

@njit(f4[:](f4[:], f4[:], f4[:], f4[:], f4[:,:,:,:]), cache=True, fastmath=True, nogil=True)
def c_interpolator_f4(x, y, xgrid, ygrid, interp_matrix):
    Nx = len(xgrid)
    Ny = len(ygrid)
    minx = xgrid[0]
    miny = ygrid[0]
    dx = xgrid[1]-xgrid[0]
    dy = ygrid[1]-ygrid[0]

    N = len(x)
    out = np.zeros_like(x).astype(np.float32)
    for ix in range(N):
        xl = x[ix]
        yl = y[ix]
        i = int(np.floor((xl - minx) / dx))
        i = np.minimum(np.maximum(i, 0), Nx - 2)
        j = int(np.floor((yl - miny) / dy))
        j = np.minimum(np.maximum(j, 0), Ny - 2)

        A = interp_matrix[:, :, i, j]

        xi = (xl - xgrid[i])/dx
        yi = (yl - ygrid[j])/dy
        
        ysq = yi**2.0
        xsq = xi**2.0
        yqb = yi**3.0
        xqb = xi**3.0

        M1 = A[0,0] + A[0,1]*yi + A[0,2]*ysq + A[0,3]*yqb
        M2 = A[1,0] + A[1,1]*yi + A[1,2]*ysq + A[1,3]*yqb
        M3 = A[2,0] + A[2,1]*yi + A[2,2]*ysq + A[2,3]*yqb
        M4 = A[3,0] + A[3,1]*yi + A[3,2]*ysq + A[3,3]*yqb

        out[ix] = M1 + M2*xi + M3*xsq + M4*xqb
    return out

@njit(f8[:](f4[:], f4[:], f4[:], f4[:], f8[:,:,:,:]), cache=True, fastmath=True, nogil=True)
def c_interpolator_f8(x, y, xgrid, ygrid, interp_matrix):
    Nx = len(xgrid)
    Ny = len(ygrid)
    minx = xgrid[0]
    miny = ygrid[0]
    dx = xgrid[1]-xgrid[0]
    dy = ygrid[1]-ygrid[0]

    N = len(x)
    out = np.zeros_like(x).astype(np.float64)
    for ix in range(N):
        xl = x[ix]
        yl = y[ix]
        i = int(np.floor((xl - minx) / dx))
        i = np.minimum(np.maximum(i, 0), Nx - 2)
        j = int(np.floor((yl - miny) / dy))
        j = np.minimum(np.maximum(j, 0), Ny - 2)

        A = interp_matrix[:, :, i, j]

        xi = (xl - xgrid[i])/dx
        yi = (yl - ygrid[j])/dy
        
        ysq = yi**2.0
        xsq = xi**2.0
        yqb = yi**3.0
        xqb = xi**3.0

        M1 = A[0,0] + A[0,1]*yi + A[0,2]*ysq + A[0,3]*yqb
        M2 = A[1,0] + A[1,1]*yi + A[1,2]*ysq + A[1,3]*yqb
        M3 = A[2,0] + A[2,1]*yi + A[2,2]*ysq + A[2,3]*yqb
        M4 = A[3,0] + A[3,1]*yi + A[3,2]*ysq + A[3,3]*yqb

        out[ix] = M1 + M2*xi + M3*xsq + M4*xqb
    return out

@njit(c8[:](f4[:], f4[:], f4[:], f4[:], c8[:,:,:,:]), cache=True, fastmath=True, nogil=True)
def c_interpolator_c8(x, y, xgrid, ygrid, interp_matrix):
    Nx = len(xgrid)
    Ny = len(ygrid)
    minx = xgrid[0]
    miny = ygrid[0]
    dx = xgrid[1]-xgrid[0]
    dy = ygrid[1]-ygrid[0]

    N = len(x)
    out = np.zeros_like(x).astype(np.complex64)
    for ix in range(N):
        xl = x[ix]
        yl = y[ix]
        i = int(np.floor((xl - minx) / dx))
        i = np.minimum(np.maximum(i, 0), Nx - 2)
        j = int(np.floor((yl - miny) / dy))
        j = np.minimum(np.maximum(j, 0), Ny - 2)

        A = interp_matrix[:, :, i, j]

        xi = (xl - xgrid[i])/dx
        yi = (yl - ygrid[j])/dy
        
        ysq = yi**2.0
        xsq = xi**2.0
        yqb = yi**3.0
        xqb = xi**3.0

        M1 = A[0,0] + A[0,1]*yi + A[0,2]*ysq + A[0,3]*yqb
        M2 = A[1,0] + A[1,1]*yi + A[1,2]*ysq + A[1,3]*yqb
        M3 = A[2,0] + A[2,1]*yi + A[2,2]*ysq + A[2,3]*yqb
        M4 = A[3,0] + A[3,1]*yi + A[3,2]*ysq + A[3,3]*yqb

        out[ix] = M1 + M2*xi + M3*xsq + M4*xqb
    return out

@njit(c16[:](f4[:], f4[:], f4[:], f4[:], c16[:,:,:,:]), cache=True, fastmath=True, nogil=True)
def c_interpolator_c16(x, y, xgrid, ygrid, interp_matrix):
    Nx = len(xgrid)
    Ny = len(ygrid)
    minx = xgrid[0]
    miny = ygrid[0]
    dx = xgrid[1]-xgrid[0]
    dy = ygrid[1]-ygrid[0]

    N = len(x)
    out = np.zeros_like(x).astype(np.complex128)
    for ix in range(N):
        xl = x[ix]
        yl = y[ix]
        i = int(np.floor((xl - minx) / dx))
        i = np.minimum(np.maximum(i, 0), Nx - 2)
        j = int(np.floor((yl - miny) / dy))
        j = np.minimum(np.maximum(j, 0), Ny - 2)

        A = interp_matrix[:, :, i, j]

        xi = (xl - xgrid[i])/dx
        yi = (yl - ygrid[j])/dy
        
        ysq = yi**2.0
        xsq = xi**2.0
        yqb = yi**3.0
        xqb = xi**3.0

        M1 = A[0,0] + A[0,1]*yi + A[0,2]*ysq + A[0,3]*yqb
        M2 = A[1,0] + A[1,1]*yi + A[1,2]*ysq + A[1,3]*yqb
        M3 = A[2,0] + A[2,1]*yi + A[2,2]*ysq + A[2,3]*yqb
        M4 = A[3,0] + A[3,1]*yi + A[3,2]*ysq + A[3,3]*yqb

        out[ix] = M1 + M2*xi + M3*xsq + M4*xqb
    return out

def create_interpolator(x_ax: np.ndarray, y_ax: np.ndarray, zs: np.ndarray, mode: intmode = intmode.CubicRunout):
    Nx = len(x_ax)
    Ny = len(y_ax)
    minx = min(x_ax.flatten())
    miny = min(y_ax.flatten())
    dx = x_ax[1]-x_ax[0]
    dy = y_ax[1]-y_ax[0]

    a_coeffs = compute_interpolator_matrix(x_ax,y_ax,zs,mode=mode)

    return_type_dict = {
        np.dtype('float32'): f4[:],
        np.dtype('float64'): f8[:],
        np.dtype('complex64'): c8[:],
        np.dtype('complex128'): c16[:],
    }

    dtype = return_type_dict[zs.dtype]

    @njit(dtype(f4[:], f4[:]), fastmath=True, nogil=True)
    def _c_interpolator(x, y):
        N = len(x)
        out = np.zeros_like(x).astype(zs.dtype)
        for ix in range(N):
            xl = x[ix]
            yl = y[ix]
            i = int(np.floor((xl - minx) / dx))
            i = np.minimum(np.maximum(i, 0), Nx - 2)
            j = int(np.floor((yl - miny) / dy))
            j = np.minimum(np.maximum(j, 0), Ny - 2)

            A = a_coeffs[:, :, i, j]

            xi = (xl - x_ax[i])/dx
            yi = (yl - y_ax[j])/dy
            
            ysq = yi**2.0
            xsq = xi**2.0
            yqb = yi**3.0
            xqb = xi**3.0

            M1 = A[0,0] + A[0,1]*yi + A[0,2]*ysq + A[0,3]*yqb
            M2 = A[1,0] + A[1,1]*yi + A[1,2]*ysq + A[1,3]*yqb
            M3 = A[2,0] + A[2,1]*yi + A[2,2]*ysq + A[2,3]*yqb
            M4 = A[3,0] + A[3,1]*yi + A[3,2]*ysq + A[3,3]*yqb

            out[ix] = M1 + M2*xi + M3*xsq + M4*xqb
        return out
    
    def interpolator(_x,_y):
        return _c_interpolator(_x.astype(np.float32).flatten(), _y.astype(np.float32).flatten()).reshape(_x.shape)
    
    return interpolator