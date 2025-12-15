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
from numba import njit, vectorize, f8, c16, f8
from numba.types import Tuple as TypeTuple

Z0 = np.float64(376.73031366857)
Y0 = np.float64(1 / Z0)
Eiso = np.float64(np.sqrt(Z0 / (2 * np.pi)))
Eomni = np.float64(np.sqrt(3 * Z0 / (4 * np.pi)))

@njit(cache=True, fastmath=True, parallel=True, nogil=True)
def dipole_pattern_ff(theta: np.ndarray, phi: np.ndarray, k0) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    theta = np.pi/2 - theta
    cst = np.cos(theta)
    csp = np.cos(phi)
    snp = np.sin(phi)
    A = (Eomni * 0.5 * np.sin(2 * theta))
    
    ex = -A * csp
    ey = -A * snp
    ez = 0.5 * Eomni * (np.cos(2 * theta) + 1.0)

    hx = Eomni * Y0 * cst * snp
    hy = -Eomni * Y0 * cst * csp
    hz = 0 * hx

    return ex, ey, ez, hx, hy, hz

@njit(cache=True, fastmath=True, parallel=True, nogil=True)
def dipole_pattern_nf(theta, phi, r, k0) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    theta = np.pi/2 - theta
    cst = np.cos(theta)
    snt = np.sin(theta)
    csp = np.cos(phi)
    snp = np.sin(phi)

    A = Eomni

    F = 1 / (np.complex64(1j) * k0 * r)
    QRr = F + F**2
    QRt = 1 + QRr

    ex = A * (-QRt * snt * csp * cst - QRr * cst * csp * 2 * snt)
    ey = A * (-QRt * snt * snp * cst - QRr * cst * snp * 2 * snt)
    ez = A * (QRt * cst * cst - QRr * 2 * snt**2)

    hx = (1 + F) * A * Y0 * cst * snp
    hy = -(1 + F) * A * Y0 * cst * csp
    hz = 0 * hx

    return ex, ey, ez, hx, hy, hz

def generate_gaussian_pattern(atangle, attenuation, k0):
    
    b = (attenuation - 20*np.log10(1+np.cos(atangle)))/(k0*np.cos(atangle)*20*np.log10(np.e))
    def ffp(theta: np.ndarray, phi: np.ndarray, k0) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        theta = np.pi/2 - theta
        cst = np.cos(theta)
        csp = np.cos(phi)
        snp = np.sin(phi)
        A = (Eomni * 0.5 * np.sin(2 * theta))
        
        tpr = np.exp(k0*b*cst*csp)
        ex = -A * csp * tpr
        ey = -A * snp * tpr
        ez = 0.5 * Eomni * (np.cos(2 * theta) + 1.0) * tpr

        hx = Eomni * Y0 * cst * snp * tpr
        hy = -Eomni * Y0 * cst * csp * tpr
        hz = 0 * hx * tpr

        return (ex, ey, ez, hx, hy, hz)

    #@njit(TypeTuple((c16[:], c16[:], c16[:], c16[:], c16[:], c16[:]))(f8[:], f8[:], f8[:], f8), cache=True, fastmath=True, parallel=True, nogil=True)
    def nfp(theta, phi, r, k0) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        theta = np.pi/2 - theta
        cst = np.cos(theta)
        snt = np.sin(theta)
        csp = np.cos(phi)
        snp = np.sin(phi)

        tpr = np.exp(k0*b*cst*csp)
        A = Eomni * tpr

        F = 1 / (np.complex64(1j) * k0 * r)
        QRr = F + F**2
        QRt = 1 + QRr

        ex = A * (-QRt * snt * csp * cst - QRr * cst * csp * 2 * snt)
        ey = A * (-QRt * snt * snp * cst - QRr * cst * snp * 2 * snt)
        ez = A * (QRt * cst * cst - QRr * 2 * snt**2)

        hx = (1 + F) * A * Y0 * cst * snp
        hy = -(1 + F) * A * Y0 * cst * csp
        hz = 0 * hx

        return (ex, ey, ez, hx, hy, hz)

    return ffp, nfp

def generate_gaussian_pattern_z(atangle, attenuation, k0):
    
    b = (20*np.log10((1+np.cos(atangle))/2)-attenuation)/(20*k0*(1-np.cos(atangle))*np.log10(np.e))
    def ffp(theta: np.ndarray, phi: np.ndarray, k0: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        theta = np.pi/2 - theta
        rz = np.sin(theta)
        ry = np.cos(theta)*np.sin(phi)+1e-20*np.random.rand(*theta.shape)
        rx = np.cos(theta)*np.cos(phi)+1e-20*np.random.rand(*theta.shape)
        A = (1/k0)*np.exp(k0*b*rx)
        ex = -A*rz*(rx + 1)
        ey = A*ry*rz*(1 - rx**2)/(ry**2 + rz**2)
        ez = A*(rx + 1)*(rx*rz**2 + ry**2)/(ry**2 + rz**2)

        hx = Y0 * (ry * ez - rz * ey)
        hy = Y0 * (rz * ex - rx * ez)
        hz = Y0 * (rx * ey - ry * ex)

        return (ex, ey, ez, hx, hy, hz)

    #@njit(TypeTuple((c16[:], c16[:], c16[:], c16[:], c16[:], c16[:]))(f8[:], f8[:], f8[:], f8), cache=True, fastmath=True, parallel=True, nogil=True)
    def nfp(theta, phi, r, k0) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        theta = np.pi/2 - theta
        rz = np.sin(theta)
        ry = np.cos(theta)*np.sin(phi)+1e-20*np.random.rand(*theta.shape)
        rx = np.cos(theta)*np.cos(phi)+1e-20*np.random.rand(*theta.shape)
        A = (1/k0)*np.exp(k0*b*rx)
        ex = -A*rz*(rx + 1)
        ey = A*ry*rz*(1 - rx**2)/(ry**2 + rz**2)
        ez = A*(rx + 1)*(rx*rz**2 + ry**2)/(ry**2 + rz**2)

        hx = Y0 * (ry * ez - rz * ey)
        hy = Y0 * (rz * ex - rx * ez)
        hz = Y0 * (rx * ey - ry * ex)

        return (ex, ey, ez, hx, hy, hz)

    return ffp, nfp

@njit(cache=True, fastmath=True, parallel=True, nogil=True)
def half_dipole_pattern_ff(theta: np.ndarray, phi: np.ndarray, k0) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    theta = np.pi/2 - theta
    cst = np.cos(theta)
    csp = np.cos(phi)
    snp = np.sin(phi)
    Active = np.abs(phi) < np.pi / 2
    
    A = (Eomni * 0.5 * np.sin(2 * theta))
    
    ex = -A * csp * Active
    ey = -A * snp * Active
    ez = 0.5 * Eomni * (np.cos(2 * theta) + 1.0) * Active

    hx = Eomni * Y0 * cst * snp * Active
    hy = -Eomni * Y0 * cst * csp * Active
    hz = 0 * hx * Active

    return ex, ey, ez, hx, hy, hz


@njit(cache=True, fastmath=True, parallel=True, nogil=True)
def half_dipole_pattern_nf(theta, phi, r, k0) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    theta = np.pi/2 - theta
    cst = np.cos(theta)
    snt = np.sin(theta)
    csp = np.cos(phi)
    snp = np.sin(phi)
    
    Active = np.abs(phi) < np.pi / 2
    A = Eomni*Active

    F = 1 / (np.complex64(1j) * k0 * r)
    QRr = F + F**2
    QRt = 1 + QRr

    ex = A * (-QRt * snt * csp * cst - QRr * cst * csp * 2 * snt)
    ey = A * (-QRt * snt * snp * cst - QRr * cst * snp * 2 * snt)
    ez = A * (QRt * cst * cst - QRr * 2 * snt**2)

    hx = (1 + F) * A * Y0 * cst * snp
    hy = -(1 + F) * A * Y0 * cst * csp
    hz = 0 * hx

    return ex, ey, ez, hx, hy, hz



@njit(cache=True, fastmath=True, parallel=False, nogil=True)
def dipole_pattern_nf_kxyz(kx, ky, kz, r, k0):

    A = Eomni

    F = 1 / (np.complex64(1j) * k0 * r)
    QRr = F + F**2
    QRt = 1 + QRr

    ex = A * (-QRt * kz * kx - QRr * kx * 2 * kz)
    ey = A * (-QRt * kz * ky - QRr * ky * 2 * kz)
    ez = A * (QRt * (kx**2 + ky**2) - QRr * 2 * kz**2)

    hx = (1 + F) * A * Y0 * ky
    hy = -(1 + F) * A * Y0 * kx
    hz = np.zeros_like(hx, dtype=np.complex64)

    return ex, ey, ez, hx, hy, hz


@njit(cache=True, fastmath=True, parallel=False, nogil=True)
def dipole_pattern_ff_kxyz(kx, ky, kz, k0) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

    A = Eomni

    ex = A * (-kz * kx )
    ey = A * (-kz * ky )
    ez = A * (kx**2 + ky**2)

    hx =  A * Y0 * ky
    hy = - A * Y0 * kx
    hz = np.zeros_like(hx, dtype=np.float64)

    return ex, ey, ez, hx, hy, hz



@njit(cache=True, fastmath=True, parallel=True, nogil=True)
def patch_pattern_nf(theta, phi, r, k0) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    theta = np.pi/2 - theta
    cst = np.cos(theta)
    snt = np.sin(theta)
    csp = np.cos(phi)
    snp = np.sin(phi)
    rx = csp * cst
    ry = snp * cst
    rz = snt

    T = np.sqrt(np.abs(cst * csp))
    Active = np.abs(phi) < np.pi / 2

    ex = -np.cos(np.pi / 2 * snt) * snt * np.sinc(np.pi / 2 * snp) * Active * T
    ey = np.zeros(ex.shape)
    ez = np.cos(np.pi / 2 * snt) * (csp * cst) * np.sinc(np.pi / 2 * snp) * Active * T

    hx = Y0 * (ry * ez - rz * ey)
    hy = Y0 * (rz * ex - rx * ez)
    hz = Y0 * (rx * ey - ry * ex)

    return ex, ey, ez, hx, hy, hz



@njit(cache=True, fastmath=True, parallel=True, nogil=True)
def patch_pattern_ff(theta, phi, k0) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    theta = np.pi/2 - theta
    cst = np.cos(theta)
    snt = np.sin(theta)
    csp = np.cos(phi)
    snp = np.sin(phi)
    rx = csp * cst
    ry = snp * cst
    rz = snt

    T = np.sqrt(np.abs(cst * csp))
    Active = np.abs(phi) < np.pi / 2

    ex = -np.cos(np.pi / 2 * snt) * snt * np.sinc(np.pi / 2 * snp) * Active * T
    ey = np.zeros(ex.shape)
    ez = np.cos(np.pi / 2 * snt) * (csp * cst) * np.sinc(np.pi / 2 * snp) * Active * T

    hx = Y0 * (ry * ez - rz * ey)
    hy = Y0 * (rz * ex - rx * ez)
    hz = Y0 * (rx * ey - ry * ex)

    return ex, ey, ez, hx, hy, hz


def generate_triang_pattern(Gain_peak, Gain_level, angle_level, curve_radius_angle, k0):
    DG = Gain_level-Gain_peak
    ang = angle_level * np.pi/180
    curve_factor = curve_radius_angle*np.pi/180
    factor = -DG/(ang**2) *np.sqrt(1 + (ang/curve_factor)**2)

    @njit(cache=True, fastmath=True, parallel=True, nogil=True)
    def _triang_pattern(theta, phi, r, k0) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        theta = np.pi/2 - theta
        cst = np.cos(theta)
        snt = np.sin(theta)
        csp = np.cos(phi)
        snp = np.sin(phi)
        rx = csp * cst
        ry = snp * cst
        rz = snt

        ux = cst*csp
        uy = cst*snp
        uz = snt
        off_angle = np.arctan2(np.sqrt(uy**2+uz**2), ux)
        
        etheta = 7.735*10**((-factor*(off_angle)**2/np.sqrt(1+(off_angle/curve_factor)**2) + Gain_peak) / 20)

        ex = -etheta*snt*csp
        ey = -etheta*snt*snp
        ez = etheta*cst

        hx = Y0 * (ry * ez - rz * ey)
        hy = Y0 * (rz * ex - rx * ez)
        hz = Y0 * (rx * ey - ry * ex)

        (ex, ey, ez, hx, hy, hz) = [x for x in (ex, ey, ez, hx, hy, hz)]
        return (ex, ey, ez, hx, hy, hz)


    @njit(cache=True, fastmath=True, parallel=True, nogil=True)
    def _triang_pattern_ff(theta, phi, k0) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        theta = np.pi/2 - theta
        cst = np.cos(theta)
        snt = np.sin(theta)
        csp = np.cos(phi)
        snp = np.sin(phi)
        rx = csp * cst
        ry = snp * cst
        rz = snt

        ux = cst*csp
        uy = cst*snp
        uz = snt
        off_angle = np.arctan2(np.sqrt(uy**2+uz**2), ux)
        
        etheta = 7.735*10**((-factor*(off_angle)**2/np.sqrt(1+(off_angle/curve_factor)**2) + Gain_peak) / 20)

        ex = -etheta*snt*csp
        ey = -etheta*snt*snp
        ez = etheta*cst

        hx = Y0 * (ry * ez - rz * ey)
        hy = Y0 * (rz * ex - rx * ez)
        hz = Y0 * (rx * ey - ry * ex)
        (ex, ey, ez, hx, hy, hz) = [x for x in (ex, ey, ez, hx, hy, hz)]
        return (ex, ey, ez, hx, hy, hz)

    return _triang_pattern, _triang_pattern_ff
    
def generate_patch_pattern(Width, Length, k0):
    kW = k0*Width
    kL = k0*Length
    @njit(cache=True, fastmath=True, parallel=True, nogil=True)
    def _patch_pattern(theta, phi, r, k0) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        theta = np.pi/2 - theta
        cst = np.cos(theta)
        snt = np.sin(theta)
        csp = np.cos(phi)
        snp = np.sin(phi)
        rx = csp * cst
        ry = snp * cst
        rz = snt

        T = (np.abs(cst * csp))**0.02
        Active = np.abs(phi) < np.pi / 2

        ex = -np.cos(kL / 2 * snt) * snt * np.sinc(kW / 2 * snp) * Active * T
        ey = np.zeros(ex.shape)
        ez = np.cos(kL / 2 * snt) * (csp * cst) * np.sinc(kW/ 2 * snp) * Active * T

        hx = Y0 * (ry * ez - rz * ey)
        hy = Y0 * (rz * ex - rx * ez)
        hz = Y0 * (rx * ey - ry * ex)

        return ex, ey, ez, hx, hy, hz



    @njit(cache=True, fastmath=True, parallel=True, nogil=True)
    def _patch_pattern_ff(theta, phi, k0) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        theta = np.pi/2 - theta
        cst = np.cos(theta)
        snt = np.sin(theta)
        csp = np.cos(phi)
        snp = np.sin(phi)
        rx = csp * cst
        ry = snp * cst
        rz = snt

        T = (np.abs(cst * csp))**0.02
        Active = np.abs(phi) < np.pi / 2

        ex = -np.cos(kL / 2 * snt) * snt * np.sinc(kW/ 2 * snp) * Active * T
        ey = np.zeros(ex.shape)
        ez = np.cos(kL / 2 * snt) * (csp * cst) * np.sinc(kW / 2 * snp) * Active * T

        hx = Y0 * (ry * ez - rz * ey)
        hy = Y0 * (rz * ex - rx * ez)
        hz = Y0 * (rx * ey - ry * ex)

        return ex, ey, ez, hx, hy, hz

    
    return _patch_pattern, _patch_pattern_ff