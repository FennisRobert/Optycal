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

from .interpolator import compute_interpolator_matrix, c_interpolator_c8, c_interpolator_c16
from ..settings import GLOBAL_SETTINGS, Precision
import numpy as np
from typing import Callable
from numba import njit, f8, c8, f4, c16
from numba.types import Tuple
from loguru import logger

@njit(Tuple((c8[:],c8[:],c8[:],c8[:],c8[:],c8[:]))(f4[:], f4[:], f4[:], f4[:], c8[:,:,:,:,:]),cache=True, fastmath=True, parallel=False)
def pattern_interp_single(theta, phi, theta_grid, phi_grid, A_matrices):
    ex = c_interpolator_c8(theta, phi, theta_grid, phi_grid, A_matrices[0,:,:,:,:])
    ey = c_interpolator_c8(theta, phi, theta_grid, phi_grid, A_matrices[1,:,:,:,:])
    ez = c_interpolator_c8(theta, phi, theta_grid, phi_grid, A_matrices[2,:,:,:,:])
    hx = c_interpolator_c8(theta, phi, theta_grid, phi_grid, A_matrices[3,:,:,:,:])
    hy = c_interpolator_c8(theta, phi, theta_grid, phi_grid, A_matrices[4,:,:,:,:])
    hz = c_interpolator_c8(theta, phi, theta_grid, phi_grid, A_matrices[5,:,:,:,:])

    return (ex, ey, ez, hx, hy, hz)

@njit(Tuple((c16[:],c16[:],c16[:],c16[:],c16[:],c16[:]))(f4[:], f4[:], f4[:], f4[:], c16[:,:,:,:,:]),cache=True, fastmath=True, parallel=False)
def pattern_interp_double(theta, phi, theta_grid, phi_grid, A_matrices):
    ex = c_interpolator_c16(theta, phi, theta_grid, phi_grid, A_matrices[0,:,:,:,:])
    ey = c_interpolator_c16(theta, phi, theta_grid, phi_grid, A_matrices[1,:,:,:,:])
    ez = c_interpolator_c16(theta, phi, theta_grid, phi_grid, A_matrices[2,:,:,:,:])
    hx = c_interpolator_c16(theta, phi, theta_grid, phi_grid, A_matrices[3,:,:,:,:])
    hy = c_interpolator_c16(theta, phi, theta_grid, phi_grid, A_matrices[4,:,:,:,:])
    hz = c_interpolator_c16(theta, phi, theta_grid, phi_grid, A_matrices[5,:,:,:,:])

    return (ex, ey, ez, hx, hy, hz)

class AntennaPattern:

    def __init__(self):
        self.Aex: np.ndarray = None
        self.Aey: np.ndarray = None
        self.Aez: np.ndarray = None
        self.Ahx: np.ndarray = None
        self.Ahy: np.ndarray = None
        self.Ahz: np.ndarray = None
        self.theta_grid: np.ndarray = None
        self.phi_grid: np.ndarray = None

    def full_matrix(self, precision: Precision = None) -> np.ndarray:
        if precision is None:
            precision = GLOBAL_SETTINGS.precision
        if precision == Precision.SINGLE:
            A_full = np.zeros((6, 4,4,self.theta_grid.shape[0]-1, self.phi_grid.shape[0]-1), dtype=np.complex64)
            A_full[0,:,:,:,:] = self.Aex.astype(np.complex64)
            A_full[1,:,:,:,:] = self.Aey.astype(np.complex64)
            A_full[2,:,:,:,:] = self.Aez.astype(np.complex64)
            A_full[3,:,:,:,:] = self.Ahx.astype(np.complex64)
            A_full[4,:,:,:,:] = self.Ahy.astype(np.complex64)
            A_full[5,:,:,:,:] = self.Ahz.astype(np.complex64)
        elif precision == Precision.DOUBLE:
            A_full = np.zeros((6, 4,4,self.theta_grid.shape[0]-1, self.phi_grid.shape[0]-1), dtype=np.complex128)
            A_full[0,:,:,:,:] = self.Aex.astype(np.complex128)
            A_full[1,:,:,:,:] = self.Aey.astype(np.complex128)
            A_full[2,:,:,:,:] = self.Aez.astype(np.complex128)
            A_full[3,:,:,:,:] = self.Ahx.astype(np.complex128)
            A_full[4,:,:,:,:] = self.Ahy.astype(np.complex128)
            A_full[5,:,:,:,:] = self.Ahz.astype(np.complex128)
        return A_full
    
    @staticmethod
    def from_function(ff_function: Callable, theta_grid: np.ndarray, phi_grid: np.ndarray, k0: float):
        """
        Creates an AntennaPattern object from the given near-field and far-field functions.
        """
        antenna = AntennaPattern()
        antenna.theta_grid = theta_grid.astype(np.float32)
        antenna.phi_grid = phi_grid.astype(np.float32)

        TH, PH = np.meshgrid(theta_grid, phi_grid, indexing='ij')

        Ex, Ey, Ez, Hx, Hy, Hz = ff_function(TH.flatten().astype(np.float32), PH.flatten().astype(np.float32), k0)

        Ex = Ex.reshape(TH.shape)
        Ey = Ey.reshape(TH.shape)
        Ez = Ez.reshape(TH.shape)
        Hx = Hx.reshape(TH.shape)
        Hy = Hy.reshape(TH.shape)
        Hz = Hz.reshape(TH.shape)

        logger.debug('Creating interpolation matrices')
        antenna.Aex = compute_interpolator_matrix(theta_grid, phi_grid, Ex)
        antenna.Aey = compute_interpolator_matrix(theta_grid, phi_grid, Ey)
        antenna.Aez = compute_interpolator_matrix(theta_grid, phi_grid, Ez)
        antenna.Ahx = compute_interpolator_matrix(theta_grid, phi_grid, Hx)
        antenna.Ahy = compute_interpolator_matrix(theta_grid, phi_grid, Hy)
        antenna.Ahz = compute_interpolator_matrix(theta_grid, phi_grid, Hz)
        logger.debug('Interpolation matrices created')
        return antenna
    
    def nf_pattern(self) -> Callable:
        """
        Returns a callable that evaluates the near-field pattern of the antenna.
        """
        if GLOBAL_SETTINGS.precision == Precision.SINGLE:
            func = c_interpolator_c8
        elif GLOBAL_SETTINGS.precision == Precision.DOUBLE:
            func = c_interpolator_c16

        def pattern(theta, phi, r, k0):
            """
            Evaluates the near-field pattern of the antenna at the given coordinates.
            """
            ex = func(theta, phi, self.theta_grid, self.phi_grid, self.Aex)
            ey = func(theta, phi, self.theta_grid, self.phi_grid, self.Aey)
            ez = func(theta, phi, self.theta_grid, self.phi_grid, self.Aez)
            hx = func(theta, phi, self.theta_grid, self.phi_grid, self.Ahx)
            hy = func(theta, phi, self.theta_grid, self.phi_grid, self.Ahy)
            hz = func(theta, phi, self.theta_grid, self.phi_grid, self.Ahz)

            return (ex, ey, ez, hx, hy, hz)
        return pattern

    def ff_pattern(self) -> Callable:
        """
        Returns a callable that evaluates the far-field pattern of the antenna.
        """
        if GLOBAL_SETTINGS.precision == Precision.SINGLE:
            func = c_interpolator_c8
        elif GLOBAL_SETTINGS.precision == Precision.DOUBLE:
            func = c_interpolator_c16
        
        def pattern(theta, phi, k0):
            """
            Evaluates the far-field pattern of the antenna at the given coordinates.
            """
            ex = func(theta, phi, self.theta_grid, self.phi_grid, self.Aex)
            ey = func(theta, phi, self.theta_grid, self.phi_grid, self.Aey)
            ez = func(theta, phi, self.theta_grid, self.phi_grid, self.Aez)
            hx = func(theta, phi, self.theta_grid, self.phi_grid, self.Ahx)
            hy = func(theta, phi, self.theta_grid, self.phi_grid, self.Ahy)
            hz = func(theta, phi, self.theta_grid, self.phi_grid, self.Ahz)

            return (ex, ey, ez, hx, hy, hz)
        return pattern
    