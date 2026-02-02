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
import numpy as np
import pyvista as pv
from typing import Literal
from ..geo.mesh import Mesh
from ..surface import Surface
from emsutil.pyvista import EMergeDisplay, cmap_names, setdefault, _AnimObject

from ..antennas.antenna import Antenna
from ..antennas.array import AntennaArray

def _do_nothing(*args, **kwargs):
    return "not implemented"

class OptycalDisplay(EMergeDisplay):

    def __post_init__(self):
        self._selector._set_encoder_function(_do_nothing)
        self._edge_length = 1.0
        
    def _get_edge_length(self):
        return self._edge_length
    
    def _ug_from_mesh(self, mesh: Mesh) -> pv.UnstructuredGrid:
        ntris = mesh.triangles.shape[1]
        cells = np.zeros((ntris,4), dtype=np.int64)
        cells[:,1:] = mesh.triangles.T
        cells[:,0] = 3
        celltypes = np.full(ntris, fill_value=pv.CellType.TRIANGLE, dtype=np.uint8)
        points = mesh.g.vertices.T
        grid = pv.UnstructuredGrid(cells, celltypes, points)
        return grid
    
    ## CUSTOM METHODS
    
    def add_mesh_object(self, mesh: Mesh, color: str = "#aaaaaa", opacity = 1.0) -> pv.UnstructuredGrid:
        """Adds a mesh object to the plot.

        Args:
            mesh (Mesh): The mesh to add.
            color (str, optional): The color of the mesh. Defaults to "#aaaaaa".
            opacity (float, optional): The opacity of the mesh. Defaults to 1.0.

        Returns:
            pv.UnstructuredGrid: The added mesh object.
        """
        grid = self._ug_from_mesh(mesh)
        self._plot.add_mesh(grid, pickable=True, color=color, opacity=opacity, show_edges=True)
    
    def add_surface_object(self, surf: Surface, field: str = None, quantity: Literal['real','imag','abs'] = 'abs', opacity: float = None) -> pv.UnstructuredGrid:
        """Adds a surface object to the plot.

        Field options: ['E', 'H', 'Ex','Sy','normE', etc.]
        Args:
            surf (Surface): The surface to add.
            field (str, optional): The field to visualize. Defaults to None.
            quantity (Literal['real','imag','abs'], optional): The quantity to visualize. Defaults to 'abs'.
            opacity (float, optional): The opacity of the surface. Defaults to None.

        Returns:
            pv.UnstructuredGrid: The added surface object.
        """
        
        surf1, surf2 = surf._get_side_surfs()
        g1 = self._ug_from_mesh(surf1.mesh)
        f1 = surf1.fresnel.mat1
        self._add_obj(g1, 2, plot_mesh=False,
                    volume_mesh=False,
                    metal=f1._metal,
                    color=f1._color_rgb,
                    opacity=f1.opacity)
        if surf2 is not None:
            g2 = self._ug_from_mesh(surf2.mesh)
            f2 = surf2.fresnel.mat2
            self._add_obj(g2, 2, plot_mesh=False,
                        volume_mesh=False,
                        metal=f2._metal,
                        color=f2._color_rgb,
                        opacity=f2.opacity)
        return
        
    def add_antenna_object(self, antenna: Antenna, color: Literal['none','amp','phase'] = 'amp'):
        """Adds an antenna object to the plot.

        Args:
            antenna (Antenna): The antenna to add.
            color (Literal['none','amp','phase'], optional): The color of the antenna. Defaults to 'amp'.
        """
        x, y, z = antenna.gxyz
        pc = np.array([x,y,z])
        pol = antenna.cs.gzhat
        length = 0.5* 299792458/antenna.frequency
        ant = pv.Arrow(pc-pol*length/2,pol, scale=length)
        
        self._plot.add_mesh(ant, scalars=abs(antenna.amplitude)*np.ones((ant.n_cells,)))

    def add_array_object(self, array: AntennaArray, color: Literal['none','amp','phase'] = 'amp'):
        for ant in array.antennas:
            self.add_antenna_object(ant)

    def add(self, *objects: Surface | Mesh | Antenna | AntennaArray, **kwargs):
        """Adds one or more objects to the plot.
        """
        for obj in objects:
            if isinstance(obj, Surface):
                self.add_surface_object(obj, **kwargs)
            elif isinstance(obj, Mesh):
                self.add_mesh_object(obj, **kwargs)
            elif isinstance(obj, Antenna):
                self.add_antenna_object(obj, **kwargs)
            elif isinstance(obj, AntennaArray):
                self.add_array_object(obj, **kwargs)

    def add_surf(self, 
                 x: np.ndarray,
                 y: np.ndarray,
                 z: np.ndarray,
                 field: np.ndarray,
                 scale: Literal['lin','log','symlog'] = 'lin',
                 cmap: cmap_names | None = None,
                 clim: tuple[float, float] | None = None,
                 opacity: float = 1.0,
                 symmetrize: bool = False,
                 _fieldname: str | None = None,
                 **kwargs,):
        """Add a surface plot to the display
        The X,Y,Z coordinates must be a 2D grid of data points. The field must be a real field with the same size.

        Args:
            x (np.ndarray): The X-grid array
            y (np.ndarray): The Y-grid array
            z (np.ndarray): The Z-grid array
            field (np.ndarray): The scalar field to display
            scale (Literal["lin","log","symlog"], optional): The colormap scaling¹. Defaults to 'lin'.
            cmap (cmap_names, optional): The colormap. Defaults to 'coolwarm'.
            clim (tuple[float, float], optional): Specific color limits (min, max). Defaults to None.
            opacity (float, optional): The opacity of the surface. Defaults to 1.0.
            symmetrize (bool, optional): Wether to force a symmetrical color limit (-A,A). Defaults to True.
        
        (¹): lin: f(x)=x, log: f(x)=log₁₀(|x|), symlog: f(x)=sgn(x)·log₁₀(1+|x·ln(10)|)
        """
        
        grid = pv.StructuredGrid(x,y,z)
        field_flat = field.flatten(order='F')
        
        if scale=='log':
            T = lambda x: np.log10(np.abs(x+1e-12))
        elif scale=='symlog':
            T = lambda x: np.sign(x) * np.log10(1 + np.abs(x*np.log(10)))
        else:
            T = lambda x: x
        
        static_field = T(np.real(field_flat))
        
        if _fieldname is None:
            name = 'anim'+str(self._ctr)
        else:
            name = _fieldname
        self._ctr += 1
        
        grid[name] = static_field

        grid_no_nan = grid.threshold(scalars=name)
        
        default_cmap = self.set
        # Determine color limits
        if clim is None:
            if self._cbar_lim is not None:
                clim = self._cbar_lim
            else:
                fmin = np.nanmin(static_field)
                fmax = np.nanmax(static_field)
                clim = (fmin, fmax)
        
        if symmetrize:
            lim = max(abs(clim[0]), abs(clim[1]))
            clim = (-lim, lim)
            default_cmap = self.set.theme.default_wave_cmap
        
        if cmap is None:
            cmap = default_cmap
        
        kwargs = setdefault(kwargs, cmap=cmap, clim=clim, opacity=opacity, pickable=False, multi_colors=True)
        actor = self._plot.add_mesh(grid_no_nan, scalars=name, scalar_bar_args=self._cbar_args, **kwargs)


        if self._animate_next:
            def on_update(obj: _AnimObject, phi: complex):
                field_anim = obj.T(np.real(obj.field * phi))
                obj.grid[name] = field_anim
                obj.fgrid[name] = obj.grid.threshold(scalars=name)[name]
                #obj.fgrid replace with thresholded scalar data.
            self._objs.append(_AnimObject(field_flat, T, grid, grid_no_nan, actor, on_update))
            self._animate_next = False
        self._reset_cbar()