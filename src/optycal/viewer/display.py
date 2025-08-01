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
import time
import numpy as np
import pyvista as pv
from typing import Iterable, Literal, Callable
from .display_settings import PVDisplaySettings
from matplotlib.colors import ListedColormap
from ..geo.mesh import Mesh
from ..surface import Surface
from ..field import Field
from ..antennas.antenna import Antenna
from ..antennas.array import AntennaArray

# Define the colors we want to use
col1 = np.array([57, 179, 227, 255])/255
col2 = np.array([22, 36, 125, 255])/255
col3 = np.array([33, 33, 33, 255])/255
col4 = np.array([173, 76, 7, 255])/255
col5 = np.array([250, 75, 148, 255])/255

cmap_names = Literal['bgy','bgyw','kbc','blues','bmw','bmy','kgy','gray','dimgray','fire','kb','kg','kr',
                     'bkr','bky','coolwarm','gwv','bjy','bwy','cwr','colorwheel','isolum','rainbow','fire',
                     'cet_fire','gouldian','kbgyw','cwr','CET_CBL1','CET_CBL3','CET_D1A']

def gen_cmap(mesh, N: int = 256):
    # build a linear grid of data‐values (not strictly needed for pure colormap)
    vmin, vmax = mesh['values'].min(), mesh['values'].max()
    mapping = np.linspace(vmin, vmax, N)
    
    # prepare output
    newcolors = np.empty((N, 4))
    
    # normalized positions of control points: start, middle, end
    control_pos = np.array([0.0, 0.25, 0.5, 0.75, 1]) * (vmax - vmin) + vmin
    # stack control colors
    controls = np.vstack([col1, col2, col3, col4, col5])
    
    # interp each RGBA channel independently
    for chan in range(4):
        newcolors[:, chan] = np.interp(mapping, control_pos, controls[:, chan])
    
    return ListedColormap(newcolors)

def setdefault(options: dict, **kwargs) -> dict:
    """Shorthand for overwriting non-existent keyword arguments with defaults

    Args:
        options (dict): The kwargs dict

    Returns:
        dict: the kwargs dict
    """
    for key in kwargs.keys():
        if options.get(key,None) is None:
            options[key] = kwargs[key]
    return options

def _logscale(dx, dy, dz):
    """
    Logarithmically scales vector magnitudes so that the largest remains unchanged
    and others are scaled down logarithmically.
    
    Parameters:
        dx, dy, dz (np.ndarray): Components of vectors.
        
    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: Scaled dx, dy, dz arrays.
    """
    dx = np.asarray(dx)
    dy = np.asarray(dy)
    dz = np.asarray(dz)

    # Compute original magnitudes
    mags = np.sqrt(dx**2 + dy**2 + dz**2)
    mags_nonzero = np.where(mags == 0, 1e-10, mags)  # avoid log(0)

    # Logarithmic scaling (scaled to max = original max)
    log_mags = np.log10(mags_nonzero)
    log_min = np.min(log_mags)
    log_max = np.max(log_mags)

    if log_max == log_min:
        # All vectors have the same length
        return dx, dy, dz

    # Normalize log magnitudes to [0, 1]
    log_scaled = (log_mags - log_min) / (log_max - log_min)

    # Scale back to original max magnitude
    max_mag = np.max(mags)
    new_mags = log_scaled * max_mag

    # Compute unit vectors
    unit_dx = dx / mags_nonzero
    unit_dy = dy / mags_nonzero
    unit_dz = dz / mags_nonzero

    # Apply scaled magnitudes
    scaled_dx = unit_dx * new_mags
    scaled_dy = unit_dy * new_mags
    scaled_dz = unit_dz * new_mags

    return scaled_dx, scaled_dy, scaled_dz

def _min_distance(xs, ys, zs):
    """
    Compute the minimum Euclidean distance between any two points
    defined by the 1D arrays xs, ys, zs.
    
    Parameters:
        xs (np.ndarray): x-coordinates of the points
        ys (np.ndarray): y-coordinates of the points
        zs (np.ndarray): z-coordinates of the points
    
    Returns:
        float: The minimum Euclidean distance between any two points
    """
    # Stack the coordinates into a (N, 3) array
    points = np.stack((xs, ys, zs), axis=-1)

    # Compute pairwise squared distances using broadcasting
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists_squared = np.sum(diff ** 2, axis=-1)

    # Set diagonal to infinity to ignore zero distances to self
    np.fill_diagonal(dists_squared, np.inf)

    # Get the minimum distance
    min_dist = np.sqrt(np.min(dists_squared))
    return min_dist

def _norm(x, y, z):
    return np.sqrt(np.abs(x)**2 + np.abs(y)**2 + np.abs(z)**2)

class _AnimObject:
    """ A private class containing the required information for plot items in a view
    that can be animated.
    """
    def __init__(self, 
                 field: np.ndarray,
                 T: Callable,
                 grid: pv.Grid,
                 actor: pv.Actor,
                 on_update: Callable):
        self.field: np.ndarray = field
        self.T: Callable = T
        self.grid: pv.Grid = grid
        self.actor: pv.Actor = actor
        self.on_update: Callable = on_update

    def update(self, phi: complex):
        self.on_update(self, phi)

class OptycalDisplay:

    def __init__(self):
        self.set: PVDisplaySettings = PVDisplaySettings()
        
        # Animation options
        self._facetags: list[int] = []
        self._stop: bool = False
        self._objs: list[_AnimObject] = []
        self._do_animate: bool = False
        self._Nsteps: int = None
        self._fps: int = 25
        self._ruler: ScreenRuler = ScreenRuler(self, 0.01)
        self._stop = False
        self._objs = []

        self._plot = pv.Plotter()

        self._plot.add_key_event("m", self.activate_ruler)
        self._plot.add_key_event("f", self.activate_object)

        self._ctr: int = 0

    def activate_ruler(self):
        self._plot.disable_picking()
        self._selector.turn_off()
        self._ruler.toggle()

    def activate_object(self):
        self._plot.disable_picking()
        self._ruler.turn_off()
        self._selector.toggle()

    def show(self):
        """ Shows the Pyvista display. """
        self._ruler.min_length = 1e-3
        self._add_aux_items()
        if self._do_animate:
            self._plot.show(auto_close=False, interactive_update=True, before_close_callback=self._close_callback)
            self._animate()
        else:
            self._plot.show()
        self._reset()

    def _reset(self):
        """ Resets key display parameters."""
        self._plot.close()
        self._plot = pv.Plotter()
        self._stop = False
        self._objs = []

    def _close_callback(self):
        """The private callback function that stops the animation.
        """
        self._stop = True

    def _animate(self) -> None:
        """Private function that starts the animation loop.
        """
        self._plot.update()
        while not self._stop:
            for step in range(self._Nsteps):
                if self._stop:
                    break
                for aobj in self._objs:
                    phi = np.exp(1j*(step/self._Nsteps)*2*np.pi)
                    aobj.update(phi)
                self._plot.update()
                time.sleep(1/self._fps)

    def animate(self, Nsteps: int = 35, fps: int = 25) -> OptycalDisplay:
        """ Turns on the animation mode with the specified number of steps and FPS.

        All subsequent plot calls will automatically be animated. This method can be
        method chained.
        
        Args:
            Nsteps (int, optional): The number of frames in the loop. Defaults to 35.
            fps (int, optional): The number of frames per seocond, Defaults to 25

        Returns:
            PVDisplay: The same PVDisplay object

        Example:
        >>> display.animate().surf(...)
        >>> display.show()
        """
        self._Nsteps = Nsteps
        self._fps = fps
        self._do_animate = True
        return self
    
    ## CUSTOM METHODS
    def add_mesh_object(self, mesh: Mesh) -> pv.UnstructuredGrid:
        ntris = mesh.triangles.shape[1]
        cells = np.zeros((ntris,4), dtype=np.int64)
        cells[:,1:] = mesh.triangles.T
        cells[:,0] = 3
        celltypes = np.full(ntris, fill_value=pv.CellType.TRIANGLE, dtype=np.uint8)
        points = mesh.vertices.T
        grid = pv.UnstructuredGrid(cells, celltypes, points)
    
        self._plot.add_mesh(grid, pickable=True)
    
    def add_surface_object(self, surf: Surface, field: str = None, quantity: Literal['real','imag','abs'] = 'abs', opacity: float = None) -> pv.UnstructuredGrid:
        mesh = surf.mesh
        ntris = mesh.triangles.shape[1]
        cells = np.zeros((ntris,4), dtype=np.int64)
        cells[:,1:] = mesh.triangles.T
        cells[:,0] = 3
        celltypes = np.full(ntris, fill_value=pv.CellType.TRIANGLE, dtype=np.uint8)
        points = mesh.vertices.T
        grid = pv.UnstructuredGrid(cells, celltypes, points)

        field_obj = surf.vertex_field(0)
        
        if field is None:
            mat = surf.fresnel
            if opacity is None:
                opacity = mat.opacity
            self._plot.add_mesh(grid, color=mat.color, opacity=opacity, pickable=True)
            return
        else:
            scalars = getattr(field_obj, field)
            cmap = 'viridis'
            if opacity is None:
                opacity = 1.0
            if quantity=='real':
                scalars = scalars.real
                cmap = 'coolwarm'
                mv = np.max(np.abs(scalars))
                clim = (-mv, mv)
            elif quantity=='abs':
                scalars = np.abs(scalars)
                mv = np.max(np.abs(scalars))
                clim = (0, mv)
            elif quantity=='imag':
                scalars = scalars.imag
                cmap = 'coolwarm'
                mv = np.max(np.abs(scalars))
                clim = (-mv, mv)
            self._plot.add_mesh(grid, scalars=np.real(scalars), pickable=True, cmap=cmap, clim=clim, opacity=opacity)

    def add_antenna_object(self, antenna: Antenna, color: Literal['none','amp','phase'] = 'amp'):
        x, y, z = antenna.gxyz
        pc = np.array([x,y,z])
        pol = antenna.cs.gzhat
        length = 0.5* 299792458/antenna.frequency
        ant = pv.Arrow(pc-pol*length/2,pol, scale=length)
        
        self._plot.add_mesh(ant, scalars=abs(antenna.amplitude)*np.ones((ant.n_cells,)))

    def add_array_object(self, array: AntennaArray, color: Literal['none','amp','phase'] = 'amp'):
        for ant in array.antennas:
            self.add_antenna_object(ant)

    ## OBLIGATORY METHODS
    def add(self, *objects: Surface | Mesh | Antenna | AntennaArray, **kwargs):
        for obj in objects:
            if isinstance(obj, Surface):
                self.add_surface_object(obj, **kwargs)
            elif isinstance(obj, Mesh):
                self.add_mesh_object(obj, **kwargs)
            elif isinstance(obj, Antenna):
                self.add_antenna_object(obj, **kwargs)
            elif isinstance(obj, AntennaArray):
                self.add_array_object(obj, **kwargs)

    def add_scatter(self, xs: np.ndarray, ys: np.ndarray, zs: np.ndarray):
        """Adds a scatter point cloud

        Args:
            xs (np.ndarray): The X-coordinate
            ys (np.ndarray): The Y-coordinate
            zs (np.ndarray): The Z-coordinate
        """
        cloud = pv.PolyData(np.array([xs,ys,zs]).T)
        self._plot.add_points(cloud)

    def add_surf(self, 
                 x: np.ndarray,
                 y: np.ndarray,
                 z: np.ndarray,
                 field: np.ndarray,
                 scale: Literal['lin','log','symlog'] = 'lin',
                 cmap: cmap_names = 'coolwarm',
                 clim: tuple[float, float] = None,
                 opacity: float = 1.0,
                 symmetrize: bool = True,
                 _fieldname: str = None,
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
            T = lambda x: np.log10(np.abs(x))
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

        if clim is None:
            fmin = np.min(static_field)
            fmax = np.max(static_field)
            clim = (fmin, fmax)
        
        if symmetrize:
            lim = max(abs(clim[0]),abs(clim[1]))
            clim = (-lim, lim)

        kwargs = setdefault(kwargs, cmap=cmap, clim=clim, opacity=opacity, pickable=False, multi_colors=True)
        actor = self._plot.add_mesh(grid, scalars=name, **kwargs)

        if self._animate:
            def on_update(obj: _AnimObject, phi: complex):
                field = obj.T(np.real(obj.field*phi))
                obj.grid['anim'] = field
            self._objs.append(_AnimObject(field_flat, T, grid, actor, on_update))
        
        
    def add_title(self, title: str) -> None:
        """Adds a title

        Args:
            title (str): The title name
        """
        self._plot.add_text(
            title,
            position='upper_edge',
            font_size=18)

    def add_text(self, text: str, 
                 color: str = 'black', 
                 position: Literal['lower_left', 'lower_right', 'upper_left', 'upper_right', 'lower_edge', 'upper_edge', 'right_edge', 'left_edge']='upper_right',
                 abs_position: tuple[float, float, float] = None):
        viewport = False
        if abs_position is not None:
            position = abs_position
            viewport = True
        self._plot.add_text(
            text,
            position=position,
            color=color,
            font_size=18,
            viewport=viewport)
        
    def add_quiver(self, x: np.ndarray, y: np.ndarray, z: np.ndarray,
              dx: np.ndarray, dy: np.ndarray, dz: np.ndarray,
              scale: float = 1,
              color: tuple[float, float, float] = None,
              scalemode: Literal['lin','log'] = 'lin'):
        """Add a quiver plot to the display

        Args:
            x (np.ndarray): The X-coordinates
            y (np.ndarray): The Y-coordinates
            z (np.ndarray): The Z-coordinates
            dx (np.ndarray): The arrow X-magnitude
            dy (np.ndarray): The arrow Y-magnitude
            dz (np.ndarray): The arrow Z-magnitude
            scale (float, optional): The arrow scale. Defaults to 1.
            scalemode (Literal['lin','log'], optional): Wether to scale lin or log. Defaults to 'lin'.
        """
        x = x.flatten()
        y = y.flatten()
        z = z.flatten()
        dx = dx.flatten().real
        dy = dy.flatten().real
        dz = dz.flatten().real
        dmin = _min_distance(x,y,z)

        dmax = np.max(_norm(dx,dy,dz))
        
        Vec = scale * np.array([dx,dy,dz]).T / dmax * dmin 
        Coo = np.array([x,y,z]).T
        if scalemode=='log':
            dx, dy, dz = _logscale(Vec[:,0], Vec[:,1], Vec[:,2])
            Vec[:,0] = dx
            Vec[:,1] = dy
            Vec[:,2] = dz
        
        kwargs = dict()
        if color is not None:
            kwargs['color'] = color
        pl = self._plot.add_arrows(Coo, Vec, scalars=None, clim=None, cmap=None, **kwargs)

    def add_contour(self,
                     X: np.ndarray,
                     Y: np.ndarray,
                     Z: np.ndarray,
                     V: np.ndarray,
                     Nlevels: int = 5,
                     symmetrize: bool = True,
                     cmap: str = 'viridis'):
        """Adds a 3D volumetric contourplot based on a 3D grid of X,Y,Z and field values


        Args:
            X (np.ndarray): A 3D Grid of X-values
            Y (np.ndarray): A 3D Grid of Y-values
            Z (np.ndarray): A 3D Grid of Z-values
            V (np.ndarray): The scalar quantity to plot ()
            Nlevels (int, optional): The number of contour levels. Defaults to 5.
            symmetrize (bool, optional): Wether to symmetrize the countour levels (-V,V). Defaults to True.
            cmap (str, optional): The color map. Defaults to 'viridis'.
        """
        Vf = V.flatten()
        vmin = np.min(np.real(Vf))
        vmax = np.max(np.real(Vf))
        if symmetrize:
            level = max(np.abs(vmin),np.abs(vmax))
            vmin, vmax = (-level, level)
        grid = pv.StructuredGrid(X,Y,Z)
        field = V.flatten(order='F')
        grid['anim'] = np.real(field)
        levels = np.linspace(vmin, vmax, Nlevels)
        contour = grid.contour(isosurfaces=levels)
        actor = self._plot.add_mesh(contour, opacity=0.25, cmap=cmap, pickable=False)

        if self._animate:
            def on_update(obj: _AnimObject, phi: complex):
                new_vals = np.real(obj.field * phi)
                obj.grid['anim'] = new_vals
                new_contour = obj.grid.contour(isosurfaces=levels)
                obj.actor.GetMapper().SetInputData(new_contour)
            
            self._objs.append(_AnimObject(field, lambda x: x, grid, actor, on_update))

    def _add_aux_items(self) -> None:
        saved_camera = {
            "position": self._plot.camera.position,
            "focal_point": self._plot.camera.focal_point,
            "view_up": self._plot.camera.up,
            "view_angle": self._plot.camera.view_angle,
            "clipping_range": self._plot.camera.clipping_range
        }
        #self._plot.add_logo_widget('src/_img/logo.jpeg',position=(0.89,0.89), size=(0.1,0.1))    
        bounds = self._plot.bounds
        max_size = max([abs(dim) for dim in [bounds.x_max, bounds.x_min, bounds.y_max, bounds.y_min, bounds.z_max, bounds.z_min]])
        length = self.set.plane_ratio*max_size*2
        if self.set.draw_xplane:
            plane = pv.Plane(
                center=(0, 0, 0),
                direction=(1, 0, 0),    # normal vector pointing along +X
                i_size=length,
                j_size=length,
                i_resolution=1,
                j_resolution=1
            )
            self._plot.add_mesh(
                plane,
                color='red',
                opacity=self.set.plane_opacity,
                show_edges=False,
                pickable=False,
            )
            self._plot.add_mesh(
                plane,
                edge_opacity=1.0,
                edge_color='red',
                color='red',
                line_width=self.set.plane_edge_width,
                style='wireframe',
                pickable=False,
            )
            
        if self.set.draw_yplane:
            plane = pv.Plane(
                center=(0, 0, 0),
                direction=(0, 1, 0),    # normal vector pointing along +X
                i_size=length,
                j_size=length,
                i_resolution=1,
                j_resolution=1
            )
            self._plot.add_mesh(
                plane,
                color='green',
                opacity=self.set.plane_opacity,
                show_edges=False,
                pickable=False,
            )
            self._plot.add_mesh(
                plane,
                edge_opacity=1.0,
                edge_color='green',
                color='green',
                line_width=self.set.plane_edge_width,
                style='wireframe',
                pickable=False,
            )
        if self.set.draw_zplane:
            plane = pv.Plane(
                center=(0, 0, 0),
                direction=(0, 0, 1),    # normal vector pointing along +X
                i_size=length,
                j_size=length,
                i_resolution=1,
                j_resolution=1
            )
            self._plot.add_mesh(
                plane,
                color='blue',
                opacity=self.set.plane_opacity,
                show_edges=False,
                pickable=False,
            )
            self._plot.add_mesh(
                plane,
                edge_opacity=1.0,
                edge_color='blue',
                color='blue',
                line_width=self.set.plane_edge_width,
                style='wireframe',
                pickable=False,
            )
        # Draw X-axis
        if getattr(self.set, 'draw_xax', False):
            x_line = pv.Line(
                pointa=(-length, 0, 0),
                pointb=(length, 0, 0),
            )
            self._plot.add_mesh(
                x_line,
                color='red',
                line_width=self.set.axis_line_width,
                pickable=False,
            )

        # Draw Y-axis
        if getattr(self.set, 'draw_yax', False):
            y_line = pv.Line(
                pointa=(0, -length, 0),
                pointb=(0, length, 0),
            )
            self._plot.add_mesh(
                y_line,
                color='green',
                line_width=self.set.axis_line_width,
                pickable=False,
            )

        # Draw Z-axis
        if getattr(self.set, 'draw_zax', False):
            z_line = pv.Line(
                pointa=(0, 0, -length),
                pointb=(0, 0, length),
            )
            self._plot.add_mesh(
                z_line,
                color='blue',
                line_width=self.set.axis_line_width,
                pickable=False,
            )

        exponent = np.floor(np.log10(length))
        gs = 10 ** exponent
        N = np.ceil(length/gs)
        if N < 5:
            gs = gs/10
        L = (2*np.ceil(length/(2*gs))+1)*gs

        # XY grid at Z=0
        if self.set.show_zgrid:
            x_vals = np.arange(-L, L+gs, gs)
            y_vals = np.arange(-L, L+gs, gs)

            # lines parallel to X
            for y in y_vals:
                line = pv.Line(
                    pointa=(-L, y, 0),
                    pointb=(L, y, 0)
                )
                self._plot.add_mesh(line, color=self.set.grid_line_color, line_width=self.set.grid_line_width, opacity=0.5, edge_opacity=0.5,pickable=False)

            # lines parallel to Y
            for x in x_vals:
                line = pv.Line(
                    pointa=(x, -L, 0),
                    pointb=(x, L, 0)
                )
                self._plot.add_mesh(line, color=self.set.grid_line_color, line_width=self.set.grid_line_width, opacity=0.5, edge_opacity=0.5,pickable=False)


        # YZ grid at X=0
        if self.set.show_xgrid:
            y_vals = np.arange(-L, L+gs, gs)
            z_vals = np.arange(-L, L+gs, gs)

            # lines parallel to Y
            for z in z_vals:
                line = pv.Line(
                    pointa=(0, -L, z),
                    pointb=(0, L, z)
                )
                self._plot.add_mesh(line, color=self.set.grid_line_color, line_width=self.set.grid_line_width, opacity=0.5, edge_opacity=0.5, pickable=False)

            # lines parallel to Z
            for y in y_vals:
                line = pv.Line(
                    pointa=(0, y, -L),
                    pointb=(0, y, L)
                )
                self._plot.add_mesh(line, color=self.set.grid_line_color, line_width=self.set.grid_line_width, opacity=0.5, edge_opacity=0.5, pickable=False)


        # XZ grid at Y=0
        if self.set.show_ygrid:
            x_vals = np.arange(-L, L+gs, gs)
            z_vals = np.arange(-L, L+gs, gs)

            # lines parallel to X
            for z in z_vals:
                line = pv.Line(
                    pointa=(-length, 0, z),
                    pointb=(length, 0, z)
                )
                self._plot.add_mesh(line, color=self.set.grid_line_color, line_width=self.set.grid_line_width, opacity=0.5, edge_opacity=0.5, pickable=False)

            # lines parallel to Z
            for x in x_vals:
                line = pv.Line(
                    pointa=(x, 0, -length),
                    pointb=(x, 0, length)
                )
                self._plot.add_mesh(line, color=self.set.grid_line_color, line_width=self.set.grid_line_width, opacity=0.5, edge_opacity=0.5, pickable=False)

        if self.set.add_light:
            light = pv.Light()
            light.set_direction_angle(*self.set.light_angle)
            self._plot.add_light(light)

        self._plot.set_background(self.set.background_bottom, top=self.set.background_top)
        self._plot.add_axes()

        self._plot.camera.position = saved_camera["position"]
        self._plot.camera.focal_point = saved_camera["focal_point"]
        self._plot.camera.up = saved_camera["view_up"]
        self._plot.camera.view_angle = saved_camera["view_angle"]
        self._plot.camera.clipping_range = saved_camera["clipping_range"]

def freeze(function):

    def new_function(self, *args, **kwargs):
        cam = self.disp._plot.camera_position[:]
        self.disp._plot.suppress_rendering = True
        function(self, *args, **kwargs)
        self.disp._plot.camera_position = cam
        self.disp._plot.suppress_rendering = False
        self.disp._plot.render()
    return new_function
        
class ScreenRuler:

    def __init__(self, display: OptycalDisplay, min_length: float):
        self.disp: OptycalDisplay = display
        self.points: list[tuple] = [(0,0,0),(0,0,0)]
        self.text: pv.Text = None
        self.ruler = None
        self.state = False
        self.min_length: float = min_length
    
    @freeze
    def toggle(self):
        if not self.state:
            self.state = True
            self.disp._plot.enable_point_picking(self._add_point, left_clicking=True, tolerance=self.min_length)
        else:
            self.state = False
            self.disp._plot.disable_picking()

    @freeze
    def turn_off(self):
        self.state = False
        self.disp._plot.disable_picking()
    
    @property
    def dist(self) -> float:
        p1, p2 = self.points
        return ((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2 + (p1[2]-p2[2])**2)**(0.5)
    
    @property
    def middle(self) -> tuple[float, float, float]:
        p1, p2 = self.points
        return ((p1[0]+p2[0])/2, (p1[1]+p2[1])/2, (p1[2]+p2[2])/2)
    
    @property
    def measurement_string(self) -> str:
        dist = self.dist
        p1, p2 = self.points
        dx = p2[0]-p1[0]
        dy = p2[1]-p1[1]
        dz = p2[2]-p1[2]
        return f'{dist*1000:.2f}mm (dx={1000.*dx:.4f}mm, dy={1000.*dy:.4f}mm, dz={1000.*dz:.4f}mm)'
    
    def set_ruler(self) -> None:
        if self.ruler is None:
            self.ruler = self.disp._plot.add_ruler(self.points[0], self.points[1], title=f'{1000*self.dist:.2f}mm')
        else:
            p1 = self.ruler.GetPositionCoordinate()
            p2 = self.ruler.GetPosition2Coordinate()
            p1.SetValue(*self.points[0])
            p2.SetValue(*self.points[1])
            self.ruler.SetTitle(f'{1000*self.dist:.2f}mm')
    
    @freeze
    def _add_point(self, point: tuple[float, float, float]):
        self.points = [point,self.points[0]]
        self.text = self.disp._plot.add_text(self.measurement_string, self.middle, name='RulerText')
        self.set_ruler()