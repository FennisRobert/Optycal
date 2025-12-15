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
from typing import Literal

class PVDisplaySettings:

    def __init__(self):
        self.draw_xplane: bool = True
        self.draw_yplane: bool = True
        self.draw_zplane: bool = True
        self.draw_xax: bool = True
        self.draw_yax: bool = True
        self.draw_zax: bool = True
        self.plane_ratio: float = 0.5
        self.plane_opacity: float = 0.00
        self.plane_edge_width: float = 1.0
        self.axis_line_width: float = 1.5
        self.show_xgrid: bool = False
        self.show_ygrid: bool = False
        self.show_zgrid: bool = True
        self.grid_line_width: float = 1.0
        self.add_light: bool = False
        self.light_angle: tuple[float, float] = (20., -20.)
        self.cast_shadows: bool = True
        self.background_bottom: str = "#c0d2e8"
        self.background_top: str = "#ffffff"
        self.grid_line_color: str = "#8e8e8e"
        self.z_boost: float = 0.0
        self.depth_peeling: bool = True
        self.anti_aliassing: Literal["msaa","ssaa",'fxaa'] = "msaa"
        self.metal_roughness: float = 0.3