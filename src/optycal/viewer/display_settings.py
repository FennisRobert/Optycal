
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
        self.light_angle: float = (20, -20)
        self.cast_shadows: bool = True
        self.background_bottom: str = "#c0d2e8"
        self.background_top: str = "#ffffff"
        self.grid_line_color: str = "#8e8e8e"
