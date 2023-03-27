from dataclasses import dataclass
from typing import List, Optional
import numpy as np
import numpy.typing as npt
import plotly.graph_objects as go
from scipy.interpolate import BSpline
import numpy as np
import numpy.typing as npt
from ezmesh import Geometry, CurveLoop, PlaneSurface, TransfiniteCurveField, TransfiniteSurfaceField, visualize_mesh

def get_bspline(ctrl_pnts: npt.NDArray, degree: int):
    "get a bspline with clamped knots"
    num_ctrl_pnts = ctrl_pnts.shape[0]
    knots = np.pad(
        array=np.linspace(0, 1, (num_ctrl_pnts + 1) - degree),
        pad_width=(degree, degree),
        mode='constant',
        constant_values=(0, 1)
    )
    return BSpline(knots, ctrl_pnts, degree, extrapolate=False)

@dataclass
class FlowPassage:

    inlet_radius: float
    "radius of throat"

    area_ratio: float
    "area ratio of throat to exit"

    axial_length: float
    "length of base nozzle in axial direction"

    contour_angles: Optional[List[float]] = None
    "angle of outlet"

    contour_props: Optional[List[float]] = None
    "proportions of points along"

    inlet_length: float = 0.0
    "length of inlet"

    outlet_length: float = 0.0
    "length of outlet"

    num_shroud_pnts: int = 50
    "number of points along shroud"

    degree: int = 3
    "degree of bspline"
    

    def __post_init__(self):
        self.symetry_line = np.array([[0.0, 0.0], [0.0, self.axial_length]])

        self.exit_radius = self.area_ratio*self.inlet_radius
        self.exit_angle = np.arctan((self.exit_radius - self.inlet_radius)/self.axial_length)

        if self.contour_angles is None:
            self.contour_angles = [self.exit_angle, self.exit_angle]

        if self.contour_props is None:
            self.contour_props = [0,0]

        self.countour_lengths = np.asarray(self.contour_props)*self.axial_length
        mid_ctrl_pnts = np.array(
            [
                self.countour_lengths, 
                self.inlet_radius + self.countour_lengths*np.tan(self.contour_angles)
            ]
        ).T

        self.ctrl_pnts = np.array(
            [
                [0.0, self.inlet_radius],
                *mid_ctrl_pnts,
                [self.axial_length, self.exit_radius]
            ]
        )

        self.shroud_bspline = get_bspline(self.ctrl_pnts, self.degree)
        self.shroud_line = self.shroud_bspline(np.linspace(0, 1, self.num_shroud_pnts))


    def get_mesh(self, mesh_size=0.01):
        with Geometry() as geo:
            curve_loop = CurveLoop.from_coords(
                [
                    ("BSpline", self.ctrl_pnts),
                    np.array([[self.axial_length, 0.0], [0.0, 0.0]])
                ],
                mesh_size=mesh_size,
                labels=["wall", "outflow", "symmetry", "inflow"],
                fields=[
                    TransfiniteCurveField(
                        node_counts={"wall": 20, "inflow": 100, "symmetry": 20, "outflow": 100}
                    )
                ]
            )
            
            surface = PlaneSurface(
                outlines=[curve_loop], 
                is_quad_mesh=True, 
                fields=[
                    TransfiniteSurfaceField(corners=[*curve_loop.get_points("wall"), *curve_loop.get_points("symmetry")])
                ],
            )

            mesh = geo.generate(surface)
            geo.write("test.geo_unrolled")

            visualize_mesh(mesh)
        return mesh


    def visualize(self, title: Optional[str] = None, include_ctrl_pnts=False):
        fig = go.Figure(
            layout=go.Layout(title=go.layout.Title(text=title or "Flow Passage"))
        )

        if include_ctrl_pnts:
            fig.add_trace(go.Scatter(
                x=self.ctrl_pnts[:, 0],
                y=self.ctrl_pnts[:, 1],
                name=f"Control Points"
            ))

        fig.add_trace(go.Scatter(
            x=self.shroud_line[:, 0],
            y=self.shroud_line[:, 1],
            name=f"Shroud Top"
        ))

        fig.add_trace(go.Scatter(
            x=self.shroud_line[:, 0],
            y=-self.shroud_line[:, 1],
            name=f"Shroud Bottom"
        ))

        fig.layout.yaxis.scaleanchor = "x"  # type: ignore
        fig.show()


# %%
