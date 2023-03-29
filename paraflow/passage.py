from dataclasses import dataclass
from functools import cached_property
from typing import List, Optional
import numpy as np
import numpy.typing as npt
import plotly.graph_objects as go
from scipy.interpolate import BSpline
import numpy as np
import numpy.typing as npt
from ezmesh import Geometry, CurveLoop, PlaneSurface, TransfiniteCurveField, TransfiniteSurfaceField


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
    axial_length: float
    "length of base nozzle in axial direction"

    inlet_radius: float
    "radius of contour inlet"

    contour_angles: Optional[List[float]] = None
    "angle of outlet"

    contour_props: Optional[List[float]] = None
    "proportions of points along"

    area_ratio: Optional[float] = None
    "ratio of exit area to inlet area"

    def __post_init__(self):
        self.symetry_line = np.array([[0.0, 0.0], [self.axial_length, 0.0]])
        


        if self.area_ratio is not None:
            self.exit_radius = self.area_ratio*self.inlet_radius
            self.exit_angle = np.arctan((self.exit_radius - self.inlet_radius)/self.axial_length)
        else:
            assert self.contour_angles is not None, "must specify contour angles if area ratio is not specified"
            self.exit_angle = self.contour_angles[-1]
            self.exit_radius = np.tan(self.exit_angle)*self.axial_length + self.inlet_radius
            self.area_ratio = self.exit_radius/self.inlet_radius

        if self.contour_props is None:
            self.contour_props = [0,0]
        if self.contour_angles is None:
            self.contour_angles = [self.exit_angle, self.exit_angle]

        contour_lengths = np.asarray(self.contour_props)*self.axial_length
        contour_ctrl_pnts = np.array(
            [
                contour_lengths,
                self.inlet_radius + contour_lengths*np.tan(self.contour_angles)
            ]
        ).T

        self.ctrl_pnts = np.array(
            [
                [0.0, self.inlet_radius],
                *contour_ctrl_pnts,
                [self.axial_length, self.exit_radius]
            ]
        )

    def get_contour_line(self, num_points=50):
        contour_bspline = get_bspline(self.ctrl_pnts, 3)
        return contour_bspline(np.linspace(0, 1, num_points))

    def get_mesh(self, mesh_size=0.01):
        with Geometry() as geo:
            curve_loop = CurveLoop.from_coords(
                [
                    ("BSpline", self.ctrl_pnts),
                    self.symetry_line[::-1]
                ],
                mesh_size=mesh_size,
                labels=["wall", "outflow", "symmetry", "inflow"],
                fields=[
                    TransfiniteCurveField(
                        node_counts={"wall": 100, "inflow": 100, "symmetry": 100, "outflow": 100},
                        coefs={"wall": 1.0, "inflow": 1/1.1, "symmetry": 1.0, "outflow": 1.1}
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

            return geo.generate(surface)

    def visualize(self, title: str = "Flow Passage", include_ctrl_pnts=False, show=True):
        fig = go.Figure(layout=go.Layout(title=go.layout.Title(text=title)))

        if include_ctrl_pnts:
            fig.add_trace(go.Scatter(x=self.ctrl_pnts[:, 0], y=self.ctrl_pnts[:, 1], name=f"Control Points"))
        
        contour_line = self.get_contour_line()
        fig.add_trace(go.Scatter(x=contour_line[:, 0], y=contour_line[:, 1], name=f"Contour Top"))
        fig.add_trace(go.Scatter(x=contour_line[:, 0], y=-contour_line[:, 1], name=f"Contour Bottom"))

        fig.layout.yaxis.scaleanchor = "x"  # type: ignore
        fig.show()

