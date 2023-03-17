# %%
from dataclasses import dataclass
from typing import List, Optional, Union
import numpy as np
import numpy.typing as npt
import plotly.graph_objects as go
from scipy.interpolate import BSpline
import numpy as np
import numpy.typing as npt

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

    throat_radius: float
    "radius of throat"

    area_ratio: float
    "area ratio of throat to exit"

    axial_length: float
    "length of base nozzle in axial direction"

    mid_angles: Optional[List[float]] = None
    "angle of outlet"

    mid_props: Optional[List[float]] = None
    "proportions of points along"

    inlet_length: float = 0.0
    "length of inlet"

    outlet_length: float = 0.0
    "length of outlet"

    degree: int = 3
    "degree of bspline"
    

    def __post_init__(self):

        self.exit_radius = self.area_ratio*self.throat_radius
        self.exit_angle = np.arctan((self.exit_radius - self.throat_radius)/self.axial_length)

        if self.mid_angles is None:
            self.mid_angles = [self.exit_angle]

        if self.mid_props is None:
            self.mid_props = [0.5]

        self.mid_length = np.asarray(self.mid_props)*self.axial_length
        mid_ctrl_pnts = np.array(
            [
                self.mid_length, 
                self.throat_radius + self.mid_length*np.tan(self.mid_angles)
            ]
        ).T

        self.shroud_ctrl_pnts = np.array(
            [
                [0.0, self.throat_radius],
                *mid_ctrl_pnts,
                [self.axial_length, self.exit_radius]
            ]
        )

        self.shroud_bspline = get_bspline(self.shroud_ctrl_pnts, self.degree)


    def visualize(self):
        fig = go.Figure(
            layout=go.Layout(title=go.layout.Title(text="Nozzle Cross Section"))
        )

        shroud_pnts = self.shroud_bspline(np.linspace(0, 1, 100))

        fig.add_trace(go.Scatter(
            x=self.shroud_ctrl_pnts[:, 0],
            y=self.shroud_ctrl_pnts[:, 1],
            name=f"Control Points"
        ))

        fig.add_trace(go.Scatter(
            x=shroud_pnts[:, 0],
            y=shroud_pnts[:, 1],
            name=f"Shroud"
        ))

        fig.layout.yaxis.scaleanchor = "x"  # type: ignore
        fig.show()


passage = FlowPassage(
    throat_radius=0.1,
    area_ratio=3.0,
    axial_length=1,
    mid_props=[0.25, 0.5, 0.75],
    mid_angles=np.radians([-15.0, 15.0, 15.0]).tolist()
)

passage.visualize()
# %%
