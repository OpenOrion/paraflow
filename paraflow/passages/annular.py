from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional
import plotly.graph_objects as go
from ezmesh import Geometry, CurveLoop, PlaneSurface, TransfiniteCurveField, TransfiniteSurfaceField
from paraflow.flow_state import FlowState
from paraflow.passages.passage import Passage
from paraflow.passages.symmetric import SymmetricPassage


@dataclass
class AnnularPassage(Passage):
    axial_length: float
    "axial length of diffuser (m)"

    inlet_hub_radius: float
    "radius of hub at inlet (m)"

    inlet_shroud_radius: float
    "radius of shroud at inlet (m)"

    hub_angle: float
    "angles between hub and symetry line (rad)"

    shroud_angle: float
    "angles between shroud and symetry line (rad)"

    def __post_init__(self):
        assert self.shroud_angle >= self.hub_angle, "shroud angle must be greater or equal to hub angle"
        self.hub_passage = SymmetricPassage(
            axial_length=self.axial_length,
            inlet_radius=self.inlet_hub_radius,
            contour_angles=[self.hub_angle, self.hub_angle],
        )

        self.shroud_passage = SymmetricPassage(
            axial_length=self.axial_length,
            inlet_radius=self.inlet_shroud_radius,
            contour_angles=[self.shroud_angle, self.shroud_angle],
        )

        self.inlet_radius = self.inlet_shroud_radius
        self.outlet_radius = self.shroud_passage.outlet_radius

    def get_mesh(self, mesh_size=0.01):
        with Geometry() as geo:
            curve_loop = CurveLoop.from_coords(
                [
                    ("BSpline", self.shroud_passage.ctrl_pnts),
                    ("BSpline", self.hub_passage.ctrl_pnts[::-1]),
                ],
                mesh_size=mesh_size,
                labels=["wall/top", "outflow", "wall/bottom", "inflow"],
                fields=[
                    TransfiniteCurveField(
                        node_counts={"wall/*": 100, "inflow": 100, "outflow": 100},
                        coefs={"wall/*": 1.0, "inflow": 1/1.1, "outflow": 1.1}
                    )
                ]
            )

            surface = PlaneSurface(
                outlines=[curve_loop],
                is_quad_mesh=True,
                fields=[
                    TransfiniteSurfaceField(corners=curve_loop.get_points("wall"))
                ],
            )

            mesh = geo.generate(surface)
            mesh.add_target_point("mid_outflow", "outflow", 0.5)
            return mesh

    def visualize(self, title: str = "Flow Passage", include_ctrl_pnts=False, show=True, save_path: Optional[str] = None):
        fig = go.Figure(layout=go.Layout(title=go.layout.Title(text=title)))

        if include_ctrl_pnts:
            fig.add_trace(go.Scatter(x=self.hub_passage.ctrl_pnts[:, 0], y=self.hub_passage.ctrl_pnts[:, 1], name=f"Hub Control Points"))
            fig.add_trace(go.Scatter(x=self.shroud_passage.ctrl_pnts[:, 0], y=self.shroud_passage.ctrl_pnts[:, 1], name=f"Shroud Control Points"))

        hub_line = self.hub_passage.get_contour_line()
        fig.add_trace(go.Scatter(x=hub_line[:, 0], y=hub_line[:, 1], name=f"Hub Top"))
        fig.add_trace(go.Scatter(x=hub_line[:, 0], y=-hub_line[:, 1], name=f"Hub Bottom"))

        shroud_line = self.shroud_passage.get_contour_line()
        fig.add_trace(go.Scatter(x=shroud_line[:, 0], y=shroud_line[:, 1], name=f"Shroud Top"))
        fig.add_trace(go.Scatter(x=shroud_line[:, 0], y=-shroud_line[:, 1], name=f"Shroud Bottom"))

        fig.layout.yaxis.scaleanchor = "x"  # type: ignore

        if save_path:
            fig.write_image(save_path) 
        if show:
            fig.show()


    @staticmethod
    def get_config(inflow: FlowState, working_directory: str, id: str):
        config = SymmetricPassage.get_config(inflow, working_directory, id)
        del config["MARKER_SYM"]
        return config

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)