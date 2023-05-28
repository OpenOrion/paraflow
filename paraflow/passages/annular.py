from dataclasses import asdict, dataclass, field
from functools import cached_property
from typing import Any, Dict, Optional
import plotly.graph_objects as go
from ezmesh import CurveLoop, PlaneSurface, TransfiniteCurveField, TransfiniteSurfaceField
from paraflow.passages.passage import Passage, ConfigParameters
from paraflow.passages.symmetric import SymmetricPassage


@dataclass
class AnnularPassageMeshParams:
    mesh_size: float = 0.01
    "mesh size"

    wall_label: str = "wall"
    "label for wall"

    inflow_label: str = "inflow"
    "label for inflow"

    outflow_label: str = "outflow"
    "label for outflow"

    target_mid_outflow_label: str = "target_mid_outflow"
    "label for target mid outflow"


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

    mesh_params: AnnularPassageMeshParams = field(default_factory=AnnularPassageMeshParams)
    "mesh parameters"

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

    @cached_property
    def surfaces(self):
        curve_loop = CurveLoop.from_coords(
            [
                ("BSpline", self.shroud_passage.ctrl_pnts),
                ("BSpline", self.hub_passage.ctrl_pnts[::-1]),
            ],
            mesh_size=self.mesh_params.mesh_size,
            curve_labels=[f"{self.mesh_params.wall_label}/top", self.mesh_params.outflow_label, f"{self.mesh_params.wall_label}/bottom", self.mesh_params.inflow_label],
            fields=[
                TransfiniteCurveField(
                    node_counts={f"{self.mesh_params.wall_label}/*": 100, self.mesh_params.inflow_label: 100, self.mesh_params.outflow_label: 100},
                    coefs={f"{self.mesh_params.wall_label}/*": 1.0, self.mesh_params.inflow_label: 1/1.1, self.mesh_params.outflow_label: 1.1}
                )
            ]
        )

        return [PlaneSurface(
            outlines=[curve_loop],
            is_quad_mesh=True,
            fields=[
                TransfiniteSurfaceField(corners=curve_loop.get_points("wall"))
            ],
        )]

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

    def get_config(
        self,
        config_params: ConfigParameters,
        working_directory: str,
        id: str,
    ):
        self.mesh_params.symmetry_label = None  # type: ignore
        config = SymmetricPassage.get_config(self, config_params, working_directory, id)  # type: ignore
        del config["MARKER_SYM"]
        return config

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
