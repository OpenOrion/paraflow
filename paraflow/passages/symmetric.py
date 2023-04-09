from dataclasses import dataclass, field
from typing import List, Optional, cast
import numpy as np
import numpy.typing as npt
import plotly.graph_objects as go
from scipy.interpolate import BSpline
import numpy as np
import numpy.typing as npt
from ezmesh import Geometry, CurveLoop, PlaneSurface, TransfiniteCurveField, TransfiniteSurfaceField
from paraflow.flow_station import FlowStation
from paraflow.passages.passage import Passage


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
class SymmetricPassage(Passage):
    axial_length: float
    "length of base nozzle in axial direction"

    inlet_radius: float
    "radius of contour inlet"

    contour_angles: List[float] = field(default_factory=list)
    "angle of outlet"

    contour_props: List[float] = field(default_factory=list)
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

        if self.contour_props and self.contour_angles:
            assert len(self.contour_props) == len(self.contour_angles), "must specify same number of contour props and angles"
            if len(self.contour_angles) == 1:
                self.contour_angles = [self.contour_angles[0], self.contour_angles[0]]
                self.contour_props = [self.contour_props[0], self.contour_props[0]]
        else:
            self.contour_angles = [self.exit_angle, self.exit_angle]
            self.contour_props = [1.0, 1.0]

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

            mesh = geo.generate(surface)
            mesh.add_target_point("top_outflow", "outflow", 0)
            mesh.add_target_point("mid_outflow", "outflow", 0.5)
            mesh.add_target_point("bottom_outflow", "outflow", 1.0)
            return mesh

    def visualize(self, title: str = "Flow Passage", include_ctrl_pnts=False, show=True, save_path: Optional[str] = None):
        fig = go.Figure(layout=go.Layout(title=go.layout.Title(text=title)))

        if include_ctrl_pnts:
            fig.add_trace(go.Scatter(x=self.ctrl_pnts[:, 0], y=self.ctrl_pnts[:, 1], name=f"Control Points"))

        contour_line = self.get_contour_line()
        fig.add_trace(go.Scatter(x=contour_line[:, 0], y=contour_line[:, 1], name=f"Contour Top"))
        fig.add_trace(go.Scatter(x=contour_line[:, 0], y=-contour_line[:, 1], name=f"Contour Bottom"))



        fig.layout.yaxis.scaleanchor = "x"  # type: ignore

        if save_path:
            fig.write_image(save_path) 
        if show:
            fig.show()

    @staticmethod
    def get_config(inflow: FlowStation, working_directory: str, id: str):
        outflow_static_pressure = 200000.0
        return {
            "SOLVER": "RANS",
            "KIND_TURB_MODEL": "SST",
            "MATH_PROBLEM": "DIRECT",
            "RESTART_SOL": "NO",
            "SYSTEM_MEASUREMENTS": "SI",
            "MACH_NUMBER": inflow.mach_number,
            "AOA": 0.0,
            "SIDESLIP_ANGLE": 0.0,
            "INIT_OPTION": "TD_CONDITIONS",
            "FREESTREAM_OPTION": "TEMPERATURE_FS",
            "FREESTREAM_PRESSURE": inflow.total_pressure,
            "FREESTREAM_TEMPERATURE": inflow.total_temperature,
            "REF_DIMENSIONALIZATION": "DIMENSIONAL",
            "FLUID_MODEL": "PR_GAS",
            "GAMMA_VALUE": inflow.gamma,
            "GAS_CONSTANT": inflow.specific_gas_constant,
            "CRITICAL_TEMPERATURE": inflow.total_state.pseudo_Tc(),
            "CRITICAL_PRESSURE": inflow.total_state.pseudo_Pc(),
            "ACENTRIC_FACTOR": inflow.total_state.pseudo_omega(),
            "VISCOSITY_MODEL": "CONSTANT_VISCOSITY",
            "MU_CONSTANT": inflow.total_state.mu(),                                  # type: ignore
            "CONDUCTIVITY_MODEL": "CONSTANT_CONDUCTIVITY",
            "THERMAL_CONDUCTIVITY_CONSTANT": inflow.total_state.k(),                 # type: ignore
            "MARKER_HEATFLUX": "( wall, 0.0 )",
            "MARKER_SYM": "symmetry",
            "MARKER_RIEMANN": f"( inflow, TOTAL_CONDITIONS_PT, {inflow.total_pressure}, {inflow.total_temperature}, 1.0, 0.0, 0.0, outflow, STATIC_PRESSURE, {outflow_static_pressure}, 0.0, 0.0, 0.0, 0.0 )",
            "NUM_METHOD_GRAD": "GREEN_GAUSS",
            "CFL_NUMBER": 1.0,
            "CFL_ADAPT": "YES",
            "CFL_ADAPT_PARAM": "( 0.1, 2.0, 10.0, 1000.0 )",
            "MAX_DELTA_TIME": 1E6,
            "MUSCL_FLOW": "YES",
            "SLOPE_LIMITER_FLOW": "NONE",
            "MUSCL_TURB": "NO",
            "LINEAR_SOLVER": "FGMRES",
            "LINEAR_SOLVER_PREC": "ILU",
            "LINEAR_SOLVER_ILU_FILL_IN": 0,
            "LINEAR_SOLVER_ERROR": 1E-6,
            "LINEAR_SOLVER_ITER": 10,
            "MGLEVEL": 0,
            "CONV_NUM_METHOD_FLOW": "ROE",
            "ENTROPY_FIX_COEFF": 0.1,
            "TIME_DISCRE_FLOW": "EULER_IMPLICIT",
            "CONV_NUM_METHOD_TURB": "SCALAR_UPWIND",
            "TIME_DISCRE_TURB": "EULER_IMPLICIT",
            "CFL_REDUCTION_TURB": 1.0,
            "ITER": 100,
            "CONV_RESIDUAL_MINVAL": -24,
            "CONV_STARTITER": 10,
            "MESH_FILENAME": f"{working_directory}/passage{id}.su2",
            "MESH_FORMAT": "SU2",
            "TABULAR_FORMAT": "CSV",
            "VOLUME_FILENAME": f"{working_directory}/flow{id}",
            "OUTPUT_WRT_FREQ": 1000,
            "SCREEN_OUTPUT": "(INNER_ITER, RMS_DENSITY, RMS_TKE, RMS_DISSIPATION, LIFT, DRAG)",
        }
