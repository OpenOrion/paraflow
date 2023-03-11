from dataclasses import dataclass
from typing import Tuple
import numpy as np
import plotly.graph_objects as go
from thermo import EquilibriumState
from diffuse.diffusers.diffuser import Diffuser


def get_hub_shroud_radii(
    density: float,
    velocity: float,
    mass_flow_rate: float,
    blockage_factor: float,
    hub_to_tip: float
) -> Tuple[float, float]:
    physical_area = (blockage_factor + 1) * mass_flow_rate/(density*velocity)
    shroud_radius = np.sqrt(physical_area / (np.pi*(1-hub_to_tip**2)))
    hub_radius = hub_to_tip * shroud_radius
    return hub_radius, shroud_radius


def get_initial_diffuser(
    inlet_total_state: EquilibriumState,
    outlet_total_state: EquilibriumState,
    inlet_mach_number: float,
    mass_flow_rate: float,
    hub_to_tip: float = 0.5,
    blockage_factor: float = 0.0,
    axial_length_to_shroud_radius: float = 1,
):
    """
    Get initial diffuser based on targets to start optimizing
    """
    inlet_velocity = inlet_mach_number * inlet_total_state.speed_of_sound()  # type: ignore
    inlet_hub_radius, inlet_shroud_radius = get_hub_shroud_radii(inlet_total_state.rho_mass(), mass_flow_rate, inlet_velocity, blockage_factor, hub_to_tip)
    outlet_hub_radius, outlet_shroud_radius = get_hub_shroud_radii(outlet_total_state.rho_mass(), mass_flow_rate, inlet_velocity, blockage_factor, hub_to_tip)

    axial_length = axial_length_to_shroud_radius*2*max(outlet_shroud_radius, inlet_shroud_radius)

    hub_angle = np.arctan((outlet_hub_radius - inlet_hub_radius)/axial_length)
    shroud_angle = np.arctan((outlet_shroud_radius - inlet_shroud_radius)/axial_length)
    return AnnuluarDiffuser(axial_length, hub_angle, shroud_angle, inlet_hub_radius, inlet_shroud_radius)


@dataclass
class AnnuluarDiffuser(Diffuser):
    axial_length: float
    "axial length of diffuser (m)"

    hub_angle: float
    "angle of hub (rad)"

    shroud_angle: float
    "angle of shroud (rad)"

    inlet_hub_radius: float
    "radius of hub at inlet (m)"

    inlet_shroud_radius: float
    "radius of shroud at inlet (m)"

    def __post_init__(self):
        self.symetry_line = np.vstack(
            [
                [0, 0],
                [self.axial_length, 0]
            ]
        )

        self.outlet_hub_radius = self.inlet_hub_radius + self.axial_length*np.tan(self.hub_angle)
        self.bottom_line = self.hub_line = np.vstack(
            [
                [0, self.inlet_hub_radius],
                [self.axial_length, self.outlet_hub_radius],
            ]
        )

        self.outlet_shroud_radius = self.inlet_shroud_radius + self.axial_length*np.tan(self.shroud_angle)
        self.shroud_line = np.vstack(
            [
                [0, self.inlet_shroud_radius],
                [self.axial_length, self.outlet_shroud_radius],
            ]
        )

    def visualize(self):
        fig = go.Figure(
            layout=go.Layout(title=go.layout.Title(text="Annular Diffuser"))
        )
        fig.add_trace(go.Scatter(
            x=self.symetry_line[:, 0],
            y=self.symetry_line[:, 1],
            name=f"Symetry Line"
        ))

        fig.add_trace(go.Scatter(
            x=self.hub_line[:, 0],
            y=self.hub_line[:, 1],
            name=f"Hub Line"
        ))

        fig.add_trace(go.Scatter(
            x=self.shroud_line[:, 0],
            y=self.shroud_line[:, 1],
            name=f"Shroud Line"
        ))

        fig.layout.yaxis.scaleanchor = "x"  # type: ignore
        fig.show()
