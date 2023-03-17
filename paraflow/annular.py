from dataclasses import dataclass
from typing import Tuple
import numpy as np
from thermo import EquilibriumState
from paraflow.flow_passage import FlowPassage


def get_hub_shroud_radii(
    density: float,
    velocity: float,
    mass_flow_rate: float,
    blockage_factor: float,
    hub_to_tip: float
) -> Tuple[float, float]:
    """
    Get hub and shroud radii based on mass flow rate, density, velocity, blockage factor, and hub to tip ratio
    """
    physical_area = (blockage_factor + 1) * mass_flow_rate/(density*velocity)
    shroud_radius = np.sqrt(physical_area / (np.pi*(1-hub_to_tip**2)))
    hub_radius = hub_to_tip * shroud_radius
    return hub_radius, shroud_radius


@dataclass
class AnnularPassage(FlowPassage):
    axial_length: float
    "axial length of diffuser (m)"

    hub_angle: float
    "angle between hub and symetry line (rad)"

    shroud_angle: float
    "angle between shroud and symetry line (rad)"

    inlet_hub_radius: float
    "radius of hub at inlet (m)"

    inlet_shroud_radius: float
    "radius of shroud at inlet (m)"

    inlet_length: float = 0.0
    "length of inlet (m)"

    outlet_length: float = 0.0
    "length of outlet (m)"

    def __post_init__(self):
        self.outlet_hub_radius = self.inlet_hub_radius + self.axial_length*np.tan(self.hub_angle)
        self.hub_line = np.vstack(
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
        super().__init__(self.axial_length, self.hub_line, self.shroud_line)

    @staticmethod
    def initial(
        inlet_total_state: EquilibriumState,
        outlet_total_state: EquilibriumState,
        inlet_mach_number: float,
        mass_flow_rate: float,
        hub_to_tip: float = 0.5,
        blockage_factor: float = 0.0,
        axial_length_to_max_radius: float = 1.0,
        inlet_length: float = 0.0,
        outlet_length: float = 0.0
    ):
        """
        Get initial diffuser based on targets to start optimizing
        """
        inlet_velocity = inlet_mach_number * inlet_total_state.speed_of_sound()  # type: ignore
        inlet_hub_radius, inlet_shroud_radius = get_hub_shroud_radii(inlet_total_state.rho_mass(), mass_flow_rate, inlet_velocity, blockage_factor, hub_to_tip)
        outlet_hub_radius, outlet_shroud_radius = get_hub_shroud_radii(outlet_total_state.rho_mass(), mass_flow_rate, inlet_velocity, blockage_factor, hub_to_tip)

        axial_length = axial_length_to_max_radius*2*max(outlet_shroud_radius, inlet_shroud_radius)

        hub_angle = np.arctan((outlet_hub_radius - inlet_hub_radius)/axial_length)
        shroud_angle = np.arctan((outlet_shroud_radius - inlet_shroud_radius)/axial_length)
        return AnnularPassage(axial_length, hub_angle, shroud_angle, inlet_hub_radius, inlet_shroud_radius, inlet_length, outlet_length)
