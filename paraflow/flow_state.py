from typing import List, Literal, Optional, Union, cast
import numpy as np
from dataclasses import dataclass
from functools import cached_property
from thermo.phases import DryAirLemmon
from thermo.chemical_package import lemmon2000_constants, lemmon2000_correlations
from thermo import EquilibriumState, FlashPureVLS, ChemicalConstantsPackage, CEOSLiquid, CEOSGas, SRKMIX, Flash
from chemicals import R, mixing_simple
from thermo import EquilibriumState


def get_flasher(
    fluid_type: Union[str, List[str]],
    state_type: Literal["gas", "liquid"],
):
    if fluid_type == "air":
        gas = DryAirLemmon(T=np.nan, P=np.nan)
        liquid = []
        constants, correlations = lemmon2000_constants, lemmon2000_correlations
    else:
        constants, correlations = ChemicalConstantsPackage.from_IDs(IDs=fluid_type if isinstance(fluid_type, list) else [fluid_type])
        eos_kwargs = dict(Tcs=constants.Tcs, Pcs=constants.Pcs, omegas=constants.omegas)
        gas, liquid = [], []
        if state_type == "liquid":
            liquid = CEOSLiquid(SRKMIX, HeatCapacityGases=correlations.HeatCapacityGases, eos_kwargs=eos_kwargs)
        else:
            gas = CEOSGas(SRKMIX, HeatCapacityGases=correlations.HeatCapacityGases, eos_kwargs=eos_kwargs)

    return FlashPureVLS(constants=constants, correlations=correlations, gas=gas, liquids=liquid, solids=[])


@dataclass
class FlowState:
    total_state: EquilibriumState
    "total state"

    mach_number: float
    "freestream mach number (dimensionless)"

    mass_flow_rate: Optional[float] = None
    "mass flow rate (kg/s)"

    absolute_angle: Optional[float] = None
    "absolute angle of flow (rad)"

    translation_velocity: Optional[float] = None
    "translation velocity of flow passage (m/s)"

    def __post_init__(self) -> None:
        self.flasher = cast(Flash, self.total_state.flasher)
        self.gamma = cast(float, self.total_state.Cp_Cv_ratio())  # type: ignore

    def reinitialize_flow(self, flow_area: float, density: Optional[float] = None):
        if density is None:
            density = self.total_state.rho_mass()
        self.flow_area = flow_area
        self.mass_flow_rate = flow_area*self.freestream_velocity*density  # type: ignore

    @cached_property
    def gas_constant(self):
        molar_weight = cast(float, mixing_simple(self.total_state.zs, self.total_state.constants.MWs))     # type: ignore
        return R * 1000 / molar_weight

    @cached_property
    def freestream_velocity(self) -> float:
        "freestream velocity (m/s)"
        return self.mach_number*self.total_state.speed_of_sound_mass()  # type: ignore

    @cached_property
    def static_state(self) -> EquilibriumState:
        "static state of fluid in passage"
        static_temperature = self.total_state.T - (self.freestream_velocity**2)/(2*self.total_state.Cp_mass())  # type: ignore
        static_pressure = self.total_state.P*(static_temperature/self.total_state.T)**(self.gamma/(self.gamma - 1))
        return self.flasher.flash(T=static_temperature, P=static_pressure)

    @cached_property
    def flow_area(self) -> float:
        "flow area (m^2)"
        assert self.mass_flow_rate, "mass flow rate must be defined"
        return self.mass_flow_rate*(self.static_state.rho_mass()*self.freestream_velocity)

    @cached_property
    def absolute_tangential_velocity(self):
        "absolute tangential velocity (m/s)"
        assert self.absolute_angle, "absolute angle must be defined"
        return self.freestream_velocity*np.tan(self.absolute_angle)

    @cached_property
    def absolute_velocity(self):
        "absolute flow velocity (m/s)"
        assert self.absolute_angle, "absolute angle must be defined"
        if np.isnan(self.absolute_angle).all():
            return self.freestream_velocity
        return self.freestream_velocity/np.cos(self.absolute_angle)

    @cached_property
    def realtive_tangential_velocity(self):
        "relative tangential flow velocity (m/s)"
        return self.absolute_tangential_velocity - self.translation_velocity

    @cached_property
    def relative_angle(self):
        "relative flow angle (rad)"
        return np.arctan(self.realtive_tangential_velocity/self.freestream_velocity)

    @cached_property
    def relative_velocity(self):
        "relative flow velocity (m/s)"
        return self.freestream_velocity/np.cos(self.relative_angle)
