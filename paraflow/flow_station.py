from typing import List, Literal, Optional, Union, cast
import numpy as np
from dataclasses import dataclass
from functools import cached_property
from thermo.phases import DryAirLemmon
from thermo.chemical_package import lemmon2000_constants, lemmon2000_correlations
from thermo import EquilibriumState, FlashPureVLS, ChemicalConstantsPackage, CEOSLiquid, CEOSGas, SRKMIX
from chemicals import R, mixing_simple
from fluids.compressible import T_critical_flow, P_critical_flow
from thermo import EquilibriumState

@dataclass
class FlowStation:
    fluid_type: Union[str, List[str]]
    "type of fluid in passage"

    total_temperature: float
    "total temperature of fluid in passage"

    total_pressure: float
    "total pressure of fluid in passage"
    
    mach_number: float
    "meridonal mach number of fluid in passage"

    state_type: Optional[Literal["liquid", "gas"]] = None
    "type of fluid in passage"

    absolute_angle: Optional[float] = None
    "absolute angle of passage in degrees"

    linear_velocity: Optional[float] = None
    "linear velocity of fluid in passage"


    def __post_init__(self) -> None:
        if self.fluid_type == "air":
            gas = DryAirLemmon(T=self.total_temperature, P=self.total_pressure)
            liquid = []
            constants, correlations = lemmon2000_constants, lemmon2000_correlations
        else:
            constants, correlations = ChemicalConstantsPackage.from_IDs(IDs=self.fluid_type if isinstance(self.fluid_type, list) else [self.fluid_type])
            eos_kwargs = dict(Tcs=constants.Tcs, Pcs=constants.Pcs, omegas=constants.omegas)
            gas, liquid = [], []
            if self.state_type == "liquid":
                liquid = CEOSLiquid(SRKMIX, HeatCapacityGases=correlations.HeatCapacityGases, eos_kwargs=eos_kwargs)
            else:
                gas = CEOSGas(SRKMIX, HeatCapacityGases=correlations.HeatCapacityGases, eos_kwargs=eos_kwargs)

        self.flasher = FlashPureVLS(constants=constants, correlations=correlations, gas=gas, liquids=liquid, solids=[])
        self.total_state = self.flasher.flash(T=self.total_temperature, P=self.total_pressure)
        self.meridonal_velocity = self.mach_number*self.total_state.speed_of_sound()

        self.gamma = cast(float, self.total_state.Cp_Cv_ratio())  # type: ignore
        self.specific_heat_const_pressure = self.total_state.Cp() # type: ignore
        self.molar_weight = cast(float, mixing_simple(self.total_state.zs, self.total_state.constants.MWs))     # type: ignore
        self.specific_gas_constant = R * 1000 / self.molar_weight
        self.critical_temperature = T_critical_flow(self.total_temperature, self.gamma)
        self.critical_pressure = P_critical_flow(self.total_pressure, self.gamma)

        if self.absolute_angle is None:
            velocity = self.meridonal_velocity
        else:
            velocity = self.absolute_tangential_velocity

        self.static_temperature = self.total_temperature - (velocity**2)/(2*self.specific_heat_const_pressure)
        self.static_pressure = self.total_pressure*(self.static_temperature/self.total_temperature)**(self.gamma/(self.gamma - 1))

    @cached_property
    def static_state(self) -> EquilibriumState:
        "static state of fluid in passage"
        return self.flasher.flash(T=self.static_temperature, P=self.static_pressure)

    @cached_property
    def absolute_tangential_velocity(self):
        "absolute tangential velocity (m/s)"
        assert self.absolute_angle, "absolute angle must be defined"
        return self.meridonal_velocity*np.tan(self.absolute_angle)

    @cached_property
    def absolute_velocity(self):
        "absolute flow velocity (m/s)"
        assert self.absolute_angle, "absolute angle must be defined"
        if np.isnan(self.absolute_angle).all():
            return self.meridonal_velocity
        return self.meridonal_velocity/np.cos(self.absolute_angle)

    @cached_property
    def realtive_tangential_velocity(self):
        "relative tangential flow velocity (m/s)"
        return self.absolute_tangential_velocity - self.linear_velocity

    @cached_property
    def relative_angle(self):
        "relative flow angle (rad)"
        return np.arctan(self.realtive_tangential_velocity/self.meridonal_velocity)

    @cached_property
    def relative_velocity(self):
        "relative flow velocity (m/s)"
        return self.meridonal_velocity/np.cos(self.relative_angle)
