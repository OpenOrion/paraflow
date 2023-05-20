from typing import List, Literal, Optional, Union, cast
import numpy as np
from dataclasses import dataclass, field
from functools import cached_property, lru_cache
from thermo.phases import DryAirLemmon
from thermo.chemical_package import lemmon2000_constants, lemmon2000_correlations
from thermo import EquilibriumState, FlashPureVLS, ChemicalConstantsPackage, CEOSLiquid, CEOSGas, SRKMIX, default_settings
from chemicals import mixing_simple
from thermo import EquilibriumState, Flash

R = 8.31446261815324

class FlowState(EquilibriumState):
    mach_number: Optional[float] = None
    "freestream mach number (dimensionless)"

    mass_flow_rate: Optional[float] = None
    "mass flow rate (kg/s)"

    absolute_angle: Optional[float] = None
    "absolute angle of flow (rad)"

    translation_velocity: Optional[float] = None
    "translation velocity of flow passage (m/s)"

    flasher: "FlowFlashPureVLS"
    "flasher object"


    def __init__(self, T, P, zs, gas, liquids, solids, betas, mach_number=None, mass_flow_rate=None, absolute_angle=None, translational_velocity=None, flash_specs=None, flash_convergence=None, constants=None, correlations=None, flasher=None, settings=...):
        super().__init__(T, P, zs, gas, liquids, solids, betas, flash_specs, flash_convergence, constants, correlations, flasher, settings)
        self.mach_number = mach_number
        self.mass_flow_rate = mass_flow_rate
        self.absolute_angle = absolute_angle
        self.translation_velocity = translational_velocity
        self.gamma = cast(float, self.Cp_Cv_ratio())  # type: ignore

    @cached_property
    def gas_constant(self):
        molar_weight = cast(float, mixing_simple(self.zs, self.constants.MWs))     # type: ignore
        return R * 1000 / molar_weight

    @cached_property
    def freestream_velocity(self) -> float:
        "freestream velocity (m/s)"
        return self.mach_number*self.speed_of_sound_mass()  # type: ignore

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


class FlowFlashPureVLS(FlashPureVLS):
    def __init__(self, constants, correlations, gas, liquids, solids, settings=default_settings):
        super().__init__(constants, correlations, gas, liquids, solids, settings)

    def flash(self, mach_number=None, mass_flow_rate=None, absolute_angle=None, translational_velocity=None, zs=None, T=None, P=None, VF=None, SF=None, V=None, H=None, S=None, G=None, U=None, A=None, solution=None, hot_start=None, retry=False, dest=None, rho=None, rho_mass=None, H_mass=None, S_mass=None, G_mass=None, U_mass=None, A_mass=None, spec_fun=None):
        eq_state = super().flash(zs, T, P, VF, SF, V, H, S, G, U, A, solution, hot_start, retry, dest, rho, rho_mass, H_mass, S_mass, G_mass, U_mass, A_mass, spec_fun)
        flow_state =  FlowState(eq_state.T, eq_state.P, eq_state.zs, eq_state.gas, eq_state.liquids, eq_state.solids, eq_state.betas, mach_number, mass_flow_rate, absolute_angle, translational_velocity, eq_state.flash_specs, eq_state.flash_convergence, eq_state.constants, eq_state.correlations, eq_state.flasher, eq_state.settings)
        flow_state.flasher = self
        return flow_state

@lru_cache(maxsize=None)
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

    return FlowFlashPureVLS(constants=constants, correlations=correlations, gas=gas, liquids=liquid, solids=[])