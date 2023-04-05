from typing import Any, Dict, Protocol
from dataclasses import dataclass
from typing import List, Literal, Optional, Union
from thermo.phases import DryAirLemmon
from thermo.chemical_package import lemmon2000_constants, lemmon2000_correlations
from thermo import FlashPureVLS, ChemicalConstantsPackage, CEOSLiquid, CEOSGas, SRKMIX
from ezmesh import Mesh

@dataclass
class PassageFluid:
    fluid_type: Union[str, List[str]]
    inlet_total_pressure: float
    inlet_total_temperature: float
    inlet_mach_number: float
    state_type: Optional[Literal["liquid", "gas"]] = None

    def __post_init__(self):
        if self.fluid_type == "air":
            gas = DryAirLemmon(T=self.inlet_total_temperature, P=self.inlet_total_pressure)
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
        self.inlet_total_state = self.flasher.flash(T=self.inlet_total_temperature, P=self.inlet_total_pressure)



class Passage(Protocol):
    def get_mesh(self, mesh_size: float = 0.01) -> Mesh: # type: ignore
        pass
    
    def visualize(self, title: str = "Passage", include_ctrl_pnts=False, show=True):
        pass
    
    @staticmethod
    def get_config(fluid: PassageFluid) -> Dict[str, Any]: # type: ignore
        pass