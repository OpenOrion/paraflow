from dataclasses import dataclass
from typing import Any, Dict, Optional, Protocol, List
from ezmesh import Geometry, PlaneSurface, Mesh
import numpy.typing as npt
from paraflow.flow_state import FlowState
import json


@dataclass
class SimulationParams:
    inlet_total_state: FlowState
    target_outlet_static_state: Optional[FlowState] = None
    angle_of_attack: float = 0.0
    translation: Optional[List[Optional[npt.NDArray]]] = None

class Passage(Protocol):
    def visualize(self, title: str = "Passage", include_ctrl_pnts=False, show=True, save_path: Optional[str] = None):
        pass

    def get_config(
        self,
        params: SimulationParams,
        working_directory: str,
        id: str,
    ) -> Dict[str, Any]:  # type: ignore
        pass

    def get_surfaces(self, params: Optional[SimulationParams] = None) -> List[PlaneSurface]:
        ...

    def get_meshes(self, params: Optional[SimulationParams] = None):
        meshes: List[Mesh] = []
        for surface in self.get_surfaces(params):
            with Geometry() as geo:
                mesh = geo.generate(surface)
                meshes.append(mesh)
        return meshes

    def to_dict(self) -> Dict[str, Any]:  # type: ignore
        ...

    def write(self, path: str):
        with open(path, "w") as f:
            json.dump(self.to_dict(), f)
