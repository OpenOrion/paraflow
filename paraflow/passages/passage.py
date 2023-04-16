from typing import Any, Dict, Optional, Protocol
from ezmesh import Mesh
import numpy as np
from paraflow.flow_state import FlowState
import json

class Passage(Protocol):
    inlet_radius: float
    "inlet radius (m)"

    outlet_radius: float
    "outlet radius (m)"


    def get_mesh(self, mesh_size: float = 0.01) -> Mesh: # type: ignore
        pass
    
    def visualize(self, title: str = "Passage", include_ctrl_pnts=False, show=True, save_path: Optional[str] = None):
        pass
    
    @staticmethod
    def get_config(inflow: FlowState, working_directory: str, id: str) -> Dict[str, Any]: # type: ignore
        pass
    
    def to_dict(self) -> Dict[str, Any]: # type: ignore
        pass

    def write(self, path: str):
        with open(path, "w") as f:
            json.dump(self.to_dict(), f)
