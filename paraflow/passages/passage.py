from typing import Any, Dict, Optional, Protocol
from ezmesh import Mesh
from paraflow.flow_station import FlowStation
import json

class Passage(Protocol):
    def get_mesh(self, mesh_size: float = 0.01) -> Mesh: # type: ignore
        pass
    
    def visualize(self, title: str = "Passage", include_ctrl_pnts=False, show=True, save_path: Optional[str] = None):
        pass
    
    @staticmethod
    def get_config(inflow: FlowStation, working_directory: str, id: str) -> Dict[str, Any]: # type: ignore
        pass
    
    def write(self, path: str):
        with open(path, "w") as f:
            json.dump(self.__dict__, f)