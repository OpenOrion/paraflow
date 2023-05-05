from typing import Any, Dict, Optional, Protocol, Union, List
from ezmesh import Mesh, PlaneSurface
from paraflow.flow_state import FlowState
import json


class Passage(Protocol):
    surface: PlaneSurface
    def get_mesh(self) -> Union[Mesh | List[Mesh]]:  # type: ignore
        pass

    def visualize(self, title: str = "Passage", include_ctrl_pnts=False, show=True, save_path: Optional[str] = None):
        pass

    def get_config(self, inlet_total_state: FlowState, working_directory: str, id: str, target_outlet_static_state: Optional[FlowState] = None) -> Dict[str, Any]:  # type: ignore
        pass

    def to_dict(self) -> Dict[str, Any]:  # type: ignore
        pass

    def write(self, path: str):
        with open(path, "w") as f:
            json.dump(self.to_dict(), f)
