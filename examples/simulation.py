# %%
from pathlib import Path
import sys
project_path = Path(__file__).parent.resolve().parents[0]
sys.path.append(f"{project_path}")

import numpy as np
from paraflow import SymmetricPassage, run_simulation, FlowStation

passage = SymmetricPassage(
    inlet_radius=0.2,
    area_ratio=2,
    axial_length=1,
    contour_props=[0.2, 0.5, 0.75],
    contour_angles=np.radians([-20, 15.0, 15.0]).tolist()
)

inflow = FlowStation(
    fluid_type="Octamethyltrisiloxane",
    total_temperature=542.13,
    total_pressure=904388,
    mach_number=1E-9,
    state_type="gas"
)

run_simulation(passage, inflow, "/workspaces/paraflow/simulation")
