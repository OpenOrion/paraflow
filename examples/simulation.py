# %%
from pathlib import Path
import sys

project_path = Path(__file__).parent.resolve().parents[0]
sys.path.append(f"{project_path}")

# %%
from paraflow import SymmetricPassage, run_simulation, FlowStation
from paraflow.optimize import OptimizationSpecification, optimize
import ray
from ezmesh import visualize_mesh
import numpy as np


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

# passage.visualize()


remote_result = run_simulation.remote(passage, inflow, "/workspaces/paraflow/simulation", "1")
sim_results = ray.get(remote_result)
print(sim_results)
# spec = OptimizationSpecification(
#     working_directory="/workspaces/paraflow/simulation",
#     inflow=inflow,
#     inlet_radius=0.2,
#     num_ctrl_pts=3,
#     num_throat_pts= 1,
#     objectives=[("mach", "max")],
#     area_ratio=2
# )

# optimize(spec)

# %%
