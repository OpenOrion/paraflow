# %%
import numpy as np
from paraflow import SymmetricPassage, get_flasher, PassageOptimizer


passage = SymmetricPassage(
    inlet_radius=0.2,
    area_ratio=2,
    axial_length=1,
    contour_props=[0.2, 0.5, 0.75],
    contour_angles=np.radians([-20, 15.0, 15.0]).tolist()
)

flasher = get_flasher("Octamethyltrisiloxane", "gas")
inlet_total_state = flasher.flash(P=904388, T=542.13, mach_number=1E-9, radius=passage.inlet_radius)
outflow_static_state = flasher.flash(P=200000, T=293.15, mass_flow_rate=inlet_total_state.mass_flow_rate, radius=passage.outlet_radius)
PassageOptimizer(
    "/workspaces/paraflow/simulation", 
    inlet_total_state, 
    outflow_static_state, 
    passage.inlet_radius,
    num_ctrl_pts=3,
    num_throat_pts=1,
    objectives=[("mach", "max")],
).optimize()

# sim_results = run_simulation(passage, inlet_total_state, outflow_static_state, "/workspaces/paraflow/simulation", "1")
# print(sim_results)

# %%
