# %%
from paraflow import SymmetricPassage, PassageFluid
import numpy as np
from ezmesh import visualize_mesh


# passage = SymmetricPassage(
#     inlet_radius=0.016,
#     axial_length=0.1407,
#     area_ratio=3
#     # contour_props=[0.25, 0.25, 0.5, 0.75],
#     # contour_angles=np.radians([-15.0, -15.0, 15.0, 15.0]).tolist()
# )
passage = SymmetricPassage(
    inlet_radius=0.2,
    area_ratio=2,
    axial_length=1,
    contour_props=[0.2, 0.5, 0.75],
    contour_angles=np.radians([-20, 15.0, 15.0]).tolist()
)
# passage.visualize("Bell Nozzle")

fluid = PassageFluid(
    fluid_type="Octamethyltrisiloxane",
    inlet_total_pressure=904388,
    inlet_total_temperature=542.13,
    inlet_mach_number=1E-9,
    state_type="gas"
)

visualize_mesh(passage.get_mesh())
# %%
# run_simulation("/workspaces/paraflow/simulation", passage, fluid)
