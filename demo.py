# %%
from ezmesh import CurveLoop, PlaneSurface, TransfiniteSurfaceField, TransfiniteCurveField, Geometry, visualize_mesh
import numpy as np

with Geometry() as geo:
    wedge_coords = np.array([[0, 1], [1.5, 1], [1.5, 0.2], [0.5, 0], [0, 0]])
    
    wedge_curve_loop = CurveLoop.from_coords(
        wedge_coords, 
        mesh_size = 0.05,
        curve_labels=["upper", "outlet", "lower/1", "lower/2", "inlet"],
        fields=[
           TransfiniteCurveField(node_counts=[15,20,10,5,20])
        ]
    )
    surface = PlaneSurface(
        outlines=[wedge_curve_loop],
        is_quad_mesh=True,
        fields=[
            TransfiniteSurfaceField(corners=[
                *wedge_curve_loop.get_points("upper"), 
                *wedge_curve_loop.get_points("lower")
            ])
        ]
    )
    mesh = geo.generate(surface)
    visualize_mesh(mesh)
    # geo.write("mesh_wedge_inv.su2")

# %% Read in Mesh
from phi.flow import *
mesh = geom.load_su2(mesh) # type: ignore
# show(mesh)



# %% Add Field (Random, No Slip)
from phi.field._resample import centroid_to_faces


bcs = {'inlet': 1, 'upper': ZERO_GRADIENT, 'lower': 0, 'outlet': 0}
field = Field(mesh, 0, bcs)
show(field)

from phi.field._resample import centroid_to_faces
# Values at each face (lerps all neighboring centroids, N lerp values back per index)
face_val = centroid_to_faces(field, field.extrapolation, 'linear')

# %%
print(face_val.cells[:1].cells.dual[0:1])
# ~cells = neighboring cells, dual dimension

# %%
new_val = math.sum(face_val, '~cells')
show(Field(mesh, new_val, bcs))

# %% Action Items
# https://github.com/tum-pbs/PhiFlow/blob/9d11b27dba48525e9b6a28161fa9768b4216d726/phi/field/_field_math.py#L49
# * spatial gradient
# * divergence
# * laplace

# Extra
# * upwind-linear



