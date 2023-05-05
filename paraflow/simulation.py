from dataclasses import dataclass
from typing import Callable, Dict, List, Type, Any, Optional
import numpy as np
from scipy.interpolate import LinearNDInterpolator
import matplotlib.pyplot as plt
from ezmesh import Mesh
from ezmesh.exporters import export_to_su2
from paraflow.flow_state import FlowState
from paraflow.passages.passage import Passage
from shapely.geometry import Polygon, Point
from matplotlib import cm

import ray
import pathlib


def get_mach_number(iVertex: int, primitives: Callable, primitiveIndices: Dict[str, int]):
    velocity_x = primitives(iVertex, primitiveIndices["VELOCITY_X"])
    velocity_y = primitives(iVertex, primitiveIndices["VELOCITY_Y"])
    sound_speed = primitives(iVertex, primitiveIndices["SOUND_SPEED"])
    freestream_velocity = np.sqrt(velocity_x**2 + velocity_y**2)
    return freestream_velocity / sound_speed


@dataclass
class SimulationResult:
    points: List[np.ndarray]
    "points for provided point attributes"

    primitive_values: Dict[str, List[float]]
    "values for provided point attributes in same order as self.points"

    target_values: Dict[str, FlowState]
    "values for marker target points"

    eval_values: Dict[str, List[float]]
    "values for provided eval attributes"


    def get_primitive_frame(self, passage: Passage, primitive_property: str, num_pnts: int, size: Optional[float] = None):
        points = np.array(self.points)
        primitive_values = np.array(self.primitive_values[primitive_property])

        x = points[:, 0]
        y = points[:, 1]
        cartcoord = list(zip(x, y))
        X, Y = np.meshgrid(
            np.linspace(-size/2 if size else min(x), size/2 if size else max(x), num_pnts),
            np.linspace(-size/2 if size else min(y), size/2 if size else max(y), num_pnts)
        )
        interp = LinearNDInterpolator(cartcoord, primitive_values, fill_value=0)
        primitive_interp = interp(X, Y)
        

        polygons = passage.surface.get_polygons()
        for polygon in polygons:
            for i in range(X.shape[0]):
                for j in range(X.shape[1]):
                    p = Point(X[i,j], Y[i,j])
                    if not polygon.contains(p):
                        primitive_interp[i,j] = 0



        plt.figure()
        plt.pcolormesh(X, Y, primitive_interp, cmap=cm.get_cmap("seismic"))
        plt.colorbar() # Color Bar
        plt.show()
                
        return np.array([X, Y, primitive_interp]).T


def setup_simulation(
    meshes: List[Mesh],
    config: Dict,
    config_path: str,
):
    config_path_directory = pathlib.Path(config_path).parent
    with open(config_path, "w") as fp_config:
        for key, value in config.items():
            if key == "CONFIG_LIST":
                assert isinstance(value, dict)
                for zone_config_save_path, zone_config_dict in value.items():
                    with open(f"{config_path_directory}/{zone_config_save_path}", "w") as fp_zone_config:
                        for zone_config_key, zone_config_value in zone_config_dict.items():
                            fp_zone_config.write(f"{zone_config_key}= {zone_config_value}\n")
                fp_config.write(f"CONFIG_LIST= ({','.join(f'{config_path_directory}/{zone_config_path}' for zone_config_path in value.keys())})\n")
            else:
                fp_config.write(f"{key}= {value}\n")

        export_to_su2(meshes, config['MESH_FILENAME'])


@ray.remote
def execute_su2(
    meshes: List[Mesh],
    config_path: str,
    inlet_total_state: FlowState,
    eval_properties: Optional[List[str]],
    primitive_properties: Optional[List[str]],
    outlet_static_state: Optional[FlowState],
    driver: Optional[Type[Any]],
):
    import pysu2
    from mpi4py import MPI

    if driver is None:
        driver = pysu2.CSinglezoneDriver

    # Import mpi4py for parallel run
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    num_zones = len(meshes)
    # Initialize the corresponding driver of SU2, this includes solver preprocessing
    SU2Driver: pysu2.CFluidDriver = driver(config_path, num_zones, comm)  # type: ignore

    # Time loop is defined in Python so that we have acces to SU2 functionalities at each time step
    comm.Barrier()

    # Time iteration preprocessing
    SU2Driver.Preprocess(0)
    # Run one time-step (static: one simulation)
    SU2Driver.Run()

    # Update the solver for the next time iteration
    SU2Driver.Update()

    # Monitor the solver and output solution to file if required
    SU2Driver.Monitor(0)

    points: List[np.ndarray] = []
    primitive_values: Dict[str, List[float]] = {}
    eval_values: Dict[str, List[float]] = {}
    target_values: Dict[str, FlowState] = {}
    for izone, mesh in enumerate(meshes):
        # maps primitive names to their indices.
        primitiveIndices = SU2Driver.GetPrimitiveIndices()
        primitives = SU2Driver.Primitives()

        # SU2Driver.SelectZone(izone) # TODO: coming soon in next pysu2
        # Get all the markers defined on this rank and their associated indices.
        allMarkerIDs = SU2Driver.GetMarkerIndices()

        if eval_properties:
            for eval_property in eval_properties:
                if eval_property not in eval_values:
                    eval_values[eval_property] = []
                eval_value = SU2Driver.GetOutputValue(eval_property)
                eval_values[eval_property].append(eval_value)

        if primitive_properties:
            coord_handler = SU2Driver.Coordinates()
            for iVertex in range(SU2Driver.GetNumberNodes()):
                coord = np.array([coord_handler(iVertex, 0), coord_handler(iVertex, 1)])
                points.append(coord)
                for primitive_property in primitive_properties:
                    if primitive_property not in primitive_values:
                        primitive_values[primitive_property] = []
                    if primitive_property == "MACH":
                        property_value = get_mach_number(iVertex, primitives, primitiveIndices)
                    else:
                        property_value = primitives(iVertex, primitiveIndices[primitive_property])
                    primitive_values[primitive_property].append(property_value)

        for marker_name, target in mesh.target_points.items():
            marker_id = allMarkerIDs[marker_name]
            nVertex_Marker = SU2Driver.GetNumberMarkerNodes(marker_id)
            for iVertex in range(nVertex_Marker):
                marker_coords = np.array(SU2Driver.MarkerCoordinates(marker_id).Get(iVertex), dtype=np.float64)
                for target_key, target_name in target.items():
                    target_name = target[target_key]
                    target_coords = mesh.points[target_key]
                    if np.allclose(target_coords[:2], marker_coords):
                        # outlet conditions
                        pressure = primitives(iVertex, primitiveIndices["PRESSURE"])
                        temperature = primitives(iVertex, primitiveIndices["TEMPERATURE"])
                        mach_number = get_mach_number(iVertex, primitives, primitiveIndices)

                        target_values[target_name] = inlet_total_state.flasher.flash(T=temperature, P=pressure, mach_number=mach_number, radius=outlet_static_state.radius if outlet_static_state else None)

    # Output the solution to file
    SU2Driver.Output(0)

    # Finalize the solver and exit cleanly
    SU2Driver.Finalize()
    return SimulationResult(points, primitive_values, target_values, eval_values)


def run_simulation(
    passage: Passage,
    inlet_total_state: FlowState,
    working_directory: str,
    id: str,
    eval_properties: Optional[List[str]] = None,
    primitive_properties: Optional[List[str]] = None,
    outlet_static_state: Optional[FlowState] = None,
    driver: Optional[Type[Any]] = None,  # type: ignore
):
    config_path = f"{working_directory}/config{id}.cfg"
    config = passage.get_config(inlet_total_state, working_directory, id, outlet_static_state)
    meshes = passage.get_mesh()
    if not isinstance(meshes, list):
        meshes = [meshes]
    setup_simulation(meshes, config, config_path)
    remote_result = execute_su2.remote(meshes, config_path, inlet_total_state, eval_properties, primitive_properties, outlet_static_state, driver)
    sim_results = ray.get(remote_result)

    return sim_results
