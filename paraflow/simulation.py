import pickle
import ray
import pathlib
from dataclasses import dataclass
from typing import Callable, Dict, List, Type, Any, Optional
import numpy as np
import numpy.typing as npt
from scipy.interpolate import LinearNDInterpolator
import matplotlib.pyplot as plt
from ezmesh import Mesh
from ezmesh.exporters import export_to_su2
from paraflow.flow_state import FlowState
from paraflow.passages.passage import Passage
from matplotlib import cm
from paraflow.utils import check_points_in_polygon

def get_mach_number(iVertex: int, primitives: Callable, primitiveIndices: Dict[str, int]):
    velocity_x = primitives(iVertex, primitiveIndices["VELOCITY_X"])
    velocity_y = primitives(iVertex, primitiveIndices["VELOCITY_Y"])
    sound_speed = primitives(iVertex, primitiveIndices["SOUND_SPEED"])
    freestream_velocity = np.sqrt(velocity_x**2 + velocity_y**2)
    return freestream_velocity / sound_speed

@dataclass
class SimulationResult:
    points: List[List[float]]
    "points for provided point attributes"

    primitive_values: Dict[str, List[float]]
    "values for provided point attributes in same order as self.points"

    target_values: Dict[str, FlowState]
    "values for marker target points"

    eval_values: Dict[str, List[float]]
    "values for provided eval attributes"


    @staticmethod
    def from_file(file_path: str) -> "SimulationResult":
        with open(file_path, 'rb') as handle:
            return pickle.load(handle)

def get_primitive_frame(
        sim_result: SimulationResult,
        passage: Passage,
        property_name: str,
        num_pnts: int,
        size: Optional[float] = None,
        offset: Optional[npt.NDArray[np.float64]] = None,
        num_spline_pnts: int = 20,
        is_cosine_sampling: bool = True,
        show: bool = False,

):
    if offset is None:
        offset = np.array([0, 0])
    points_np = np.array(sim_result.points)
    primitive_values = np.array(sim_result.primitive_values[property_name])

    points_x = points_np[:, 0]
    points_y = points_np[:, 1]
    cartcoord = list(zip(points_x, points_y))
    X, Y = np.meshgrid(
        np.linspace(-size/2 if size else min(points_x), size/2 if size else max(points_x), num_pnts) + offset[0],
        np.linspace(-size/2 if size else min(points_y), size/2 if size else max(points_y), num_pnts) + offset[1]
    )
    interp = LinearNDInterpolator(cartcoord, primitive_values, fill_value=0)
    primitive_interp = interp(X, Y)

    for outline in passage.surface.outlines:
        coords = outline.get_exterior_coords(num_spline_pnts, is_cosine_sampling)
        mask = check_points_in_polygon((X, Y), coords).astype(int)
        primitive_interp = primitive_interp*mask

    for hole in passage.surface.holes:
        coords = hole.get_exterior_coords(num_spline_pnts, is_cosine_sampling)
        mask = (~check_points_in_polygon((X, Y), coords)).astype(int)
        primitive_interp = primitive_interp*mask

    if show:
        plt.figure()
        plt.pcolormesh(X, Y, primitive_interp, cmap=cm.get_cmap("seismic"))
        plt.colorbar()
        plt.axis('equal')
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
    driver: Optional[Type[Any]],
    eval_properties: Optional[List[str]],
    primitive_properties: Optional[List[str]],
):
    import pysu2
    from mpi4py import MPI

    if driver is None:
        driver = pysu2.CSinglezoneDriver

    # Import mpi4py for parallel run
    comm = MPI.COMM_WORLD
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

    points: List[List[float]] = []
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
                points.append([coord_handler(iVertex, 0), coord_handler(iVertex, 1)])
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
                        target_values[target_name] = inlet_total_state.flasher.flash(T=temperature, P=pressure, mach_number=mach_number)

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
    driver: Optional[Type[Any]] = None,  # type: ignore
    save_path: Optional[str] = None,
    eval_properties: Optional[List[str]] = None,
    primitive_properties: Optional[List[str]] = None,
    outlet_static_state: Optional[FlowState] = None,
    angle_of_attack: float = 0.0,
):
    config_path = f"{working_directory}/config{id}.cfg"
    config = passage.get_config(inlet_total_state, working_directory, id, outlet_static_state, angle_of_attack)
    meshes = passage.get_mesh()
    if not isinstance(meshes, list):
        meshes = [meshes]
    setup_simulation(meshes, config, config_path)
    remote_result = execute_su2.remote(meshes, config_path, inlet_total_state, driver, eval_properties, primitive_properties)
    sim_results = ray.get(remote_result)

    if save_path:
        with open(save_path, 'wb') as handle:
            pickle.dump(sim_results, handle, protocol=pickle.HIGHEST_PROTOCOL)


    return sim_results
