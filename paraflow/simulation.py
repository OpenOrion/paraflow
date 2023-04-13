from typing import Dict
import numpy as np
import pysu2
from mpi4py import MPI
from ezmesh import Mesh
from ezmesh.exporters import export_to_su2
from paraflow.flow_state import FlowState
from paraflow.passages.passage import Passage
import ray


def setup_simulation(
    mesh: Mesh,
    config: Dict,
    save_path: str,
):
    with open(save_path, "w") as f:
        for key, value in config.items():
            f.write(f"{key}= {value}\n")
        export_to_su2(mesh, config['MESH_FILENAME'])


@ray.remote
def run_simulation(
    passage: Passage,
    inflow: FlowState,
    working_directory: str,
    id: str
):

    config_path = f"{working_directory}/config{id}.cfg"
    config = passage.get_config(inflow, working_directory, id)
    mesh = passage.get_mesh()
    setup_simulation(mesh, config, config_path)

    # Import mpi4py for parallel run
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    # Initialize the corresponding driver of SU2, this includes solver preprocessing
    SU2Driver = pysu2.CSinglezoneDriver(config_path, 1, comm)

    # Get all the boundary tags
    MarkerList = SU2Driver.GetMarkerTags()

    # Get all the markers defined on this rank and their associated indices.
    allMarkerIDs = SU2Driver.GetMarkerIndices()

    primitiveIndices = SU2Driver.GetPrimitiveIndices()  # maps primitive names to their indices.
    temperatureIndex = primitiveIndices["TEMPERATURE"]
    pressureIndex = primitiveIndices["PRESSURE"]
    soundSpeedIndex = primitiveIndices["SOUND_SPEED"]
    velocityXIndex = primitiveIndices["VELOCITY_X"]
    velocityYIndex = primitiveIndices["VELOCITY_Y"]
    densityIndex = primitiveIndices["DENSITY"]

    primitives = SU2Driver.Primitives()

    # Time loop is defined in Python so that we have acces to SU2 functionalities at each time step
    comm.Barrier()

    # Time iteration preprocessing
    SU2Driver.Preprocess(0)

    # Run one time-step (static: one simulation)
    SU2Driver.Run()

    # Postprocess
    SU2Driver.Postprocess()

    # Update the solver for the next time iteration
    SU2Driver.Update()

    # Monitor the solver and output solution to file if required
    SU2Driver.Monitor(0)

    target_values: Dict[str, FlowState] = {}
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
                    pressure = primitives(iVertex, pressureIndex)
                    temperature = primitives(iVertex, temperatureIndex)
                    velocity_x = primitives(iVertex, velocityXIndex)
                    velocity_y = primitives(iVertex, velocityYIndex)
                    sound_speed = primitives(iVertex, soundSpeedIndex)
                    density = primitives(iVertex, densityIndex)

                    freestream_velocity = np.sqrt(velocity_x**2 + velocity_y**2)
                    mach_number = freestream_velocity / sound_speed

                    total_state = inflow.flasher.flash(T=temperature, P=pressure)
                    outflow = FlowState(total_state, mach_number)
                    outflow.reinitialize_flow(passage.outlet_area, density)
                    target_values[target_name] = outflow

    # Output the solution to file
    SU2Driver.Output(0)

    # Finalize the solver and exit cleanly
    SU2Driver.Finalize()
    return target_values
