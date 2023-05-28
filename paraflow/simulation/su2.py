from dataclasses import dataclass
import pathlib
from typing import Any, Dict, List, Optional
import ray
from paraflow.simulation.output import SimulationResult, read_vtu_data
from ezmesh.exporters import export_to_su2
from ezmesh import Mesh

@dataclass
class SU2SimulationValues:
    eval_values: Dict[str, List[float]]

def setup_su2_simulation(
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


def execute_su2_simulation(
    config_path: str,
    vtu_path: str,
    eval_properties: Optional[List[str]],
    num_zones: int = 1,
):
    import pysu2
    from mpi4py import MPI

    # Initialize the corresponding driver of SU2, this includes solver preprocessing
    driver = pysu2.CSinglezoneDriver if num_zones == 1 else pysu2.CMultizoneDriver
    comm = MPI.COMM_WORLD
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

    eval_values: Dict[str, List[float]] = {}
    for izone in range(num_zones):
        # SU2Driver.SelectZone(izone) # TODO: coming soon in next pysu2
        if eval_properties:
            for eval_property in eval_properties:
                if eval_property not in eval_values:
                    eval_values[eval_property] = []
                eval_value = SU2Driver.GetOutputValue(eval_property)
                eval_values[eval_property].append(eval_value)

    # Output the solution to file
    SU2Driver.Output(0)

    # Finalize the solver and exit cleanly
    SU2Driver.Finalize()

    return SU2SimulationValues(eval_values)


@ray.remote
def execute_su2_simulation_in_subprocess(
    config_path: str,
    vtu_path: str,
    eval_properties: Optional[List[str]],
    num_zones: int = 1,
):
    return execute_su2_simulation(config_path, vtu_path, eval_properties, num_zones)


def run_su2_simulation(
    meshes: List[Mesh],
    config: Dict[str, Any],
    config_path: str,
    eval_properties: Optional[List[str]] = None,
    is_subprocess: bool = True,
):

    vtu_path = config["VOLUME_FILENAME"]
    setup_su2_simulation(meshes, config, config_path)

    if is_subprocess:
        remote_result = execute_su2_simulation_in_subprocess.remote(config_path, vtu_path, eval_properties)
        sim_values = ray.get(remote_result)
    else:
        sim_values = execute_su2_simulation(config_path, vtu_path, eval_properties)
    vtu = read_vtu_data(vtu_path)
    return SimulationResult(vtu, sim_values.eval_values)