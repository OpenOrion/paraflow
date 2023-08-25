import os
from typing import Dict, Literal
from paraflow.passages.passage import Passage, SimulationOptions

def run_simulation(
    passage: Passage,
    sim_options: SimulationOptions,
    working_directory: str,
    id: str,
    sim_type: Literal['su2'] = 'su2',
    auto_delete: bool = True,
    verbose: bool = False,
    num_procs: int = 1,
    sim_config: Dict = {}
):
    working_directory = os.path.abspath(working_directory)
    if not os.path.exists(working_directory):
        os.makedirs(working_directory)
    config_path = f"{working_directory}/config{id}.cfg"
    config = passage.get_config(sim_options, working_directory, id)

    meshes = passage.get_meshes(sim_options)

    if sim_type == 'su2':
        from paraflow.simulation.su2 import run_su2_simulation
        sim_results = run_su2_simulation(meshes, config, config_path, auto_delete, verbose, num_procs, **sim_config)

    return sim_results
