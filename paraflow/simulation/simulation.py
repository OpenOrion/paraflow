import os
from paraflow.passages.passage import Passage, SimulationParams
from paraflow.simulation.su2 import Su2SimulationConfig

def run_simulation(
    passage: Passage,
    params: SimulationParams,
    working_directory: str,
    id: str,
    auto_delete: bool = True,
    verbose: bool = False,
    num_procs: int = 1,
    cfg: Su2SimulationConfig = Su2SimulationConfig()
):
    working_directory = os.path.abspath(working_directory)
    if not os.path.exists(working_directory):
        os.makedirs(working_directory)

    if isinstance(cfg, Su2SimulationConfig):
        from paraflow.simulation.su2 import run_su2_simulation
        sim_results = run_su2_simulation(passage, params, working_directory, id, auto_delete, verbose, num_procs, cfg)

    return sim_results
