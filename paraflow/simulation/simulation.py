from dataclasses import dataclass
import os
from typing import List, Literal, Optional
from paraflow.passages.passage import Passage, ConfigParameters
from ezmesh import Geometry


def run_simulation(
    passage: Passage,
    config_params: ConfigParameters,
    working_directory: str,
    id: str,
    sim_type: Literal['su2'] = 'su2',
    auto_delete: bool = True,
    is_subprocess: bool = True,
    eval_properties: Optional[List[str]] = None,
):
    meshes = []
    for surface in passage.surfaces:
        with Geometry() as geo:
            mesh = geo.generate(surface)
            meshes.append(mesh)

    config_path = f"{working_directory}/config{id}.cfg"
    config = passage.get_config(config_params, working_directory, id)

    if sim_type == 'su2':
        from paraflow.simulation.su2 import run_su2_simulation
        sim_results = run_su2_simulation(meshes, config, config_path, eval_properties, is_subprocess)

    if auto_delete:
        for key, value in config.items():
            if key.endswith("FILENAME"):
                os.remove(value)
        os.remove(config_path)

    return sim_results
