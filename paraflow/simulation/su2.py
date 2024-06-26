from dataclasses import dataclass
import pathlib
import shutil
from typing import Any, Dict, List, Optional
from paraflow.passages.passage import Passage, SimulationParams
from paraflow.simulation.output import SimulationResult, read_vtu_data
from ezmesh.exporters import export_to_su2
import pandas as pd
from ezmesh import Mesh
import subprocess
import urllib.request 
import zipfile
import io
import os
import vtk
import re

from paraflow.simulation.platform import PlatformType, get_platform

DEFAULT_INSTALL_DIR = f"{pathlib.Path.home()}/simulators/su2"
DEFAULT_VERSION = "7.5.1"

@dataclass
class SU2SimulationValues:
    eval_values: Dict[str, List[float]]

def setup_su2_simulation(
    meshes: List[Mesh],
    config: Dict,
    config_path: str,
    platform: PlatformType
):
    with open(config_path, "w") as fp_config:
        for key, value in config.items():
            if key == "CONFIG_LIST":
                assert isinstance(value, dict)
                for zone_config_save_path, zone_config_dict in value.items():
                    with open(zone_config_save_path, "w") as fp_zone_config:
                        for zone_config_key, zone_config_value in zone_config_dict.items():
                            fp_zone_config.write(f"{zone_config_key}= {zone_config_value}\n")
                fp_config.write(f"CONFIG_LIST= ({','.join(zone_config_path for zone_config_path in value.keys())})\n")
            else:
                fp_config.write(f"{key}= {value}\n")
    export_to_su2(meshes, config['MESH_FILENAME'])


def install_su2(url: str, install_dir: str, platform: PlatformType, executable_name: str = "SU2_CFD"):
    if os.path.exists(install_dir):
        shutil.rmtree(install_dir)
    os.makedirs(install_dir)

    print(f"Downloading SU2 from {url} to {install_dir}/{executable_name}")
    
    with urllib.request.urlopen(url) as response:
        with zipfile.ZipFile(io.BytesIO(response.read())) as zip_file:
            for file_info in zip_file.infolist():
                # Extract the file to the extract directory
                if platform != "win" and file_info.filename == "bin/SU2_CFD" or platform == "win" and file_info.filename == "bin/SU2_CFD.exe":
                    file_info.filename = executable_name
                    zip_file.extract(file_info, install_dir)
                    if platform != "win":
                        os.chmod(f"{install_dir}/{executable_name}", 0o755)

def delete_output_files(
    config: Dict[str, Any],
    config_path: str,
    num_zones: int
):
    print(f"Cleaning up ..")
    for key, value in config.items():
        if key == "CONFIG_LIST":
            for config_file in value.keys():
                os.remove(config_file)
        if key.endswith("FILENAME"):
            if num_zones > 1 and key not in ["CONV_FILENAME", "MESH_FILENAME"]:
                try:
                    for i in range(num_zones):
                        filename = value.replace(".", f"_{i}.")
                        os.remove(filename)
                except Exception as e:
                    print(e)
            else:
                os.remove(value)
    os.remove(config_path)

@dataclass
class Su2SimulationConfig:
    install_dir: str = DEFAULT_INSTALL_DIR
    version: str = DEFAULT_VERSION
    custom_download_url: Optional[str] = None
    custom_repo_url: str = f"https://github.com/su2code/SU2"
    custom_mpirun_path: Optional[str] = None
    custom_executable_path: Optional[str] = None

def run_su2_simulation(
    passage: Passage,
    params: SimulationParams,
    working_directory: str,
    id: str,
    auto_delete: bool = True,
    verbose: bool = False,
    num_procs: int = 1,
    cfg: Su2SimulationConfig = Su2SimulationConfig(),
):
    platform = get_platform()
    su2_cfg_directory = working_directory
    if platform == "win":
        # remove C:\\ and use forward slashes
        regex = r"(.*):\\(.*)"
        path_match = re.match(regex, su2_cfg_directory)
        assert path_match, "Invalid working directory"
        su2_cfg_directory = path_match.group(2).replace("\\", "/")

    if not os.path.exists(working_directory):
        os.makedirs(working_directory)
    config_path = f"{working_directory}/config{id}.cfg"
    config = passage.get_config(params, su2_cfg_directory, id)

    meshes = passage.get_meshes(params)

    is_mpi = num_procs > 1
    num_zones=len(meshes)

    if cfg.custom_executable_path:
        executable_path = cfg.custom_executable_path
    else:
        if cfg.custom_download_url:
            url = cfg.custom_download_url
        else:
            url = f"{cfg.custom_repo_url}/releases/download/v{cfg.version}/SU2-v{cfg.version}-{platform}64{'-mpi' if is_mpi else ''}.zip"
        executable_name = url.split("/")[-1].replace(".zip", "")
        executable_path = f"{cfg.install_dir}/{executable_name}"
        if not os.path.exists(executable_path):
            install_su2(url, cfg.install_dir, platform, executable_name)

    print(f"Setting up SU2 Simulation for {config_path}")
    setup_su2_simulation(meshes, config, config_path, platform)

    print(f"Running SU2 Simulation for {config_path}")
    if is_mpi:
        mpi_cmd = cfg.custom_mpirun_path or 'mpirun'
        output = subprocess.run([mpi_cmd, "-n", f"{num_procs}", executable_path, config_path], capture_output=True, text=True)    
    else:
        output = subprocess.run([executable_path, config_path], capture_output=True, text=True)


    log_output = output.stdout
    if verbose:
        print(log_output)

    if output.stderr:
        raise Exception(output.stderr)
    grids: List[vtk.vtkUnstructuredGrid] = []
    for i in range(num_zones):
        
        vtu_filename = config["VOLUME_FILENAME"]
        if num_zones > 1:
            vtu_filename = vtu_filename.replace(".", f"_{i}.")

        grids += [read_vtu_data(vtu_filename)]
    eval_values = pd.read_csv(config["CONV_FILENAME"])

    
    if auto_delete:
        delete_output_files(config, config_path, num_zones)

    return SimulationResult(grids, eval_values, log_output)