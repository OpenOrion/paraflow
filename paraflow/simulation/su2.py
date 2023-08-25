from dataclasses import dataclass
import pathlib
import shutil
import sys
from typing import Any, Dict, List, Optional
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

DEFAULT_INSTALL_DIR = f"{pathlib.Path.home()}/simulators/su2"
DEFAULT_VERSION = "7.5.1"

def get_platform():
    if sys.platform.startswith('darwin'):
        return "macos"
    elif sys.platform.startswith('win'):
        return "win "
    elif sys.platform.startswith('linux'):
        return "linux"
    else:
        raise Exception(f"Unsupported platform {sys.platform}")


@dataclass
class SU2SimulationValues:
    eval_values: Dict[str, List[float]]

def setup_su2_simulation(
    meshes: List[Mesh],
    config: Dict,
    config_path: str,
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


def install_su2(url: str, install_dir: str, executable_name: str = "SU2_CFD"):
    if os.path.exists(install_dir):
        shutil.rmtree(install_dir)
    os.makedirs(install_dir)

    print(f"Downloading SU2 from {url} to {install_dir}/{executable_name}")
    
    with urllib.request.urlopen(url) as response:
        with zipfile.ZipFile(io.BytesIO(response.read())) as zip_file:
            for file_info in zip_file.infolist():
                # Extract the file to the extract directory
                if file_info.filename == "bin/SU2_CFD":
                    file_info.filename = executable_name
                    zip_file.extract(file_info, install_dir)
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

def run_su2_simulation(
    meshes: List[Mesh],
    config: Dict[str, Any],
    config_path: str,
    auto_delete: bool = True,
    verbose: bool = False,
    num_procs: int = 1,
    install_dir: str = DEFAULT_INSTALL_DIR,
    version: str = DEFAULT_VERSION,
    custom_download_url: Optional[str] = None,
    custom_repo_url: str = f"https://github.com/su2code/SU2",
    custom_mpirun_path: Optional[str] = None,
    custom_executable_path: Optional[str] = None
):
    is_mpi = num_procs > 1
    num_zones=len(meshes)

    platform = get_platform()
    if custom_executable_path:
        executable_path = custom_executable_path
    else:
        if custom_download_url:
            url = custom_download_url
        else:
            url = f"{custom_repo_url}/releases/download/v{version}/SU2-v{version}-{platform}64{'-mpi' if is_mpi else ''}.zip"
        executable_name = url.split("/")[-1].replace(".zip", "")
        executable_path = f"{install_dir}/{executable_name}"
        if not os.path.exists(executable_path):
            install_su2(url, install_dir, executable_name)

    print(f"Setting up SU2 Simulation for {config_path}")
    setup_su2_simulation(meshes, config, config_path)

    print(f"Running SU2 Simulation for {config_path}")
    if is_mpi:
        mpi_cmd = custom_mpirun_path or 'mpirun'
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