from dataclasses import dataclass
import pathlib
import shutil
from typing import Any, Dict, List
from paraflow.simulation.output import SimulationResult, read_vtu_data
from ezmesh.exporters import export_to_su2
import pandas as pd
from ezmesh import Mesh
import subprocess
import urllib.request 
import zipfile
import io
import os

DEFAULT_INSTALL_DIR = f"{pathlib.Path.home()}/simulators/su2"

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


def install_su2(install_dir: str = DEFAULT_INSTALL_DIR):
    print(f"Downloading SU2 to {install_dir}")
    if os.path.exists(install_dir):
        shutil.rmtree(install_dir)
    os.makedirs(install_dir)

    url = "https://github.com/OpenOrion/su2_build/releases/download/7.5.1-python-3.10-bullseye-develop/build.zip"
    with urllib.request.urlopen(url) as response:
        with zipfile.ZipFile(io.BytesIO(response.read())) as zip_file:
            for file_info in zip_file.infolist():
                # Extract the file to the extract directory
                if file_info.filename == "build/SU2_CFD/src/SU2_CFD":
                    file_info.filename = file_info.filename.replace(f"build/SU2_CFD/src", "")
                    zip_file.extract(file_info, install_dir)
                    os.chmod(f"{install_dir}/SU2_CFD", 0o755)

def run_su2_simulation(
    meshes: List[Mesh],
    config: Dict[str, Any],
    config_path: str,
    install_dir: str = DEFAULT_INSTALL_DIR
):
    print(f"Running SU2 Simulation for {config_path}")

    executable_path = f"{install_dir}/SU2_CFD"
    if not os.path.exists(executable_path):
       install_su2(install_dir)

    setup_su2_simulation(meshes, config, config_path)
    output = subprocess.run([f"{install_dir}/SU2_CFD", config_path], capture_output=True, text=True)
    vtu = read_vtu_data(config["VOLUME_FILENAME"])
    eval_values = pd.read_csv(config["CONV_FILENAME"])

    return SimulationResult(vtu, eval_values)
