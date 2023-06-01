from dataclasses import dataclass
import pickle
from typing import List
import pandas as pd
import gzip
import vtk


def read_vtu_data(filename: str) -> vtk.vtkUnstructuredGrid:
    reader = vtk.vtkXMLUnstructuredGridReader()
    reader.SetFileName(filename)
    reader.Update()
    return reader.GetOutput()


def serialize_vtu(mesh: vtk.vtkUnstructuredGrid):
    writer = vtk.vtkDataSetWriter()
    writer.SetInputDataObject(mesh)
    writer.SetWriteToOutputString(True)
    writer.SetFileTypeToASCII()
    writer.Write()
    return writer.GetOutputString()


def deserialize_vtu(vtu_str: str):
    reader = vtk.vtkDataSetReader()
    reader.ReadFromInputStringOn()
    reader.SetInputString(vtu_str)
    reader.Update()
    return reader.GetOutput()


@dataclass
class SimulationResult:
    grids: List[vtk.vtkUnstructuredGrid]
    "vtu unsctructured grid"

    eval_values: pd.DataFrame
    "values for provided eval attributes"

    @staticmethod
    def from_file(file_path: str) -> "SimulationResult":
        with gzip.open(file_path, 'rb') as handle:
            deserialized_dict = pickle.load(handle)
            deserialized_grids = [deserialize_vtu(grid) for grid in deserialized_dict["grids"]]
            return SimulationResult(
                grids=deserialized_grids,
                eval_values=deserialized_dict["eval_values"]
            )

    def to_file(self, file_path: str):
        serialized_grids = [serialize_vtu(grid) for grid in self.grids]
        with gzip.open(file_path, 'wb') as handle:
            pickle.dump(
                obj={
                    "grids": serialized_grids,
                    "eval_values": self.eval_values
                },
                file=handle,
                protocol=pickle.HIGHEST_PROTOCOL
            )
