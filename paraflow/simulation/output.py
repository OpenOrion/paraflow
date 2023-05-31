from dataclasses import dataclass
import pickle
import pandas as pd
import gzip
import vtk


def read_vtu_data(filename: str):
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
    grid: vtk.vtkUnstructuredGrid
    "vtu unsctructured grid"

    eval_values: pd.DataFrame
    "values for provided eval attributes"

    @staticmethod
    def from_file(file_path: str) -> "SimulationResult":
        with gzip.open(file_path, 'rb') as handle:
            deserialized_dict = pickle.load(handle)
            deserialized_vtu = deserialize_vtu(deserialized_dict["vtu"])
            return SimulationResult(
                grid=deserialized_vtu,
                eval_values=deserialized_dict["eval_values"]
            )

    def to_file(self, file_path: str):
        serialized_vtu = serialize_vtu(self.grid)
        with gzip.open(file_path, 'wb') as handle:
            pickle.dump(
                obj={
                    "vtu": serialized_vtu,
                    "eval_values": self.eval_values
                },
                file=handle,
                protocol=pickle.HIGHEST_PROTOCOL
            )
