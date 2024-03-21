
from dataclasses import dataclass
import matplotlib.pyplot as plt
import vtk
import numpy as np
import numpy.typing as npt
from typing import Dict, List, Optional, Union
from paraflow.simulation.output import SimulationResult
from vtkmodules.util.numpy_support import numpy_to_vtk, vtk_to_numpy

def get_points(grid: vtk.vtkUnstructuredGrid):
    # Get the points from the grid
    vtk_points = grid.GetPoints()
    return vtk_to_numpy(vtk_points.GetData())

def get_point_data(grid: vtk.vtkUnstructuredGrid, interp_points: npt.NDArray[np.float64]):
    """
    Get interpolated data for points_interpol using vtks built-in interpolation methods
    """

    temp_grid = vtk.vtkUnstructuredGrid()
    temp_grid_vtk_points = vtk.vtkPoints()
    vtk_point_data = numpy_to_vtk(interp_points)
    temp_grid_vtk_points.SetData(vtk_point_data)
    temp_grid.SetPoints(temp_grid_vtk_points)
    locator = vtk.vtkStaticPointLocator()
    locator.SetDataSet(grid)
    locator.BuildLocator()


    interpolator = vtk.vtkProbeFilter()
    interpolator.SetInputData(temp_grid)
    interpolator.SetSourceData(grid)
    interpolator.Update()

    return interpolator.GetOutput().GetPointData()

def export_paraview(sim_result: SimulationResult):
    sim_result.grids
    print('check')

@dataclass
class DataFrame:
    X: npt.NDArray[np.double]
    Y: npt.NDArray[np.double]
    values: Dict[str, npt.NDArray[np.double]]

def get_frames(
    sim_result: SimulationResult,
    property_names: List[str],
    num_pnts: int,
    size: Optional[float] = None,
    offset: Optional[npt.NDArray[np.double]] = None,
):    
    if offset is None:
        offset = np.array([0, 0], dtype=np.double)

    frames: List[DataFrame] = []
    for i in range(len(sim_result.grids)):
        points = get_points(sim_result.grids[i])
        x = points[:, 0]
        y = points[:, 1]
        X, Y = np.meshgrid(
            np.linspace(-size/2 if size else min(x), size/2 if size else max(x), num_pnts) + offset[0],
            np.linspace(-size/2 if size else min(y), size/2 if size else max(y), num_pnts) + offset[1]
        )

        # Convert the meshgrid to a flattened array
        interp_points = np.vstack((X.flatten(), Y.flatten(), np.zeros(num_pnts**2))).T
        point_data = get_point_data(sim_result.grids[i], interp_points)
        
        # Convert the flattened array back to a meshgrid
        values = {}
        for property_name in property_names:
            values[property_name] = np.reshape(vtk_to_numpy(point_data.GetArray(property_name)), X.shape)

        frame = DataFrame(X, Y, values)
        frames.append(frame)
    if len(frames) == 1:
        return frames[0]
    return frames


def display_frame(frames: Union[List[DataFrame], DataFrame], property_name: str):
    frames = frames if isinstance(frames, list) else [frames]

    # Create subplots
    fig, axes = plt.subplots(1, len(frames), figsize=(8, 4))

    if len(frames) == 1:
        axes: npt.NDArray = [axes] # type: ignore

    for i, frame in enumerate(frames):

        # Plot pcolormesh on each subplot
        im = axes[i].pcolormesh(frame.X, frame.Y, frame.values[property_name])

        # Customize colorbars
        fig.colorbar(im, ax=axes[i])

        # Add titles and labels
        axes[i].set_title(f"Zone {i+1} - {property_name}")
        axes[i].set_xlabel('X')
        axes[i].set_ylabel('Y')
        axes[i].axis('equal')

    # Adjust spacing between subplots
    plt.tight_layout()

    # Show the plot
    plt.show()
