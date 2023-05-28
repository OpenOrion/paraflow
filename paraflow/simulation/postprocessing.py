
from dataclasses import dataclass
from matplotlib import cm, pyplot as plt
import vtk
import numpy as np
import numpy.typing as npt
from typing import Dict, List, Optional
from paraflow.simulation.output import SimulationResult
from vtkmodules.util.numpy_support import numpy_to_vtk, vtk_to_numpy


def get_points(grid: vtk.vtkUnstructuredGrid):
    # Get the points from the grid
    vtk_points = grid.GetPoints()
    return vtk_to_numpy(vtk_points.GetData())


def get_point_data(grid: vtk.vtkUnstructuredGrid, interp_points: npt.NDArray[np.float64], interpolation_method: str):
    """
    Get interpolated data for points_interpol using vtks built-in interpolation methods
    """
    kernels = {
        "linear": vtk.vtkLinearKernel(),
        "shepard": vtk.vtkShepardKernel(),
        "voronoi": vtk.vtkVoronoiKernel(),
        "gaussian": vtk.vtkGaussianKernel(),
    }

    temp_grid = vtk.vtkUnstructuredGrid()
    temp_grid_vtk_points = vtk.vtkPoints()
    vtk_point_data = numpy_to_vtk(interp_points)
    temp_grid_vtk_points.SetData(vtk_point_data)
    temp_grid.SetPoints(temp_grid_vtk_points)
    locator = vtk.vtkStaticPointLocator()
    locator.SetDataSet(grid)
    locator.BuildLocator()

    interpolator = vtk.vtkPointInterpolator()
    interpolator.SetInputData(temp_grid)
    interpolator.SetSourceData(grid)
    interpolator.SetKernel(kernels[interpolation_method])
    interpolator.SetLocator(locator)
    interpolator.Update()

    return interpolator.GetOutput().GetPointData()


@dataclass
class DataFrames:
    X: npt.NDArray[np.double]
    Y: npt.NDArray[np.double]
    values: Dict[str, npt.NDArray[np.double]]

def get_frames(
        sim_result: SimulationResult,
        property_names: List[str],
        num_pnts: int,
        size: Optional[float] = None,
        offset: Optional[npt.NDArray[np.double]] = None,
        interpolation_method="voronoi"
):
    if offset is None:
        offset = np.array([0, 0], dtype=np.double)

    points = get_points(sim_result.grid)
    x = points[:, 0]
    y = points[:, 1]
    X, Y = np.meshgrid(
        np.linspace(-size/2 if size else min(x), size/2 if size else max(x), num_pnts) + offset[0],
        np.linspace(-size/2 if size else min(y), size/2 if size else max(y), num_pnts) + offset[1]
    )

    # Convert the meshgrid to a flattened array
    interp_points = np.vstack((X.flatten(), Y.flatten(), np.zeros(num_pnts**2))).T
    point_data = get_point_data(sim_result.grid, interp_points, interpolation_method)
    
    # Convert the flattened array back to a meshgrid
    values = {}
    for property_name in property_names:
        values[property_name] = np.reshape(vtk_to_numpy(point_data.GetArray(property_name)), X.shape)

    return DataFrames(X, Y, values)


def display_frame(frames: DataFrames, property_name: str):
    plt.figure()
    plt.pcolormesh(frames.X, frames.Y, frames.values[property_name])
    plt.colorbar()
    plt.axis('equal')
    plt.show()
