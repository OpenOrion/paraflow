from typing import cast
import numpy as np
from matplotlib.path import Path

def check_points_in_polygon(meshgrid, polygon):
    # meshgrid is a tuple of two 2D arrays of x and y coordinates
    # polygon is a list of (x,y) tuples defining the polygon vertices
    
    # flatten the meshgrid into two 1D arrays of x and y coordinates
    x, y = meshgrid
    x_flat = x.flatten()
    y_flat = y.flatten()
    
    # create a Path object from the polygon vertices
    vertices = np.array(polygon)
    path = Path(vertices)
    
    # check if each point in the meshgrid is within the polygon
    points = np.column_stack((x_flat, y_flat))
    mask_flat = cast(np.ndarray, path.contains_points(points))
    return mask_flat.reshape(x.shape)
    