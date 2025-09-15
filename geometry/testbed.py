import numpy as np
import pyvista as pv
import networkx as nx

points = np.array(
    [
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1],
    ]
)

dtype = np.dtype([('v1', points.dtype), ('v2', points.dtype)])

points_view = points.view(dtype)

point = np.array([1, 0])

point_view = point.view(dtype)

print(np.where(points_view == point_view)[0])