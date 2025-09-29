"""
Provides subdivision and smoothing algorithms for 3D meshes using the trimesh library. Install normal dependencies with `pip install subdivision_smoothing` and dev dependencies (for using debug_view) with `pip install subdivision_subdivide[dev].

This module offers a collection of static methods for manipulating 3D meshes.
The main methods handle common subdivision and smoothing tasks, while other public methods are intended for advanced or internal use.

Main Provided Methods
---------------------
- :func:`SubdivisionSmoothing.simple_subdivide`: Performs a basic subdivision.
- :func:`SubdivisionSmoothing.simple_smooth`: Applies a modified Laplacian smoothing algorithm.
- :func:`SubdivisionSmoothing.edge_erosion_subdivision`: Performs a specialized subdivision with edge erosion.

.. note::
   Some public methods are primarily intended for internal use and may have less stable APIs and minimal documentation.

Main Exports
------------
- :class:`SubdivisionSmoothing`: A class containing static methods for mesh subdivision and smoothing.

Dependencies
------------
- :mod:`numpy`
- :mod:`trimesh`
- :mod:`numba`

Examples
--------
To perform a simple subdivision on a mesh (cutting each triangle into 4 smaller triangles):

>>> import trimesh
>>> from subdivision_smoothing import SubdivisionSmoothing
>>> mesh = trimesh.creation.box()
>>> subdivided_mesh = SubdivisionSmoothing.simple_subdivide(mesh)

To smooth a mesh using a modified Laplacian smoothing algorithm:

>>> import trimesh
>>> from subdivision_smoothing import SubdivisionSmoothing
>>> mesh = trimesh.creation.box()
>>> mesh = SubdivisionSmoothing.simple_smooth(mesh)

To smooth a mesh using intelligent vertex addition:

>>> import trimesh
>>> from subdivision_smoothing import SubdivisionSmoothing
>>> mesh = trimesh.creation.box()
>>> SubdivisionSmoothing.edge_erosion_subdivision(mesh, alpha=0.5, beta=0.5)
"""

from subdivision_smoothing1.src.subdivision_smoothing.geometry import SubdivisionSmoothing

__all__ = ["SubdivisionSmoothing"]

__version__ = "0.1.0"
__author__ = "EclipseShadow55"