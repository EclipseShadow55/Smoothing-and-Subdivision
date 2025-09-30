# Subdivision and Smoothing
This library contains a class `SubdivisionSmoothing` that provides methods for performing subdivision and smoothing operations on 3D meshes. The class includes methods for Simple Subdivsion, Laplacian Smoothing, and a custom Subdivision Method that takes into account vertex position and edge sharpness. It also includes some helpful methods for making a subdivision or smoothing method yourself.

## Installation
To use this library, you need to have Python installed.
1. Create and navigate to your project directory.
2. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```
3. Install the package using pip:
   For normal installation:
   ```bash
   pip install subdivision-smoothing
   ```
    For development installation:
    ```bash
    pip install subdivision-smoothing[dev]
    ```
4. For development, clone the `subdivision_smoothing\debug` dir from the repository. You can use the `debug.py` script to visualize the intermediate steps of the algorithms.

## Usage
Here's a basic example of how to use the `SubdivisionSmoothing` class:
```python
from subdivision_smoothing import SubdivisionSmoothing
import trimesh

mesh = trimesh.load_mesh('path_to_your_mesh.obj') # Load a mesh
subdivided_mesh = SubdivisionSmoothing.simple_subdivision(mesh, iterations=2) # Use a sudivision method on the mesh
subdivided_mesh.export('subdivided_mesh.obj') # Save or visualize the subdivided mesh
```